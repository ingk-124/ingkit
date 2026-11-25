"""Contracts for the new independent low-level VMEC API."""
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from ingkit.physics.plasma.vmec import VMECData, rpz_to_xyz


@pytest.fixture
def wout(tmp_path):
    path = tmp_path / 'wout_test.nc'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('radius', 3)
        ds.createDimension('mode', 2)
        for name, value in [('ns', 3), ('nfp', 2), ('lasym__logical__', 0)]:
            ds.createVariable(name, 'i4')[...] = value
        for name, values in [('xm', [0, 1]), ('xn', [0, 2])]:
            ds.createVariable(name, 'f8', ('mode',))[:] = values
        ds.createVariable('xm_nyq', 'f8', ('mode',))[:] = [0, 1]
        ds.createVariable('xn_nyq', 'f8', ('mode',))[:] = [0, 2]
        ds.createVariable('rmnc', 'f8', ('radius', 'mode'))[:] = [
            [2, 0], [2, .25], [2, .5]]
        ds.createVariable('zmns', 'f8', ('radius', 'mode'))[:] = [
            [0, 0], [0, .25], [0, .5]]
        ds.description = 'analytic helical geometry'
    return path


@pytest.mark.parametrize('scalar_s', [False, True])
@pytest.mark.parametrize('scalar_theta', [False, True])
@pytest.mark.parametrize('scalar_phi', [False, True])
def test_forward_geometry_tensor_product(wout, scalar_s, scalar_theta, scalar_phi):
    vmec = VMECData(wout)
    s, theta, phi = np.array([.2, .7]), np.array([.5, 1.]), np.array([.2, .4])
    args = tuple(float(values[0]) if scalar else values
                 for scalar, values in zip((scalar_s, scalar_theta, scalar_phi), (s, theta, phi)))
    s_grid, theta_grid, phi_grid = np.meshgrid(*args, indexing='ij')
    phase = theta_grid - 2*phi_grid
    expected = np.stack((2 + .5*s_grid*np.cos(phase), phi_grid, .5*s_grid*np.sin(phase)), axis=-1)
    scalar_axes = tuple(i for i, scalar in enumerate((scalar_s, scalar_theta, scalar_phi)) if scalar)
    expected = np.squeeze(expected, axis=scalar_axes)

    rpz = vmec.flux_to_rpz(*args)
    assert rpz.shape == expected.shape
    np.testing.assert_allclose(rpz, expected, atol=1e-14)
    xyz = vmec.flux_surface(args[0], theta=args[1], phi=args[2])
    np.testing.assert_allclose(xyz, rpz_to_xyz(expected), atol=1e-14)


def test_readonly_grids_data_and_path(wout):
    vmec = VMECData(wout)
    for value in (vmec.rho_grid, vmec.theta_grid, vmec.wout['rmnc']):
        with pytest.raises(ValueError):
            value.flat[0] = 0.1
    assert isinstance(vmec.metadata['description'], str)
    with pytest.raises(AttributeError):
        vmec.path = Path('other.nc')
    with pytest.raises(AttributeError):
        vmec.rho_grid = np.array([0., .2, 1.])
    with pytest.raises(TypeError):
        vmec.wout['ns'] = 10
    assert vmec.path == wout.resolve()


def test_grid_counts_and_endpoints(wout):
    vmec = VMECData(wout, oversample_theta=2, oversample_phi=3, oversample_rho=1)
    np.testing.assert_allclose(vmec.rho_grid, np.sqrt([0, .5, 1]))
    np.testing.assert_allclose(vmec.theta_grid, np.linspace(0, 2*np.pi, 7))
    np.testing.assert_allclose(vmec.phi_grid, np.linspace(0, 2*np.pi, 16))
    for name in ('oversample_theta', 'oversample_phi', 'oversample_rho'):
        with pytest.raises(ValueError, match=name):
            VMECData(wout, **{name: 0})


def test_missing_geometry_does_not_block_loading(wout):
    with netCDF4.Dataset(wout, 'a') as ds:
        ds.renameVariable('rmnc', 'unrelated')
    vmec = VMECData(wout)
    with pytest.raises(ValueError, match='rmnc'):
        vmec.flux_to_rpz(.5, 0, 0)


@pytest.mark.parametrize('filename', [
    'wout_helical_rfp_zero_beta.nc', 'wout_symmetric_rfp_zero_beta.nc',
    'wout_symmetric_tokamak_zero_beta.nc',
])
def test_real_wout_stored_surface_fourier_sum(filename):
    path = Path(__file__).parents[1] / 'src' / 'ingkit' / 'io' / filename
    vmec = VMECData(path)
    theta, phi = .731, .217
    phase = vmec.wout['xm']*theta - vmec.wout['xn']*phi
    def total(c, s):
        return np.sum(vmec.wout[c][-1]*np.cos(phase)) + (
            np.sum(vmec.wout[s][-1]*np.sin(phase)) if s in vmec.wout else 0)
    expected_r = total('rmnc', 'rmns')
    expected_z = np.sum(vmec.wout['zmns'][-1]*np.sin(phase))
    if 'zmnc' in vmec.wout:
        expected_z += np.sum(vmec.wout['zmnc'][-1]*np.cos(phase))
    np.testing.assert_allclose(vmec.flux_to_rpz(1, theta, phi),
                               [expected_r, phi, expected_z], atol=1e-13)


def test_surface_shapes_sections_and_periodic_closure(wout):
    vmec = VMECData(wout, oversample_theta=2, oversample_phi=2)
    theta = np.array([0., np.pi / 2, np.pi])
    section = vmec.flux_surface([.2, 1.], theta=theta, phi=0., coordinates='rpz')
    assert section.shape == (2, 3, 3)
    np.testing.assert_allclose(section[..., 0],
                               2 + .5*np.array([.2, 1.])[:, None]*np.cos(theta))
    np.testing.assert_allclose(section[..., 2],
                               .5*np.array([.2, 1.])[:, None]*np.sin(theta))
    assert vmec.flux_surface(.5).shape == (7, 11, 3)
    assert vmec.flux_surface([.5]).shape == (1, 7, 11, 3)
    closed = vmec.flux_surface()
    assert closed.shape == (7, 11, 3)
    np.testing.assert_allclose(closed[0], closed[-1], atol=1e-14)
    np.testing.assert_allclose(closed[:, 0], closed[:, -1], atol=1e-14)
    for kwargs in ({'s': [[.5]]}, {'phi': [np.nan]},
                   {'coordinates': 'invalid'}, {'s': 1.1}):
        with pytest.raises(ValueError):
            vmec.flux_surface(**kwargs)


def test_flux_surface_rejects_multidimensional_theta(wout):
    vmec = VMECData(wout)
    with pytest.raises(ValueError):
        vmec.flux_surface(theta=[[0.]])


def test_axis_asymmetric_fourier_sign_and_closure(wout):
    with netCDF4.Dataset(wout, 'a') as ds:
        ds['lasym__logical__'][...] = 1
        for name, values in [('raxis_cc', [2., .1]), ('raxis_cs', [0., .2]),
                             ('zaxis_cc', [.05, .3]), ('zaxis_cs', [0., .4])]:
            ds.createVariable(name, 'f8', ('mode',))[:] = values
    vmec = VMECData(wout, oversample_phi=2)
    # -n*nfp*phi = -pi/2: asymmetric sine coefficients must enter negatively.
    axis = vmec.magnetic_axis(phi=np.pi / 4, coordinates='rpz')
    np.testing.assert_allclose(axis, [[1.8, np.pi/4, -.35]], atol=1e-14)
    axis = vmec.magnetic_axis()
    assert axis.shape == (11, 3)
    np.testing.assert_allclose(axis[0], axis[-1], atol=1e-14)
    for kwargs in ({'phi': [np.inf]}, {'phi': [[0.]]}, {'coordinates': 'invalid'}):
        with pytest.raises(ValueError):
            vmec.magnetic_axis(**kwargs)


def test_axis_missing_and_symmetric_omissions(wout):
    vmec = VMECData(wout)
    with pytest.raises(ValueError, match='raxis_cc'):
        vmec.magnetic_axis()
    with netCDF4.Dataset(wout, 'a') as ds:
        ds.createDimension('axis_mode', 1)
        ds.createVariable('raxis_cc', 'f8', ('axis_mode',))[:] = [2.]
        ds.createVariable('zaxis_cs', 'f8', ('axis_mode',))[:] = [0.]
    vmec = VMECData(wout)
    rpz = vmec.magnetic_axis(phi=[0., .4], coordinates='rpz')
    np.testing.assert_allclose(rpz, [[2., 0., 0.], [2., .4, 0.]])


@pytest.mark.parametrize('filename', [
    'wout_helical_rfp_zero_beta.nc', 'wout_symmetric_rfp_zero_beta.nc',
    'wout_symmetric_tokamak_zero_beta.nc',
])
def test_real_axis_matches_degenerate_surface_and_one_period_rotation(filename):
    path = Path(__file__).parents[1] / 'src' / 'ingkit' / 'io' / filename
    vmec = VMECData(path)
    phi = np.array([0., .17, .51, 2*np.pi])
    axis = vmec.magnetic_axis(phi=phi, coordinates='rpz')
    for theta in (0., .7, 2.):
        np.testing.assert_allclose(axis, vmec.flux_to_rpz(0., theta, phi), atol=1e-12)
    one_period = vmec.magnetic_axis(phi=[0, 2*np.pi/vmec.nfp], coordinates='rpz')
    np.testing.assert_allclose(one_period[0, [0, 2]], one_period[-1, [0, 2]],
                               atol=1e-12)
    assert one_period[-1, 1] == 2*np.pi/vmec.nfp


@pytest.fixture
def lcfs_wout(tmp_path):
    """Elliptical LCFS centered on the axis; optional non-star-shaped dent."""
    def create(*, a=.5, b=.25, dent=0., axis_radius=2.):
        path = tmp_path / 'wout_lcfs.nc'
        with netCDF4.Dataset(path, 'w') as ds:
            ds.createDimension('radius', 2)
            ds.createDimension('mode', 3)
            ds.createDimension('axis_mode', 1)
            for name, value in [('ns', 2), ('nfp', 2), ('lasym__logical__', 0)]:
                ds.createVariable(name, 'i4')[...] = value
            ds.createVariable('xm', 'f8', ('mode',))[:] = [0, 1, 2]
            ds.createVariable('xn', 'f8', ('mode',))[:] = [0, 0, 0]
            ds.createVariable('xm_nyq', 'f8', ('mode',))[:] = [0, 1, 2]
            ds.createVariable('xn_nyq', 'f8', ('mode',))[:] = [0, 0, 0]
            ds.createVariable('rmnc', 'f8', ('radius', 'mode'))[:] = [
                [axis_radius, 0, 0], [axis_radius, a, dent]]
            ds.createVariable('zmns', 'f8', ('radius', 'mode'))[:] = [
                [0, 0, 0], [0, b, 0]]
            ds.createVariable('raxis_cc', 'f8', ('axis_mode',))[:] = [axis_radius]
            ds.createVariable('zaxis_cs', 'f8', ('axis_mode',))[:] = [0]
        return path
    return create


def test_lcfs_angle_derivative_matches_ellipse(lcfs_wout):
    a, b = .5, .25
    vmec = VMECData(lcfs_wout(a=a, b=b), oversample_theta=13, oversample_phi=3)
    theta = np.array([0., .3, np.pi/2, np.pi, 2*np.pi])
    derivative = vmec.lcfs_angle_derivative(theta, [0., .7])
    expected = a*b / ((a*np.cos(theta))**2 + (b*np.sin(theta))**2)
    assert derivative.shape == (5, 2)
    np.testing.assert_allclose(derivative, np.broadcast_to(expected[:, None], (5, 2)))
    np.testing.assert_array_equal(vmec.is_star_shaped(), np.ones(vmec.n_phi, dtype=bool))
    assert vmec.lcfs_angle_derivative(0., .2).shape == (1, 1)
    np.testing.assert_array_equal(vmec.theta_grid, np.linspace(0, 2*np.pi, 66))


@pytest.mark.parametrize('scale', [1e-3, 1., 1e3])
def test_star_shape_is_scale_and_orientation_independent(lcfs_wout, scale):
    vmec = VMECData(lcfs_wout(a=.5*scale, b=-.25*scale, axis_radius=2*scale),
                   oversample_theta=13)
    assert np.all(vmec.lcfs_angle_derivative() < 0)
    assert vmec.is_star_shaped().all()


def test_star_shape_detects_fold(lcfs_wout):
    # A second harmonic produces a fold while the LCFS remains a simple curve.
    vmec = VMECData(lcfs_wout(dent=.4), oversample_theta=13, oversample_phi=1)
    np.testing.assert_array_equal(vmec.is_star_shaped(), np.zeros(vmec.n_phi, dtype=bool))


def test_star_shape_strict_signs_per_section(monkeypatch):
    vmec = VMECData.__new__(VMECData)
    # Each column is a phi section: positive, negative, mixed, zero, tiny positive.
    derivative = np.array([[1., -1., 1., 0., 1e-16],
                           [2., -2., -1., 1., 2e-16]])
    monkeypatch.setattr(VMECData, 'lcfs_angle_derivative',
                        lambda self, theta, phi, *, s: derivative)
    np.testing.assert_array_equal(vmec.is_star_shaped(),
                                  [True, True, False, False, True])


def test_star_shape_axis_intersection_and_missing_axis(lcfs_wout, wout):
    vmec = VMECData(lcfs_wout(a=0., b=.5), oversample_theta=13)
    with pytest.raises(ValueError, match='intersects'):
        vmec.is_star_shaped()
    with pytest.raises(ValueError, match='raxis_cc'):
        VMECData(wout).is_star_shaped()


@pytest.mark.parametrize('filename', [
    'wout_helical_rfp_zero_beta.nc', 'wout_symmetric_rfp_zero_beta.nc',
    'wout_symmetric_tokamak_zero_beta.nc',
])
def test_real_lcfs_fourier_derivative_matches_angle_difference(filename):
    path = Path(__file__).parents[1] / 'src' / 'ingkit' / 'io' / filename
    vmec = VMECData(path, oversample_theta=2)
    theta, phi = np.array([.13, 1.2, 3.7]), np.array([.07, .23])
    step = 1e-5
    axis = vmec.magnetic_axis(phi=phi, coordinates='rpz')
    before = vmec.flux_surface(theta=theta-step, phi=phi, coordinates='rpz')
    after = vmec.flux_surface(theta=theta+step, phi=phi, coordinates='rpz')
    # Angle difference via cross/dot avoids atan2's branch cut.
    x0, y0 = before[..., 0]-axis[:, 0], before[..., 2]-axis[:, 2]
    x1, y1 = after[..., 0]-axis[:, 0], after[..., 2]-axis[:, 2]
    expected = np.arctan2(x0*y1-y0*x1, x0*x1+y0*y1)/(2*step)
    np.testing.assert_allclose(vmec.lcfs_angle_derivative(theta, phi), expected,
                               rtol=1e-7, atol=1e-8)
    assert vmec.is_star_shaped().shape == (len(vmec.phi_grid),)


def test_inside_circle_shapes_and_boundary(lcfs_wout):
    vmec = VMECData(lcfs_wout(a=.5, b=.5), oversample_theta=13, oversample_phi=1)
    points = np.array([[[2., 0., 0.], [2.5, 0., 0.]],
                       [[2.4, .3, 0.], [2.6, .7, 0.]]])
    expected = [[True, True], [True, False]]
    np.testing.assert_array_equal(vmec.is_inside_lcfs(points), expected)
    assert vmec.is_inside_lcfs(points[0, 0]).shape == ()
    assert vmec.is_inside_lcfs(np.empty((0, 3))).shape == (0,)


def test_inside_ellipse_and_reverse_orientation(lcfs_wout):
    for b in (.25, -.25):
        vmec = VMECData(lcfs_wout(b=b), oversample_theta=26, oversample_phi=4)
        alpha = np.linspace(-2*np.pi, 4*np.pi, 90)
        phi = np.linspace(-3*np.pi, 3*np.pi, 90)
        boundary = 1/np.sqrt((np.cos(alpha)/.5)**2 + (np.sin(alpha)/abs(b))**2)
        for factor, expected in ((.98, True), (1.02, False)):
            rpz = np.column_stack((2+factor*boundary*np.cos(alpha), phi,
                                   factor*boundary*np.sin(alpha)))
            np.testing.assert_array_equal(vmec.is_inside_lcfs(rpz), np.full(90, expected))
            np.testing.assert_array_equal(vmec.is_inside_lcfs(rpz_to_xyz(rpz), coordinates='xyz'),
                                           vmec.is_inside_lcfs(rpz))


def test_inside_varying_phi_and_moving_axis(tmp_path):
    path = tmp_path / 'wout_varying.nc'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('radius', 2)
        ds.createDimension('mode', 5)
        ds.createDimension('axis_mode', 2)
        for name, value in [('ns', 2), ('nfp', 2), ('lasym__logical__', 0)]:
            ds.createVariable(name, 'i4')[...] = value
        for name, values in [('xm', [0, 0, 1, 1, 1]), ('xn', [0, 2, 0, 2, -2]),
                             ('xm_nyq', [0, 0, 1, 1, 1]), ('xn_nyq', [0, 2, 0, 2, -2])]:
            ds.createVariable(name, 'f8', ('mode',))[:] = values
        ds.createVariable('rmnc', 'f8', ('radius', 'mode'))[:] = [
            [2, .1, 0, 0, 0], [2, .1, .5, .05, .05]]
        ds.createVariable('zmns', 'f8', ('radius', 'mode'))[:] = [
            [0, -.05, 0, 0, 0], [0, -.05, .5, .05, .05]]
        ds.createVariable('raxis_cc', 'f8', ('axis_mode',))[:] = [2, .1]
        ds.createVariable('zaxis_cs', 'f8', ('axis_mode',))[:] = [0, -.05]
    vmec = VMECData(path, oversample_theta=13, oversample_phi=32)
    # Includes both angular seams, arbitrary phi, and multiple field periods.
    alpha = np.array([-1e-8, 0., 1e-8, 2., 4., 6.])
    phi = np.array([-1e-8, 0., 1e-8, np.pi-1e-8, np.pi+1e-8, 7.3])
    radius = .5 + .1*np.cos(2*phi)
    for factor, expected in ((.99, True), (1.01, False)):
        points = np.column_stack((2+.1*np.cos(2*phi)+factor*radius*np.cos(alpha), phi,
                                  .05*np.sin(2*phi)+factor*radius*np.sin(alpha)))
        np.testing.assert_array_equal(vmec.is_inside_lcfs(points), np.full(6, expected))
        shifted = points.copy()
        shifted[:, 1] += 2*np.pi/vmec.nfp
        np.testing.assert_array_equal(vmec.is_inside_lcfs(points), vmec.is_inside_lcfs(shifted))


def test_inside_invalid_input_and_fold(lcfs_wout):
    vmec = VMECData(lcfs_wout(dent=.4), oversample_theta=13)
    with pytest.raises(ValueError, match='not star-shaped'):
        vmec.is_inside_lcfs([2., 0., 0.])
    for points, coordinates in (([1., 2.], 'rpz'), ([np.nan, 0, 0], 'xyz'),
                                 ([-1., 0, 0], 'rpz'), ([2., 0, 0], 'invalid')):
        with pytest.raises(ValueError):
            vmec.is_inside_lcfs(points, coordinates=coordinates)

def test_points_to_flux_circle_axis_boundary_and_tolerance(lcfs_wout):
    vmec = VMECData(lcfs_wout(a=.5, b=.5))
    tol = 1e-3
    points = [[2., 0., 0.], [2.5, .3, 0.], [2.50025, -1., 0.], [2.501, 0., 0.]]
    np.testing.assert_allclose(vmec.points_to_flux(points, tol=tol), [0., 1., 1., np.nan])
    np.testing.assert_array_equal(vmec.is_inside_lcfs(points, tol=tol),
                                  [True, True, True, False])
    assert vmec.points_to_flux(points[0]).shape == ()
    assert vmec.points_to_flux(np.empty((0, 3))).shape == (0,)


def test_points_to_flux_between_surfaces_and_periodic_angles(lcfs_wout):
    vmec = VMECData(lcfs_wout(), oversample_theta=26, oversample_rho=20)
    s = np.array([.07, .32, .73, .95])
    theta = np.array([-1e-7, .61, 2*np.pi+1e-7])
    phi = np.array([-1e-7, .4, np.pi+1e-7, 4*np.pi+.2])
    points = vmec.flux_surface(s, theta=theta, phi=phi, coordinates='rpz')
    rpz = vmec.points_to_flux(points)
    xyz = vmec.points_to_flux(rpz_to_xyz(points), coordinates='xyz')
    assert rpz.shape == (s.size, theta.size, phi.size)
    np.testing.assert_allclose(rpz, np.broadcast_to(s[:, None, None], rpz.shape), atol=3e-3)
    np.testing.assert_allclose(xyz, rpz, atol=1e-13)


def test_s_on_rz_grid_orientation_and_outside(lcfs_wout):
    vmec = VMECData(lcfs_wout(a=.5, b=.5), oversample_rho=30)
    R = np.array([2., 2.3, 2.6])
    Z = np.array([0., .2])
    s = vmec.s_on_rz_grid(R, Z, phi=-.3)
    expected = np.hypot(R[:, None]-2, Z[None, :]) / .5
    expected[expected > 1] = np.nan
    assert s.shape == (R.size, Z.size)
    np.testing.assert_allclose(s, expected, atol=3e-3)


@pytest.mark.parametrize("b", [.5, -.5])
def test_inverse_theta_round_trip_at_seams(lcfs_wout, b):
    vmec = VMECData(lcfs_wout(a=.5, b=b), oversample_theta=13, oversample_rho=20)
    s_expected = np.array([[.0005, .32], [.73, 1.]])
    theta_expected = np.array([[-1e-7, 2*np.pi + 1e-7], [3.1, 6.2]])
    phi = np.array([[-1e-7, np.pi + 1e-7], [2*np.pi - 1e-7, -3*np.pi + .2]])
    points = np.stack((2 + .5*s_expected*np.cos(theta_expected), phi,
                       b*s_expected*np.sin(theta_expected)), axis=-1)
    s, theta = vmec.points_to_flux(points, return_theta=True)
    s_xyz, theta_xyz = vmec.points_to_flux(rpz_to_xyz(points), coordinates="xyz",
                                         return_theta=True)
    assert s.shape == theta.shape == s_expected.shape
    np.testing.assert_allclose(s, s_expected, atol=2e-3)
    np.testing.assert_allclose(s_xyz, s, atol=1e-13)
    np.testing.assert_allclose(np.exp(1j*theta_xyz), np.exp(1j*theta), atol=1e-13)
    np.testing.assert_allclose(np.exp(1j*theta), np.exp(1j*theta_expected), atol=1e-12)
    restored = np.array([vmec.flux_to_rpz(s_i, theta_i, phi_i)
                         for s_i, theta_i, phi_i in zip(s.flat, theta.flat, phi.flat)])
    np.testing.assert_allclose(restored.reshape(points.shape), points, atol=1e-3)


@pytest.fixture
def field_wout(lcfs_wout):
    path = lcfs_wout(a=.5, b=.5)
    with netCDF4.Dataset(path, "a") as ds:
        # Constant contravariant B and J on an axisymmetric circular surface.
        # gmnc=-2 and stored currents include this signed Jacobian.
        for name, value in (("bsupumnc", .4), ("bsupvmnc", 1.5), ("bmnc", 4.),
                            ("gmnc", -2.)):
            ds.createVariable(name, "f8", ("radius", "mode"))[:] = [
                [1000, 0, 0], [value, 0, 0]]
        for name, value in (("currumnc", -6.), ("currvmnc", 4.)):
            ds.createVariable(name, "f8", ("radius", "mode"))[:] = [[value, 0, 0]] * 2
        ds.createVariable("presf", "f8", ("radius",))[:] = [200., 50.]
    return path


def test_flux_fields_physical_components_and_basis(field_wout):
    vmec = VMECData(field_wout)
    flux_points = np.array([[[0., .4, -.6], [.3, .7, .25]],
                            [[.6, 1.2, .4], [1., 2., 1.1]]])
    s, theta, phi = np.moveaxis(flux_points, -1, 0)
    R = 2 + .5*s*np.cos(theta)
    tangent_theta = np.stack((-.5*s*np.sin(theta), np.zeros_like(s),
                              .5*s*np.cos(theta)), axis=-1)
    tangent_phi = np.stack((np.zeros_like(s), R, np.zeros_like(s)), axis=-1)
    B = .4*tangent_theta + 1.5*tangent_phi
    J = 3*tangent_theta - 2*tangent_phi
    J[0, 0] = np.nan
    for method, expected in ((vmec.magnetic_field, B), (vmec.current_density, J)):
        np.testing.assert_allclose(method(flux_points), expected, atol=1e-13)
        radial, toroidal, vertical = np.moveaxis(expected, -1, 0)
        cartesian = np.stack((radial*np.cos(phi) - toroidal*np.sin(phi),
                              radial*np.sin(phi) + toroidal*np.cos(phi), vertical), axis=-1)
        np.testing.assert_allclose(method(flux_points, basis="cartesian"), cartesian, atol=1e-13)
    np.testing.assert_allclose(vmec.magnetic_field_strength(flux_points), np.full(s.shape, 4.))


def test_real_space_fields_at_axis_lcfs_and_outside(field_wout):
    vmec = VMECData(field_wout, oversample_rho=2)
    points = np.array([[2., -.5, 0.], [2.125, .4, 0.], [2.5, 1., 0.], [2.6, 0., 0.]])
    s, theta = vmec.points_to_flux(points, return_theta=True)
    np.testing.assert_allclose(s, [0., .25, 1., np.nan], atol=1e-13)
    np.testing.assert_allclose(theta, [0., 0., 0., np.nan], atol=1e-13)
    B = [[0., 3., 0.], [0., 3.1875, .05], [0., 3.75, .2], [np.nan]*3]
    J = [[np.nan]*3, [0., -4.25, .375], [0., -5., 1.5], [np.nan]*3]
    for method, expected in ((vmec.magnetic_field_at, B), (vmec.current_at, J),
                             (vmec.magnetic_field_strength_at, [4., 4., 4., np.nan]),
                             (vmec.pressure_at, [200., 162.5, 50., np.nan])):
        np.testing.assert_allclose(method(points), expected, atol=1e-12)
        np.testing.assert_allclose(method(rpz_to_xyz(points), coordinates="xyz"),
                                   expected, atol=1e-12)


@pytest.mark.parametrize("filename", [
    "wout_helical_rfp_zero_beta.nc", "wout_symmetric_rfp_zero_beta.nc",
    "wout_symmetric_tokamak_zero_beta.nc",
])
def test_real_wout_field_norm_matches_stored_strength(filename):
    path = Path(__file__).parents[1] / "src" / "ingkit" / "io" / filename
    vmec = VMECData(path)
    points = [[.14, .3, .2], [.37, 3.2, .45], [.8, 2*np.pi - 1e-6, np.pi - 1e-6],
              [.92, 4.4, 2*np.pi + .2]]
    B = vmec.magnetic_field(points)
    strength = vmec.magnetic_field_strength(points)
    np.testing.assert_allclose(np.linalg.norm(B, axis=-1), strength, rtol=1e-3)
    assert np.isfinite(vmec.current_density(points)).all()
