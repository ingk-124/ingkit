from pathlib import Path

import numpy as np
import pytest

from ingkit.io.read_vmec import VMECData, VMECValidationError
from ingkit.myplot.vmec_plot import plot_cylindrical_field, plot_flux_surfaces_3d


DATA_DIR = Path(__file__).parents[1] / "src" / "ingkit" / "io"
WOUT_FILES = [
    "wout_helical_rfp_zero_beta.nc",
    "wout_symmetric_rfp_zero_beta.nc",
    "wout_symmetric_tokamak_zero_beta.nc",
]


def circular_equilibrium() -> VMECData:
    """Return an analytic torus whose minor radius is linear in s."""
    s = np.array([0.0, 0.4, 1.0])
    rmnc = np.column_stack((np.full(s.size, 2.0), 0.5 * s))
    zmns = np.column_stack((np.zeros(s.size), 0.5 * s))
    return VMECData.from_fourier(
        s=s,
        xm=np.array([0, 1]),
        xn=np.array([0, 0]),
        rmnc=rmnc,
        zmns=zmns,
    )


def test_forward_map_scalar_paired_and_broadcast_inputs():
    vmec = circular_equilibrium()

    scalar = vmec.to_cylindrical(0.5, np.pi / 2, 0.25)
    assert scalar.R.shape == ()
    assert scalar.Z.shape == ()
    np.testing.assert_allclose((scalar.R, scalar.Z), (2.0, 0.25), atol=1e-14)

    paired = vmec.to_cylindrical(
        np.array([0.2, 0.8]),
        np.array([0.0, np.pi]),
        np.array([0.1, 0.2]),
    )
    assert paired.R.shape == (2,)
    np.testing.assert_allclose(paired.R, [2.1, 1.6], atol=1e-14)

    broadcast = vmec.to_cylindrical(
        np.array([[0.2], [0.8]]),
        np.array([[0.0, np.pi / 2, np.pi]]),
        0.0,
    )
    assert broadcast.R.shape == (2, 3)
    assert broadcast.Z.shape == (2, 3)
    np.testing.assert_allclose(broadcast.R[0], [2.1, 2.0, 1.9], atol=1e-14)


def test_forward_map_interpolates_coefficients_in_s():
    vmec = circular_equilibrium()
    coordinates = vmec.to_cylindrical(0.7, 0.0, 0.0)
    np.testing.assert_allclose(coordinates.R, 2.35, atol=1e-14)
    np.testing.assert_allclose(coordinates.Z, 0.0, atol=1e-14)


def test_forward_map_periodicity_and_vmec_phase_convention():
    s = np.array([0.0, 1.0])
    # alpha = theta - 2*phi.  The nonzero asymmetric terms fix both signs.
    vmec = VMECData.from_fourier(
        s=s,
        xm=np.array([0, 1]),
        xn=np.array([0, 2]),
        nfp=2,
        lasym=True,
        rmnc=np.array([[2.0, 0.0], [2.0, 0.4]]),
        rmns=np.array([[0.0, 0.0], [0.0, 0.2]]),
        zmnc=np.array([[0.0, 0.0], [0.0, 0.1]]),
        zmns=np.array([[0.0, 0.0], [0.0, 0.3]]),
    )
    theta = 0.7
    phi = 0.2
    alpha = theta - 2.0 * phi
    actual = vmec.to_cylindrical(1.0, theta, phi)
    np.testing.assert_allclose(
        actual.R, 2.0 + 0.4 * np.cos(alpha) + 0.2 * np.sin(alpha)
    )
    np.testing.assert_allclose(
        actual.Z, 0.1 * np.cos(alpha) + 0.3 * np.sin(alpha)
    )
    theta_period = vmec.to_cylindrical(1.0, theta + 2 * np.pi, phi)
    field_period = vmec.to_cylindrical(1.0, theta, phi + np.pi)
    np.testing.assert_allclose(actual.R, theta_period.R, atol=1e-14)
    np.testing.assert_allclose(actual.Z, theta_period.Z, atol=1e-14)
    np.testing.assert_allclose(actual.R, field_period.R, atol=1e-14)
    np.testing.assert_allclose(actual.Z, field_period.Z, atol=1e-14)


def test_cartesian_and_cylindrical_maps_are_consistent():
    vmec = circular_equilibrium()
    s = np.array([0.3, 0.9])
    theta = np.array([0.4, 2.1])
    phi = np.array([-0.2, 1.3])
    cylindrical = vmec.to_cylindrical(s, theta, phi)
    cartesian = vmec.to_cartesian(s, theta, phi)
    np.testing.assert_allclose(
        cartesian.x, cylindrical.R * np.cos(cylindrical.phi)
    )
    np.testing.assert_allclose(
        cartesian.y, cylindrical.R * np.sin(cylindrical.phi)
    )
    np.testing.assert_allclose(cartesian.z, cylindrical.Z)


def test_forward_inverse_roundtrip_with_broadcast_arrays():
    vmec = circular_equilibrium()
    s = np.array([[0.25], [0.65]])
    theta = np.array([[0.3, 1.7, 4.8]])
    phi = np.array([[0.2], [5.9]])
    cartesian = vmec.to_cartesian(s, theta, phi)

    result = vmec.from_cartesian(*cartesian, tol=1e-10)

    assert result.s.shape == (2, 3)
    assert result.valid.all()
    assert np.all(result.status == "converged")
    np.testing.assert_allclose(result.s, np.broadcast_to(s, (2, 3)), atol=1e-9)
    np.testing.assert_allclose(
        np.exp(1j * result.theta),
        np.exp(1j * np.broadcast_to(theta, (2, 3))),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.exp(1j * result.phi),
        np.exp(1j * np.broadcast_to(phi, (2, 3))),
        atol=1e-12,
    )
    np.testing.assert_allclose(result.rho, np.sqrt(result.s), atol=1e-14)
    assert np.all(result.residual <= 1e-10)


def test_inverse_distinguishes_outside_axis_and_nonconvergence():
    vmec = circular_equilibrium()

    outside = vmec.from_cartesian(3.0, 0.0, 0.0)
    assert not outside.valid
    assert outside.status == "outside_lcfs"
    assert np.isnan(outside.s)
    assert outside.residual > 0.49

    axis_xyz = vmec.to_cartesian(0.0, 1.8, -0.3)
    axis = vmec.from_cartesian(*axis_xyz)
    assert axis.valid
    assert axis.status == "axis"
    assert axis.s == 0.0
    assert axis.rho == 0.0
    assert axis.theta == 0.0
    np.testing.assert_allclose(axis.phi, np.mod(-0.3, 2 * np.pi))

    point = vmec.to_cartesian(0.63, 1.234, 0.4)
    failed = vmec.from_cartesian(*point, max_nfev=1, tol=1e-12)
    assert not failed.valid
    assert failed.status == "nonconverged"
    assert np.isfinite(failed.residual)


def test_forward_map_rejects_lcfs_extrapolation_by_default():
    vmec = circular_equilibrium()
    with pytest.raises(ValueError, match="outside"):
        vmec.to_cylindrical(1.01, 0.0, 0.0)
    clipped = vmec.to_cylindrical(1.01, 0.0, 0.0, bounds="clip")
    np.testing.assert_allclose(clipped.R, 2.5)


def test_vmec_validation_rejects_bad_radial_grid_and_asymmetric_omissions():
    coefficients = np.ones((3, 1))
    with pytest.raises(VMECValidationError, match="strictly increasing"):
        VMECData.from_fourier(
            s=np.array([0.0, 0.8, 0.7]),
            xm=np.array([0]),
            xn=np.array([0]),
            rmnc=coefficients,
            zmns=coefficients,
        )
    with pytest.raises(VMECValidationError, match="rmns"):
        VMECData.from_fourier(
            s=np.array([0.0, 0.5, 1.0]),
            xm=np.array([0]),
            xn=np.array([0]),
            rmnc=coefficients,
            zmns=coefficients,
            lasym=True,
        )


def test_vmec_validation_rejects_coefficient_shape_and_nfp_convention():
    with pytest.raises(VMECValidationError, match="shape"):
        VMECData.from_fourier(
            s=np.array([0.0, 1.0]),
            xm=np.array([0, 1]),
            xn=np.array([0, 0]),
            rmnc=np.ones((2, 1)),
            zmns=np.ones((2, 2)),
        )
    with pytest.raises(VMECValidationError, match="multiples"):
        VMECData.from_fourier(
            s=np.array([0.0, 1.0]),
            xm=np.array([0]),
            xn=np.array([1]),
            nfp=2,
            rmnc=np.ones((2, 1)),
            zmns=np.ones((2, 1)),
        )


def test_symmetric_equilibrium_fills_only_asymmetric_coefficients_with_zero():
    vmec = circular_equilibrium()
    assert not vmec.lasym
    np.testing.assert_array_equal(vmec.data["rmns"], 0.0)
    np.testing.assert_array_equal(vmec.data["zmnc"], 0.0)


@pytest.mark.parametrize("filename", WOUT_FILES)
def test_vmec_coordinate_and_field_calculations(filename):
    vmec = VMECData(DATA_DIR / filename)
    u = np.linspace(0.0, 2.0 * np.pi, 9)
    v = np.linspace(0.0, 2.0 * np.pi / vmec.nfp, 5)
    expected_shape = (vmec.ns, u.size, v.size)

    coordinates, *_ = vmec.get_derivatives(u, v)
    _, cylindrical_field = vmec.get_B_field_cylindrical(u, v)
    _, cylindrical_current = vmec.get_current_cylindrical(u, v)
    cartesian_coordinates, cartesian_field = vmec.get_B_field_cartesian(u, v)

    for array in (*coordinates, *cylindrical_field, *cylindrical_current,
                  *cartesian_coordinates, *cartesian_field):
        assert array.shape == expected_shape
        assert np.isfinite(array).all()

    # Fourier surfaces are periodic in both angular coordinates.
    np.testing.assert_allclose(coordinates[0][:, 0], coordinates[0][:, -1], atol=1e-12)
    np.testing.assert_allclose(coordinates[1][:, 0], coordinates[1][:, -1], atol=1e-12)
    np.testing.assert_allclose(coordinates[0][:, :, 0], coordinates[0][:, :, -1], atol=1e-12)
    np.testing.assert_allclose(coordinates[1][:, :, 0], coordinates[1][:, :, -1], atol=1e-12)


@pytest.mark.parametrize("filename", WOUT_FILES)
def test_real_wout_forward_map_matches_explicit_vmec_fourier_sum(filename):
    vmec = VMECData(DATA_DIR / filename)
    surface = min(13, vmec.ns - 1)
    theta = 0.731
    phi = 0.217
    phase = vmec.xm * theta - vmec.xn * phi
    expected_r = np.sum(
        vmec.data["rmnc"][surface] * np.cos(phase)
        + vmec.data["rmns"][surface] * np.sin(phase)
    )
    expected_z = np.sum(
        vmec.data["zmnc"][surface] * np.cos(phase)
        + vmec.data["zmns"][surface] * np.sin(phase)
    )

    actual = vmec.to_cylindrical(vmec.s_arr[surface], theta, phi)

    np.testing.assert_allclose(actual.R, expected_r, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(actual.Z, expected_z, rtol=0.0, atol=1e-13)


def test_existing_tensor_grid_api_matches_forward_map():
    vmec = VMECData(DATA_DIR / WOUT_FILES[0])
    theta = np.array([0.2, 1.1, 3.7])
    phi = np.array([0.1, 0.8])
    legacy, *_ = vmec.get_derivatives(theta, phi)
    s_grid = vmec.s_arr[:, None, None]
    theta_grid = theta[None, :, None]
    phi_grid = phi[None, None, :]

    modern = vmec.to_cylindrical(s_grid, theta_grid, phi_grid)

    np.testing.assert_allclose(modern.R, legacy[0], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(modern.Z, legacy[1], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(modern.phi, legacy[2], rtol=0.0, atol=0.0)


def test_real_wout_forward_inverse_regression():
    vmec = VMECData(DATA_DIR / "wout_symmetric_tokamak_zero_beta.nc")
    expected_s = np.array([0.25, 0.7])
    expected_theta = np.array([0.7, 2.2])
    expected_phi = np.array([0.2, 1.1])
    cartesian = vmec.to_cartesian(expected_s, expected_theta, expected_phi)

    result = vmec.from_cartesian(*cartesian, tol=1e-8)

    assert result.valid.all()
    np.testing.assert_allclose(result.s, expected_s, atol=2e-8)
    np.testing.assert_allclose(result.theta, expected_theta, atol=2e-8)
    np.testing.assert_allclose(result.phi, expected_phi, atol=1e-14)


def test_vmec_rejects_non_1d_angle_arrays():
    vmec = VMECData(DATA_DIR / WOUT_FILES[0])

    with pytest.raises(ValueError, match="one-dimensional"):
        vmec.get_derivatives(np.zeros((2, 2)), np.zeros(2))


def test_vmec_plot_helpers_return_figures():
    pytest.importorskip("plotly")
    vmec = VMECData(DATA_DIR / WOUT_FILES[0])

    figure_3d = plot_flux_surfaces_3d(
        vmec, surface_indices=[5, -1], nu=9, nv=9, vector_step=4
    )
    figure_2d, axes_2d = plot_cylindrical_field(
        vmec, nu=17, radial_step=10, angular_step=4
    )

    assert len(figure_3d.data) == 3  # two surfaces and one vector trace
    assert axes_2d.figure is figure_2d
    assert axes_2d.collections


def test_vmec_plot_rejects_invalid_surface_index():
    pytest.importorskip("plotly")
    vmec = VMECData(DATA_DIR / WOUT_FILES[0])

    with pytest.raises(IndexError, match="outside"):
        plot_flux_surfaces_3d(vmec, surface_indices=[vmec.ns])
