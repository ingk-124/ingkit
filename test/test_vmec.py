from pathlib import Path

import numpy as np
import pytest

from ingkit.io.read_vmec import VMECData
from ingkit.myplot.vmec_plot import plot_cylindrical_field, plot_flux_surfaces_3d


DATA_DIR = Path(__file__).parents[1] / "src" / "ingkit" / "io"
WOUT_FILES = [
    "wout_helical_rfp_zero_beta.nc",
    "wout_symmetric_rfp_zero_beta.nc",
    "wout_symmetric_tokamak_zero_beta.nc",
]


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
