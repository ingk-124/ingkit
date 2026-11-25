"""Plotting integration with the low-level VMEC API."""
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ingkit.myplot.vmec_plot import plot_cylindrical_field, plot_flux_surfaces_3d
from ingkit.physics.plasma.vmec import VMECData


DATA_DIR = Path(__file__).parents[1] / "samples" / "vmec"
WOUT_FILE = "wout_helical_rfp_zero_beta.nc"


def test_vmec_plot_helpers_return_figures():
    pytest.importorskip("plotly")
    vmec = VMECData(DATA_DIR / WOUT_FILE)

    figure_3d = plot_flux_surfaces_3d(
        vmec, surface_indices=[5, -1], nu=9, nv=9, vector_step=4
    )
    figure_2d, axes_2d = plot_cylindrical_field(
        vmec, nu=17, radial_step=10, angular_step=4
    )

    assert len(figure_3d.data) == 3  # two surfaces and one vector trace
    assert axes_2d.figure is figure_2d
    assert axes_2d.collections
    plt.close(figure_2d)


def test_vmec_plot_rejects_invalid_surface_index():
    pytest.importorskip("plotly")
    vmec = VMECData(DATA_DIR / WOUT_FILE)

    with pytest.raises(IndexError, match="outside"):
        plot_flux_surfaces_3d(vmec, surface_indices=[vmec.ns])
