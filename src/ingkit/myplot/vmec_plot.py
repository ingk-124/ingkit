"""Plotting helpers for :class:`ingkit.physics.plasma.vmec.VMECData`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from ingkit.physics.plasma.vmec import VMECData


def _default_surface_indices(ns: int) -> list[int]:
    """Return representative nested surfaces including the boundary."""
    interior = np.linspace(max(1, ns // 12), max(1, ns - 2), 6, dtype=int)
    return [*np.unique(interior).tolist(), ns - 1]


def plot_flux_surfaces_3d(
        vmec: VMECData,
        surface_indices: Sequence[int] | None = None,
        *,
        nu: int = 76,
        nv: int = 101,
        vector_step: int = 5,
        v_end: float = 5 * np.pi / 3,
        sizeref: float = 2.0,
) -> Any:
    """Plot nested flux surfaces and magnetic-field vectors with Plotly.

    Plotly is imported only when this function is called. The returned figure
    is not shown automatically.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plot_flux_surfaces_3d requires plotly") from exc

    if nu < 2 or nv < 2:
        raise ValueError("nu and nv must be at least 2")
    if vector_step < 1:
        raise ValueError("vector_step must be positive")

    indices = list(_default_surface_indices(vmec.ns) if surface_indices is None
                   else surface_indices)
    if not indices:
        raise ValueError("surface_indices must not be empty")
    if any(index < -vmec.ns or index >= vmec.ns for index in indices):
        raise IndexError("surface index is outside the VMEC radial grid")
    indices = [index % vmec.ns for index in indices]
    if len(set(indices)) != len(indices):
        raise ValueError("surface_indices must not contain duplicates")

    u_arr = np.linspace(0.0, 2.0 * np.pi, nu)
    colors = px.colors.sample_colorscale("jet_r", np.linspace(0.0, 1.0, vmec.ns))
    vector_coordinates: list[list[np.ndarray]] = [[], [], []]
    vector_components: list[list[np.ndarray]] = [[], [], []]
    fig = go.Figure()

    for index in indices:
        # A radial cutaway keeps inner nested surfaces visible.
        v_start = (index / vmec.ns) ** 0.6 * 2.0 * np.pi / 3.0
        v_arr = np.linspace(v_start, v_end, nv)
        s = vmec.s_grid[index]
        xyz = vmec.flux_surface(s, theta=u_arr, phi=v_arr, coordinates="xyz")
        X, Y, Z = np.moveaxis(xyz, -1, 0)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.zeros_like(X),
            colorscale=[[0, colors[index]], [1, colors[index]]],
            showscale=False, opacity=1, name=f"Flux surface {index}",
        ))

        if index == vmec.ns - 1:
            continue
        selection = np.s_[::vector_step, 1:-1:vector_step]
        flux_points = np.stack(np.meshgrid(s, u_arr[::vector_step],
                                           v_arr[1:-1:vector_step], indexing="ij"), axis=-1)[0]
        field = vmec.magnetic_field(flux_points, basis="cartesian")
        for target, array in zip(vector_coordinates, np.moveaxis(xyz[selection], -1, 0)):
            target.append(array)
        for target, array in zip(vector_components, np.moveaxis(field, -1, 0)):
            target.append(array)

    if vector_coordinates[0]:
        X_vec, Y_vec, Z_vec = [np.concatenate(values).ravel()
                               for values in vector_coordinates]
        B_X_vec, B_Y_vec, B_Z_vec = [np.concatenate(values).ravel()
                                     for values in vector_components]
        fig.add_trace(go.Cone(
            x=X_vec, y=Y_vec, z=Z_vec,
            u=B_X_vec, v=B_Y_vec, w=B_Z_vec,
            sizemode="scaled", sizeref=sizeref, anchor="tail",
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False, name="B field",
        ))

    fig.update_layout(
        title="VMEC Flux Surfaces with Magnetic Field Vectors",
        scene={
            "xaxis_title": "X [m]",
            "yaxis_title": "Y [m]",
            "zaxis_title": "Z [m]",
            "aspectmode": "data",
        },
        width=1200,
        height=800,
    )
    fig.update_traces(hoverinfo="name")
    return fig


def plot_cylindrical_field(vmec: VMECData, *, v: float = 0.0, nu: int = 101,
                           radial_step: int = 5, angular_step: int = 5, levels: int = 21,
                           scale: float = 1.0,                           ax: plt.Axes | None = None, ) -> tuple[
    plt.Figure, plt.Axes]:
    """Plot toroidal field and poloidal vectors on one toroidal plane."""
    if nu < 2:
        raise ValueError("nu must be at least 2")
    if radial_step < 1 or angular_step < 1:
        raise ValueError("radial_step and angular_step must be positive")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    u_arr = np.linspace(0.0, 2.0 * np.pi, nu)
    rpz = vmec.flux_surface(vmec.s_grid, theta=u_arr, phi=[v], coordinates="rpz")
    flux_points = np.stack(np.meshgrid(vmec.s_grid, u_arr, [v], indexing="ij"), axis=-1)
    R, _, Z = np.moveaxis(rpz, -1, 0)
    B_R, B_phi, B_Z = np.moveaxis(vmec.magnetic_field(flux_points), -1, 0)
    contour = ax.contourf(
        R[1:, :, 0], Z[1:, :, 0], B_phi[1:, :, 0] * 1e3,
        levels=levels, cmap="viridis",
    )
    fig.colorbar(contour, ax=ax, label=r"$B_\phi$ [mT]")
    ax.quiver(
        R[1::radial_step, ::angular_step, 0],
        Z[1::radial_step, ::angular_step, 0],
        B_R[1::radial_step, ::angular_step, 0],
        B_Z[1::radial_step, ::angular_step, 0],
        color="k", scale=scale,
    )
    ax.set(xlabel="R [m]", ylabel="Z [m]",
           title=f"Magnetic field at v={v:.3g} rad")
    ax.set_aspect("equal")
    ax.grid(True)
    return fig, ax


if __name__ == "__main__":
    from ingkit.physics.plasma.vmec import VMECData

    data_path = Path(__file__).parents[3] / "samples/vmec/wout_bstd89.nt800.nc"
    vmec_data = VMECData(data_path)
    plot_flux_surfaces_3d(vmec_data, sizeref=0.3).show()
    plot_cylindrical_field(vmec_data)
    plt.show()
