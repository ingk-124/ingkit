"""Low-level VMEC equilibrium API.

Geometry uses ``xm * theta_vmec - xn * phi``; xn already includes nfp.
Positions and vectors have shape (..., 3). Cylindrical ordering is (R, phi, Z).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import netCDF4
import numpy as np
from numpy.typing import ArrayLike

Coordinates = Literal["xyz", "rpz"]

__all__ = ["VMECData"]


# def _validate_count(value: int, name: str, minimum: int) -> int:
#     if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
#         raise ValueError(f"{name} must be an integer")
#     if value < minimum:
#         raise ValueError(f"{name} must be at least {minimum}")
#     return int(value)


def xyz_to_rpz(xyz: ArrayLike) -> np.ndarray:
    """Return cylindrical positions, shape (..., 3), in metres/radians/metres."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim < 1 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape (..., 3)")
    x, y, z = np.moveaxis(xyz, -1, 0)
    radius = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return np.stack((radius, phi, z), axis=-1)


def rpz_to_xyz(rpz: ArrayLike) -> np.ndarray:
    """Return Cartesian positions, shape (..., 3), in metres."""
    rpz = np.asarray(rpz, dtype=float)
    if rpz.ndim < 1 or rpz.shape[-1] != 3:
        raise ValueError("rpz must have shape (..., 3)")
    r, phi, z = np.moveaxis(rpz, -1, 0)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack((x, y, z), axis=-1)


class VMECData:
    """Read-only equilibrium and sampling grids loaded from a NetCDF wout.

    Parameters
    ----------
    path : str or Path
        Source wout NetCDF file.
    oversample_theta, oversample_phi : int, optional
        Angular sampling factors relative to the stored Fourier spectrum.
    oversample_rho : int, optional
        Radial interval multiplier for real-space queries. Samples are
        uniform in rho = sqrt(s).

    """

    def __init__(self, path: str | Path, *, oversample_theta: int = 5, oversample_phi: int = 5,
                 oversample_rho: int = 3) -> None:

        for name, value in (("oversample_theta", oversample_theta),
                            ("oversample_phi", oversample_phi),
                            ("oversample_rho", oversample_rho)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer")

        self._path = Path(path).expanduser().resolve()
        self._read_wout()

        self._ns = int(self.wout["ns"])
        self._nfp = int(self.wout["nfp"])
        self._lasym = bool(self.wout["lasym__logical__"])
        self._n_theta, self._n_phi, self._theta_grid, self._phi_grid = None, None, None, None
        self._initialize_radial_grids()
        self._initialize_angular_grids(oversample_theta=oversample_theta, oversample_phi=oversample_phi)
        self._inverse_s_grid = np.linspace(0, 1, (self.ns - 1) * oversample_rho + 1) ** 2
        self._inverse_theta_grid = self.theta_grid
        n_phi = (self.n_phi - 1 + self.nfp - 1) // self.nfp + 1
        self._inverse_phi_grid = np.linspace(0, 2 * np.pi / self.nfp, n_phi)

    def _read_wout(self) -> None:
        """Load variables and attributes without changing VMEC variable names."""
        variables = {}
        variable_attributes = {}

        with netCDF4.Dataset(self.path) as dataset:
            for name, variable in dataset.variables.items():
                values = variable[...]
                if np.ma.isMaskedArray(values) and np.any(values.mask):
                    if np.issubdtype(values.dtype, np.number):
                        values = values.astype(float).filled(np.nan)
                    elif values.dtype.kind == "S":
                        values = values.filled(b"")
                    else:
                        values = values.filled("")

                values = np.array(values, copy=True)
                values.setflags(write=False)
                variables[name] = values

                attributes = {key: variable.getncattr(key) for key in variable.ncattrs()}
                variable_attributes[name] = MappingProxyType(attributes)

            metadata = {key: dataset.getncattr(key) for key in dataset.ncattrs()}

        self._wout = MappingProxyType(variables)
        self._variable_attributes = MappingProxyType(variable_attributes)
        self._metadata = MappingProxyType(metadata)

    def _initialize_radial_grids(self) -> None:
        self._s_grid = np.linspace(0, 1, self.ns)
        self._rho_grid = np.sqrt(self.s_grid)
        self._s_grid.setflags(write=False)
        self._rho_grid.setflags(write=False)

    def _initialize_angular_grids(self, oversample_theta: int, oversample_phi: int) -> None:
        """Determine angular sample counts and create grids."""

        max_m = int(np.max(np.abs(self.wout["xm_nyq"])))
        max_n = int(np.max(np.abs(self.wout["xn_nyq"])))

        min_theta, min_phi = 2 * max_m + 1, 2 * max_n + 1
        self._n_theta = min_theta * oversample_theta + 1
        self._n_phi = min_phi * oversample_phi + 1

        self._theta_grid = np.linspace(0, 2 * np.pi, self._n_theta, endpoint=True)  # [0, 2*pi]
        self._phi_grid = np.linspace(0, 2 * np.pi, self._n_phi, endpoint=True)  # [0, 2*pi]
        self._theta_grid.setflags(write=False)
        self._phi_grid.setflags(write=False)

    # region Properties
    @property
    def path(self) -> Path:
        return self._path

    @property
    def wout(self) -> Mapping[str, np.ndarray]:
        """Read-only VMEC variables in their original NetCDF layout."""
        return self._wout

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def variable_attributes(self) -> Mapping[str, Mapping[str, Any]]:
        return self._variable_attributes

    @property
    def ns(self) -> int:
        return self._ns

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def lasym(self) -> bool:
        return self._lasym

    @property
    def s_grid(self) -> np.ndarray:
        """Original normalized toroidal-flux grid."""
        return self._s_grid

    @property
    def rho_grid(self) -> np.ndarray:
        """Radial sampling locations, rho = sqrt(s)."""
        return self._rho_grid

    @property
    def theta_grid(self) -> np.ndarray:
        """Grid of VMEC poloidal angles in radians. [0, 2*pi)"""
        return self._theta_grid

    @property
    def phi_grid(self) -> np.ndarray:
        """Grid of toroidal angles in radians. [0, 2*pi)"""
        return self._phi_grid

    @property
    def n_theta(self) -> int:
        return self._n_theta

    @property
    def n_phi(self) -> int:
        return self._n_phi

    # endregion

    def __repr__(self) -> str:
        grid = tuple(len(x) for x in (self.rho_grid, self.theta_grid, self.phi_grid))
        return (f"VMECData(path={str(self.path)!r}, ns={self.ns}, nfp={self.nfp}, "
                f"lasym={self.lasym}, grid={grid})")

    def _geometry(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Validate and retrieve named full-mesh geometry coefficients."""
        for name in ("xm", "xn", "rmnc", "zmns"):
            if name not in self.wout:
                raise ValueError(f"Geometry requires missing variable {name!r}")
        xm, xn = self.wout["xm"], self.wout["xn"]
        if xm.size == 0 or xm.shape != xn.shape:
            raise ValueError("Geometry modes must be nonempty matching arrays")
        coefficients = {}
        for name in ("rmnc", "rmns", "zmnc", "zmns"):
            if name not in self.wout:
                if self.lasym or name in ("rmnc", "zmns"):
                    raise ValueError(f"Geometry requires missing variable {name!r}")
                values = np.zeros((self.ns, xm.size))
            else:
                values = self.wout[name]
            if values.shape != (self.ns, xm.size) or not np.isfinite(values).all():
                raise ValueError(f"Invalid geometry coefficients {name!r}")
            coefficients[name] = values
        return xm, xn, coefficients

    def magnetic_axis(self, *, phi: ArrayLike | None = None, coordinates: Coordinates = "xyz") -> np.ndarray:
        """Return axis positions (n_phi, 3), without constructing a mapper.

        Parameters
        ----------
        coordinates : {'xyz', 'rpz'}, optional
            Output coordinate system. 'xyz' returns Cartesian positions in metres.
            'rpz' returns cylindrical positions in metres/radians/metres. Default is 'xyz'.
        phi : array_like, optional
            Finite scalar or 1-D toroidal angles in radians. Defaults to phi_grid.
        """

        phi = self.phi_grid if phi is None else np.atleast_1d(np.asarray(phi, dtype=float))
        if phi.ndim != 1 or phi.size == 0 or not np.isfinite(phi).all():
            raise ValueError("phi must be a finite scalar or nonempty 1-D array")
        if coordinates not in ("xyz", "rpz"):
            raise ValueError("coordinates must be 'xyz' or 'rpz'")

        R, Z = self._axis_rz(phi)
        rpz = np.stack((R, phi, Z), axis=-1)
        return rpz_to_xyz(rpz) if coordinates == "xyz" else rpz

    def _axis_rz(self, phi: np.ndarray, *, derivative: bool = False
                 ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = {}
        for name in ("raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc"):
            if name not in self.wout:
                if self.lasym or name in ("raxis_cc", "zaxis_cs"):
                    raise ValueError(f"Magnetic axis requires missing variable {name!r}")
                continue
            values = self.wout[name]
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"Invalid magnetic-axis coefficients {name!r}")
            coefficients[name] = values

        raxis_cc = coefficients.get("raxis_cc", 0.0)  # (n_modes,) or 0
        raxis_cs = coefficients.get("raxis_cs", 0.0)  # (n_modes,) or 0
        zaxis_cc = coefficients.get("zaxis_cc", 0.0)  # (n_modes,) or 0
        zaxis_cs = coefficients.get("zaxis_cs", 0.0)  # (n_modes,) or 0

        xn = np.arange(raxis_cc.size) * self.nfp
        E_phi = np.exp(-1j * phi[:, None] * xn)  # (n_phi, n_modes)
        if derivative:
            E_phi *= -1j * xn
        c_R = raxis_cc - 1j * raxis_cs  # (n_modes,)
        c_Z = zaxis_cc - 1j * zaxis_cs  # (n_modes,)
        return (E_phi @ c_R).real, (E_phi @ c_Z).real

    def flux_to_rpz(self, s: ArrayLike, theta_vmec: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """Evaluate geometry on the tensor product of the supplied coordinates.

        Parameters
        ----------
        s : array_like
            Scalar or nonempty 1-D array of normalized toroidal flux in [0, 1].
        theta_vmec, phi : array_like
            Scalar or nonempty 1-D arrays of VMEC poloidal and geometric toroidal
            angles in radians. Angle arrays are sampled independently.

        Returns
        -------
        numpy.ndarray
            Shape (n_s, n_theta, n_phi, 3), omitting the axis for each
            scalar input. Components are (R, phi, Z) in metres/radians/metres.
            Surface coefficients are interpolated linearly in s.
        """
        s = np.asarray(s, dtype=float)  # (n_s)
        theta = np.asarray(theta_vmec, dtype=float)  # (n_theta)
        phi = np.asarray(phi, dtype=float)  # (n_phi)

        for name, values in (("s", s), ("theta_vmec", theta), ("phi", phi)):
            if values.ndim > 1 or values.size == 0:
                raise ValueError(f"{name} must be a scalar or nonempty 1-D array")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite values")
        xm, xn, (rmnc, rmns, zmnc, zmns) = self._geometry_at_s(s)

        E_theta = np.exp(1j * theta[..., None] * xm)  # (n_theta, n_modes)
        E_phi = np.exp(-1j * phi[..., None] * xn).T  # (n_modes, n_phi)
        c_R = rmnc - 1j * rmns  # (n_s, n_modes)
        c_Z = zmnc - 1j * zmns  # (n_s, n_modes)
        if theta.ndim == 1:
            c_R = c_R[..., None, :]  # (n_s, 1, n_modes)
            c_Z = c_Z[..., None, :]  # (n_s, 1, n_modes)
        R_pol = c_R * E_theta  # (n_s, n_theta, n_modes)
        Z_pol = c_Z * E_theta  # (n_s, n_theta, n_modes)
        R = np.tensordot(R_pol, E_phi, axes=([-1], [0])).real  # (n_s, n_theta, n_phi)
        Z = np.tensordot(Z_pol, E_phi, axes=([-1], [0])).real  # (n_s, n_theta, n_phi)
        return np.stack(np.broadcast_arrays(R, phi, Z), axis=-1)  # (..., 3)

    def _geometry_at_s(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not np.all((0 <= s) & (s <= 1)):
            raise ValueError("s must lie in [0, 1]")
        xm, xn, coefficients = self._geometry()
        coeff_array = np.stack(tuple(coefficients.values()), axis=1)  # (ns, 4, n_modes)
        upper = np.clip(np.searchsorted(self.s_grid, s, side="right"), 1, self.ns - 1)
        lower = upper - 1
        weight = ((s - self.s_grid[lower]) / (self.s_grid[upper] - self.s_grid[lower]))[..., None, None]
        interpolated = coeff_array[lower] * (1 - weight) + coeff_array[upper] * weight
        return xm, xn, np.moveaxis(interpolated, -2, 0)  # (4, ..., n_modes)

    def flux_surface(self, s: ArrayLike = 1.0, *, theta: ArrayLike | None = None,
                     phi: ArrayLike | None = None, coordinates: Coordinates = "xyz"
                     ) -> np.ndarray:
        """Return positions on a tensor grid of flux coordinates.

        Parameters
        ----------
        s : array_like, optional
            Normalized toroidal flux in [0, 1]. Scalar or nonempty 1-D array.
        theta, phi : array_like, optional
            Finite scalar or nonempty 1-D arrays of angles in radians.
            Defaults to theta_grid and phi_grid.
        coordinates : {'xyz', 'rpz'}, optional
            Output coordinate system. 'xyz' returns Cartesian positions in metres.
            'rpz' returns cylindrical positions in metres/radians/metres. Default is 'xyz'.

        Returns
        -------
        numpy.ndarray
            Shape (n_s, n_theta, n_phi, 3), omitting the axis for each
            scalar input. Components follow the requested coordinate system.

        Raises
        ------
        ValueError
            If s, theta, or phi are invalid, or if coordinates is not 'xyz' or 'rpz'.

        """

        theta = self.theta_grid if theta is None else theta
        phi = self.phi_grid if phi is None else phi
        rpz = self.flux_to_rpz(s, theta, phi)
        if coordinates not in ("xyz", "rpz"):
            raise ValueError("coordinates must be 'xyz' or 'rpz'")
        return rpz_to_xyz(rpz) if coordinates == "xyz" else rpz

    def lcfs_angle_derivative(self, theta: ArrayLike | None = None,
                              phi: ArrayLike | None = None, *, s: ArrayLike = 1.0) -> np.ndarray:
        """Evaluate d(alpha)/d(theta_vmec) on magnetic surfaces, dimensionless.

        Parameters
        ----------
        theta, phi : array_like, optional
            Finite scalar or 1-D angles in radians. Defaults to theta_grid
            and phi_grid. The two inputs form a tensor product, not paired
            points; scalar inputs keep a length-one dimension.
        s : array_like, optional
            Scalar or 1-D normalized flux in (0, 1]. Defaults to the LCFS.

        Returns
        -------
        numpy.ndarray
            Shape (n_theta, n_phi) for scalar s, or (n_s, n_theta, n_phi).
            alpha is the geometric poloidal angle
            measured from the magnetic axis at the same phi. Derivatives
            come directly from the Fourier coefficients, not differences.

        Raises
        ------
        ValueError
            If axis/geometry data are missing, or a surface touches the axis
            on the evaluated grid, making alpha undefined.
        """
        theta = self.theta_grid if theta is None else np.atleast_1d(np.asarray(theta, dtype=float))
        phi = self.phi_grid if phi is None else np.atleast_1d(np.asarray(phi, dtype=float))
        axis = self.magnetic_axis(phi=phi, coordinates="rpz")  # (n_phi, 3)

        for name, samples in (("theta", theta), ("phi", phi)):
            if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError(f"{name} must be a finite scalar or nonempty 1-D array")
        s = np.asarray(s, dtype=float)
        if s.ndim > 1 or s.size == 0 or np.any(s == 0):
            raise ValueError("s must be a scalar or nonempty 1-D array in (0, 1]")
        xm, xn, (rmnc, rmns, zmnc, zmns) = self._geometry_at_s(s)

        E_theta = np.exp(1j * theta[..., None] * xm)  # (n_theta, n_modes)
        E_phi = np.exp(-1j * phi[..., None] * xn).T  # (n_modes, n_phi)
        R_pol = (rmnc - 1j * rmns)[..., None, :] * E_theta  # (..., n_theta, n_modes)
        Z_pol = (zmnc - 1j * zmns)[..., None, :] * E_theta
        dR_dtheta = ((1j * xm * R_pol) @ E_phi).real  # (..., n_theta, n_phi)
        dZ_dtheta = ((1j * xm * Z_pol) @ E_phi).real
        X = (R_pol @ E_phi).real - axis[:, 0]
        Y = (Z_pol @ E_phi).real - axis[:, 2]

        radius_squared = X ** 2 + Y ** 2
        if np.any(radius_squared == 0):
            raise ValueError("A magnetic surface intersects the axis on the evaluated grid")
        derivative = (X * dZ_dtheta - Y * dR_dtheta) / radius_squared
        if not np.isfinite(derivative).all():
            raise ValueError("Geometric-angle derivative is not finite")
        return derivative

    def is_star_shaped(self, *, theta: ArrayLike | None = None,
                       phi: ArrayLike | None = None, s: ArrayLike = 1.0) -> np.ndarray:
        """Check sampled magnetic surfaces for angular folding about the axis.

        Parameters
        ----------
        theta, phi : array_like, optional
            Tensor-product angular samples in radians; defaults to saved grids.
        s : array_like, optional
            Scalar or 1-D normalized flux in (0, 1]. Defaults to the LCFS.

        Returns
        -------
        numpy.ndarray
            Boolean array (n_phi,) for scalar s, or (n_s, n_phi).
            True if all sampled derivatives in a
            section are strictly positive or all strictly negative. A zero
            derivative or mixed signs returns False; no tolerance is applied.

        Notes
        -----
        Assumes simple closed sections around the magnetic axis. This is
        a sampled check for strict angular monotonicity, not a proof between
        theta/phi samples. A star-shaped section with a zero derivative can
        return False. Undefined derivatives (such as an axis intersection)
        still raise ValueError through lcfs_angle_derivative. No mapper is
        constructed and stored grids are not changed.
        """
        derivative = self.lcfs_angle_derivative(theta, phi, s=s)
        positive = np.all(derivative > 0, axis=-2)
        negative = np.all(derivative < 0, axis=-2)
        return positive | negative

    def is_inside_lcfs(self, points: ArrayLike, *, coordinates: Coordinates = "rpz",
                       tol: float = 1e-5) -> np.ndarray:
        """Return the LCFS mask of shape points.shape[:-1].

        Points are ordered (R, phi, Z) in metres/radians/metres, or (x, y, z)
        in metres. The relative radial tolerance is applied at the LCFS.
        """
        return self._map_points(points, coordinates, tol, boundary_only=True)

    def points_to_flux(self, points: ArrayLike, *, coordinates: Coordinates = "rpz",
                       tol: float = 1e-5, return_theta: bool = False
                       ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return normalized toroidal flux s with shape points.shape[:-1].

        Points are ordered (R, phi, Z) in metres/radians/metres, or (x, y, z)
        in metres. Points outside the LCFS are NaN, with a relative radial
        tolerance tol at the boundary. Points on the magnetic axis have s=0.
        With return_theta=True, return (s, theta_vmec) with the same shapes.
        Theta is in radians in [0, 2*pi), zero on the axis, and NaN outside.
        """
        return self._map_points(points, coordinates, tol, boundary_only=False,
                                return_theta=return_theta)

    def _map_points(self, points: ArrayLike, coordinates: Coordinates, tol: float, *,
                    boundary_only: bool, return_theta: bool = False
                    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite array of shape (..., 3)")
        if coordinates not in ("xyz", "rpz"):
            raise ValueError("coordinates must be 'xyz' or 'rpz'")
        rpz = xyz_to_rpz(points) if coordinates == "xyz" else points
        if np.any(rpz[..., 0] < 0):
            raise ValueError("Cylindrical radius must be nonnegative")

        rpz = rpz.reshape(-1, 3)
        result = np.zeros(len(rpz), dtype=bool) if boundary_only else np.full(len(rpz), np.nan)
        theta_result = np.full(len(rpz), np.nan) if return_theta else None
        shape = points.shape[:-1]
        if not len(rpz):
            return (result.reshape(shape), theta_result.reshape(shape)
                    ) if theta_result is not None else result.reshape(shape)

        theta, phi_grid = self._inverse_theta_grid[:-1], self._inverse_phi_grid
        period = 2 * np.pi

        def theta_on_surface(alpha: np.ndarray, weight: np.ndarray,
                             alpha_0: np.ndarray, alpha_1: np.ndarray) -> np.ndarray:
            # Invert each periodic alpha grid, keeping theta paired with its sample.
            angles = []
            for samples in (alpha_0, alpha_1):
                samples = np.mod(samples, period)
                order = np.argsort(samples)
                samples, theta_sorted = samples[order], theta[order]
                query = np.mod(alpha - samples[0], period) + samples[0]
                samples = np.r_[samples, samples[0] + period]
                theta_sorted = np.r_[theta_sorted, theta_sorted[0]]
                lower = np.searchsorted(samples, query, side="right") - 1
                lower = np.minimum(lower, theta.size - 1)
                upper = lower + 1
                fraction = (query - samples[lower]) / (samples[upper] - samples[lower])
                delta = (theta_sorted[upper] - theta_sorted[lower] + np.pi) % period - np.pi
                angles.append(theta_sorted[lower] + fraction * delta)
            delta = (angles[1] - angles[0] + np.pi) % period - np.pi
            return angles[0] + weight * delta

        s_grid = self._inverse_s_grid[-1:] if boundary_only else self._inverse_s_grid[1:]
        phi = np.mod(rpz[:, 1], phi_grid[-1])

        # Determine which phi_grid intervals contain each point.
        intervals = np.searchsorted(phi_grid[1:-1], phi, side="right")  # (n_points,), in [0, n_phi-2]
        # Sort intervals to group points by the same section.
        sorted_point_indices = np.argsort(intervals, kind="stable")
        sorted_intervals = intervals[sorted_point_indices]
        # Find the left and right edges of each unique interval in sorted_intervals.
        edges = np.r_[0, np.flatnonzero(np.diff(sorted_intervals)) + 1, len(sorted_intervals)]

        axis = self.magnetic_axis(phi=phi_grid, coordinates="rpz")  # (n_phi, 3)

        # Store original angular samples as (phi, s, theta), without resampling alpha.
        alpha_grid = np.empty((phi_grid.size, s_grid.size, theta.size))
        radius_grid = np.empty_like(alpha_grid)
        for start in range(0, s_grid.size, 8):
            rows = slice(start, start + 8)
            s_samples = s_grid[rows]
            if not self.is_star_shaped(s=s_samples, theta=theta, phi=phi_grid).all():
                raise ValueError("A sampled magnetic surface is not star-shaped about the axis")
            surface = self.flux_surface(s_samples, theta=theta, phi=phi_grid, coordinates="rpz")
            X = surface[..., 0] - axis[:, 0]
            Y = surface[..., 2] - axis[:, 2]
            alpha_grid[:, rows, :] = np.moveaxis(np.arctan2(Y, X), -1, 0)
            radius_grid[:, rows, :] = np.moveaxis(np.hypot(X, Y), -1, 0)

        for start, stop in zip(edges[:-1], edges[1:]):
            i_phi = sorted_intervals[start]
            indices = sorted_point_indices[start:stop]
            phi_0, phi_1 = phi_grid[i_phi:i_phi + 2]
            weight = (phi[indices] - phi_0) / (phi_1 - phi_0)
            axis = self.magnetic_axis(phi=phi[indices], coordinates="rpz")
            X, Y = rpz[indices, 0] - axis[:, 0], rpz[indices, 2] - axis[:, 2]
            alpha, r = np.arctan2(Y, X), np.hypot(X, Y)
            alpha_0, alpha_1 = alpha_grid[i_phi:i_phi + 2]  # alpha_grid at phi_0 and phi_1 (n_s, n_theta)
            radius_0, radius_1 = radius_grid[i_phi:i_phi + 2]  # radius_grid at phi_0 and phi_1 (n_s, n_theta)

            r_0 = np.interp(alpha, alpha_0[-1], radius_0[-1], period=2 * np.pi)  # boundary radius at phi_0, alpha, s=1
            r_1 = np.interp(alpha, alpha_1[-1], radius_1[-1], period=2 * np.pi)  # boundary radius at phi_1, alpha, s=1
            r_b = r_0 + weight * (r_1 - r_0)  # interpolated boundary radius at phi, alpha, s=1
            inside = r <= r_b * (1 + tol)  # whether points are inside the LCFS with tolerance
            if boundary_only:
                result[indices] = inside
                continue

            # r_b <= r <= r_b * (1 + tol) -> boundary points (s=1)
            on_boundary = inside & (r >= r_b)
            result[indices[on_boundary]] = 1
            # r <= L * 8 * eps -> axis points (s=0) (L = sqrt(R_axis^2 + Z_axis^2))
            on_axis = r <= 8 * np.finfo(float).eps * np.hypot(axis[:, 0], axis[:, 2])
            result[indices[inside & on_axis]] = 0
            if theta_result is not None:
                boundary = on_boundary & ~on_axis
                if boundary.any():
                    angle = theta_on_surface(alpha[boundary], weight[boundary],
                                             alpha_0[-1], alpha_1[-1])
                    theta_result[indices[boundary]] = np.mod(angle, period)
                theta_result[indices[inside & on_axis]] = 0

            interior = ~on_axis & (r < r_b)
            if not interior.any():
                continue
            indices, alpha, r = indices[interior], alpha[interior], r[interior]
            weight, r_b = weight[interior], r_b[interior]
            r_lower = np.zeros_like(r)
            s_lower = 0.0
            for i_s, s_upper in enumerate(s_grid):
                if i_s == s_grid.size - 1:
                    r_upper = r_b
                else:
                    # Get i_s-th surface's radii at points' (phi, alpha) locations, r_upper.
                    r_0 = np.interp(alpha, alpha_0[i_s], radius_0[i_s], period=2 * np.pi)
                    r_1 = np.interp(alpha, alpha_1[i_s], radius_1[i_s], period=2 * np.pi)
                    r_upper = r_0 + weight * (r_1 - r_0)
                if np.any(r_upper <= r_lower):
                    raise ValueError("Interpolated surface radii must increase with s")
                # Check which points are between lower and upper s surfaces.
                bracket = (r_lower < r) & (r <= r_upper)
                # Compute the fractional radius for points in the bracket.
                fraction = (r[bracket] - r_lower[bracket]) / (r_upper[bracket] - r_lower[bracket])
                # Interpolate rho = sqrt(s) linearly and assign s = rho**2 to these points.
                result[indices[bracket]] = (np.sqrt(s_lower) + fraction * (np.sqrt(s_upper) - np.sqrt(s_lower))) ** 2
                if theta_result is not None and bracket.any():
                    alpha_points, phi_weight = alpha[bracket], weight[bracket]
                    upper_theta = theta_on_surface(alpha_points, phi_weight,
                                                   alpha_0[i_s], alpha_1[i_s])
                    # From the axis to the first surface, use the first surface angle.
                    angle = upper_theta
                    if i_s > 0:
                        lower_theta = theta_on_surface(alpha_points, phi_weight,
                                                       alpha_0[i_s - 1], alpha_1[i_s - 1])
                        delta = (upper_theta - lower_theta + np.pi) % period - np.pi
                        angle = lower_theta + fraction * delta
                    theta_result[indices[bracket]] = np.mod(angle, period)
                # Update lower surface's radii and s values for the next iteration.
                r_lower, s_lower = r_upper, s_upper

        result = result.reshape(shape)
        if theta_result is not None:
            return result, theta_result.reshape(shape)
        return result

    def s_on_rz_grid(self, R: ArrayLike, Z: ArrayLike, *, phi: float = 0.0, tol: float = 1e-5) -> np.ndarray:
        """Return normalized toroidal flux s on a 2-D R-Z grid at fixed phi.

        Parameters
        ----------
        R, Z : array_like
            1-D arrays of cylindrical coordinates in metres. Must be finite and nonempty.
        phi : float, optional
            Toroidal angle in radians. Defaults to 0.0.
        tol : float, optional
            Relative radial tolerance for points outside the LCFS. Defaults to 1e-5.

        Returns
        -------
        numpy.ndarray
            2-D array of shape (len(R), len(Z)) containing normalized toroidal flux s.
            Points outside the LCFS are NaN, and points on the magnetic axis have s=0.
        """
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if R.ndim != 1 or Z.ndim != 1 or R.size == 0 or Z.size == 0:
            raise ValueError("R and Z must be finite nonempty 1-D arrays")
        if not np.isfinite(R).all() or not np.isfinite(Z).all():
            raise ValueError("R and Z must contain finite values")
        if not np.isfinite(phi):
            raise ValueError("phi must be a finite scalar")

        # Create a meshgrid of R and Z coordinates for evaluation.
        R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
        points = np.stack((R_grid.ravel(), np.full(R_grid.size, phi), Z_grid.ravel()), axis=-1)

        # Map points to normalized toroidal flux s using the existing method.
        s_values = self.points_to_flux(points, coordinates="rpz", tol=tol)

        return s_values.reshape(R_grid.shape)

    def _half_to_full(self, half: np.ndarray) -> np.ndarray:
        """Map valid half-mesh rows to the full mesh, including the endpoints."""
        full = np.empty((self.ns, *half.shape[1:]), dtype=half.dtype)
        if self.ns == 2:
            full[:] = half[0]
        else:
            full[1:-1] = 0.5 * (half[:-1] + half[1:])
            if self.ns >= 5:
                full[0] = 3 * full[1] - 3 * full[2] + full[3]
                full[-1] = 3 * full[-2] - 3 * full[-3] + full[-4]
            else:
                full[0], full[-1] = full[1], full[-2]
        return full

    def magnetic_field_strength(self, flux_points: ArrayLike) -> np.ndarray:
        """Evaluate |B| at paired flux coordinates.

        Parameters
        ----------
        flux_points : array_like
            Shape (..., 3), ordered (s, theta_vmec, phi). s is in [0, 1]
            and both angles are in radians.

        Returns
        -------
        numpy.ndarray
            Magnetic field strength in tesla, shape flux_points.shape[:-1].
        """
        points = np.asarray(flux_points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("flux_points must be a finite array of shape (..., 3)")
        s, theta, phi = np.moveaxis(points, -1, 0)
        if np.any((s < 0) | (s > 1)):
            raise ValueError("s must lie in [0, 1]")

        xm, xn = self.wout["xm_nyq"], self.wout["xn_nyq"]
        bmnc = self.wout["bmnc"]
        bmns = self.wout["bmns"] if self.lasym else 0.0
        half = (bmnc - 1j * bmns)[1:]  # First wout row is unused.
        full = self._half_to_full(half)
        full[0, xm > 0] = 0

        # Interpolate all modes with the same radial weights at each point.
        upper = np.clip(np.searchsorted(self.s_grid, s, side="right"), 1, self.ns - 1)
        lower = upper - 1
        weight = ((s - self.s_grid[lower]) / (self.s_grid[upper] - self.s_grid[lower]))[..., None]
        coefficients = full[lower] * (1 - weight) + full[upper] * weight
        phase = theta[..., None] * xm - phi[..., None] * xn
        return np.sum(coefficients * np.exp(1j * phase), axis=-1).real

    def magnetic_field_strength_at(self, points: ArrayLike, *, coordinates: Coordinates = "rpz",
                                   tol: float = 1e-5) -> np.ndarray:
        """Evaluate |B| at real-space points.

        Parameters
        ----------
        points : array_like
            Shape (..., 3), ordered (R, phi, Z) in metres/radians/metres
            or (x, y, z) in metres.
        coordinates : {'rpz', 'xyz'}, optional
            Input coordinate system.
        tol : float, optional
            Relative LCFS radius tolerance passed to points_to_flux.

        Returns
        -------
        numpy.ndarray
            Field strength in tesla, shape points.shape[:-1]. Points
            outside the LCFS tolerance return NaN.
        """
        points = np.asarray(points, dtype=float)
        s, theta = self.points_to_flux(points, coordinates=coordinates, tol=tol,
                                       return_theta=True)
        result = np.full(s.shape, np.nan)
        valid = np.isfinite(s) & np.isfinite(theta)
        if valid.any():
            if coordinates == "xyz":
                phi = np.arctan2(points[..., 1][valid], points[..., 0][valid])
            else:
                phi = points[..., 1][valid]
            flux_points = np.stack((s[valid], theta[valid], phi), axis=-1)
            result[valid] = self.magnetic_field_strength(flux_points)
        return result

    def magnetic_field(self, flux_points: ArrayLike, *,
                       basis: Literal["cylindrical", "cartesian"] = "cylindrical"
                       ) -> np.ndarray:
        """Evaluate the physical magnetic field at paired flux coordinates.

        Parameters
        ----------
        flux_points : array_like
            Shape (..., 3), ordered (s, theta_vmec, phi). Angles are in
            radians and s is in [0, 1].
        basis : {'cylindrical', 'cartesian'}, optional
            Output basis. Defaults to the cylindrical unit vectors.

        Returns
        -------
        numpy.ndarray
            Shape (..., 3), with (B_R, B_phi, B_Z) or (B_x, B_y, B_z) in tesla.

        Notes
        -----
        Field coefficients are moved to the full mesh using the same endpoint
        rule as magnetic_field_strength. On the axis the vector follows the
        axis tangent with the m=0 contribution to the contravariant B^phi.
        """
        points = np.asarray(flux_points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("flux_points must be a finite array of shape (..., 3)")
        if basis not in ("cylindrical", "cartesian"):
            raise ValueError("basis must be 'cylindrical' or 'cartesian'")
        s, theta, phi = np.moveaxis(points, -1, 0)
        if np.any((s < 0) | (s > 1)):
            raise ValueError("s must lie in [0, 1]")

        xm_nyq, xn_nyq = self.wout["xm_nyq"], self.wout["xn_nyq"]
        phase = theta[..., None] * xm_nyq - phi[..., None] * xn_nyq
        E = np.exp(1j * phase)
        upper = np.clip(np.searchsorted(self.s_grid, s, side="right"), 1, self.ns - 1)
        lower = upper - 1
        weight = ((s - self.s_grid[lower]) / (self.s_grid[upper] - self.s_grid[lower]))[..., None]

        components = []
        for name in ("bsupum", "bsupvm"):
            cosine = self.wout[name + "nc"][1:]  # The first wout row is unused.
            sine = self.wout[name + "ns"][1:] if self.lasym else 0.0
            full = self._half_to_full(cosine - 1j * sine)
            if name == "bsupvm":
                full[0, xm_nyq > 0] = 0
            coefficients = full[lower] * (1 - weight) + full[upper] * weight
            components.append(np.sum(coefficients * E, axis=-1).real)
        B_theta, B_phi = components  # Contravariant components.
        return self._tangent_vector(s, theta, phi, B_theta, B_phi, basis)

    def _tangent_vector(self, s: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                        V_theta: np.ndarray, V_phi: np.ndarray, basis: str) -> np.ndarray:
        """Convert a vector with zero radial component to physical components."""
        xm, xn, (rmnc, rmns, zmnc, zmns) = self._geometry_at_s(s)
        E = np.exp(1j * (theta[..., None] * xm - phi[..., None] * xn))
        R_modes, Z_modes = (rmnc - 1j * rmns) * E, (zmnc - 1j * zmns) * E
        R = np.sum(R_modes, axis=-1).real
        factor = 1j * (V_theta[..., None] * xm - V_phi[..., None] * xn)
        V_R = np.sum(R_modes * factor, axis=-1).real
        V_Z = np.sum(Z_modes * factor, axis=-1).real
        vector = np.stack((V_R, R * V_phi, V_Z), axis=-1)
        on_axis = s == 0
        if np.any(on_axis):
            axis_phi = phi[on_axis]
            R_axis, _ = self._axis_rz(axis_phi)
            dR_dphi, dZ_dphi = self._axis_rz(axis_phi, derivative=True)
            tangent = np.stack((dR_dphi, R_axis, dZ_dphi), axis=-1)
            vector[on_axis] = V_phi[on_axis, None] * tangent
        if basis == "cartesian":
            V_R, V_toroidal, V_Z = np.moveaxis(vector, -1, 0)
            cosine, sine = np.cos(phi), np.sin(phi)
            return np.stack((V_R * cosine - V_toroidal * sine,
                             V_R * sine + V_toroidal * cosine, V_Z), axis=-1)
        return vector

    def current_density(self, flux_points: ArrayLike, *,
                        basis: Literal["cylindrical", "cartesian"] = "cylindrical"
                        ) -> np.ndarray:
        """Evaluate the flux-surface-tangent current density at paired flux points.

        Parameters
        ----------
        flux_points : array_like
            Shape (..., 3), ordered (s, theta_vmec, phi), with s in [0, 1]
            and angles in radians.
        basis : {'cylindrical', 'cartesian'}, optional
            Output basis. Defaults to the cylindrical unit vectors.

        Returns
        -------
        numpy.ndarray
            Shape (..., 3), ordered (J_R, J_phi, J_Z) or (J_x, J_y, J_z),
            in A/m². Points on the magnetic axis return NaN.

        Notes
        -----
        The wout current coefficients already include 1/mu_0. The signed
        Jacobian is moved to the full mesh with the same endpoint rule as
        magnetic_field_strength, then evaluated at the same points as curr*.
        """
        points = np.asarray(flux_points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("flux_points must be a finite array of shape (..., 3)")
        if basis not in ("cylindrical", "cartesian"):
            raise ValueError("basis must be 'cylindrical' or 'cartesian'")
        if np.any((points[..., 0] < 0) | (points[..., 0] > 1)):
            raise ValueError("s must lie in [0, 1]")

        valid = points[..., 0] > 0
        result = np.full(points.shape, np.nan)
        if not valid.any():
            return result
        s, theta, phi = points[valid].T

        xm, xn = self.wout["xm_nyq"], self.wout["xn_nyq"]
        E = np.exp(1j * (theta[:, None] * xm - phi[:, None] * xn))
        upper = np.searchsorted(self.s_grid, s, side="left")
        lower = upper - 1
        weight = ((s - self.s_grid[lower]) / (self.s_grid[upper] - self.s_grid[lower]))[:, None]

        values = []
        for name in ("currum", "currvm", "gm"):
            cosine = self.wout[name + "nc"]
            sine = self.wout[name + "ns"] if self.lasym else 0.0
            full = cosine - 1j * sine
            if name == "gm":
                full = self._half_to_full(full[1:])  # First wout row is unused.
            coefficients = full[lower] * (1 - weight) + full[upper] * weight
            values.append(np.sum(coefficients * E, axis=-1).real)
        sqrtg_J_theta, sqrtg_J_phi, sqrtg = values
        sqrtg = np.where(sqrtg != 0, sqrtg, np.nan)
        J_theta, J_phi = sqrtg_J_theta / sqrtg, sqrtg_J_phi / sqrtg
        result[valid] = self._tangent_vector(s, theta, phi, J_theta, J_phi, basis)
        return result

    def magnetic_field_at(self, points: ArrayLike, *, coordinates: Coordinates = "rpz",
                          basis: Literal["cylindrical", "cartesian"] = "cylindrical",
                          tol: float = 1e-5) -> np.ndarray:
        """Evaluate the physical magnetic field at real-space points.

        Parameters
        ----------
        points : array_like
            Shape (..., 3), ordered (R, phi, Z) in metres/radians/metres
            or (x, y, z) in metres.
        coordinates : {'rpz', 'xyz'}, optional
            Input coordinate system.
        basis : {'cylindrical', 'cartesian'}, optional
            Output basis. Defaults to the cylindrical unit vectors.
        tol : float, optional
            Relative LCFS radius tolerance passed to points_to_flux.

        Returns
        -------
        numpy.ndarray
            Shape (..., 3), with (B_R, B_phi, B_Z) or (B_x, B_y, B_z) in tesla.
            Points outside the LCFS tolerance return NaN.
        """
        points = np.asarray(points, dtype=float)
        s, theta = self.points_to_flux(points, coordinates=coordinates, tol=tol,
                                       return_theta=True)
        valid = np.isfinite(s) & np.isfinite(theta)
        phi = np.arctan2(points[..., 1], points[..., 0]) if coordinates == "xyz" else points[..., 1]
        flux_points = np.stack((s[valid], theta[valid], phi[valid]), axis=-1)
        result = np.full(points.shape, np.nan)
        result[valid] = self.magnetic_field(flux_points, basis=basis)
        return result

    def current_at(self, points: ArrayLike, *, coordinates: Coordinates = "rpz",
                   basis: Literal["cylindrical", "cartesian"] = "cylindrical",
                   tol: float = 1e-5) -> np.ndarray:
        """Evaluate the flux-surface-tangent current density in real space.

        Parameters
        ----------
        points : array_like
            Shape (..., 3), ordered (R, phi, Z) in metres/radians/metres
            or (x, y, z) in metres.
        coordinates : {'rpz', 'xyz'}, optional
            Input coordinate system.
        basis : {'cylindrical', 'cartesian'}, optional
            Output basis. Defaults to the cylindrical unit vectors.
        tol : float, optional
            Relative LCFS radius tolerance passed to points_to_flux.

        Returns
        -------
        numpy.ndarray
            Shape (..., 3) in A/m², with NaN on the axis and outside the LCFS tolerance.
        """
        points = np.asarray(points, dtype=float)
        s, theta = self.points_to_flux(points, coordinates=coordinates, tol=tol,
                                       return_theta=True)
        valid = np.isfinite(s) & np.isfinite(theta)
        phi = np.arctan2(points[..., 1], points[..., 0]) if coordinates == "xyz" else points[..., 1]
        flux_points = np.stack((s[valid], theta[valid], phi[valid]), axis=-1)
        result = np.full(points.shape, np.nan)
        result[valid] = self.current_density(flux_points, basis=basis)
        return result

    def pressure_at(self, points: ArrayLike, *, coordinates: Coordinates = "rpz",
                    tol: float = 1e-5) -> np.ndarray:
        """Evaluate equilibrium pressure at real-space points.

        Parameters
        ----------
        points : array_like
            Shape (..., 3), ordered (R, phi, Z) in metres/radians/metres
            or (x, y, z) in metres.
        coordinates : {'rpz', 'xyz'}, optional
            Input coordinate system.
        tol : float, optional
            Relative LCFS radius tolerance passed to points_to_flux.

        Returns
        -------
        numpy.ndarray
            Pressure in Pa, shape points.shape[:-1]. Pressure is linearly
            interpolated in s from the full-mesh presf profile. Points
            outside the LCFS tolerance return NaN.
        """
        s = self.points_to_flux(points, coordinates=coordinates, tol=tol)
        return np.asarray(np.interp(s, self.s_grid, self.wout["presf"]))

    def dvolume_ds(self, s: ArrayLike) -> np.ndarray:
        """Return dV/ds in m³; TODO: specify source normalization and endpoints."""
        raise NotImplementedError("dV/ds normalization and endpoint policy pending")

    @property
    def alpha_grid(self) -> np.ndarray:
        raise RuntimeError("Boundary interpolation grid is not built")

    @property
    def boundary_radius_grid(self) -> np.ndarray:
        raise RuntimeError("Boundary interpolation grid is not built")


if __name__ == '__main__':
    # file_path = ("/Users/inagakishinichiro/Documents/KyotoUniv/4_works/4_1_codes/misc/vmec_calc/vmectest"
    #              "/symmetric_tokamak/zero_beta/wout_symmetric_tokamak_zero_beta.nc")
    file_path = (
        "/Users/inagakishinichiro/Documents/KyotoUniv/2_experiments/2_4_exp_analysis/HJ_MECH_2025/vmec_out"
        "/wout_bstd89.nt800.nc")
    vmec_data = VMECData(file_path)

    print(vmec_data.flux_surface(s=1, phi=0, coordinates="rpz").shape)
    print(vmec_data.flux_surface(s=[0, 0.5, 1], phi=0, coordinates="rpz").shape)
    print(vmec_data.flux_surface(s=1, theta=0, coordinates="rpz").shape)

    from matplotlib import pyplot as plt

    s_arr = np.linspace(0, 1, 21)
    s_reversed = vmec_data.points_to_flux(
        vmec_data.flux_surface(s=s_arr, coordinates="rpz", phi=vmec_data._inverse_phi_grid))

    s_ref = s_arr[1:]
    fig, axes = plt.subplots(2, 1, sharex=True)
    err = (s_ref[:, None, None] - s_reversed[1:]) / s_ref[:, None, None]
    axes[0].errorbar(s_ref, err.mean(axis=(1, 2)), yerr=err.std(axis=(1, 2)),
                 fmt="o--")
    axes[0].axhline(0, color="0.5", linewidth=1)
    axes[0].set_ylabel("(s_ref - s_reversed) / s_ref")

    err = s_ref[:, None, None] - s_reversed[1:]
    axes[1].errorbar(s_ref, err.mean(axis=(1, 2)), yerr=err.std(axis=(1, 2)),
                 fmt="o--")
    axes[1].axhline(0, color="0.5", linewidth=1)
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("s_ref - s_reversed")
    plt.tight_layout()
    plt.show()

    axis = vmec_data.magnetic_axis(coordinates="xyz")  # (n_phi, 3)
    lcfs = vmec_data.flux_surface(s=1.0, coordinates="xyz")  # (n_theta, n_phi, 3)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
    ax.plot_wireframe(lcfs[..., 0], lcfs[..., 1], lcfs[..., 2],
                      rcount=24, ccount=24, color="tab:blue", linewidth=0.5)
    # ax.plot_wireframe(lcfs[..., 0], lcfs[..., 1], lcfs[..., 2],
                      # rcount=0, ccount=36, color="red", linewidth=0.5)
    ax.plot(*axis.T, color="tab:red", linewidth=2.5, label="Magnetic axis")

    ax.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    ax.set_box_aspect(np.ptp(lcfs.reshape(-1, 3), axis=0))
    # ax.view_init(elev=90, azim=0)
    ax.legend()
    fig.tight_layout()
    plt.show()

    R = np.linspace(0.5, 1.5, 101)
    Z = np.linspace(-0.5, 0.5, 101)
    B_mod = vmec_data.magnetic_field_strength_at(np.stack(np.meshgrid(R, 0.0, Z, indexing="ij"), axis=-1))[:, 0, :]
    s_contour = vmec_data.s_on_rz_grid(R, Z, phi=0.0)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(R, Z, B_mod.T, shading="auto")
    plt.contour(R, Z, s_contour.T, levels=np.linspace(0, 1, 11), colors="k", linewidths=0.5)
    plt.colorbar(label="|B| [T]")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title("Magnetic field strength at phi=0")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
