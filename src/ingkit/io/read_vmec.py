"""Read VMEC ``wout_*.nc`` files and evaluate VMEC coordinates.

This module intentionally provides no VMEC writer.  VMEC geometry uses the
Fourier phase ``xm * theta - xn * phi``.  The ``xn`` values stored by VMEC
already include the field-period factor ``nfp``; they must not be multiplied by
``nfp`` again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import netCDF4 as nc
import numpy as np
from scipy.optimize import least_squares


class VMECValidationError(ValueError):
    """Report invalid or inconsistent VMEC equilibrium data."""


class CylindricalCoordinates(NamedTuple):
    """Store cylindrical coordinates with a shared broadcast shape."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray


class CartesianCoordinates(NamedTuple):
    """Store Cartesian coordinates with a shared broadcast shape."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class InverseCoordinateResult:
    """Store the result of a Cartesian-to-VMEC coordinate inversion.

    Attributes
    ----------
    s : np.ndarray (...)
        Normalized toroidal flux.
    rho : np.ndarray (...)
        Normalized flux radius, ``sqrt(s)``.
    theta : np.ndarray (...)
        VMEC poloidal angle (rad), normalized to ``[0, 2*pi)``.
    phi : np.ndarray (...)
        Geometric toroidal angle (rad), normalized to ``[0, 2*pi)``.
    valid : np.ndarray (...)
        Whether the returned magnetic coordinates satisfy the tolerance.
    residual : np.ndarray (...)
        Euclidean residual in the cylindrical ``R-Z`` plane (m).
    status : np.ndarray (...)
        One of ``"converged"``, ``"axis"``, ``"outside_lcfs"``, or
        ``"nonconverged"``.
    """

    s: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    valid: np.ndarray
    residual: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class VMECFourierCoefficients:
    """Store validated VMEC surface Fourier coefficients.

    All coefficient arrays have shape ``(ns, mnmax)`` and units of metres.
    The radial grid ``s`` has shape ``(ns,)``.
    """

    s: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    rmnc: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    zmns: np.ndarray
    nfp: int
    lasym: bool


def _read_netcdf(file: Path) -> dict[str, Any]:
    """Return all variables from a VMEC NetCDF output file."""
    try:
        with nc.Dataset(file, "r") as dataset:
            return {
                name: np.asarray(variable[:])
                for name, variable in dataset.variables.items()
            }
    except OSError as exc:
        raise OSError(f"Unable to read VMEC NetCDF file {file}: {exc}") from exc


class VMECData:
    """Represent a VMEC equilibrium read from a NetCDF ``wout`` file.

    Parameters
    ----------
    file : str or pathlib.Path
        VMEC NetCDF output file.  Only the ``.nc`` backend is supported.

    Notes
    -----
    ``s`` is normalized toroidal flux and ``rho = sqrt(s)``.  ``theta`` is the
    VMEC poloidal angle and ``phi`` is the geometric cylindrical toroidal
    angle, both in radians.  Surface geometry is evaluated as

    ``R = sum(rmnc*cos(alpha) + rmns*sin(alpha))``

    ``Z = sum(zmnc*cos(alpha) + zmns*sin(alpha))``

    where ``alpha = xm*theta - xn*phi``.  VMEC stores ``xn`` with ``nfp``
    already included, so no additional field-period factor is applied.
    Coordinate coefficients and returned lengths are in metres.
    """

    def __init__(self, file: str | Path):
        self.file = Path(file)
        self.data: dict[str, Any]
        self._read_file()
        self._initialize()

    @classmethod
    def from_fourier(
            cls,
            *,
            s: np.ndarray,
            xm: np.ndarray,
            xn: np.ndarray,
            rmnc: np.ndarray,
            zmns: np.ndarray,
            nfp: int = 1,
            lasym: bool = False,
            rmns: np.ndarray | None = None,
            zmnc: np.ndarray | None = None,
    ) -> VMECData:
        """Construct an equilibrium from in-memory Fourier coefficients.

        Parameters
        ----------
        s : array_like (ns,)
            Strictly increasing normalized toroidal-flux grid from 0 to 1.
        xm : array_like (mnmax,)
            Integer poloidal mode numbers.
        xn : array_like (mnmax,)
            Toroidal mode numbers with ``nfp`` already included.
        rmnc : array_like (ns, mnmax)
            Cosine coefficients of cylindrical radius (m).
        zmns : array_like (ns, mnmax)
            Sine coefficients of vertical position (m).
        nfp : int, optional
            Number of field periods.  Default is 1.
        lasym : bool, optional
            Whether non-stellarator-symmetric coefficients are required.
            Default is False.
        rmns : array_like (ns, mnmax), optional
            Sine coefficients of cylindrical radius (m).  Default is zeros
            for symmetric equilibria.
        zmnc : array_like (ns, mnmax), optional
            Cosine coefficients of vertical position (m).  Default is zeros
            for symmetric equilibria.

        Returns
        -------
        equilibrium : VMECData
            Validated equilibrium.
        """
        obj = cls.__new__(cls)
        obj.file = None
        ns = np.asarray(s).size
        obj.data = {
            "ns": np.asarray(ns),
            "nfp": np.asarray(nfp),
            "lasym__logical__": np.asarray(int(lasym)),
            "s": np.asarray(s),
            "xm": np.asarray(xm),
            "xn": np.asarray(xn),
            "rmnc": np.asarray(rmnc),
            "zmns": np.asarray(zmns),
        }
        if rmns is not None:
            obj.data["rmns"] = np.asarray(rmns)
        if zmnc is not None:
            obj.data["zmnc"] = np.asarray(zmnc)
        obj._initialize()
        return obj

    def _read_file(self) -> None:
        if self.file.suffix.lower() == ".nc":
            self.data = _read_netcdf(self.file)
        else:
            raise ValueError(
                f"Unsupported VMEC file format {self.file.suffix!r}; "
                "only NetCDF .nc files are supported"
            )

    @staticmethod
    def _scalar_integer(data: dict[str, Any], name: str) -> int:
        if name not in data:
            raise VMECValidationError(f"Missing required VMEC variable {name!r}")
        value = np.asarray(data[name])
        if value.size != 1 or not np.isfinite(value).all():
            raise VMECValidationError(
                f"VMEC variable {name!r} must be one finite scalar"
            )
        number = float(value.reshape(()))
        if not number.is_integer():
            raise VMECValidationError(f"VMEC variable {name!r} must be an integer")
        return int(number)

    def _initialize(self) -> None:
        self.ns = self._scalar_integer(self.data, "ns")
        self.nfp = self._scalar_integer(self.data, "nfp")
        lasym = self._scalar_integer(self.data, "lasym__logical__")
        if self.ns < 2:
            raise VMECValidationError("VMEC variable 'ns' must be at least 2")
        if self.nfp < 1:
            raise VMECValidationError("VMEC variable 'nfp' must be positive")
        if lasym not in (0, 1):
            raise VMECValidationError(
                "VMEC variable 'lasym__logical__' must be 0 or 1"
            )
        self.lasym = lasym

        self.xm = self._mode_array("xm")
        self.xn = self._mode_array("xn")
        if self.xm.shape != self.xn.shape:
            raise VMECValidationError(
                "VMEC variables 'xm' and 'xn' must have the same shape"
            )
        if np.any(self.xm < 0):
            raise VMECValidationError("VMEC poloidal modes 'xm' must be nonnegative")
        if not np.allclose(self.xn / self.nfp, np.rint(self.xn / self.nfp)):
            raise VMECValidationError(
                "VMEC toroidal modes 'xn' must be integer multiples of 'nfp'"
            )

        if "s" in self.data:
            self.s_arr = np.asarray(self.data["s"], dtype=float)
        else:
            self.s_arr = np.linspace(0.0, 1.0, self.ns)
        self._validate_radial_grid()

        rmnc = self._coefficient_array("rmnc", required=True)
        zmns = self._coefficient_array("zmns", required=True)
        rmns = self._coefficient_array("rmns", required=bool(self.lasym))
        zmnc = self._coefficient_array("zmnc", required=bool(self.lasym))
        self.data["rmnc"] = rmnc
        self.data["rmns"] = rmns
        self.data["zmnc"] = zmnc
        self.data["zmns"] = zmns
        self.fourier = VMECFourierCoefficients(
            s=self.s_arr.copy(),
            xm=self.xm.copy(),
            xn=self.xn.copy(),
            rmnc=rmnc,
            rmns=rmns,
            zmnc=zmnc,
            zmns=zmns,
            nfp=self.nfp,
            lasym=bool(self.lasym),
        )

        self.xm_nyq = np.asarray(
            self.data.get("xm_nyq", self.xm), dtype=float
        )
        self.xn_nyq = np.asarray(
            self.data.get("xn_nyq", self.xn), dtype=float
        )
        if (
                self.xm_nyq.ndim != 1
                or self.xn_nyq.shape != self.xm_nyq.shape
                or not np.isfinite(self.xm_nyq).all()
                or not np.isfinite(self.xn_nyq).all()
        ):
            raise VMECValidationError(
                "VMEC Nyquist mode arrays must be finite matching 1-D arrays"
            )

    def _mode_array(self, name: str) -> np.ndarray:
        if name not in self.data:
            raise VMECValidationError(f"Missing required VMEC variable {name!r}")
        values = np.asarray(self.data[name], dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise VMECValidationError(
                f"VMEC variable {name!r} must be a nonempty 1-D array"
            )
        if not np.isfinite(values).all():
            raise VMECValidationError(
                f"VMEC variable {name!r} contains non-finite values"
            )
        if not np.allclose(values, np.rint(values)):
            raise VMECValidationError(
                f"VMEC variable {name!r} must contain integer mode numbers"
            )
        return values

    def _coefficient_array(self, name: str, *, required: bool) -> np.ndarray:
        expected = (self.ns, self.xm.size)
        if name not in self.data:
            if required:
                raise VMECValidationError(
                    f"Missing required VMEC coefficient variable {name!r}"
                )
            return np.zeros(expected, dtype=float)
        values = np.asarray(self.data[name], dtype=float)
        if values.shape != expected:
            raise VMECValidationError(
                f"VMEC coefficient {name!r} has shape {values.shape}; "
                f"expected {expected}"
            )
        if not np.isfinite(values).all():
            raise VMECValidationError(
                f"VMEC coefficient {name!r} contains non-finite values"
            )
        return values

    def _validate_radial_grid(self) -> None:
        if self.s_arr.shape != (self.ns,):
            raise VMECValidationError(
                f"VMEC radial coordinate 's' has shape {self.s_arr.shape}; "
                f"expected ({self.ns},)"
            )
        if not np.isfinite(self.s_arr).all():
            raise VMECValidationError(
                "VMEC radial coordinate 's' contains non-finite values"
            )
        if np.any(np.diff(self.s_arr) <= 0):
            raise VMECValidationError(
                "VMEC radial coordinate 's' must be strictly increasing"
            )
        if not np.isclose(self.s_arr[0], 0.0) or not np.isclose(
                self.s_arr[-1], 1.0
        ):
            raise VMECValidationError(
                "VMEC radial coordinate 's' must span normalized flux [0, 1]"
            )

    def _interpolate_coefficients(
            self,
            coefficients: np.ndarray,
            s: np.ndarray,
    ) -> np.ndarray:
        flat_s = s.ravel()
        upper = np.searchsorted(self.s_arr, flat_s, side="right")
        upper = np.clip(upper, 1, self.ns - 1)
        lower = upper - 1
        fraction = (
            (flat_s - self.s_arr[lower])
            / (self.s_arr[upper] - self.s_arr[lower])
        )
        values = (
            coefficients[lower]
            + fraction[:, None]
            * (coefficients[upper] - coefficients[lower])
        )
        return values.reshape(s.shape + (self.xm.size,))

    def to_cylindrical(
            self,
            s: Any,
            theta: Any,
            phi: Any,
            *,
            bounds: Literal["raise", "clip"] = "raise",
    ) -> CylindricalCoordinates:
        """Evaluate cylindrical coordinates at broadcast magnetic coordinates.

        Parameters
        ----------
        s : array_like (...)
            Normalized toroidal flux.
        theta : array_like (...)
            VMEC poloidal angle (rad).
        phi : array_like (...)
            Geometric cylindrical toroidal angle (rad).
        bounds : {'raise', 'clip'}, optional
            Handling of ``s`` outside ``[0, 1]``.  ``"raise"`` rejects such
            values and ``"clip"`` evaluates the nearest boundary.  Default is
            ``"raise"``.

        Returns
        -------
        coordinates : CylindricalCoordinates
            ``R``, ``Z``, and ``phi`` arrays in metres, metres, and radians.
            Each array has the broadcast shape of ``s``, ``theta``, and
            ``phi``.

        Raises
        ------
        ValueError
            If inputs cannot be broadcast, contain non-finite values, or ``s``
            lies outside the equilibrium with ``bounds="raise"``.

        Notes
        -----
        Coefficients are linearly interpolated in ``s`` between stored
        surfaces.  The Fourier phase is ``xm*theta - xn*phi``; ``xn`` already
        includes ``nfp``.  Angles are periodic and are not otherwise modified.
        No radial extrapolation is performed.
        """
        if bounds not in ("raise", "clip"):
            raise ValueError("bounds must be 'raise' or 'clip'")
        try:
            s_array, theta_array, phi_array = np.broadcast_arrays(
                np.asarray(s, dtype=float),
                np.asarray(theta, dtype=float),
                np.asarray(phi, dtype=float),
            )
        except ValueError as exc:
            raise ValueError(
                "s, theta, and phi must be broadcast-compatible"
            ) from exc
        if not (
                np.isfinite(s_array).all()
                and np.isfinite(theta_array).all()
                and np.isfinite(phi_array).all()
        ):
            raise ValueError("s, theta, and phi must contain only finite values")
        if bounds == "raise" and np.any((s_array < 0.0) | (s_array > 1.0)):
            raise ValueError("s lies outside the VMEC equilibrium interval [0, 1]")
        evaluated_s = np.clip(s_array, 0.0, 1.0)
        phase = (
            theta_array[..., None] * self.xm
            - phi_array[..., None] * self.xn
        )
        cosine = np.cos(phase)
        sine = np.sin(phase)
        rmnc = self._interpolate_coefficients(self.fourier.rmnc, evaluated_s)
        rmns = self._interpolate_coefficients(self.fourier.rmns, evaluated_s)
        zmnc = self._interpolate_coefficients(self.fourier.zmnc, evaluated_s)
        zmns = self._interpolate_coefficients(self.fourier.zmns, evaluated_s)
        radius = np.sum(rmnc * cosine + rmns * sine, axis=-1)
        vertical = np.sum(zmnc * cosine + zmns * sine, axis=-1)
        return CylindricalCoordinates(radius, vertical, phi_array.copy())

    def to_cartesian(
            self,
            s: Any,
            theta: Any,
            phi: Any,
            *,
            bounds: Literal["raise", "clip"] = "raise",
    ) -> CartesianCoordinates:
        """Evaluate Cartesian coordinates at broadcast magnetic coordinates.

        Parameters
        ----------
        s : array_like (...)
            Normalized toroidal flux.
        theta : array_like (...)
            VMEC poloidal angle (rad).
        phi : array_like (...)
            Geometric cylindrical toroidal angle (rad).
        bounds : {'raise', 'clip'}, optional
            Radial-bound handling passed to :meth:`to_cylindrical`.  Default is
            ``"raise"``.

        Returns
        -------
        coordinates : CartesianCoordinates
            ``x``, ``y``, and ``z`` arrays (m), each with the broadcast input
            shape.
        """
        radius, vertical, toroidal = self.to_cylindrical(
            s, theta, phi, bounds=bounds
        )
        return CartesianCoordinates(
            radius * np.cos(toroidal),
            radius * np.sin(toroidal),
            vertical,
        )

    @staticmethod
    def _inside_polygon(
            radius: float,
            vertical: float,
            boundary_r: np.ndarray,
            boundary_z: np.ndarray,
    ) -> bool:
        shifted = np.roll(np.arange(boundary_r.size), 1)
        r0 = boundary_r[shifted]
        z0 = boundary_z[shifted]
        r1 = boundary_r
        z1 = boundary_z
        crosses = (z0 > vertical) != (z1 > vertical)
        denominator = np.where(z1 == z0, np.finfo(float).eps, z1 - z0)
        intersections = r0 + (vertical - z0) * (r1 - r0) / denominator
        return bool(np.count_nonzero(crosses & (radius < intersections)) % 2)

    def from_cartesian(
            self,
            x: Any,
            y: Any,
            z: Any,
            *,
            tol: float = 1e-9,
            max_nfev: int = 100,
            coarse_s: int = 17,
            coarse_theta: int = 64,
            boundary_theta: int = 256,
            axis_tol: float | None = None,
    ) -> InverseCoordinateResult:
        """Invert broadcast Cartesian points to VMEC magnetic coordinates.

        Parameters
        ----------
        x : array_like (...)
            Cartesian x coordinate (m).
        y : array_like (...)
            Cartesian y coordinate (m).
        z : array_like (...)
            Cartesian z coordinate (m).
        tol : float, optional
            Solver and accepted geometric residual tolerance (m).  Default is
            1e-9.
        max_nfev : int, optional
            Maximum local-solver function evaluations per point.  Default is
            100.
        coarse_s : int, optional
            Number of radial initial-guess samples.  Default is 17.
        coarse_theta : int, optional
            Number of poloidal initial-guess samples.  Default is 64.
        boundary_theta : int, optional
            Number of LCFS samples used for inside/outside classification.
            Default is 256.
        axis_tol : float, optional
            Distance from the magnetic axis treated as axis-degenerate (m).
            Default is ``max(10*tol, 1e-9)``.

        Returns
        -------
        result : InverseCoordinateResult
            Magnetic coordinates, validity, residual (m), and status arrays
            with the broadcast input shape.

        Raises
        ------
        ValueError
            If inputs are invalid or solver controls are out of range.

        Notes
        -----
        ``phi = atan2(y, x)`` is normalized to ``[0, 2*pi)``.  At each fixed
        ``phi``, LCFS polygon classification precedes a coarse ``(s, theta)``
        search and bounded nonlinear least-squares solve in the ``R-Z`` plane.
        The magnetic-axis poloidal angle is indeterminate; axis points return
        the representative value ``theta=0`` and status ``"axis"``.
        """
        try:
            x_array, y_array, z_array = np.broadcast_arrays(
                np.asarray(x, dtype=float),
                np.asarray(y, dtype=float),
                np.asarray(z, dtype=float),
            )
        except ValueError as exc:
            raise ValueError("x, y, and z must be broadcast-compatible") from exc
        if not (
                np.isfinite(x_array).all()
                and np.isfinite(y_array).all()
                and np.isfinite(z_array).all()
        ):
            raise ValueError("x, y, and z must contain only finite values")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_nfev < 1:
            raise ValueError("max_nfev must be positive")
        if coarse_s < 2 or coarse_theta < 8 or boundary_theta < 16:
            raise ValueError(
                "coarse_s must be at least 2, coarse_theta at least 8, "
                "and boundary_theta at least 16"
            )
        if axis_tol is None:
            axis_tol = max(10.0 * tol, 1e-9)
        if axis_tol < 0:
            raise ValueError("axis_tol must be nonnegative")
        solver_tol = max(0.1 * tol, 10.0 * np.finfo(float).eps)

        shape = x_array.shape
        flat_r = np.hypot(x_array, y_array).ravel()
        flat_z = z_array.ravel()
        flat_phi = np.mod(np.arctan2(y_array, x_array), 2.0 * np.pi).ravel()
        out_s = np.full(flat_r.shape, np.nan)
        out_theta = np.full(flat_r.shape, np.nan)
        out_residual = np.full(flat_r.shape, np.inf)
        out_valid = np.zeros(flat_r.shape, dtype=bool)
        out_status = np.full(flat_r.shape, "nonconverged", dtype="<U16")
        theta_boundary = np.linspace(
            0.0, 2.0 * np.pi, boundary_theta, endpoint=False
        )
        guess_s, guess_theta = np.meshgrid(
            np.linspace(0.0, 1.0, coarse_s),
            np.linspace(0.0, 2.0 * np.pi, coarse_theta, endpoint=False),
            indexing="ij",
        )

        for index, (target_r, target_z, toroidal) in enumerate(
                zip(flat_r, flat_z, flat_phi)
        ):
            boundary_r, boundary_z, _ = self.to_cylindrical(
                1.0, theta_boundary, toroidal
            )
            boundary_distance = np.hypot(
                boundary_r - target_r, boundary_z - target_z
            )
            on_boundary = float(np.min(boundary_distance)) <= tol
            inside_lcfs = self._inside_polygon(
                target_r, target_z, boundary_r, boundary_z
            )
            if not on_boundary and not inside_lcfs:
                nearest = int(np.argmin(boundary_distance))

                def boundary_error(value: np.ndarray) -> np.ndarray:
                    r_value, z_value, _ = self.to_cylindrical(
                        1.0, value[0], toroidal
                    )
                    return np.array([r_value - target_r, z_value - target_z])

                boundary_solution = least_squares(
                    boundary_error,
                    np.array([theta_boundary[nearest]]),
                    xtol=solver_tol,
                    ftol=solver_tol,
                    gtol=solver_tol,
                    max_nfev=max_nfev,
                )
                boundary_residual = float(np.linalg.norm(boundary_solution.fun))
                if boundary_residual > tol:
                    out_residual[index] = boundary_residual
                    out_status[index] = "outside_lcfs"
                    continue
                out_s[index] = 1.0
                out_theta[index] = np.mod(
                    boundary_solution.x[0], 2.0 * np.pi
                )
                out_residual[index] = boundary_residual
                out_valid[index] = True
                out_status[index] = "converged"
                continue

            axis_r, axis_z, _ = self.to_cylindrical(
                0.0, theta_boundary, toroidal
            )
            axis_distance = np.hypot(axis_r - target_r, axis_z - target_z)
            axis_residual = float(np.min(axis_distance))
            if axis_residual <= axis_tol:
                out_s[index] = 0.0
                out_theta[index] = 0.0
                out_residual[index] = axis_residual
                out_valid[index] = True
                out_status[index] = "axis"
                continue

            coarse_r, coarse_z, _ = self.to_cylindrical(
                guess_s, guess_theta, toroidal
            )
            coarse_distance = np.hypot(
                coarse_r - target_r, coarse_z - target_z
            )
            nearest = np.unravel_index(
                int(np.argmin(coarse_distance)), coarse_distance.shape
            )
            initial = np.array(
                [guess_s[nearest], guess_theta[nearest]], dtype=float
            )

            def error(value: np.ndarray) -> np.ndarray:
                r_value, z_value, _ = self.to_cylindrical(
                    value[0], np.mod(value[1], 2.0 * np.pi), toroidal
                )
                return np.array([r_value - target_r, z_value - target_z])

            solution = least_squares(
                error,
                initial,
                bounds=([0.0, -np.inf], [1.0, np.inf]),
                xtol=solver_tol,
                ftol=solver_tol,
                gtol=solver_tol,
                max_nfev=max_nfev,
            )
            residual = float(np.linalg.norm(solution.fun))
            out_s[index] = solution.x[0]
            out_theta[index] = np.mod(solution.x[1], 2.0 * np.pi)
            out_residual[index] = residual
            if solution.success and residual <= tol:
                out_valid[index] = True
                out_status[index] = "converged"

        reshaped_s = out_s.reshape(shape)
        return InverseCoordinateResult(
            s=reshaped_s,
            rho=np.sqrt(reshaped_s),
            theta=out_theta.reshape(shape),
            phi=flat_phi.reshape(shape),
            valid=out_valid.reshape(shape),
            residual=out_residual.reshape(shape),
            status=out_status.reshape(shape),
        )

    @staticmethod
    def _validate_angles(u_arr: np.ndarray, v_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_arr = np.asarray(u_arr, dtype=float)
        v_arr = np.asarray(v_arr, dtype=float)
        if u_arr.ndim != 1 or v_arr.ndim != 1:
            raise ValueError("u_arr and v_arr must be one-dimensional arrays")
        return u_arr, v_arr

    def get_derivatives(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the surface coordinates (R, Z) and their derivatives with respect to s, u, v.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        x : tuple of np.ndarray
            (R, Z, Zeta) arrays of shape (ns, nu, nv).
        dx_ds : tuple of np.ndarray
            (dR/ds, dZ/ds) arrays of shape (ns, nu, nv).
        dx_du : tuple of np.ndarray
            (dR/du, dZ/du) arrays of shape (ns, nu, nv).
        dx_dv : tuple of np.ndarray
            (dR/dv, dZ/dv) arrays of shape (ns, nu, nv).
        """

        u_arr, v_arr = self._validate_angles(u_arr, v_arr)

        R_mn_c = self.data.get('rmnc', np.zeros((self.ns, len(self.xm))))  # (ns, mn)
        R_mn_s = self.data.get('rmns', np.zeros((self.ns, len(self.xm))))  # (ns, mn)
        Z_mn_c = self.data.get('zmnc', np.zeros((self.ns, len(self.xm))))  # (ns, mn)
        Z_mn_s = self.data.get('zmns', np.zeros((self.ns, len(self.xm))))  # (ns, mn)

        dR_mn_c_ds = np.gradient(R_mn_c, self.s_arr, axis=0)  # (ns, mn)
        dR_mn_s_ds = np.gradient(R_mn_s, self.s_arr, axis=0)  # (ns, mn)
        dZ_mn_c_ds = np.gradient(Z_mn_c, self.s_arr, axis=0)  # (ns, mn)
        dZ_mn_s_ds = np.gradient(Z_mn_s, self.s_arr, axis=0)  # (ns, mn)

        angles = self.xm[:, None, None] * u_arr[None, :, None] \
                 - self.xn[:, None, None] * v_arr[None, None, :]  # (mn, nu, nv)
        cosines = np.cos(angles)  # (mn, nu, nv)
        sines = np.sin(angles)  # (mn, nu, nv)

        R = np.tensordot(R_mn_c, cosines, axes=([1], [0])) \
            + np.tensordot(R_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        Z = np.tensordot(Z_mn_c, cosines, axes=([1], [0])) \
            + np.tensordot(Z_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        Zeta = np.broadcast_to(
            v_arr[None, None, :], (self.ns, u_arr.size, v_arr.size)
        )  # (ns, nu, nv)

        dR_ds = np.tensordot(dR_mn_c_ds, cosines, axes=([1], [0])) \
                + np.tensordot(dR_mn_s_ds, sines, axes=([1], [0]))  # (ns, nu, nv)
        dZ_ds = np.tensordot(dZ_mn_c_ds, cosines, axes=([1], [0])) \
                + np.tensordot(dZ_mn_s_ds, sines, axes=([1], [0]))  # (ns, nu, nv)

        dR_du = np.tensordot(R_mn_c, self.xm[:, None, None] * (-sines), axes=([1], [0])) \
                + np.tensordot(R_mn_s, self.xm[:, None, None] * cosines, axes=([1], [0]))  # (ns, nu, nv)
        dZ_du = np.tensordot(Z_mn_c, self.xm[:, None, None] * (-sines), axes=([1], [0])) \
                + np.tensordot(Z_mn_s, self.xm[:, None, None] * cosines, axes=([1], [0]))  # (ns, nu, nv)

        dR_dv = np.tensordot(R_mn_c, (-self.xn[:, None, None]) * (-sines), axes=([1], [0])) \
                + np.tensordot(R_mn_s, (-self.xn[:, None, None]) * cosines, axes=([1], [0]))  # (ns, nu, nv)
        dZ_dv = np.tensordot(Z_mn_c, (-self.xn[:, None, None]) * (-sines), axes=([1], [0])) \
                + np.tensordot(Z_mn_s, (-self.xn[:, None, None]) * cosines, axes=([1], [0]))  # (ns, nu, nv)

        return (R, Z, Zeta), (dR_ds, dZ_ds), (dR_du, dZ_du), (dR_dv, dZ_dv)

    def get_B_field_contravariant(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the contravariant components of the magnetic field (B^u, B^v) in the (u, v) coordinate system.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        B_sup : tuple of np.ndarray
            (B^u, B^v) arrays of shape (ns, nu, nv).
        """

        u_arr, v_arr = self._validate_angles(u_arr, v_arr)
        B_sup_u_mn_c = self.data.get('bsupumnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        B_sup_u_mn_s = self.data.get('bsupumns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        B_sup_v_mn_c = self.data.get('bsupvmnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        B_sup_v_mn_s = self.data.get('bsupvmns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)

        angles = self.xm_nyq[:, None, None] * u_arr[None, :, None] \
                 - self.xn_nyq[:, None, None] * v_arr[None, None, :]  # (mn_nyq, nu, nv)
        cosines = np.cos(angles)  # (mn_nyq, nu, nv)
        sines = np.sin(angles)  # (mn_nyq, nu, nv)

        B_sup_u = np.tensordot(B_sup_u_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(B_sup_u_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        B_sup_v = np.tensordot(B_sup_v_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(B_sup_v_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)

        return (B_sup_u, B_sup_v)

    def get_current_contravariant(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """Calculate contravariant current components ``(J^u, J^v)``."""
        u_arr, v_arr = self._validate_angles(u_arr, v_arr)
        J_sup_u_mn_c = self.data.get('currumnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        J_sup_u_mn_s = self.data.get('currumns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        J_sup_v_mn_c = self.data.get('currvmnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        J_sup_v_mn_s = self.data.get('currvmns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)

        angles = self.xm_nyq[:, None, None] * u_arr[None, :, None] \
                 - self.xn_nyq[:, None, None] * v_arr[None, None, :]  # (mn_nyq, nu, nv)
        cosines = np.cos(angles)  # (mn_nyq, nu, nv)
        sines = np.sin(angles)  # (mn_nyq, nu, nv)

        # \sum_mn J^u_mnc * cos(angles) + \sum_mn J^u_mns * sin(angles)
        J_sup_u = np.tensordot(J_sup_u_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(J_sup_u_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        J_sup_v = np.tensordot(J_sup_v_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(J_sup_v_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)

        return (J_sup_u, J_sup_v)

    def get_current_contraveriant(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """Deprecated misspelling of :meth:`get_current_contravariant`."""
        return self.get_current_contravariant(u_arr, v_arr)


    def get_B_field_cylindrical(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the magnetic field components (B_R, B_Z, B_phi) in cylindrical coordinates.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        B : tuple of np.ndarray
            (B_R, B_Z, B_Zeta) arrays of shape (ns, nu, nv).
        """
        (R, Z, Zeta), (dR_ds, dZ_ds), (dR_du, dZ_du), (dR_dv, dZ_dv) = self.get_derivatives(u_arr, v_arr)
        (B_sup_u, B_sup_v) = self.get_B_field_contravariant(u_arr, v_arr)

        # B_R = B^u * dR/du + B^v * dR/dv
        # B_Zeta = B^v * R
        # B_Z = B^u * dZ/du + B^v * dZ/dv
        B_R = B_sup_u * dR_du + B_sup_v * dR_dv  # (ns, nu, nv)
        B_Z = B_sup_u * dZ_du + B_sup_v * dZ_dv  # (ns, nu, nv)
        B_Zeta = B_sup_v * R  # (ns, nu, nv)

        return (R, Z, Zeta), (B_R, B_Z, B_Zeta)

    def get_current_cylindrical(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate current components (J_R, J_Z, J_phi) in cylindrical coordinates.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        J : tuple of np.ndarray
            (J_R, J_Z, J_Zeta) arrays of shape (ns, nu, nv).
        """
        (R, Z, Zeta), (dR_ds, dZ_ds), (dR_du, dZ_du), (dR_dv, dZ_dv) = self.get_derivatives(u_arr, v_arr)
        (J_sup_u, J_sup_v) = self.get_current_contravariant(u_arr, v_arr)

        # J_R = J^u * dR/du + J^v * dR/dv
        # J_Zeta = J^v * R
        # J_Z = J^u * dZ/du + J^v * dZ/dv
        J_R = J_sup_u * dR_du + J_sup_v * dR_dv  # (ns, nu, nv)
        J_Z = J_sup_u * dZ_du + J_sup_v * dZ_dv  # (ns, nu, nv)
        J_Zeta = J_sup_v * R  # (ns, nu, nv)

        return (R, Z, Zeta), (J_R, J_Z, J_Zeta)

    def get_B_field_cartesian(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the magnetic field components (B_x, B_y, B_z) in Cartesian coordinates.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        B : tuple of np.ndarray
            (B_x, B_y, B_z) arrays of shape (ns, nu, nv).
        """
        (R, Z, Zeta), (B_R, B_Z, B_phi) = self.get_B_field_cylindrical(u_arr, v_arr)

        # Convert cylindrical to Cartesian coordinates
        X = R * np.cos(Zeta)
        Y = R * np.sin(Zeta)

        # Convert magnetic field components from cylindrical to Cartesian coordinates
        B_X = B_R * np.cos(Zeta) - B_phi * np.sin(Zeta)
        B_Y = B_R * np.sin(Zeta) + B_phi * np.cos(Zeta)

        return (X, Y, Z), (B_X, B_Y, B_Z)
