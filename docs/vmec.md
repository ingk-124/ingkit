# VMEC equilibrium and coordinate mapping

## Scope

`ingkit.io` provides a validated NetCDF `wout_*.nc` reader, a common Fourier
coefficient representation, forward coordinate maps, and a numerical inverse
map. It does not write VMEC files.

The module deliberately excludes camera and multi-pinhole concerns: voxels,
visibility, projection matrices, detector geometry, plasma profiles (`Te`,
`ne`), radiation models, and emission integration remain the responsibility of
downstream code. This keeps `ingkit` independent of any multi-pinhole package.

## Coordinate and Fourier convention

- `s` is normalized toroidal flux on `[0, 1]`.
- `rho = sqrt(s)` is the normalized flux radius.
- `theta` is the VMEC poloidal angle in radians. It is not converted to a PEST
  or straight-field-line angle.
- `phi` is the geometric cylindrical toroidal angle in radians:
  `x = R cos(phi)` and `y = R sin(phi)`.
- `R`, `Z`, `x`, `y`, and `z` are in metres, following the VMEC `wout`
  coefficient units.

For every stored mode, the phase is

```text
alpha = xm * theta - xn * phi
R = sum(rmnc * cos(alpha) + rmns * sin(alpha))
Z = sum(zmnc * cos(alpha) + zmns * sin(alpha))
```

VMEC `wout` stores `xn` with the field-period factor `nfp` already included.
`ingkit` therefore does not multiply `xn` by `nfp`. It validates that `xn/nfp`
is integral. The resulting geometry is periodic in `theta` with period `2*pi`
and in `phi` with field period `2*pi/nfp`.

For stellarator-symmetric equilibria (`lasym = 0`), `rmns` and `zmnc` may be
absent and are represented by zeros. `rmnc` and `zmns` remain required. All
four arrays are required when `lasym = 1`.

Standard `wout` files do not carry a dedicated normalized `s` variable, so the
full radial mesh is represented as `linspace(0, 1, ns)`. An explicit `s` array
is honored by the in-memory constructor and must be finite, strictly
increasing, and span `[0, 1]`.

## Public API

```python
import numpy as np
from ingkit.io import VMECData

equilibrium = VMECData("wout_example.nc")

# Paired or broadcast evaluation; every output has shape (2, 3).
s = np.array([[0.25], [0.75]])
theta = np.array([[0.0, np.pi / 2, np.pi]])
R, Z, phi = equilibrium.to_cylindrical(s, theta, 0.2)
x, y, z = equilibrium.to_cartesian(s, theta, 0.2)

inverse = equilibrium.from_cartesian(x, y, z)
usable = inverse.valid
```

`to_cylindrical` and `to_cartesian` accept scalar, paired-array, and general
NumPy-broadcast inputs. Fourier coefficients are linearly interpolated in `s`.
The default `bounds="raise"` rejects points outside `[0, 1]`; `bounds="clip"`
is available only when explicitly requested. Radial extrapolation and silent
NaN filling are not performed.

`VMECData.from_fourier(...)` constructs a validated equilibrium from analytic
or externally loaded coefficients. `VMECFourierCoefficients` is the common
validated coefficient representation held by `equilibrium.fourier`.

The existing tensor-product methods (`get_derivatives`, field/current
evaluation, and the misspelled compatibility alias
`get_current_contraveriant`) remain available. They continue to return arrays
of shape `(ns, nu, nv)`.

## Inverse algorithm and status

`from_cartesian` broadcasts `x`, `y`, and `z`, then processes each point:

1. Compute `R = hypot(x, y)` and normalize `phi = atan2(y, x)` to
   `[0, 2*pi)`.
2. Sample the LCFS at that fixed `phi` and classify the point with an `R-Z`
   polygon.
3. Detect the magnetic-axis degeneracy.
4. Search a coarse `(s, theta)` grid for an initial guess.
5. Use SciPy bounded nonlinear least squares for `(R, Z)`, with `s` constrained
   to `[0, 1]`.

Solver controls are keyword arguments: `tol`, `max_nfev`, `coarse_s`,
`coarse_theta`, `boundary_theta`, and `axis_tol`.

The result contains `s`, `rho`, `theta`, `phi`, `valid`, `residual`, and
`status`, all with the broadcast input shape. `residual` is the Euclidean
`R-Z` error in metres. Status values are:

- `converged`: a solution met the requested tolerance.
- `axis`: the point is within `axis_tol` of the magnetic axis. Since `theta`
  is indeterminate there, the representative value `theta = 0` is returned.
- `outside_lcfs`: the point is outside the sampled LCFS. `s`, `rho`, and
  `theta` are NaN, `valid` is false, and `residual` is the distance to the
  optimized boundary point.
- `nonconverged`: a local solve did not produce an accepted solution.
  The best attempted `s`, `theta`, and residual are retained while `valid` is
  false.

The LCFS test assumes that the fixed-`phi` boundary is a simple closed curve.
Very unusual self-intersecting or non-nested equilibria require a specialized
topology-aware classifier.

## Text `.out` backend assessment

The reference `load_vmec_out_fourier` parser targets a human-readable KSPDIAG
report (`<< KSPDIAG >> (1992/Y.N)`) containing tables headed by strings such as
`Fourier Coefficients R(m,n)`. The inspected Heliotron J example declares
`nfp = 4`, and its table mode numbers are multiples of four, consistent with
pre-scaled VMEC `xn`. It labels the radial column `rho` and supplies one table
each for `R`, `Z`, `B`, and `l`, but it does not provide a versioned schema or
machine-readable declarations for the radial-flux definition and Fourier
parity. In particular, there is no general asymmetric counterpart to map
unambiguously onto `rmnc`, `rmns`, `zmnc`, and `zmns`.

The additional VMEC 10.0 fixtures use another text format entirely:
`LWOUTTXT = .TRUE.` produces positional `wout_*.txt` streams alongside
NetCDF files. The positional format is not the KSPDIAG table format. The three
NetCDF fixtures packaged with `ingkit` were confirmed byte-for-byte identical
to their corresponding `vmectest` outputs, and their `LASYM` input settings
agree with the coefficient variables present in NetCDF. These fixtures support
the NetCDF convention tests but do not define either text format as a stable
API.

Publishing one generic `.out`/`.txt` parser would therefore conflate two
formats and freeze assumptions that cannot currently be validated.

No text `.out` backend is exposed in this release. A future optional backend
should first define supported producer/version signatures, parse into an
intermediate table with explicit coefficient parity and mode convention, and
then construct `VMECFourierCoefficients`. It must remain independent of the
NetCDF reader and must use format-specific fixture tests.

## Downstream multi-pinhole adapter contract

A downstream adapter should accept a `VMECData`-compatible equilibrium and:

- pass NumPy-broadcastable `s`, `theta`, and `phi` arrays in radians;
- consume Cartesian outputs in metres;
- use `inverse.valid` and `inverse.status`, rather than testing NaNs alone;
- decide whether outside/nonconverged points are discarded, masked, or
  surfaced to users;
- own all voxelization, camera, visibility, projection, and emission logic.

The adapter may depend on `ingkit`; `ingkit` must not depend on the adapter or
the multi-pinhole package.
