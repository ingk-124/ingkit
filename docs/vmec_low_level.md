# Low-level VMEC API

`ingkit.physics.plasma.vmec.VMECData` reads a VMEC `wout` NetCDF file and
evaluates the magnetic geometry. It provides magnetic-axis and surface
coordinates, LCFS membership, inverse flux coordinates, and equilibrium fields.

```python
from ingkit.physics.plasma.vmec import VMECData

vmec = VMECData("wout_example.nc")
axis = vmec.magnetic_axis()
lcfs = vmec.flux_surface(s=1.0)
```

## Data and grids

- `path` is the resolved source path. `wout`, `metadata`, and
  `variable_attributes` expose the NetCDF variables and attributes through
  read-only mappings. The stored variable arrays are marked non-writeable.
- `s_grid` is the normalized toroidal-flux grid of length `ns`, spanning
  `[0, 1]`. `rho_grid = sqrt(s_grid)`.
- `theta_grid` and `phi_grid` both span `[0, 2*pi]`, including the endpoint.
  Their interval counts are `(2*max(abs(xm_nyq)) + 1)*oversample_theta`
  and `(2*max(abs(xn_nyq)) + 1)*oversample_phi`, respectively. Both
  oversampling factors default to 5. The angular and radial grid arrays are
  marked non-writeable.
- `oversample_rho` defaults to 3. It multiplies the number of radial intervals
  used for real-space flux evaluation. These internal samples are uniform in
  `rho = sqrt(s)`.

## Geometry

Cartesian coordinates `xyz` are ordered `(x, y, z)` in metres. Cylindrical
coordinates `rpz` are ordered `(R, phi, Z)` in metres, radians, and metres.

```python
axis = vmec.magnetic_axis(phi=[0.0, 0.2], coordinates="rpz")
section = vmec.flux_surface(s=[0.2, 0.5, 1.0], phi=[0.0], coordinates="rpz")
rpz = vmec.flux_to_rpz(0.5, [0.0, 0.7], [0.0, 0.2])
```

- `magnetic_axis()` takes optional keyword-only `phi` and `coordinates`.
  It defaults to the stored toroidal grid and returns `(n_phi, 3)`. A scalar
  `phi` produces one row. The axis is evaluated from the stored axis Fourier
  coefficients using phase `-n*nfp*phi`.
- `flux_to_rpz(s, theta_vmec, phi)` evaluates the tensor product of scalar or
  1-D inputs, in axis order `(s, theta_vmec, phi, coordinate)`. Each scalar
  omits its corresponding axis. For example, the call above returns
  `(2, 2, 3)`, with the two angle arrays evaluated as a (2 × 2) grid.
- `flux_surface()` uses the same tensor-product convention. It defaults to
  `s=1`, the stored angular grids, and Cartesian output. Pass a length-one
  array such as `phi=[0.0]` to retain the toroidal axis of a single section.
  The `section` above has shape `(3, n_theta, 1, 3)`.
- Geometry coefficients are interpolated linearly in `s` over `[0, 1]`.
  The surface Fourier phase is `xm*theta_vmec - xn*phi`. The stored `xn`
  already includes the field-period factor.

## LCFS membership and real-space flux

```python
points = [[1.2, 0.1, 0.0], [1.3, 0.1, 0.0]]  # R, phi, Z
inside = vmec.is_inside_lcfs(points, coordinates="rpz")
s = vmec.points_to_flux(points, coordinates="rpz")
s, theta_vmec = vmec.points_to_flux(points, return_theta=True)
s_rz = vmec.s_on_rz_grid(R=[1.2, 1.3], Z=[-0.1, 0.0, 0.1], phi=0.1)
```

`is_inside_lcfs()` and `points_to_flux()` accept finite positions with shape
`(..., 3)`. Both default to cylindrical input and return
`points.shape[:-1]`. `s_on_rz_grid()` evaluates all combinations of 1-D
`R` and `Z` arrays at a fixed `phi`, returning `(len(R), len(Z))`.

The calculation folds `phi` into one magnetic field period, `[0, 2*pi/nfp)`.
At each toroidal section the magnetic axis defines the geometric angle
`alpha` and distance `r`. For each sampled surface, the radius is
interpolated periodically in `alpha` at the two surrounding toroidal
sections, then linearly between the sections. Surface radii are evaluated
within each call. The LCFS query uses the outermost surface alone.

The relative radial tolerance `tol` defaults to `1e-5`. The inside mask is
`r <= r_b*(1 + tol)`, where `r_b` is the interpolated LCFS radius.
`points_to_flux()` assigns `NaN` outside that bound, `s=1` between `r_b`
and the tolerance bound, and `s=0` on the magnetic axis. In the interior,
it finds the surfaces bracketing `r`, interpolates linearly in `rho`, and
squares the result to return `s`.

With `return_theta=True`, `points_to_flux()` returns `(s, theta_vmec)` with
the same shape for both arrays. It inverts the geometric direction on the
sampled surfaces and interpolates the resulting VMEC angle periodically
across toroidal sections and between surfaces. Inside the first sampled
surface it uses that surface's angle. The returned angle is in radians
in `[0, 2*pi)`, with zero on the magnetic axis and `NaN` outside the
LCFS tolerance.

The method requires star-shaped sampled surfaces about the magnetic axis and
increasing interpolated radii with increasing `s`. A failed geometry check
raises `ValueError`. LCFS locations and interior flux values depend on the
angular and radial sampling.

## Star-shape diagnostic

```python
derivative = vmec.lcfs_angle_derivative(s=1.0)  # (n_theta, n_phi)
star_shaped = vmec.is_star_shaped(s=1.0)       # (n_phi,)
```

Both methods accept scalar or 1-D `theta` and `phi` samples. A scalar
angle retains a length-one axis in the diagnostic output. An array of
`s` values in `(0, 1]` adds a leading surface axis. The derivative is
`d(alpha)/d(theta_vmec)` for the axis-relative geometric angle `alpha`,
evaluated from Fourier derivatives. A sampled section passes when all
derivatives are strictly positive or all strictly negative. Mixed signs
or any zero fail the check. The check covers the sampled angles.

## Equilibrium fields

```python
flux_points = [[0.5, 0.7, 0.2], [1.0, 0.9, 0.3]]  # s, theta_vmec, phi
B_strength = vmec.magnetic_field_strength(flux_points)
B = vmec.magnetic_field(flux_points, basis="cylindrical")
J = vmec.current_density(flux_points, basis="cartesian")

B_strength_at = vmec.magnetic_field_strength_at(points)
B_at = vmec.magnetic_field_at(points)
J_at = vmec.current_at(points)
p_at = vmec.pressure_at(points)
```

The flux-coordinate methods take paired inputs of shape `(..., 3)`, with
each triple specifying one `(s, theta_vmec, phi)` point. The real-space
methods use the same `coordinates` and `tol` arguments as `points_to_flux()`.
Real-space evaluations beyond the LCFS tolerance return `NaN`.

- `magnetic_field_strength()` and `magnetic_field_strength_at()` return the
  Fourier-evaluated field magnitude in tesla. The result has the input shape
  without its last axis. The axis value uses only the poloidal mode `m=0`.
- `magnetic_field()` and `magnetic_field_at()` return the physical vector in
  tesla, with shape `(..., 3)`. On the axis, the vector follows the axis
  tangent with the `m=0` contribution to contravariant `B^phi`.
- `current_density()` and `current_at()` return flux-surface-tangent current
  density in A/m² with shape `(..., 3)`. They divide the stored current
  coefficients evaluated at each point by the signed Jacobian. The stored
  coefficients include `1/mu_0`. The axis returns `NaN`.
- The vector methods default to the cylindrical unit vector basis and
  return `(V_R, V_phi, V_Z)`. For `basis="cartesian"`, the components are
  `(V_x, V_y, V_z)`. The output basis is independent of the input coordinates.
- `pressure_at()` returns pressure in Pa with shape `points.shape[:-1]`,
  using linear interpolation of the full-mesh `presf` profile in `s`.

The field-strength, vector-field, and Jacobian coefficients are transferred
from the valid half-mesh rows to the full mesh by averaging adjacent rows
at interior nodes. The first row of the stored half-mesh data is unused.
With at least five full-mesh nodes, each endpoint is obtained by quadratic
extrapolation from its nearest three interior full-mesh nodes. With three
or four full-mesh nodes, the endpoints use their nearest interior values.
With two nodes, both endpoints use the single valid half-mesh row.

`dvolume_ds()` is pending specification of the source normalization and
endpoint treatment.
