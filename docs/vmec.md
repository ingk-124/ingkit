# VMEC API migration

The legacy `ingkit.io.read_vmec` module and its exports from `ingkit.io`
were removed in ingkit 0.3.0. Use the independent low-level API:

```python
from ingkit.physics.plasma.vmec import VMECData

vmec = VMECData("wout_example.nc")
xyz = vmec.flux_surface(s=0.5, theta=1.0, phi=0.2)
s, theta_vmec = vmec.points_to_flux(xyz, coordinates="xyz", return_theta=True)
```

The new class is not a drop-in alias for the old class:

- `flux_to_rpz()` and `flux_surface()` evaluate the tensor product of scalar
  or one-dimensional flux-coordinate inputs. Scalar inputs omit their axes.
- Point-based field methods accept paired arrays with shape `(..., 3)`.
- Cylindrical positions and vector components use `(R, phi, Z)` ordering.
- `points_to_flux()` returns `s`, or `(s, theta_vmec)` when requested.
  Outside points return `NaN`; the legacy inverse-result/status types and
  nonlinear-solver controls are no longer provided.
- Real-space inversion requires star-shaped sampled magnetic surfaces
  and increasing radii about the magnetic axis. Its accuracy depends on
  the sampling grids; it does not have the old solver's tolerance contract.
- The legacy `from_fourier()` constructor and coordinate/coefficient
  container classes have no compatibility wrappers.

See [Low-level VMEC API](vmec_low_level.md) for the complete current API,
units, coordinate conventions, and boundary handling.
