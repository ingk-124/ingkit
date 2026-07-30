# Changelog

## 0.2.0

- Validate VMEC NetCDF geometry variables, mode conventions, coefficient
  shapes, finite values, and the normalized radial grid.
- Add broadcast-aware cylindrical and Cartesian forward coordinate maps with
  radial coefficient interpolation and explicit LCFS bounds handling.
- Add structured Cartesian-to-VMEC inversion with coarse initialization,
  local least-squares refinement, validity, residual, and failure status.
- Preserve the existing `VMECData` tensor-grid field/current APIs.
- Document VMEC phase conventions, the multi-pinhole responsibility boundary,
  and why the format-specific text `.out` parser is not public.
- Declare the package's runtime dependencies and add `netCDF4` to the
  synchronized requirements list.
