# VMEC reference data

These fixed datasets are used for real-wout regression tests and plotting
examples. Keep them here, outside the importable Python package. They are
tracked in ordinary Git and are not included in the wheel. Unit tests with
analytic equilibria generate their own temporary NetCDF files.

## Provenance and stored metadata

The helical RFP, symmetric RFP, symmetric tokamak, and legacy text files were
previously stored in `src/ingkit/io/`. The bstd89 file was supplied in the
working tree and was also duplicated in `src/ingkit/physics/plasma/`.
The original author/source and complete calculation inputs are not documented;
the exact calculation conditions cannot currently be regenerated from this
repository. The values below are read from the NetCDF files, not inferred
from their names. `lasym=1` denotes a stellarator-asymmetric calculation.

| File | Stored VMEC version | nfp | ns | lasym | betatotal |
|---|---:|---:|---:|---:|---:|
| `wout_bstd89.nt800.nc` | 10.0 | 4 | 101 | 0 | 0.0 |
| `wout_helical_rfp_zero_beta.nc` | 10.0 | 1 | 61 | 1 | 0.11639932195339672 |
| `wout_symmetric_rfp_zero_beta.nc` | 10.0 | 1 | 61 | 0 | 0.0 |
| `wout_symmetric_tokamak_zero_beta.nc` | 10.0 | 1 | 61 | 0 | 0.0 |

Despite its filename, `wout_helical_rfp_zero_beta.nc` has nonzero stored
`betatotal`. Preserve the filename for provenance; use the stored values
when interpreting the equilibrium.

`wout_symmetry_rfp.txt` is retained as a legacy reference. Its generating
VMEC version is unknown; the current API reads NetCDF, not this text format.

## Versioning policy

- Keep one copy of each reference file. Test and demo code must reference
  this directory rather than paths inside `src/` or personal directories.
- Treat these files as fixed inputs. Explain any replacement in its commit
  and update the checksums below. Distinct calculation cases get distinct names.
- Record the source, VMEC version, input parameters or input-file location,
  and intended tests when adding a new reference equilibrium. Mark unknown
  provenance explicitly rather than guessing it.
- Store personal experiment data and bulk VMEC runs outside this repository.
  Reconsider external versioned datasets or Git LFS if the reference set grows.
- Run real-data tests from a source checkout. Plotting demos that use these
  files also require a checkout; a wheel installation does not provide them.

## SHA-256

```text
981c1d2039d9bd16c5012a5d5a61d38177092e25e742d5b8326e18d101379e2f  wout_bstd89.nt800.nc
2416384f2951a70ef4359f88d1d9db84d0d0f189e8c1ba0fca9bf71f6401b3f7  wout_helical_rfp_zero_beta.nc
f5d789cd1d8d919a3740f78a7588ec3b50fb6716708578b4d9973a5868214bb0  wout_symmetric_rfp_zero_beta.nc
a9ef0d1452413e54c7d018bdfa482f9cabb3c10b851a55bace959dee421d22a1  wout_symmetric_tokamak_zero_beta.nc
2a7cf9bda152f3a568959d8cf1bf97addeaf27c462649e30fd8472a66f5bbbb5  wout_symmetry_rfp.txt
```
