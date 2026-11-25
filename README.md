# ingkit

`ingkit` is a Python utility toolkit for signal processing, file I/O, plotting, and physics-related calculations.
It is built on top of the scientific Python ecosystem (NumPy / SciPy / Matplotlib).

`ingkit` は、信号処理・ファイル入出力・可視化・物理計算を支援する Python ツールキットです。
科学技術計算向けの主要ライブラリ（NumPy / SciPy / Matplotlib）を前提に設計されています。

## Installation

```bash
git clone <repository-url>
cd ingkit
pip install -r requirements.txt
pip install -e .
```

## Package structure

- `ingkit.signals`
  - Signal analysis and filtering utilities.
- `ingkit.io`
  - File input/output helpers.
- `ingkit.physics`
  - Physics-related modules (including plasma and X-ray utilities).
- `ingkit.myplot`
  - Plot styles and colormap utilities.
- `ingkit.utils`
  - General-purpose helpers.
- `ingkit.tools`
  - Shared types and internal tools.
- `ingkit.analysis`
  - Analysis namespace package.

## Demos

Example scripts are available in `demo/`:

- `demo/filter_demo.py`
- `demo/plot_demo.py`
- `demo/spectrum_analysis_demo.py`

## VMEC coordinate mapping

`ingkit.physics.plasma.vmec.VMECData` reads VMEC NetCDF `wout_*.nc`
equilibria, evaluates magnetic geometry and equilibrium fields, and maps
real-space points to flux coordinates:

```python
from ingkit.physics.plasma.vmec import VMECData

equilibrium = VMECData("wout_example.nc")
xyz = equilibrium.flux_surface(s=0.5, theta=1.0, phi=0.2)
s, theta_vmec = equilibrium.points_to_flux(xyz, coordinates="xyz", return_theta=True)
B = equilibrium.magnetic_field_at(xyz, coordinates="xyz")
```

Surface methods evaluate a tensor product of scalar or 1-D coordinate inputs;
field methods accept paired points of shape `(..., 3)`. See
[docs/vmec_low_level.md](docs/vmec_low_level.md) for units, conventions,
boundary handling, and sampling requirements. The legacy `ingkit.io` VMEC
API was removed in 0.3.0; see [the migration guide](docs/vmec.md).

Camera geometry, voxels, visibility, projection matrices, emission profiles,
and line integration belong in downstream packages. `ingkit` does not
provide a VMEC writer.

## License

This project is licensed under the MIT License.

このソフトウェアは MIT License で提供されています。
