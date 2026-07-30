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

`ingkit.io.VMECData` reads VMEC NetCDF `wout_*.nc` equilibria and maps
magnetic coordinates to cylindrical or Cartesian coordinates. It can also
numerically invert Cartesian points:

```python
from ingkit.io import VMECData

equilibrium = VMECData("wout_example.nc")
x, y, z = equilibrium.to_cartesian(s=0.5, theta=1.0, phi=0.2)
result = equilibrium.from_cartesian(x=x, y=y, z=z)

if result.valid:
    print(result.s, result.rho, result.theta, result.phi)
```

Inputs may be scalars, paired arrays, or NumPy-broadcastable arrays. See
[docs/vmec.md](docs/vmec.md) for the coordinate convention, failure handling,
and solver controls.

The VMEC layer owns equilibrium I/O, Fourier coefficients, and coordinate
mapping only. Camera geometry, voxels, visibility, projection matrices,
emission profiles, and line integration belong in downstream packages.
`ingkit` does not provide a VMEC writer.

## License

This project is licensed under the MIT License.

このソフトウェアは MIT License で提供されています。
