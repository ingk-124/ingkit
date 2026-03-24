# src/ingkit/myplot/__init__.py
from matplotlib import pyplot
from . import styles
from .colormap import get_cmap_with_extreme

styles.use("my_default")
__all__ = ["styles", "get_cmap_with_extreme", "pyplot"]
__version__ = "0.2.0"
