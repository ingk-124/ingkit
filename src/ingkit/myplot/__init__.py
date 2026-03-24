from matplotlib import pyplot
from . import styles
from .colormap import get_cmap_with_extreme

styles.use("my_default")
print(f"Using style: my_default")
__all__ = ["styles", "get_cmap_with_extreme", "pyplot"]
__version__ = "0.1.2"
