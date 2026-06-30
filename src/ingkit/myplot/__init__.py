# src/ingkit/myplot/__init__.py
from matplotlib import pyplot

from . import styles
from .colormap import get_cmap_with_extreme

__all__ = [
    "styles",
    "get_cmap_with_extreme",
    "pyplot",
    "use_style",
    "use_my_default",
]
__version__ = "0.2.0"


def use_style(*style_name: str | styles.Path):
    """
    Use a matplotlib style by name or path.

    Parameters
    ----------
    style_name : str or Path
        Style name (without extension) or path to .mplstyle file.

    Raises
    ------
    ValueError
        If the style is not found or the file is invalid.
    """
    styles.use(*style_name)


def use_my_default():
    """
    Use the default style provided by ingkit.
    """
    styles.use("my_default")
