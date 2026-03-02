# src/ingkit/myplot/colormap.py
# Custom colormap for myplot.

from __future__ import annotations

import matplotlib
from matplotlib import pyplot as plt


def get_cmap_with_extreme(name: str,
                          bad: str | tuple | list = None,
                          under: str | tuple | list = None,
                          over: str | tuple | list = None) -> matplotlib.colors.ListedColormap:
    """
    Create a custom colormap with specified bad, under, and over colors.

    Parameters
    ----------
    name : str
        The name of the base colormap to use (e.g., 'viridis', 'plasma').
    bad : str, tuple, or list
        The color to use for NaN values.
    under : str, tuple, or list
        The color to use for values below the colormap range.
    over : str, tuple, or list
        The color to use for values above the colormap range.

    Returns
    -------
    Colormap
        A new colormap with the specified properties.
    """
    return plt.get_cmap(name).copy().with_extremes(bad=bad, under=under, over=over)
