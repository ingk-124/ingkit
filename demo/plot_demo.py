# demo/plot_demo.py
# This is a demo script to show how to use `ingkit.myplot` package.
# Copyright (c) 2026, ingk-124, License under MIT License.

import numpy as np

from ingkit.myplot import styles, get_cmap_with_extreme
from ingkit.myplot import pyplot as plt

if __name__ == '__main__':

    # ----- styles demo -----
    print("Available styles:", styles.available_styles())

    with styles.context("default"):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], label="Default")
        ax.legend()
        fig.show()

    with styles.context("ggplot"):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], label="ggplot")
        ax.legend()
        fig.show()

    with styles.context("my_default"):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], label="my_default")
        ax.legend()
        fig.show()

    # ----- colormap demo -----
    styles.use("my_default")
    cmap = get_cmap_with_extreme("viridis", over="red", under="blue", bad="gray")

    x = y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots()
    im = ax.contourf(X, Y, Z, levels=15, cmap=cmap, vmin=-1.5, vmax=1.5, extend="both")
    fig.colorbar(im, ax=ax)
    ax.set_title("Custom colormap with extreme color")
    fig.show()
