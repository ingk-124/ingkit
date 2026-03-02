# -*- coding: utf-8 -*-
"""
Matplotlib style management utilities.

Copyright (c) 2026, ingk-124
License under MIT License.
"""
# File: ingkit/tools/myplot/styles.py
# Author: ingk-124
# Date: 2026-02-22
# Description: Matplotlib style management utilities.

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MY_STYLE_DIR = Path(__file__).parent / "styles"
MY_STYLE_DIR.mkdir(parents=True, exist_ok=True)


def available_styles(user: bool = False) -> list[str]:
    """
    Return a list of available style names.

    Parameters
    ----------
    user : bool, optional
        If True, return only user-defined styles; otherwise, return both system and user styles.
    """

    styles_sys = list(plt.style.available)
    styles_user = [p.stem for p in MY_STYLE_DIR.glob("*.mplstyle")]
    return styles_user if user else (styles_sys + styles_user)


def _find_style(*style_name: str | Path) -> str | list[str]:
    """
    Check if a style is available.

    Parameters
    ----------
    style_name : str, Path, or list of str/Path
        Style name(s) (without extension), or path(s) to .mplstyle file(s).

    Returns
    -------
    style_name : str or list of str
        If found, return the style name(s) (for built-in or user styles) or the path(s) (for custom styles).
        Path(s) are returned as string(s) for compatibility with plt.style.use().

    Raises
    ------
    ValueError
        If any style is not found or any file is invalid.

    Examples
    --------
    >>> _find_style("default")
    'default'
    >>> _find_style(Path("/path/to/some_style.mplstyle"))
    '/path/to/some_style.mplstyle'
    >>> _find_style("default", "my_default")
    ['default', '/absolute/path/to/my_default.mplstyle']
    >>> _find_style("default", "nonexistent_style")
    ValueError: Style 'nonexistent_style' not found. Use available_styles() to see available styles.
    """
    if len(style_name) > 1:
        return [_find_style(s) for s in style_name]  # Recursively check each style name in the list
    else:
        style_name = style_name[0]  # if only one style name is given

        if isinstance(style_name, str):
            if style_name == "default" or style_name in plt.style.available:  # Built-in style
                return style_name
            elif style_name in available_styles(user=True):  # styles in this package
                return _find_style(MY_STYLE_DIR / f"{style_name}.mplstyle")
            else:
                return _find_style(Path(style_name))  # Try as a path (even if it's a string)
        elif isinstance(style_name, Path):  # Path
            style_path_str = str(style_name.resolve())  # Absolute path as string
            try:
                mpl.rc_params_from_file(style_path_str)  # Check if it's a valid style file
                return style_path_str  # Return the path as string for compatibility with plt.style.use()
            except Exception as e:  # Invalid style file
                raise ValueError(f"'{style_name}' is not a valid style file.") from e
        else:
            raise ValueError(
                f"Style '{style_name}' not found. "
                f"Use available_styles() to see available styles."
            )


def use(style_name: str | Path = "default", *style_names: str | Path) -> None:
    """
    Use a style by name or path.

    Parameters
    ----------
    style_name : str or Path
        Style name (without extension), or path to a .mplstyle file.
    *style_names : str or Path
        Additional style names or paths to apply (optional).
    """
    if style_names:
        style_names = [style_name] + list(style_names)  # Combine the first style name with additional ones
    else:
        style_names = [style_name]
    plt.style.use(_find_style(*style_names))


@contextmanager
def context(style_name: str | Path = "default", *style_names: str | Path) -> Iterator[None]:
    """
    Context manager to temporarily apply a style.
    
    Parameters
    ----------
    style_name : str or Path
        Style name (without extension), or path to a .mplstyle file.
    style_names : str or Path
        Additional style names or paths to apply (optional).

    Returns
    -------
    None

    Examples
    --------
    >>> with context("my_default"):
    ...     # This block will use the "my_default" style
    ...     plt.plot([1, 2, 3], [4, 5, 6])
    ...     plt.show()
    ... # After the block, the style will revert to the previous state
    ... plt.plot([1, 2, 3], [4, 5, 6])
    ... plt.show()
    """
    with plt.rc_context():
        use(style_name, *style_names)
        yield


def save_style(style_name: str) -> Path:
    style_path = MY_STYLE_DIR / f"{style_name}.mplstyle"
    if style_path.exists():
        raise ValueError(f"Style '{style_name}' already exists. Choose a different name.")

    with open(style_path, "w") as f:
        for key in sorted(plt.rcParams):
            current = plt.rcParams[key]
            default = plt.rcParamsDefault.get(key)
            if current == default:
                continue

            if key == "axes.prop_cycle":
                colors = [d.get("color") for d in current]
                f.write(f"{key} : cycler('color', {colors})\n")
            else:
                f.write(f"{key} : {current}\n")

    return style_path


def get_rcParams(style_name: str | Path) -> dict:
    """
    Return rcParams dictionary of a style
    without modifying global matplotlib state.

    Parameters
    ----------
    style_name : str or Path
        Style name (without extension), or path to a .mplstyle file.

    Returns
    -------
    rcParams : dict
        Dictionary of rcParams defined in the style.
    """
    with context(style_name):
        rcParams = plt.rcParams.copy()
    return rcParams


def quick_look(style_name: str | Path, n_lines: int | None = None) -> None:
    """
    Quick preview of a mplstyle file.

    Parameters
    ----------
    style_name : str or Path
        Style name (without extension), or path to a .mplstyle file.
    n_lines : int, optional
        If given, only show first n lines.
    """
    _rcParams = get_rcParams(style_name)
    if n_lines is None:
        n_lines = len(_rcParams)
    print(f"=== {style_name} ===")
    lines = []
    for key, value in _rcParams.items():
        default_value = plt.rcParamsDefault.get(key)
        if value != default_value:
            lines.append(f"{key} : {value}")

    for line in lines[:n_lines]:
        print(line)
    if len(lines) > n_lines:
        print(f"... ({len(lines) - n_lines} more lines)")


if __name__ == '__main__':
    print("Available styles:", available_styles())
    quick_look("myplot")
