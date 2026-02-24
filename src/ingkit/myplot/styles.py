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


def _find_style(style_name: str | Path) -> str:
    """
    Check if a style is available.

    Parameters
    ----------
    style_name : str or Path
        Style name (without extension), or path to a .mplstyle file.

    Returns
    -------
    style_name : str
        If found, return the style name (for built-in or user styles) or the path (for custom styles).
        Path is returned as string for compatibility with plt.style.use().

    Raises
    ------
    ValueError
        If the style is not found or the file is invalid.
    """
    if isinstance(style_name, Path):  # Path
        style_path = style_name.resolve()
        if style_path.exists():
            style_name_str = str(style_path)
            try:
                mpl.rc_params_from_file(style_name_str)
                return style_name_str
            except Exception:
                raise ValueError(f"'{style_name}' is not a valid style file.")
        else:
            raise ValueError(f"Style file not found: {style_name}")

    elif isinstance(style_name, str):  # str
        if style_name == "default" or style_name in plt.style.available:
            return style_name
        elif style_name in available_styles(user=True):
            return _find_style(MY_STYLE_DIR / f"{style_name}.mplstyle")
        else:
            return _find_style(Path(style_name))
    raise ValueError(
        f"Style '{style_name}' not found. "
        f"Use available_styles() to see available styles."
    )


def use(style_name: str | Path = "default") -> None:
    """
    Use a style by name or path.

    Parameters
    ----------
    style_name : str or Path
        Style name (without extension), or path to a .mplstyle file.
    """
    plt.style.use(_find_style(style_name))


@contextmanager
def context(style_name: str | Path = "default"):
    with plt.rc_context():
        use(style_name)
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


def quick_look(style_name: str | Path, n_lines: int | None = None):
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
