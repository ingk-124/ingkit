"""Utilities for selecting signal data."""

from __future__ import annotations

import operator

import numpy as np

__all__ = ["select_time"]


def select_time(
    data: np.ndarray,
    t_arr: np.ndarray,
    t_0: float | None = None,
    t_1: float | None = None,
    axis: int = -1,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Select data within a specified time range.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    t_arr : np.ndarray (n,)
        Time values corresponding to the selected axis of `data`.
    t_0 : float, optional
        Start time of the range, inclusive. Default is None, which leaves the
        lower bound unrestricted.
    t_1 : float, optional
        End time of the range, inclusive. Default is None, which leaves the
        upper bound unrestricted.
    axis : int, optional
        Axis corresponding to `t_arr`. Default is -1.
    return_mask : bool, optional
        Whether to return the boolean selection mask. Default is False.

    Returns
    -------
    selected_data : np.ndarray
        Data selected along `axis`.
    mask : np.ndarray (n,), optional
        Boolean selection mask. Returned only when `return_mask` is True.

    Raises
    ------
    ValueError
        If `t_arr` is not one-dimensional.
        If the length of `t_arr` differs from the length of `data` along
        `axis`.
        If `t_0` is greater than `t_1`.
    np.exceptions.AxisError
        If `axis` is outside the valid range for `data`.
    """
    data = np.asarray(data)
    t_arr = np.asarray(t_arr)

    if t_arr.ndim != 1:
        raise ValueError("t_arr must be one-dimensional")

    axis = operator.index(axis)
    if not -data.ndim <= axis < data.ndim:
        raise np.exceptions.AxisError(axis, data.ndim)
    axis %= data.ndim
    if data.shape[axis] != t_arr.size:
        raise ValueError(
            "t_arr must have the same length as data along the selected axis"
        )
    if t_0 is not None and t_1 is not None and t_0 > t_1:
        raise ValueError("t_0 must be less than or equal to t_1")

    mask = np.ones(t_arr.shape, dtype=bool)
    if t_0 is not None:
        mask &= t_arr >= t_0
    if t_1 is not None:
        mask &= t_arr <= t_1

    selected_data = np.compress(mask, data, axis=axis)
    if return_mask:
        return selected_data, mask
    return selected_data
