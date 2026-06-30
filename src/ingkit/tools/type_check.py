# src/ingkit/tools/type_check.py
from __future__ import annotations

from typing import Any, TypeGuard

import numpy as np


def is_int(x: Any) -> TypeGuard[int]:
    """
    Type check for integer.

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is an integer (including numpy integer types), but not a boolean; False otherwise.
    """
    # note: bool is a subclass of int, so we need to exclude boolean.
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def is_float(x: Any) -> TypeGuard[float]:
    """
    Type check for float.

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is a float (including numpy floating types); False otherwise.
    """
    return isinstance(x, (float, np.floating))


def is_number(x: Any) -> TypeGuard[int | float]:
    """
    Type check for number (int or float).

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is an integer or a float (including numpy integer and floating types), but not a boolean; False otherwise.
    """
    return is_int(x) or is_float(x)


def _as_object_array(x: Any) -> np.ndarray:
    """
    Convert x to a numpy array with dtype=object. This allows us to check the type of each element without converting them to a common type (which may cause issues if they are of different types).

    Parameters
    ----------
    x : Any
        The value to convert.

    Returns
    -------
    np.ndarray
        A numpy array with dtype=object containing the elements of x.
    """
    try:
        return np.asarray(x, dtype=object)
    except Exception as e:
        raise TypeError("expected array-like") from e


def is_array_like_of(x: Any, is_type) -> bool:
    """
    Check if x is array-like and all elements satisfy the predicate pred.

    Parameters
    ----------
    x : Any
        The value to check.
    is_type : function
        A type check function.

    Returns
    -------
    bool
        True if x is array-like and all elements satisfy the predicate; False otherwise.

    Notes
    -----
    `is_type` should be a function that takes a single argument and returns True if the argument is of the desired type, and False otherwise.
    """
    try:
        a = np.asarray(x, dtype=object)
    except Exception:
        return False
    if a.ndim == 0:
        return False
    return all(is_type(v) for v in a.ravel())


def is_array_like_of_int(x: Any) -> bool:
    """
    Check if x is array-like object of integers.

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is array-like and all elements are integers; False otherwise.
    """
    return is_array_like_of(x, is_int)


def is_array_like_of_float(x: Any) -> bool:
    """
    Check if x is array-like object of floats.

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is array-like and all elements are floats; False otherwise.
    """
    return is_array_like_of(x, is_float)


def is_array_like_of_number(x: Any) -> bool:
    """
    Check if x is array-like object of numbers (integers or floats).

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if x is array-like and all elements are numbers (integers or floats); False otherwise.
    """
    return is_array_like_of(x, is_number)


def _ensure_array_like_of(x: Any, is_type, kind: str, *, dtype=None) -> np.ndarray:
    """
    Ensure that x is an array-like object of specified type.

    Parameters
    ----------
    x : Any
        The value to check and convert.
    is_type : function
        A type check function.
    kind : str
        A string describing the expected type, used in error messages.
    dtype : data-type, optional
        Desired data-type for the array. If None, the data type will be inferred from the input.

    Returns
    -------
    np.ndarray
        A numpy array containing the elements of x, with the specified dtype if provided.

    Raises
    ------
    TypeError
        If x is not array-like or is a scalar.
    ValueError
        If at least one element of x does not satisfy the type check.
    """
    a_obj = _as_object_array(x)
    if a_obj.ndim == 0:
        raise TypeError(f"Expected array-like object of {kind}, got scalar.")

    for i, v in enumerate(a_obj.ravel()):
        if not is_type(v):
            raise ValueError(f"Expected array-like object of {kind}. At least one element is not {kind}.")

    return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)


def ensure_array_like_of_int(x: Any) -> np.ndarray:
    """
    Ensure that x is an array-like object of integers.

    Parameters
    ----------
    x : Any
        The value to check and convert.

    Returns
    -------
    np.ndarray
        A numpy array containing the elements of x as integers.

    Raises
    ------
    TypeError
        If x is not array-like or is a scalar.
    ValueError
        If at least one element of x is not an integer.
    """
    return _ensure_array_like_of(x, is_int, "int", dtype=np.int64)


def ensure_array_like_of_float(x: Any, dtype=np.float64) -> np.ndarray:
    """
    Ensure that x is an array-like object of floats.

    Parameters
    ----------
    x : Any
        The value to check and convert.
    dtype : data-type, optional (default=np.float64)
        Desired data-type for the array. If None, the data type will be inferred from the input.

    Returns
    -------
    np.ndarray
        A numpy array containing the elements of x as floats, with the specified dtype if provided.

    Raises
    ------
    TypeError
        If x is not array-like or is a scalar.
    ValueError
        If at least one element of x is not a float.
    """
    return _ensure_array_like_of(x, is_float, "float", dtype=dtype)


def ensure_array_like_of_number(x: Any, dtype=np.float64) -> np.ndarray:
    """
    Ensure that x is an array-like object of numbers (integers or floats).

    Parameters
    ----------
    x : Any
        The value to check and convert.
    dtype : data-type, optional (default=np.float64)
        Desired data-type for the array. If None, the data type will be inferred from the input.

    Returns
    -------
    np.ndarray
        A numpy array containing the elements of x as numbers (integers or floats), with the specified dtype if provided.

    Raises
    ------
    TypeError
        If x is not array-like or is a scalar.
    ValueError
        If at least one element of x is not a number (integer or float).
    """
    return _ensure_array_like_of(x, is_number, "number", dtype=dtype)
