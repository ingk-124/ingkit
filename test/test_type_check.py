import numpy as np
import pytest

from ingkit.tools import type_check


@pytest.mark.parametrize("value", [1, np.int64(1)])
def test_integer_predicates_accept_integer_scalars(value):
    assert type_check.is_int(value)
    assert type_check.is_number(value)
    assert not type_check.is_float(value)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_integer_predicate_rejects_booleans(value):
    assert not type_check.is_int(value)
    assert not type_check.is_number(value)


def test_array_like_predicates_are_type_strict():
    assert type_check.is_array_like_of_int([1, np.int64(2)])
    assert type_check.is_array_like_of_float([1.0, np.float64(2.0)])
    assert type_check.is_array_like_of_number([1, 2.0])
    assert not type_check.is_array_like_of_float([1, 2.0])


def test_ensure_array_like_converts_to_requested_dtype():
    integers = type_check.ensure_array_like_of_int([1, 2])
    numbers = type_check.ensure_array_like_of_number([1, 2.5])

    assert integers.dtype == np.int64
    assert numbers.dtype == np.float64


def test_ensure_array_like_rejects_scalars_and_wrong_elements():
    with pytest.raises(TypeError, match="scalar"):
        type_check.ensure_array_like_of_number(1.0)
    with pytest.raises(ValueError, match="not number"):
        type_check.ensure_array_like_of_number([1.0, "2"])
