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


def test_as_numeric_array_accepts_scalars_and_normalizes_rank():
    scalar = type_check.as_numeric_array(np.int64(2))
    vector = type_check.as_numeric_array(2, min_ndim=1)
    matrix = type_check.as_numeric_array([[1, 2]], ndim=2, name="matrix")

    assert scalar.shape == ()
    assert vector.shape == (1,)
    assert matrix.shape == (1, 2)
    assert matrix.dtype == np.float64


def test_as_numeric_vector_accepts_scalar_but_rejects_matrix():
    np.testing.assert_array_equal(type_check.as_numeric_vector(2), np.array([2.0]))

    with pytest.raises(ValueError, match="1-dimensional"):
        type_check.as_numeric_vector([[1, 2]], name="energy")


def test_as_numeric_array_validates_values_and_dimensions():
    with pytest.raises(TypeError, match="energy"):
        type_check.as_numeric_array([1, "2"], name="energy")
    with pytest.raises(ValueError, match="1-dimensional"):
        type_check.as_numeric_array([[1, 2]], ndim=1, name="energy")


def test_align_last_axis_for_outer_broadcast():
    spectrum = np.zeros((2, 3, 5))
    transmission = np.zeros((4, 5))

    spectrum, transmission = type_check.align_last_axis_for_broadcast(
        spectrum, transmission
    )

    assert spectrum.shape == (2, 3, 1, 5)
    assert transmission.shape == (1, 1, 4, 5)
    assert (spectrum + transmission).shape == (2, 3, 4, 5)


def test_align_last_axis_requires_matching_lengths():
    with pytest.raises(ValueError, match="matching"):
        type_check.align_last_axis_for_broadcast(np.zeros(3), np.zeros(4))
