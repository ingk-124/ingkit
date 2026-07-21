import numpy as np
import pytest

from ingkit.signals import select_time


def test_select_time_uses_inclusive_bounds():
    data = np.arange(5)
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    selected, mask = select_time(data, time, 0.5, 1.5, return_mask=True)

    np.testing.assert_array_equal(selected, [1, 2, 3])
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_select_time_supports_axis_and_open_bounds():
    data = np.arange(10).reshape(2, 5)
    time = np.arange(5.0)

    np.testing.assert_array_equal(
        select_time(data, time, t_0=3.0, axis=1), data[:, 3:]
    )
    np.testing.assert_array_equal(
        select_time(data.T, time, t_1=1.0, axis=0), data.T[:2]
    )


def test_select_time_with_no_bounds_selects_everything():
    data = np.arange(4)
    selected, mask = select_time(data, np.arange(4), return_mask=True)

    np.testing.assert_array_equal(selected, data)
    np.testing.assert_array_equal(mask, np.ones(4, dtype=bool))


@pytest.mark.parametrize(
    ("data", "time", "kwargs", "message"),
    [
        (np.zeros((2, 3)), np.zeros((1, 3)), {}, "one-dimensional"),
        (np.zeros((2, 3)), np.arange(2), {}, "same length"),
        (np.zeros(3), np.arange(3), {"t_0": 2, "t_1": 1}, "t_0"),
    ],
)
def test_select_time_validates_inputs(data, time, kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_time(data, time, **kwargs)
