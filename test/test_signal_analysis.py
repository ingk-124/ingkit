import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from ingkit.signals.analysis import sliding_window_coherence


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(42)
    x = rng.normal(size=2048)
    y = 0.7 * x + 0.3 * rng.normal(size=x.size)
    return x, y


def test_frequency_smoothing_uses_frequency_axis(signals):
    kwargs = dict(fs=1_000.0, nperseg=128, noverlap=64,
                  seg_per_win=5, seg_step=2)
    raw = sliding_window_coherence(*signals, **kwargs)
    smoothed = sliding_window_coherence(*signals, freq_smooth_bins=3, **kwargs)

    for raw_spectrum, smooth_spectrum in zip(raw[2:5], smoothed[2:5]):
        np.testing.assert_allclose(
            smooth_spectrum,
            uniform_filter1d(raw_spectrum, size=3, axis=-2),
        )


@pytest.mark.parametrize("value, error", [(0, ValueError), (1.5, TypeError)])
def test_frequency_smoothing_validates_bin_count(signals, value, error):
    with pytest.raises(error, match="freq_smooth_bins"):
        sliding_window_coherence(
            *signals, fs=1_000.0, nperseg=128, noverlap=64,
            seg_per_win=5, freq_smooth_bins=value,
        )
