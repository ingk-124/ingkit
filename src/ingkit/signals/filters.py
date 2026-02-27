# src/ingkit/signals/filters.py
# Wrapper for scipy.signal filters.

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from ingkit.tools import type_check

FILTER_TYPES = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def _cutoff_check(cutoff_freq: float | list[float], filter_type: FILTER_TYPES
                  ) -> tuple[float | np.ndarray, FILTER_TYPES]:
    """validate cutoff_freq based on filter_type.

    Parameters
    ----------
    cutoff_freq : float or list of float
        Cutoff frequency/frequencies (unit: Hz) for the filters.
    filter_type : str
        Type of filters. Supported types are following:
        - "lowpass"
        - "highpass"
        - "bandpass"
        - "bandstop"

    Returns
    -------
    cutoff_freq : float or np.ndarray
        Validated cutoff frequency/frequencies. For "lowpass" and "highpass" filters, this will be a single float.
        For "bandpass" and "bandstop" filters, this will be a numpy array of shape (2,) containing [f_lc, f_hc].
    filter_type : str
        The validated filters type (same as input).

    Raises
    ------
    TypeError
        If cutoff_freq is not of the expected type for the given filter_type.
    ValueError
        If cutoff_freq does not satisfy the expected conditions for the given filter_type.
        (e.g., f_lc < f_hc for bandpass/bandstop filters, or unsupported filter_type).

    Notes
    -----
    - For "lowpass" and "highpass" filters, cutoff_freq should be a single float.
    - For "bandpass" and "bandstop" filters, cutoff_freq should be a list of two floats [f_lc, f_hc], where f_lc < f_hc.
    """

    if filter_type in ["lowpass", "highpass"]:
        if not type_check.is_number(cutoff_freq):
            raise TypeError(
                f"cutoff_freq should be a single number for {filter_type} filters.")
    elif filter_type in ["bandpass", "bandstop"]:
        try:
            cutoff_freq = type_check.ensure_array_like_of_number(cutoff_freq, dtype=float)
        except Exception as e:
            raise TypeError(
                f"cutoff_freq should be a list of two numbers [f_lc, f_hc] for {filter_type} filters.") from e
        if cutoff_freq.shape != (2,):
            raise ValueError(
                f"cutoff_freq should be a list of two numbers [f_lc, f_hc] for {filter_type} filters.")
        if cutoff_freq[0] >= cutoff_freq[1]:
            raise ValueError(f"For {filter_type} filters, cutoff_freq should satisfy f_lc < f_hc. "
                             f"Got f_lc={cutoff_freq[0]} and f_hc={cutoff_freq[1]}.")
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}. "
                         f"Supported types are: 'lowpass', 'highpass', 'bandpass', 'bandstop'.")

    return cutoff_freq, filter_type


def filters(t: np.ndarray, y: np.ndarray, cutoff_freq: float | list[float], filter_type: FILTER_TYPES,
            filter_name: str = 'bessel', order: int = 4, window: str = 'boxcar',
            detrend: str = 'constant') -> np.ndarray:
    """
    Base function for applying filters to a signal. 
    
    Parameters
    ----------
    t : np.ndarray (n,)
        Time array (unit: s)
    y : np.ndarray (..., n)
        Signal array, where n is the length of the time array. The filters will be applied along the last axis.
    cutoff_freq : list of float
        List of cutoff frequencies (unit: Hz) for the filters.
    filter_type : str
        Type of filters. Supported types: "lowpass", "highpass", "bandpass", "bandstop".
    filter_name : str, optional (default: 'bessel')
        Name of the IIR filter design method to use.
        See scipy.signal.iirfilter for available filter types.
    order : int, optional (default: 4)
        The order of the IIR filter.
    window : str, optional (default: 'boxcar')
        Desired window to use. See scipy.signal.get_window for available window types.
    detrend : str, optional (default: 'constant')
        Specifies how to detrend the data before filtering. See scipy.signal.detrend for available options.
    **kwargs
        Additional keyword arguments to pass to scipy.signal.iirfilter.
    
        
        
    """
    cutoff_freq, filter_type = _cutoff_check(cutoff_freq, filter_type)
    t = type_check.ensure_array_like_of_number(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("Time array t should be one-dimensional.")
    if t.shape[0] != y.shape[-1]:
        raise ValueError("The length of time array should match the last dimension of signal array.")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("Time array t should have uniform spacing.")

    y = signal.detrend(y, axis=-1, type=detrend)
    window = signal.get_window(window, t.size)
    y = y * window
    fs = 1 / dt[0]
    if np.any(cutoff_freq >= 0.5 * fs):
        raise ValueError(f"Cutoff frequency should be less than Nyquist frequency ({0.5 * fs} Hz). "
                         f"Got cutoff_freq={cutoff_freq} Hz.")

    sos = signal.iirfilter(N=order, Wn=cutoff_freq,
                           btype=filter_type, ftype=filter_name, fs=fs, output='sos')

    return signal.sosfiltfilt(sos, y, axis=-1)


def lowpass_filter(t: np.ndarray, y: np.ndarray, cutoff_freq: float, **kwargs) -> np.ndarray:
    """
    Lowpass filter for y(t) with given cutoff frequency (unit: Hz).

    Parameters
    ----------
    t : np.ndarray (n,)
        Time array (unit: s)
    y : np.ndarray (..., n)
        Signal array, where n is the length of the time array. The filter will be applied along the last axis.
    cutoff_freq : float
        Cutoff frequency (unit: Hz) for the lowpass filter.
    kwargs : additional keyword arguments
        Additional keyword arguments to pass to the base filters function `filters()`. See its documentation for details.

    Returns
    -------
    np.ndarray
        Filtered signal array with the same shape as input y.

    See Also
    --------
    filters : Base function for applying various types of filters to a signal.
    """
    return filters(t, y, cutoff_freq=cutoff_freq, filter_type="lowpass", **kwargs)


def highpass_filter(t: np.ndarray, y: np.ndarray, cutoff_freq: float, **kwargs) -> np.ndarray:
    """
    Highpass filter for y(t) with given cutoff frequency (unit: Hz).

    Parameters
    ----------
    t : np.ndarray (n,)
        Time array (unit: s)
    y : np.ndarray (..., n)
        Signal array, where n is the length of the time array. The filter will be applied along the last axis.
    cutoff_freq : float
        Cutoff frequency (unit: Hz) for the highpass filter.
    kwargs : additional keyword arguments
        Additional keyword arguments to pass to the base filters function `filters()`. See its documentation for details.

    Returns
    -------
    np.ndarray
        Filtered signal array with the same shape as input y.

    See Also
    --------
    filters : Base function for applying various types of filters to a signal.
    """
    return filters(t, y, cutoff_freq=cutoff_freq, filter_type="highpass", **kwargs)


def bandpass_filter(t: np.ndarray, y: np.ndarray, cutoff_freq: list[float], **kwargs) -> np.ndarray:
    """
    Bandpass filter for y(t) with given cutoff frequencies (unit: Hz).

    Parameters
    ----------
    t : np.ndarray (n,)
        Time array (unit: s)
    y : np.ndarray (..., n)
        Signal array, where n is the length of the time array. The filter will be applied along the last axis.
    cutoff_freq : list of float
        List of two cutoff frequencies [f_lc, f_hc] (unit: Hz) for the bandpass filter, where f_lc < f_hc.
    kwargs : additional keyword arguments
        Additional keyword arguments to pass to the base filters function `filters()`. See its documentation for details.

    Returns
    -------
    np.ndarray
        Filtered signal array with the same shape as input y.

    See Also
    --------
    filters : Base function for applying various types of filters to a signal.
    """
    return filters(t, y, cutoff_freq=cutoff_freq, filter_type="bandpass", **kwargs)


def bandstop_filter(t: np.ndarray, y: np.ndarray, cutoff_freq: list[float], **kwargs) -> np.ndarray:
    """
    Bandstop filter for y(t) with given cutoff frequencies (unit: Hz).

    Parameters
    ----------
    t : np.ndarray (n,)
        Time array (unit: s)
    y : np.ndarray (..., n)
        Signal array, where n is the length of the time array. The filter will be applied along the last axis.
    cutoff_freq : list of float
        List of two cutoff frequencies [f_lc, f_hc] (unit: Hz) for the bandstop filter, where f_lc < f_hc.
    kwargs : additional keyword arguments
        Additional keyword arguments to pass to the base filters function `filters()`. See its documentation for details.

    Returns
    -------
    np.ndarray
        Filtered signal array with the same shape as input y.

    See Also
    --------
    filters : Base function for applying various types of filters to a signal.
    """
    return filters(t, y, cutoff_freq=cutoff_freq, filter_type="bandstop", **kwargs)


def filter_response(cutoff_freq: float | list[float], filter_type: FILTER_TYPES,
                    filter_name: str = 'bessel', order: int = 4, fs: float = 1e6, worN: int = 512
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency response of a filter.

    Parameters
    ----------
    cutoff_freq : float or list of float
        Cutoff frequency/frequencies (unit: Hz) for the filters.
    filter_type : str
        Type of filters. Supported types are following:
        - "lowpass"
        - "highpass"
        - "bandpass"
        - "bandstop"
    filter_name : str, optional (default: 'bessel')
        Name of the IIR filter design method to use.
        See scipy.signal.iirfilter for available filter types.
    order : int, optional (default: 4)
        The order of the IIR filter.
    worN : int, optional (default: 512)
        The number of frequencies at which to compute the response.
    fs : float, optional (default: 1.0)
        Sampling frequency (unit: Hz) of the signal to be filtered. This is used to normalize the cutoff frequencies.

    Returns
    -------
    w : np.ndarray
        Frequencies at which the response was computed (unit: Hz).
    h : np.ndarray
        Frequency response of the filter at frequencies w.

    See Also
    --------
    filters : Base function for applying various types of filters to a signal.
    """
    cutoff_freq, filter_type = _cutoff_check(cutoff_freq, filter_type)
    sos = signal.iirfilter(N=order, Wn=cutoff_freq,
                           btype=filter_type, ftype=filter_name, fs=fs, output='sos')
    w, h = signal.sosfreqz(sos, worN=worN, fs=fs)
    return w, h


if __name__ == '__main__':
    from ingkit.myplot import pyplot as plt

    cutoff_freqs = [200e3, [2e3, 100e3]]
    filter_types = ["lowpass", "bandpass"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for cutoff_freq, filter_type in zip(cutoff_freqs, filter_types):
        w, h = filter_response(cutoff_freq=cutoff_freq, filter_type=filter_type, filter_name='bessel', order=6, fs=1e6,
                               worN=512)
        axes[0].semilogx(w[1:], 20 * np.log10(np.abs(h[1:])), label=f'{filter_type} filter')
        axes[1].semilogx(w[1:], np.angle(h[1:]), label=f'{filter_type} filter')
    axes[0].set_title('Frequency Response (Magnitude)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_xlim(1e3, 5e5)
    axes[0].set_ylim(-60, 5)
    axes[0].legend()
    axes[1].set_title('Frequency Response (Phase)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_xlim(1e3, 5e5)
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].legend()
    fig.tight_layout()
    plt.show()

    np.random.seed(1234)
    f_list = [0.5e3, 10e3, 150e3]  # Hz
    A_list = [3, 2, 1]
    t = np.linspace(-1, 2, 3000) * 1e-3  # seconds
    phase_list = np.random.uniform(0, 2 * np.pi, size=len(f_list))
    y = np.sum([A * np.exp(1j * (2 * np.pi * f * t + p)) for A, f, p in zip(A_list, f_list, phase_list)],
                    axis=0).real
    y_noise = y + np.random.normal(scale=0.5, size=t.shape)

    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.plot(t*1e3, y, "k--", label='Original Signal')
    ax.plot(t*1e3, y_noise, label='Noisy Signal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    for cutoff_freq, filter_type in zip(cutoff_freqs, filter_types):
        y_filtered = filters(t, y_noise, cutoff_freq=cutoff_freq, filter_type=filter_type, filter_name='bessel', order=4,
                             window='boxcar', detrend='constant')
        ax.plot(t*1e3, y_filtered, label=f'{filter_type} Filtered Signal')
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    plt.show()
