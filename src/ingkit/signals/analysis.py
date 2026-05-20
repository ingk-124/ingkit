# src/ingkit/signals/analysis.py
# FFT utilities for signal processing.

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

__all__ = ["ens_N", "nperseg_from_ens", "coherence_analysis", "sliding_window_coherence",
           "param_from_df_window", "param_from_df_ens", "param_from_window_ens", "resolutions"]


def ens_N(points: int, nperseg: int, noverlap: int) -> int:
    """
    Compute the number of ensemble segments for Welch's method.

    Parameters
    ----------
    points : int
        The total number of data points in the signal.
    nperseg : int
        The length of each segment.
    noverlap : int
        The number of points to overlap between segments.

    Returns
    -------
    ens : int
        The number of ensemble segments that can be formed with the given parameters.
    """
    hop = nperseg - noverlap
    if hop <= 0:
        raise ValueError("noverlap must be less than nperseg")
    return (points - nperseg) // hop + 1


def nperseg_from_ens(points: int, ens: int, overlap_ratio: float = 0.5, fs: float = None) -> int:
    """
    Compute the nperseg parameter for Welch's method based on the desired number of points and ensemble segments.

    Parameters
    ----------
    points : int
        The total number of data points in the signal.
    ens : int
        The desired number of ensemble segments.
    overlap_ratio : float, optional
        The ratio of overlap between segments (default is 0.5 for 50% overlap).
    fs : float, optional
        The sampling frequency (not used in this function but included for consistency with other parameters).

    Returns
    -------
    nperseg : int
        The length of each segment.
    noverlap : int
        The number of points to overlap between segments.
    df : float
        The frequency resolution corresponding to the calculated nperseg.
    ens : int
        The actual number of ensemble segments that can be formed with the calculated nperseg and overlap.
    """
    nperseg = round(points / (ens - 1 + 1 / (1 - overlap_ratio)))
    noverlap = round(nperseg * overlap_ratio)
    ens_actual = ens_N(points, nperseg, noverlap)
    df = 1 / nperseg if fs is None else fs / nperseg
    return nperseg, noverlap, df, ens_actual


def coherence_analysis(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int = None, noverlap: int = None,
                       **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform coherence analysis between two signals.

    Parameters
    ----------
    x : np.ndarray (n,)
        Signal for reference. This array should be one-dimensional.
    y : np.ndarray (..., n)
        The second input signal. Time axis should be along the last dimension.
    fs : float
        The sampling frequency of the signals.
    nperseg : int, optional
        Length of each segment for Welch's method. If None, it will be set to `min(256, n // 8)`.
    noverlap : int, optional
        Number of points to overlap between segments. If None, it will be set to `nperseg // 2`.
    **kwargs : Any
        Additional keyword arguments to pass to scipy.signal.welch and scipy.signal.csd.

    Returns
    -------
    freqs : np.ndarray
        The frequencies at which the coherence is computed.
    Pxx : np.ndarray (n,)
        The power spectral density of x.
    Pyy : np.ndarray (..., n)
        The power spectral density of y.
    Cxy : np.ndarray (..., n)
        The cross spectral density between x and y.
    coh2 : np.ndarray (..., n)
        The squared coherence between x and y.
    phase : np.ndarray (..., n)
        The phase difference between x and y at each frequency.
    ens : int
        The ensemble number.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if y.shape[-1] != x.shape[-1]:
        raise ValueError("y must have same length as x along last axis")

    n = x.shape[-1]
    if nperseg is None:
        nperseg = min(256, n // 8)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1, **kwargs)
    _, Pyy = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1, **kwargs)
    _, Cxy = signal.csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1, **kwargs)

    coh2 = (np.abs(Cxy) ** 2) / (Pxx * Pyy)
    phase = np.angle(Cxy)

    return freqs, Pxx, Pyy, Cxy, coh2, phase, ens_N(n, nperseg, noverlap)


#
# def sliding_window_coherence(x: np.ndarray, y: np.ndarray, fs: float, window_size: int, step_size: int,
#                              nperseg: int = None, noverlap: int = None, **kwargs: Any):
#     """
#     Calculate time-resolved coherence between two signals using a sliding window approach.
#
#     Parameters
#     ----------
#     x : np.ndarray (n,)
#         Signal for reference. This array should be one-dimensional.
#     y : np.ndarray (..., n)
#         The second input signal. Time axis should be along the last dimension.
#     fs : float
#         The sampling frequency of the signals.
#     window_size : int
#         The size of the sliding window in samples.
#     step_size : int
#         The step size for sliding the window in samples.
#     nperseg : int, optional
#         Length of each segment for Welch's method within each window. If None, it will be set to `min(256, window_size // 8)`.
#     noverlap : int, optional
#         Number of points to overlap between segments within each window. If None, it will be set to `nperseg // 2`.
#     kwargs : Any
#         Additional keyword arguments to pass to scipy.signal.welch and scipy.signal.csd.
#
#     Returns
#     -------
#     time_array : np.ndarray
#         Array of time points corresponding to the center of each window.
#     freqs : np.ndarray
#         The frequencies at which the coherence is computed.
#     Pxx_array : np.ndarray (num_windows, n_freqs)
#         The power spectral density of x for each window.
#     Pyy_array : np.ndarray (num_windows, ..., n_freqs)
#         The power spectral density of y for each window.
#     Cxy_array : np.ndarray (num_windows, ..., n_freqs)
#         The cross spectral density between x and y for each window.
#     coh2_array : np.ndarray (num_windows, ..., n_freqs)
#         The squared coherence between x and y for each window.
#     phase_array : np.ndarray (num_windows, ..., n_freqs)
#         The phase difference between x and y at each frequency for each window.
#     ens : int
#         The ensemble number for each window (should be the same for all windows).
#
#     See Also
#     --------
#     coherence_analysis : Perform coherence analysis on two signals without sliding windows.
#     scipy.signal.welch : Compute the power spectral density using Welch's method.
#     scipy.signal.csd : Compute the cross spectral density between two signals.
#     """
#
#     x = np.asarray(x)
#     y = np.asarray(y)
#     if x.ndim != 1:
#         raise ValueError("x must be 1D")
#     n = x.shape[-1]
#     if y.shape[-1] != n:
#         raise ValueError("y must have same length as x along last axis")
#     window_size = int(window_size)
#     step_size = int(step_size)
#     if window_size > n:
#         raise ValueError("window_size must be less than or equal to the length of the signals")
#
#     xw = sliding_window_view(x, window_size, axis=-1)[::step_size]  # (nwin, window_size)
#     yw = sliding_window_view(y, window_size, axis=-1)[::step_size]  # (..., nwin, window_size)
#     expand_dims = [None] * (yw.ndim - 2)
#     freq, Pxx = signal.welch(xw, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1, **kwargs)  # (nwin, nfreq)
#     _, Pyy = signal.welch(yw, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1, **kwargs)  # (..., nwin, nfreq)
#     _, Cxy = signal.csd(xw[*expand_dims, ...], yw, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1,
#                         **kwargs)  # (..., nwin, nfreq)
#
#     coh2 = (np.abs(Cxy) ** 2) / (Pxx[*expand_dims, ...] * Pyy)
#     phase = np.angle(Cxy)
#
#     start_indices = np.arange(0, n - window_size + 1, step_size)
#     time_array = (start_indices + window_size / 2) / fs
#
#     return time_array, freq, Pxx, Pyy, Cxy, coh2, phase, ens_N(window_size, nperseg, noverlap)


def sliding_window_coherence(x: np.ndarray, y: np.ndarray, fs: float | int, nperseg: int, noverlap: int,
                             seg_per_win: int = None, seg_step: int = None,
                             win_param: tuple[str, Any] = None, **kwargs: Any):
    """
    Calculate time-resolved coherence between two signals using a Short-Time Fourier Transform (STFT).

    Parameters
    ----------
    x : np.ndarray (n,)
        Signal for reference. This array should be one-dimensional.
    y : np.ndarray (..., n)
        The second input signal. Time axis should be along the last dimension.
    fs : float or int
        The sampling frequency of the signals.
    nperseg : int
        Length of each segment for Welch's method within each window.
    noverlap : int
        Number of points to overlap between segments within each window.
    seg_per_win : int, optional
        The number of segments to average spectra over for each window. If None, it will be set to 10.
    seg_step : int, optional
        The number of segments to slide the window for each step. If None, it will be set to half of segment_per_window.
    win_param : tuple[str, Any], optional
        Parameters for the window function used in STFT. If None, it will use a default Hann window.
    kwargs : Any
        Additional keyword arguments to pass to the STFT calculation.

    Returns
    -------
    time_array : np.ndarray
        Array of time points corresponding to the center of each window.
    freqs : np.ndarray
        The frequencies at which the coherence is computed.
    Pxx_sliding : np.ndarray (num_windows, n_freqs)
        The power spectral density of x for each window.
    Pyy_sliding : np.ndarray (num_windows, ..., n_freqs)
        The power spectral density of y for each window.
    Cxy_sliding : np.ndarray (num_windows, ..., n_freqs)
        The cross spectral density between x and y for each window.
    coh2_sliding : np.ndarray (num_windows, ..., n_freqs)
        The squared coherence between x and y for each window.
    phase_sliding : np.ndarray (num_windows, ..., n_freqs)
        The phase difference between x and y at each frequency for each window.

    Notes
    -----
    This function calculates the power spectral densities and cross spectral density within each sliding window.
    The spectra are averaged over a specified number of segments within each window to improve the estimation of coherence.

    See Also
    --------
    scipy.signal.ShortTimeFFT : Class for computing the Short-Time Fourier Transform (STFT) with various windowing options.
    scipy.signal.ShortTimeFFT.stft_detrend : Method for computing the STFT with detrending options.
    scipy.signal.welch : Compute the power spectral density using Welch's method.
    scipy.signal.csd : Compute the cross spectral density between two signals.
    analysis.coherence_analysis : Perform coherence analysis on two signals without sliding windows.
    """
    # sliding window coherence with signal.ShortTimeFourierTransform (STFT)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    n = x.shape[-1]
    if y.shape[-1] != n:
        raise ValueError("y must have same length as x along last axis")

    if seg_per_win is None:
        seg_per_win = 10

    win_param = win_param if win_param is not None else ('hann',)
    STFT = signal.ShortTimeFFT.from_window(win_param=win_param, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                           scale_to='psd', fft_mode='onesided2X', phase_shift=None, **kwargs)

    Sxx = STFT.stft_detrend(x, k_offset=nperseg // 2, p0=0, p1=(n - nperseg) // STFT.hop, detr='constant')
    Syy = STFT.stft_detrend(y, k_offset=nperseg // 2, p0=0, p1=(n - nperseg) // STFT.hop, detr='constant')
    Pxx = np.abs(Sxx) ** 2
    Pyy = np.abs(Syy) ** 2
    Cxy = Sxx.conj() * Syy

    window_size = STFT.hop * (seg_per_win - 1) + nperseg
    if window_size > n:
        raise ValueError(f"window_size must be less than or equal to the length of the signals. \n"
                         f"{nperseg=}, {noverlap=}, {seg_per_win=}, {window_size=}, {n=}")

    seg_step = seg_per_win // 2 if seg_step is None else seg_step

    Pxx_sliding = sliding_window_view(Pxx, seg_per_win, axis=-1)[..., ::seg_step, :].mean(axis=-1)
    Pyy_sliding = sliding_window_view(Pyy, seg_per_win, axis=-1)[..., ::seg_step, :].mean(axis=-1)
    Cxy_sliding = sliding_window_view(Cxy, seg_per_win, axis=-1)[..., ::seg_step, :].mean(axis=-1)

    coh2_sliding = (np.abs(Cxy_sliding) ** 2) / (Pxx_sliding * Pyy_sliding)
    phase_sliding = np.angle(Cxy_sliding)

    time_array = (np.arange(Pxx_sliding.shape[-1]) * seg_step * STFT.hop + window_size / 2) / fs

    return time_array, STFT.f, Pxx_sliding, Pyy_sliding, Cxy_sliding, coh2_sliding, phase_sliding


def _check_params(fs: float, df: float = None, window_sec: float = None,
                  ens: int = None, overlap_ratio: float = 0.5, dt: float = None):
    if fs <= 0:
        raise ValueError("fs must be positive")
    if df is not None and df <= 0:
        raise ValueError("df must be positive")
    if window_sec is not None and window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if ens is not None and ens <= 1:
        raise ValueError("ens must be greater than 1")
    if not (0 <= overlap_ratio < 1):
        raise ValueError("overlap_ratio must be in [0,1)")
    if dt is not None and dt <= 0:
        raise ValueError("dt must be positive")


def param_from_df_window(fs: float, df: float, window_sec: float, overlap_ratio: float = 0.5, dt: float = None):
    _check_params(fs, df=df, window_sec=window_sec, overlap_ratio=overlap_ratio, dt=dt)
    nperseg = round(fs / df)
    noverlap = round(nperseg * overlap_ratio)
    hop = nperseg - noverlap
    window_size = round(fs * window_sec)
    seg_per_win = (window_size - nperseg) // hop + 1
    seg_step = seg_per_win // 2 if dt is None else max(1, round(dt * fs / hop))
    return nperseg, noverlap, seg_per_win, seg_step


def param_from_df_ens(fs: float, df: float, ens: int, overlap_ratio: float = 0.5, dt: float = None):
    _check_params(fs, df=df, ens=ens, overlap_ratio=overlap_ratio, dt=dt)
    nperseg = round(fs / df)
    noverlap = round(nperseg * overlap_ratio)
    hop = nperseg - noverlap
    seg_per_win = ens
    seg_step = seg_per_win // 2 if dt is None else max(1, round(dt * fs / hop))
    return nperseg, noverlap, seg_per_win, seg_step


def param_from_window_ens(fs: float, window_sec: float, ens: int, overlap_ratio: float = 0.5, dt: float = None):
    _check_params(fs, window_sec=window_sec, ens=ens, overlap_ratio=overlap_ratio, dt=dt)
    window_size = round(fs * window_sec)
    # window_size = nperseg + (seg_per_win - 1) * hop
    #             = nperseg + (ens - 1) * (nperseg - noverlap)
    #             = nperseg + (ens - 1) * nperseg * (1 - overlap_ratio)
    #             = nperseg * (1 + (ens - 1) * (1 - overlap_ratio))
    nperseg = round(window_size / (1 + (ens - 1) * (1 - overlap_ratio)))
    noverlap = round(nperseg * overlap_ratio)
    hop = nperseg - noverlap
    seg_per_win = ens
    seg_step = seg_per_win // 2 if dt is None else max(1, round(dt * fs / hop))
    return nperseg, noverlap, seg_per_win, seg_step


def resolutions(nperseg: int, noverlap: int, seg_per_win: int, seg_step: int, fs: float) -> tuple[float, float, float]:
    """
    Calculate the frequency and time resolutions of the sliding window coherence analysis based on the given parameters.

    Parameters
    ----------
    nperseg : int
        Length of each segment for Welch's method within each window.
    noverlap : int
        Number of points to overlap between segments within each window.
    seg_per_win : int
        The number of segments to average spectra over for each window.
    seg_step : int
        The number of segments to slide the window for each step.
    fs : float
        The sampling frequency of the signals.

    Returns
    -------
    df : float
        The frequency resolution of the analysis.
    window_sec : float
        The duration of each sliding window in seconds.
    step_sec : float
        The time step between consecutive windows in seconds.
    """
    hop = nperseg - noverlap
    window_size = nperseg + (seg_per_win - 1) * hop
    step_size = seg_step * hop
    df = fs / nperseg
    dt = step_size / fs
    window_sec = window_size / fs
    return df, window_sec, dt
