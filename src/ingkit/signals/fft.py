# src/ingkit/signals/fft.py
# FFT utilities for signal processing.

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def compute_fft(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform (FFT) of a signal.

    Parameters
    ----------
    y : np.ndarray
        The input signal. Time axis should be along the last dimension.
    fs : float
        The sampling frequency of the signal.

    Returns
    -------
    freqs : np.ndarray
        The frequencies corresponding to the FFT components.
    fft_values : np.ndarray
        The FFT values of the input signal.
    """
    n = y.shape[-1]
    if np.iscomplex(y).any():
        # For complex signals, use the full FFT.
        fft_values = np.fft.fft(y, axis=-1)
        freqs = np.fft.fftfreq(n, d=1 / fs)
    else:
        # For real signals, use the rFFT which is more efficient.
        fft_values = np.fft.rfft(y, axis=-1)
        freqs = np.fft.rfftfreq(n, d=1 / fs)
    return freqs, fft_values


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
        The computed nperseg value to achieve the desired number of ensemble segments.
    df : float, optional
        The frequency resolution corresponding to the computed nperseg if fs is provided.
    """
    if ens <= 0:
        raise ValueError("ens must be positive")
    if not (0 <= overlap_ratio < 1):
        raise ValueError("overlap_ratio must be in [0,1)")

    r = overlap_ratio
    nperseg = points // ((ens - 1) * (1 - r) + 1)
    nperseg = max(1, int(nperseg))  # Ensure nperseg is at least 1

    def ens_(nperseg_: int) -> int:
        noverlap_ = int(nperseg_ * overlap_ratio)
        step_ = nperseg_ - noverlap_
        return (points - nperseg_) // step_ + 1

    while True:
        ens_calculated = ens_(nperseg)
        if nperseg == 1:
            break
        elif ens_calculated >= ens:
            break
        nperseg -= 1

    if fs is not None:
        df = fs / nperseg
        return nperseg, ens_calculated, df
    return nperseg, ens_calculated


def coherence_analysis(x: np.ndarray, y: np.ndarray, fs: float, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
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
    **kwargs : Any
        Additional keyword arguments to pass to scipy.signal.coherence.

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
    if kwargs.get("nperseg") is None:
        kwargs["nperseg"] = min(256, n)
    if kwargs.get("noverlap") is None:
        kwargs["noverlap"] = kwargs["nperseg"] // 2

    step = kwargs["nperseg"] - kwargs["noverlap"]
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")
    ens = 1 + (n - kwargs["noverlap"]) // step

    freqs, Pxx = signal.welch(x, fs=fs, **kwargs)
    _, Pyy = signal.welch(y, fs=fs, **kwargs)
    _, Cxy = signal.csd(x, y, fs=fs, **kwargs)

    coh2 = (np.abs(Cxy) ** 2) / (Pxx * Pyy)
    phase = np.angle(Cxy)

    return freqs, Pxx, Pyy, Cxy, coh2, phase, ens
