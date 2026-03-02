# demo/filter_demo.py

import numpy as np

from ingkit.myplot import pyplot as plt
from ingkit.signals import *
from ingkit.signals.filters import filter_response

if __name__ == '__main__':
    fs = 1e6  # sampling frequency
    t_s, t_e = -1e-3, 1e-3  # time range
    t = np.arange(t_s, t_e, 1 / fs)  # time array
    f1 = 0.7e3  # low frequency noise
    f2 = 10e3  # signal frequency
    f3 = 150e3  # high frequency noise

    amps = [0.5, 0.5, 0.2]  # amplitudes of the components
    signal = (amps[0] * np.sin(2 * np.pi * f1 * t) +
              amps[1] * np.sin(2 * np.pi * f2 * t) +
              amps[2] * np.sin(2 * np.pi * f3 * t))

    # Apply low-pass filter
    cutoff_lp = 100e3  # cutoff frequency for low-pass filter
    cutoff_bp = (5e3, 100e3)  # cutoff frequencies for band-pass filter

    # Plotting filter responses
    filter_names = ["bessel", "butter"]
    for filter_name in filter_names:
        f, resp_lp = filter_response(cutoff_freq=cutoff_lp, filter_type="lowpass",
                                     fs=fs, filter_name=filter_name, order=4, worN=2048)
        _, resp_bp = filter_response(cutoff_freq=cutoff_bp, filter_type="bandpass",
                                     fs=fs, filter_name=filter_name, order=4, worN=2048)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].loglog(f, np.abs(resp_lp), label='Low-pass Filter Response')
        axes[0].loglog(f, np.abs(resp_bp), label='Band-pass Filter Response')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Magnitude')
        axes[0].set_title(f'{filter_name} filter response')
        axes[0].legend()
        axes[0].grid()
        axes[0].set_xlim(f[1], fs / 2)
        axes[0].set_ylim(1e-3, 1.5)

        axes[1].semilogx(f, np.angle(resp_lp), label='Low-pass Filter Phase')
        axes[1].semilogx(f, np.angle(resp_bp), label='Band-pass Filter Phase')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (radians)')
        axes[1].set_title(f'{filter_name} filter phase response')
        axes[1].legend()
        axes[1].grid()
        axes[1].set_ylim(-np.pi, np.pi)

        fig.tight_layout()
        fig.show()

    # Apply filters to the signal
    filtered_lp = lowpass_filter(signal, cutoff_freq=cutoff_lp, fs=fs, filter_name="bessel", order=4)
    filtered_bp = bandpass_filter(signal, cutoff_freq=cutoff_bp, fs=fs, filter_name="bessel", order=4)

    # Plotting the original and filtered signals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t * 1e3, signal, label='Original Signal')
    ax.plot(t * 1e3, filtered_lp, label=f'Low-pass ({cutoff_lp / 1e3} kHz)')
    ax.plot(t * 1e3, filtered_bp, label=f'Band-pass ({cutoff_bp[0] / 1e3}-{cutoff_bp[1] / 1e3} kHz)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal Filtering Demo')
    ax.legend()
    ax.grid()
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.show()
