# demo/spectrum_analysis_demo.py

import numpy as np
import scipy.signal as sg

from ingkit.myplot import pyplot as plt
from ingkit.signals import lowpass_filter

if __name__ == '__main__':
    np.random.seed(1234)  # for reproducibility

    f0 = 0.3e3  # low frequency noise
    f1 = 10e3  # signal frequency1
    f2 = 40e3  # signal frequency2
    f3 = 700.3e3  # high frequency noise
    f4 = 80e3  # burst signal frequency
    freqs = [f0, f1, f2, f3, f4]
    amps = [0.4, 0.3, 0.2, 0.25, 0.4]  # amplitudes of the components
    phases = np.random.uniform(0, 2 * np.pi, size=5)  # random phases
    noise_amp = 0.1  # amplitude of noise
    dc_offset = 1  # DC offset
    burst_time = (0.25e-3, 0.35e-3)  # time range for burst signal


    def pink_noise(size):
        """Generate pink noise (1/f noise)"""
        freqs = np.fft.rfftfreq(size, d=1 / 1e6)  # assuming fs=1 MHz for pink noise generation
        X = np.random.normal(size=freqs.size) * np.exp(
            2j * np.pi * np.random.uniform(size=freqs.size))  # random complex numbers

        freqs[0] = 1e-6  # avoid division by zero
        X /= np.sqrt(freqs)  # 1/f scaling
        X[0] = 0  # remove DC component
        return np.fft.irfft(X) * size / 2  # scale back to time domain


    def white_noise(size):
        """Generate white noise"""
        return np.random.normal(size=size)


    def generate_signal(t):
        y = [a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases)]
        y[4] *= np.exp(-((t - np.mean(burst_time)) ** 2) / (
                (burst_time[1] - burst_time[0]) ** 2))  # transient signal with Gaussian envelope
        # y.append(pink_noise(len(t)) * noise_amp)  # add pink noise
        y.append(white_noise(len(t)) * noise_amp)  # add white noise
        y.append(dc_offset * np.ones_like(t))  # add DC offset
        y = np.array(y)
        return y


    # plot signals
    fs_100M = 100e6  # ideal sampling frequency for demonstration
    t_100M = np.arange(-2e-3, 2e-3, 1 / fs_100M)  # time array from -1 ms to 1 ms
    y_100M = generate_signal(t_100M)  # generate signal at high sampling rate
    ideal_signal_100M = np.sum(y_100M[[1, 2]], axis=0)  # ideal signal without noise and offset
    all_signal_100M = np.sum(y_100M[:5], axis=0)  # signal with all components but no noise or offset
    lowpass_signal_100M = lowpass_filter(all_signal_100M, fs=fs_100M, cutoff_freq=200e3)  # low-pass filtered signal
    noisy_signal_100M = np.sum(y_100M, axis=0)  # signal with all components including noise and offset
    noisy_lowpass_signal_100M = lowpass_signal_100M + y_100M[5] + y_100M[
        6]  # low-pass filtered signal with noise and offset

    plt.figure(figsize=(10, 6))
    plt.plot(t_100M * 1e3, ideal_signal_100M, label='Ideal Signal (f1=10kHz, f2=40kHz)')
    plt.plot(t_100M * 1e3, all_signal_100M + 3, label='Ideal Signal + Noise (f0=0.2kHz, f3=700kHz, f4=80kHz)')
    plt.plot(t_100M * 1e3, noisy_signal_100M + 6, label='Noisy Signal (with DC offset)')
    plt.plot(t_100M * 1e3, noisy_lowpass_signal_100M + 9, label='Low-pass Filtered Signal (with noise and DC offset)')

    plt.xlim(-1, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Time-domain Signal')
    plt.legend(loc='lower left', fontsize=10, framealpha=0.7)
    plt.grid()
    plt.show()

    # サンプリング周波数 fs = 1 MHz でサンプリングした場合のスペクトルを示す．
    t_1M = t_100M[::100]  # decimate to 1 MHz
    ideal_signal_1M = ideal_signal_100M[::100]  # ideal signal (f1 & f2)
    all_signal_1M = all_signal_100M[::100]  # all frequencies (f0-f4), aliasing test
    noisy_signal_1M = noisy_signal_100M[::100]  # all f (f0-f4) + noise + DC offset, aliasing test

    lowpass_signal_1M = lowpass_signal_100M[::100]  # antialiasing filter (no noise)
    noisy_lowpass_signal_1M = noisy_lowpass_signal_100M[::100]  # antialiasing filter + noise + DC offset

    freq_1M, P_noisy = sg.periodogram(noisy_signal_1M, fs=1e6, window="boxcar", detrend=False,
                                      scaling="density")  # no window, no detrend
    _, P_noisy_window = sg.periodogram(noisy_signal_1M, fs=1e6, window="hann", detrend="constant",
                                       scaling="density")  # with window and detrend
    _, P_lowpass = sg.periodogram(noisy_lowpass_signal_1M, fs=1e6, window="boxcar", detrend=False,
                                  scaling="density")  # antialiasing filter, no window
    _, P_lowpass_window = sg.periodogram(noisy_lowpass_signal_1M, fs=1e6, window="hann", detrend="constant",
                                         scaling="density")  # antialiasing filter, with window and detrend

    freq_welch, P_noisy_welch = sg.welch(noisy_signal_1M, fs=1e6, window="hann", detrend="linear",
                                         nperseg=1000, noverlap=None, scaling="density", average="mean"
                                         )  # Welch's method for comparison
    _, P_lowpass_welch = sg.welch(noisy_lowpass_signal_1M, fs=1e6, window="hann", detrend="linear",
                                  nperseg=1000, noverlap=None, scaling="density", average="median"
                                  )  # Welch's method for low-pass filtered signal

    plt.figure(figsize=(10, 6))
    plt.axvline(f1, color='k', ls="--", alpha=0.7)
    plt.axvline(f2, color='k', ls="--", alpha=0.7)
    plt.axvline(0.5e6 - (f3 - 0.5e6), color='k', ls="--", alpha=0.7)  # aliased f3
    plt.axvline(f4, color='k', ls="--", alpha=0.7)

    plt.plot(freq_1M, P_noisy, label='No pre-proc')
    plt.plot(freq_1M, P_noisy_window, label='Windowed & detrend')
    plt.plot(freq_1M, P_lowpass * 1e5, label='Anti-aliasing')
    plt.plot(freq_1M, P_lowpass_window * 1e5, label='Anti-aliasing + Windowed & detrend')
    plt.plot(freq_welch, P_noisy_welch * 1e10, label='Welch', ls="--", c='C0')
    plt.plot(freq_welch, P_lowpass_welch * 1e10, label='Anti-aliasing + Welch', ls="--", c='C2')

    for _ in [1, 1e5, 1e10]:
        plt.axhline(_ * np.median(P_lowpass_welch),
                    color='k', ls="--", lw=1, alpha=0.7)  # reference lines for power levels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(freq_1M[1]/2, 500e3)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (a.u.)')
    plt.title('Frequency-domain Spectrum')
    plt.legend(loc='best', fontsize=10, framealpha=0.7)
    plt.grid()
    plt.tight_layout()
    plt.show()
