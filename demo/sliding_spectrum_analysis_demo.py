import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from ingkit.signals import sliding_window_coherence, param_from_df_window

if __name__ == '__main__':
    fs = 10_000
    T = 10.0
    N = int(fs * T)
    t = np.arange(N) / fs

    # 線形chirp
    x = signal.chirp(t, f0=200, f1=3000, t1=T, method='linear', )
    # 振幅包絡
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.15 * t))
    x *= env
    # 時間変化phase
    phase_offset = 0.8 * np.sin(2 * np.pi * 0.3 * t)
    # coherent成分
    y_sig = signal.chirp(t, f0=200, f1=3000, t1=T, method='linear', phi=np.rad2deg(phase_offset), )

    # coherenceが途中で落ちるように
    mask = ((t > 3) & (t < 7)).astype(float)
    y = (mask * y_sig + 0.7 * np.random.randn(N))

    # multi-channel例
    y_multi = np.stack([y, 0.7 * y + 0.3 * np.random.randn(N), np.random.randn(N), ], axis=0)

    nperseg, noverlap, seg_per_win, seg_step = param_from_df_window(fs=fs, df=10, window_sec=1,
                                                                    overlap_ratio=0.75, dt=0.5)

    t_arr, f_arr, Pxx_sliding, Pyy_sliding, Cxy_sliding, coh2_sliding, phase_sliding = \
        sliding_window_coherence(x, y_multi, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                 seg_per_win=seg_per_win, seg_step=seg_step)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, )
    axes[0].pcolormesh(t_arr, f_arr, coh2_sliding[0])
    phase_ = np.ma.masked_array(phase_sliding[0], coh2_sliding[0] < 0.3)
    axes[1].pcolormesh(t_arr, f_arr, phase_)

    plt.show()
