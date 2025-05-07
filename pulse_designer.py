import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import minimize
import nevergrad as ng
from scipy.signal.windows import chebwin
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from IPython.display import FileLink


def fft_pad(waveform, pad=0):
    # Description:
    #     Compute the fast fourier transform of the input waveform with an
    #     optional length padding. The necessarily corrections for frequency
    #     shift and sign change are implemented.
    # Arguments:
    #     waveform:
    #         The input waveform.
    #     pad:
    #         The length of zero data attached to the input waveform. Finer
    #         frequency resolution of the fourier transform correspondes
    #         to longer padding.
    # Return:
    #     The fast fourier transform of the input waveform with an
    #     optional length padding
    wf_pad = np.zeros(len(waveform) + pad, dtype=np.complex128)
    wf_pad[:len(waveform)] = waveform
    t = np.arange(len(wf_pad))
    T = len(wf_pad)
    phi = 2 * np.pi * (t / T) * ((len(waveform) - 1) / 2)
    wf_ft_pad = fft(wf_pad) * np.exp(1j * phi)
    if len(waveform) % 2 == 0:
        wf_ft_pad[(len(wf_ft_pad) // 2):] = -wf_ft_pad[(len(wf_ft_pad) // 2):]
    return wf_ft_pad


def fft_tool(waveform, sampl_rate_Hz=1e9, freq_res_Hz=1e6):
    # Description:
    #     The convenient function for computing the fast fourier transform of
    #     the input waveform at a defined sampling rate and frequency
    #     resolution. Both the returned frequency list and fourier transform
    #     starts in negative frequency and ends in the positive frequency, in
    #     other words, shifted by fftshift.
    # Arguments:
    #     waveform:
    #         The input waveform.
    #     sampl_rate_Hz:
    #         The sampling rate of the waveform in Hz. The default value
    #         is 1 GHz.
    #     freq_res_Hz:
    #         *Estimation* of the frequency resolution of the fourier
    #         transform in Hz. The default value is 1 MHz. The accurate
    #         frequency resolution should be calculated from the returned
    #         frequency list.
    # Return:
    #     freq:
    #     The frequency list starts in negative frequency and
    #     ends in the positive frequency.
    #     wf_ft_pad:
    #         The fast fourier transform of the input waveform with an
    #         optional length padding
    pad = max(0, int(sampl_rate_Hz / freq_res_Hz) - len(waveform))
    wf_ft_pad = fftshift(fft_pad(waveform, pad=pad))
    freq = fftshift(fftfreq(len(wf_ft_pad), d=1 / sampl_rate_Hz))
    return freq, wf_ft_pad


def waveform_plot(pulse, sampl_rate_Hz=1e9, freq_res_Hz=1e6, log_scale=True, freqs_MHz_vline=None):
    # variables:
    #     pulse: the input pulse
    #     freq_res: the minimum frequency resolution for the 
    #               fast fourier transform

    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.family'] = 'DejaVu Serif'
    # matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['font.size'] = 16
    
    fig = plt.figure(figsize=(12, 4.2))

    ax = plt.subplot(1, 2, 1)

    ax.plot(np.real(pulse), label='I', color='r', zorder=0.5)
    ax.plot(np.imag(pulse), label='Q', color='b', zorder=0.4)
    ax.set_xlim(0, len(pulse) - 1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend()

    ax = plt.subplot(1, 2, 2)
    # ax.plot(fftfreq(len(pulse)), np.real(fft(pulse)))
    freq, pulse_fft_pad = fft_tool(pulse, sampl_rate_Hz, freq_res_Hz)
    freq_MHz = 1e-6 * freq

    ax.axvline(0, color='gray', linestyle='--', zorder=-np.inf)
    if freqs_MHz_vline is not None:
        for freq_MHz_vline in freqs_MHz_vline:
            ax.axvline(freq_MHz_vline, color='k', linestyle=':', zorder=-np.inf)

    if log_scale:
        abs_pulse_fft_pad = np.abs(pulse_fft_pad)
        ax.plot(freq_MHz, abs_pulse_fft_pad, label='I', color='r', zorder=0.5)
        ax.set_yscale('log')
        max_pulse_fft_pad = np.max(abs_pulse_fft_pad)
        ax.set_ylim(1e-9 * max_pulse_fft_pad, 10 * max_pulse_fft_pad)

    else:
        ax.plot(freq_MHz, np.real(pulse_fft_pad), label='I', color='r', zorder=0.5)
        ax.plot(freq_MHz, np.imag(pulse_fft_pad), label='Q', color='b', zorder=0.4)
        ax.legend()
    # ax.plot(freq_MHz, np.abs(pulse_fft_pad), label='abs', zorder=0.4)

    nyquist_freq_MHz = 0.5e-6 * sampl_rate_Hz
    ax.set_xlim(-nyquist_freq_MHz, nyquist_freq_MHz)

    

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel(r'Amplitude spectrum (V$\cdot$ns)')
    
    fig.tight_layout()
    plt.show()


def u_pulse(N, omega_0, n=None):
    if n is None:
        n = np.arange(N, dtype=np.complex128)
    u = 1 - np.cos((2 * n + 1) * (np.pi / N))
    u*= np.exp((0.5j * omega_0) * (2 * n - N + 1))
    return u


def v_pulse(N, omega_0, n=None):
    if n is None:
        n = np.arange(N, dtype=np.complex128)
    v = 2j * np.sin((2 * n + 1) * (np.pi / N))
    v *= np.exp((0.5j * omega_0) * (2 * n - N + 1))
    return v


def w_pulse(N, omega_0, alpha, n=None):
    if n is None:
        n = np.arange(N, dtype=np.complex128)
    u = 1 - np.cos((2 * n + 1) * (np.pi / N))
    v_imag = 2 * np.sin((2 * n + 1) * (np.pi / N))
    w = np.cos(alpha) * u + 1j * np.sin(alpha) * v_imag
    w *= np.exp((0.5j * omega_0) * (2 * n - N + 1))
    return w


def u_fourier(N, omega_0, omega):
    omega = omega - omega_0
    if omega == 0:
        fu = N
    elif np.pi / N - 0.5 * omega == 0:
        fu = 0.5 * N
    elif np.pi / N + 0.5 * omega == 0:
        fu = 0.5 * N
    else:
        fu = (np.sin(np.pi / (2 * N)) ** 2) * np.sin(0.5 * N * omega) * (1 + 2 * np.cos(np.pi / N) + np.cos(omega)) / (np.sin(0.5 * omega) * np.sin(np.pi / N + 0.5 * omega) * np.sin(np.pi / N - 0.5 * omega))

    return fu


def v_fourier(N, omega_0, omega):
    omega = omega - omega_0
    if omega == 0:
        fv = 0
    elif np.pi / N - 0.5 * omega == 0:
        fv = -N
    elif np.pi / N + 0.5 * omega == 0:
        fv = N
    else:
        fv = -2 * np.sin(np.pi / N) * np.cos(0.5 * omega) * np.sin(0.5 * N * omega) / ( np.sin(np.pi / N + 0.5 * omega) * np.sin(np.pi / N - 0.5 * omega))

    return fv


def w_fourier(N, omega_0, alpha, omega):
    omega = omega - omega_0
    if omega == 0:
        fu = N
        fv = 0
    elif np.pi / N - 0.5 * omega == 0:
        fu = 0.5 * N
        fv = -N
    elif np.pi / N + 0.5 * omega == 0:
        fu = 0.5 * N
        fv = N
    else:
        fu = (np.sin(np.pi / (2 * N)) ** 2) * np.sin(0.5 * N * omega) * (1 + 2 * np.cos(np.pi / N) + np.cos(omega)) / (np.sin(0.5 * omega) * np.sin(np.pi / N + 0.5 * omega) * np.sin(np.pi / N - 0.5 * omega))
        fv = -2 * np.sin(np.pi / N) * np.cos(0.5 * omega) * np.sin(0.5 * N * omega) / ( np.sin(np.pi / N + 0.5 * omega) * np.sin(np.pi / N - 0.5 * omega))
        
    fw = np.cos(alpha) * fu + np.sin(alpha) * fv
    return fw


def w_solve_0(N, omega_1):
    omega_0 = 0.0
    fu = u_fourier(N, omega_0, omega_1)
    fv = v_fourier(N, omega_0, omega_1)

    alpha = np.arctan2(fu, -fv)
    if alpha > 0.5 * np.pi:
        alpha -= np.pi
    elif alpha < -0.5 * np.pi:
        alpha += np.pi

    return omega_0, alpha


def w_solve_1(N, omega_1):
    if np.abs(omega_1) < 1.6 * np.pi / N:
        print('omega_1 is too small.')
        return
    omega_0 = 0.0

    for _ in range(20):
        fu = u_fourier(N, omega_0, omega_1)
        fv = v_fourier(N, omega_0, omega_1)
    
        alpha = np.arctan2(fu, -fv)
        if alpha > 0.5 * np.pi:
            alpha -= np.pi
        elif alpha < -0.5 * np.pi:
            alpha += np.pi
    
        omega_0 = 0.5 * np.pi * np.pi * np.sin(alpha) / N
    
    return omega_0, alpha


def w_solve_2(N, omega_1):
    if np.abs(omega_1) < 2.6 * np.pi / N:
        print('omega_1 is too small.')
        return
    
    omega_0, alpha =  w_solve_1(N, omega_1)

    def loss(d_omega_0):
        omega_0_new = omega_0 + d_omega_0
        fu = u_fourier(N, omega_0_new, omega_1)
        fv = v_fourier(N, omega_0_new, omega_1)

        alpha = np.arctan2(fu, -fv)
        if alpha > 0.5 * np.pi:
            alpha -= np.pi
        elif alpha < -0.5 * np.pi:
            alpha += np.pi

        sum = 0

        fu = u_fourier(N, omega_0_new, 0.25 * np.pi * np.pi / N)
        fv = v_fourier(N, omega_0_new, 0.25 * np.pi * np.pi / N)
        sum += (np.cos(alpha) * fu + np.sin(alpha) * fv)

        fu = u_fourier(N, omega_0_new, -0.25 * np.pi * np.pi / N)
        fv = v_fourier(N, omega_0_new, -0.25 * np.pi * np.pi / N)
        sum += (np.cos(alpha) * fu + np.sin(alpha) * fv)

        fu = u_fourier(N, omega_0_new, 0.5 * np.pi * np.pi / N)
        fv = v_fourier(N, omega_0_new, 0.5 * np.pi * np.pi / N)
        sum += (np.cos(alpha) * fu + np.sin(alpha) * fv)

        fu = u_fourier(N, omega_0_new, -0.5 * np.pi * np.pi / N)
        fv = v_fourier(N, omega_0_new, -0.5 * np.pi * np.pi / N)
        sum += (np.cos(alpha) * fu + np.sin(alpha) * fv)

        return -sum
        
    res = minimize(loss, [0.0], bounds=[(-np.pi * np.pi / N, np.pi * np.pi / N)], method='L-BFGS-B', options=dict(ftol=0, gtol=0))
    d_omega_0 = res.x[0]
    omega_0= omega_0 + d_omega_0

    fu = u_fourier(N, omega_0, omega_1)
    fv = v_fourier(N, omega_0, omega_1)
    
    alpha = np.arctan2(fu, -fv)
    if alpha > 0.5 * np.pi:
        alpha -= np.pi
    elif alpha < -0.5 * np.pi:
        alpha += np.pi
    
    return omega_0, alpha


def gen_w_pulse(N, omega_1):
    omega_0, alpha = w_solve_2(N, omega_1)
    return w_pulse(N, omega_0, alpha)


def export_w_pulse(w):
    Path(r'outputs').mkdir(parents=True, exist_ok=True)
    w_csv = np.stack((np.real(w), np.imag(w)), axis=1)
    np.savetxt(r'outputs/w.csv', w_csv, delimiter=',', header='I,Q')
    return FileLink('outputs/w.csv')