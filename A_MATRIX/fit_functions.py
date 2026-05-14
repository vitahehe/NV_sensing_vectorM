import numpy as np
from scipy.signal import savgol_filter, find_peaks


def normalize_signal(y):
    y = np.asarray(y, dtype=float)
    y_min = np.min(y)
    y_max = np.max(y)
    if np.isclose(y_max, y_min):
        raise ValueError("Signal is nearly constant; cannot normalize.")
    return (y - y_min) / (y_max - y_min)


def detect_8_transition_guesses(f_mhz, y_norm, smooth_window=21, polyorder=3, prominence=0.01):
    f_mhz = np.asarray(f_mhz, dtype=float)
    y_norm = np.asarray(y_norm, dtype=float)

    if smooth_window % 2 == 0:
        smooth_window += 1

    y_smooth = savgol_filter(y_norm, smooth_window, polyorder)
    inv = 1.0 - y_smooth

    peaks, props = find_peaks(inv, prominence=prominence)
    if len(peaks) < 8:
        raise RuntimeError(f"Detected only {len(peaks)} dips, need at least 8.")

    prominences = props["prominences"]
    keep = np.argsort(prominences)[::-1][:8]
    peaks = np.sort(peaks[keep])

    return f_mhz[peaks], y_smooth, peaks, prominences[keep]


def lorentzian(x, x0, gamma_fwhm):
    x = np.asarray(x, dtype=float)
    x0 = float(x0)
    gamma_fwhm = float(gamma_fwhm)
    return 1.0 / (1.0 + 4.0 * ((x - x0) / gamma_fwhm) ** 2)


def odmr_triplet_independent(x, center, delta_hf, gamma_fwhm, A_minus, A_zero, A_plus):
    x = np.asarray(x, dtype=float)

    return (
        A_minus * lorentzian(x, center - delta_hf, gamma_fwhm) +
        A_zero  * lorentzian(x, center,            gamma_fwhm) +
        A_plus  * lorentzian(x, center + delta_hf, gamma_fwhm)
    )


def odmr_model_full(x, p):
    """
    Parameter vector:
    p[0:8]    = 8 centers
    p[8:16]   = 8 delta_hf_i
    p[16:24]  = 8 linewidths gamma_i

    p[24:48]  = 8 x 3 amplitudes
                [A1m, A1z, A1p, A2m, A2z, A2p, ..., A8m, A8z, A8p]

    p[48:51]  = baseline coefficients (b2, b1, b0) on centered x
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    centers = p[0:8]
    delta_hf = p[8:16]
    gammas = p[16:24]
    amps = p[24:48].reshape(8, 3)

    x_centered = x - np.mean(x)
    b2, b1, b0 = p[48:51]
    baseline = b2 * x_centered**2 + b1 * x_centered + b0

    y = baseline.copy()

    for i in range(8):
        c = centers[i]
        d = delta_hf[i]
        g = gammas[i]
        A_minus, A_zero, A_plus = amps[i]

        y -= odmr_triplet_independent(
            x, c, d, g, A_minus, A_zero, A_plus
        )

    return y

def residuals_odmr_full(p, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return odmr_model_full(x, p) - y