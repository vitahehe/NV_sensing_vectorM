import numpy as np
from scipy.signal import savgol_filter, find_peaks


#  -----------------------------
# Helper functions
# -----------------------------
def normalize_signal(y):
    """
    Normalize ODMR so that off-resonant baseline is near 1.
    This is convenient because dips are then modeled as downward features.
    """
    y = np.asarray(y, dtype=float)
    y_min = np.min(y)
    y_max = np.max(y)
    if np.isclose(y_max, y_min):
        raise ValueError("Signal is nearly constant; cannot normalize.")
    return (y - y_min) / (y_max - y_min)


def fit_baseline_poly(f_mhz, y, deg=2):
    """
    Fit a low-order polynomial baseline to the full spectrum.
    Since your ODMR dips are relatively narrow compared to the full span,
    a quadratic baseline is usually enough for a first notebook.
    """
    coeffs = np.polyfit(f_mhz, y, deg)
    return coeffs


def eval_baseline_poly(f_mhz, coeffs):
    return np.polyval(coeffs, f_mhz)


def detect_8_transition_guesses(f_mhz, y_norm, smooth_window=21, polyorder=3, prominence=0.01):
    """
    Detect 8 ODMR transitions as minima of the normalized spectrum.
    We invert the signal so that dips become peaks.
    """
    if smooth_window % 2 == 0:
        smooth_window += 1

    y_smooth = savgol_filter(y_norm, smooth_window, polyorder)
    inv = 1.0 - y_smooth

    peaks, props = find_peaks(inv, prominence=prominence)

    if len(peaks) < 8:
        raise RuntimeError(
            f"Detected only {len(peaks)} candidate dips. "
            "Try lowering 'prominence' or increasing smoothing."
        )

    # Keep the 8 most prominent dips
    prominences = props["prominences"]
    order = np.argsort(prominences)[::-1][:8]
    peaks = peaks[order]
    peaks = np.sort(peaks)

    return f_mhz[peaks], y_smooth, peaks, prominences[order]


def pair_centers_from_8(sorted_centers_mhz):
    """
    Pair the 8 fitted transition centers using the known ODMR ordering:
    left side:  d1, d2, d3, d4
    right side: d4, d3, d2, d1

    sorted_centers_mhz must be ascending in frequency.
    """
    c = np.asarray(sorted_centers_mhz, dtype=float)
    if len(c) != 8:
        raise ValueError("Need exactly 8 transition centers.")

    pairs = {
        "d1": (c[0], c[7]),
        "d2": (c[1], c[6]),
        "d3": (c[2], c[5]),
        "d4": (c[3], c[4]),
    }
    return pairs


# -----------------------------
# ODMR model
# -----------------------------
def lorentzian(x, x0, gamma_fwhm):
    """
    Unit-height Lorentzian with FWHM = gamma_fwhm.
    x, x0, gamma_fwhm all in MHz.
    """
    return 1.0 / (1.0 + 4.0 * ((x - x0) / gamma_fwhm) ** 2)


def odmr_triplet(x, center, contrast, delta_hf, gamma_fwhm, w_minus, w_zero, w_plus):
    """
    One hyperfine-resolved ODMR transition, modeled as a sum of 3 Lorentzians.
    All frequencies in MHz.

    center = central transition frequency
    delta_hf = hyperfine splitting between adjacent triplet components
    """
    return contrast * (
        w_minus * lorentzian(x, center - delta_hf, gamma_fwhm) +
        w_zero  * lorentzian(x, center,            gamma_fwhm) +
        w_plus  * lorentzian(x, center + delta_hf, gamma_fwhm)
    )


def odmr_model_full(x, p):
    """
    Global ODMR model.

    Parameter vector p:
    - 8 centers
    - 8 contrasts
    - 1 shared delta_hf
    - 8 linewidths gamma_fwhm_i
    - 3 shared hyperfine weights
    - baseline polynomial coefficients b2, b1, b0
    """
    centers = np.array(p[0:8], dtype=float)
    contrasts = np.array(p[8:16], dtype=float)

    delta_hf = float(p[16])
    gammas = np.array(p[17:25], dtype=float)

    w_minus, w_zero, w_plus = p[25:28]
    s = w_minus + w_zero + w_plus + 1e-15
    w_minus, w_zero, w_plus = w_minus / s, w_zero / s, w_plus / s

    b2, b1, b0 = p[28:31]
    baseline = b2 * x**2 + b1 * x + b0

    y = baseline.copy()
    for c, a, g in zip(centers, contrasts, gammas):
        y -= odmr_triplet(x, c, a, delta_hf, g, w_minus, w_zero, w_plus)

    return y

def residuals_odmr_full(p, x, y):
    return odmr_model_full(x, p) - y