import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from itertools import permutations
import re

# Model functions
def lorentzian(x, x0, gamma_fwhm):
    return 1.0 / (1.0 + 4.0 * ((x - x0) / gamma_fwhm) ** 2)

def make_global_odmr_model():
    def model(x, *params):
        n_dips = 8
        centers = np.array(params[:n_dips])
        contrasts = np.array(params[n_dips:2 * n_dips])
        delta_hf = params[2 * n_dips]
        width_L  = params[2 * n_dips + 1]
        w_m1 = params[2 * n_dips + 2]
        w_0  = params[2 * n_dips + 3]
        w_p1 = params[2 * n_dips + 4]
        wsum = w_m1 + w_0 + w_p1 + 1e-15
        w_m1 /= wsum
        w_0  /= wsum
        w_p1 /= wsum
        y = np.ones_like(x, dtype=float)
        for fc, C in zip(centers, contrasts):
            phi_m1 = lorentzian(x, fc - delta_hf, width_L)
            phi_0  = lorentzian(x, fc,             width_L)
            phi_p1 = lorentzian(x, fc + delta_hf, width_L)
            triplet = w_m1 * phi_m1 + w_0 * phi_0 + w_p1 * phi_p1
            y -= C * triplet
        return y
    return model

def normalize_signal(y):
    y = np.asarray(y, dtype=float)
    return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-15)

def smooth_signal(y, window=9, poly=3):
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=poly)

def validate_frequency_units(f_hz):
    f_hz = np.asarray(f_hz, dtype=float)
    f_min = np.min(f_hz)
    f_max = np.max(f_hz)
    if f_max < 1e6:
        raise ValueError("Frequency axis looks too small to be in Hz. Expected raw data in Hz, e.g. around 2.8e9 to 3.0e9.")
    f_ghz = f_hz / 1e9
    if not (2.0 < np.median(f_ghz) < 4.0):
        raise ValueError(f"After Hz->GHz conversion, median frequency is {np.median(f_ghz):.6f} GHz, which does not look like ODMR around 2.87 GHz.")

def detect_8_dip_center_guesses(f_ghz, y_norm, D_GHz=2.87, fmin=2.79, fmax=2.96, prominence=0.01):
    y_smooth = smooth_signal(y_norm, window=9, poly=3)
    inv = 1.0 - y_smooth
    left_mask = (f_ghz < D_GHz) & (f_ghz >= fmin)
    right_mask = (f_ghz > D_GHz) & (f_ghz <= fmax)
    df = np.median(np.diff(f_ghz))
    min_distance_pts = max(4, int(np.ceil(0.0015 / max(df, 1e-12))))
    def detect_side(mask):
        f_side = f_ghz[mask]
        inv_side = inv[mask]
        peaks, props = find_peaks(inv_side, prominence=prominence, distance=min_distance_pts)
        if len(peaks) == 0:
            return np.array([]), np.array([])
        prominences = props["prominences"]
        strongest = np.argsort(prominences)[::-1][:4]
        peak_freqs = f_side[peaks[strongest]]
        peak_proms = prominences[strongest]
        order = np.argsort(peak_freqs)
        return peak_freqs[order], peak_proms[order]
    left_centers, left_proms = detect_side(left_mask)
    right_centers, right_proms = detect_side(right_mask)
    if len(left_centers) != 4 or len(right_centers) != 4:
        raise RuntimeError(f"Expected 4 left and 4 right dips. Got {len(left_centers)} left and {len(right_centers)} right. This usually means the spectrum is too overlapped for the simple resolved-dip detector.")
    dip_centers = np.sort(np.concatenate([left_centers, right_centers]))
    dip_prominences = np.concatenate([left_proms, right_proms])
    return dip_centers, y_smooth, dip_prominences

def fit_polynomial_baseline(f_ghz, y, dip_centers, mask_halfwidth_GHz=0.006, deg=2):
    mask = np.ones_like(f_ghz, dtype=bool)
    for fc in dip_centers:
        mask &= np.abs(f_ghz - fc) > mask_halfwidth_GHz
    if np.sum(mask) < max(20, deg + 5):
        mask[:] = True
    coeffs = np.polyfit(f_ghz[mask], y[mask], deg=deg)
    baseline = np.polyval(coeffs, f_ghz)
    return baseline, coeffs, mask

def pair_dips_by_symmetry(centers_ghz, centers_err_ghz, contrasts, D_guess_GHz=2.87):
    centers_ghz = np.asarray(centers_ghz, dtype=float)
    centers_err_ghz = np.asarray(centers_err_ghz, dtype=float)
    contrasts = np.asarray(contrasts, dtype=float)
    left_idx = np.where(centers_ghz < D_guess_GHz)[0]
    right_idx = np.where(centers_ghz > D_guess_GHz)[0]
    if len(left_idx) != 4 or len(right_idx) != 4:
        raise RuntimeError(f"Expected 4 left and 4 right fitted dips, got {len(left_idx)} and {len(right_idx)}.")
    left_idx = left_idx[np.argsort(centers_ghz[left_idx])]
    right_idx = right_idx[np.argsort(centers_ghz[right_idx])]
    best_cost = np.inf
    best_perm = None
    best_D_eff = None
    for perm in permutations(right_idx):
        pair_means = 0.5 * (centers_ghz[left_idx] + centers_ghz[list(perm)])
        D_eff = np.mean(pair_means)
        symmetry_cost = np.sum(((pair_means - D_eff) * 1000.0) ** 2)
        pair_span_cost = np.sum(((centers_ghz[list(perm)] - centers_ghz[left_idx]) * 1000.0) ** 2)
        cost = symmetry_cost + 1e-6 * pair_span_cost
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
            best_D_eff = D_eff
    import pandas as pd
    rows = []
    for pair_id, (li, ri) in enumerate(zip(left_idx, best_perm), start=1):
        f_left = centers_ghz[li]
        f_right = centers_ghz[ri]
        f_left_err = centers_err_ghz[li]
        f_right_err = centers_err_ghz[ri]
        pair_center = 0.5 * (f_left + f_right)
        splitting = f_right - f_left
        zeeman_half = 0.5 * splitting
        symmetry_error = pair_center - best_D_eff
        rows.append({
            "pair_id": pair_id,
            "left_dip_id_sorted": int(li + 1),
            "right_dip_id_sorted": int(ri + 1),
            "left_center_x1_GHz": f_left,
            "right_center_x1_GHz": f_right,
            "left_center_x1_err_MHz": f_left_err * 1000.0,
            "right_center_x1_err_MHz": f_right_err * 1000.0,
            "pair_center_GHz": pair_center,
            "pair_center_minus_D_eff_MHz": symmetry_error * 1000.0,
            "splitting_GHz": splitting,
            "splitting_MHz": splitting * 1000.0,
            "zeeman_half_MHz": zeeman_half * 1000.0,
            "left_contrast": contrasts[li],
            "right_contrast": contrasts[ri],
            "D_eff_GHz": best_D_eff,
        })
    pair_df = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
    return pair_df, float(best_D_eff)

def fit_global_odmr(
    df,
    D_GHz=2.87,
    fmin=2.79,
    fmax=2.96,
    baseline_deg=2,
    baseline_mask_halfwidth_GHz=0.006,
    prominence=0.01,
    sigma=None,
):
    f_hz = df.iloc[:, 0].to_numpy(dtype=float)
    y_raw = df.iloc[:, 1].to_numpy(dtype=float)
    validate_frequency_units(f_hz)
    f_ghz = f_hz / 1e9
    roi = (f_ghz >= fmin) & (f_ghz <= fmax)
    f_hz = f_hz[roi]
    f_ghz = f_ghz[roi]
    y_raw = y_raw[roi]
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)[roi]
    y_norm = normalize_signal(y_raw)
    dip_centers_init, y_smooth, dip_prominences = detect_8_dip_center_guesses(
        f_ghz, y_norm, D_GHz=D_GHz, fmin=fmin, fmax=fmax, prominence=prominence
    )
    baseline_raw, baseline_coeffs, baseline_mask = fit_polynomial_baseline(
        f_ghz, y_raw, dip_centers_init,
        mask_halfwidth_GHz=baseline_mask_halfwidth_GHz,
        deg=baseline_deg,
    )
    y_corr = y_raw / (baseline_raw + 1e-15)
    median_top = np.median(np.sort(y_corr)[-max(10, len(y_corr) // 10):])
    y_fit = y_corr / (median_top + 1e-15)
    if sigma is not None:
        sigma = sigma / (baseline_raw + 1e-15)
        sigma = sigma / (median_top + 1e-15)
    n_dips = 8
    centers0 = dip_centers_init.copy()
    contrasts0 = []
    for fc in centers0:
        local = np.abs(f_ghz - fc) < 0.004
        if np.any(local):
            depth = 1.0 - np.min(y_fit[local])
            contrasts0.append(max(0.003, depth))
        else:
            contrasts0.append(0.02)
    contrasts0 = np.array(contrasts0)
    delta0 = 0.00216
    widthL0 = 0.0008
    w0 = [1.0, 1.0, 1.0]
    p0 = np.concatenate([centers0, contrasts0, [delta0, widthL0], w0])
    lower_centers = centers0 - 0.004
    upper_centers = centers0 + 0.004
    lower_contrasts = np.full(n_dips, 0.0)
    upper_contrasts = np.full(n_dips, 0.7)
    lower = np.concatenate([
        lower_centers,
        lower_contrasts,
        [0.0015, 0.0001],
        [0.0, 0.0, 0.0],
    ])
    upper = np.concatenate([
        upper_centers,
        upper_contrasts,
        [0.0030, 0.0040],
        [5.0, 5.0, 5.0],
    ])
    model = make_global_odmr_model()
    popt, pcov = curve_fit(
        model,
        f_ghz,
        y_fit,
        p0=p0,
        bounds=(lower, upper),
        maxfev=100000,
        sigma=sigma,
        absolute_sigma=False,
    )
    y_model = model(f_ghz, *popt)
    residuals = y_fit - y_model
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    if np.ndim(pcov) == 2 and pcov.shape[0] == len(popt):
        diag = np.diag(pcov)
        perr = np.sqrt(np.clip(diag, 0, np.inf))
    else:
        perr = np.full(len(popt), np.nan)
    centers = np.array(popt[:n_dips])
    centers_err = np.array(perr[:n_dips])
    contrasts = np.array(popt[n_dips:2 * n_dips])
    contrasts_err = np.array(perr[n_dips:2 * n_dips])
    delta_hf = float(popt[2 * n_dips])
    delta_hf_err = float(perr[2 * n_dips])
    width_L = float(popt[2 * n_dips + 1])
    width_L_err = float(perr[2 * n_dips + 1])
    w_m1, w_0, w_p1 = popt[2 * n_dips + 2:2 * n_dips + 5]
    wsum = w_m1 + w_0 + w_p1 + 1e-15
    w_m1, w_0, w_p1 = w_m1 / wsum, w_0 / wsum, w_p1 / wsum
    order = np.argsort(centers)
    centers = centers[order]
    centers_err = centers_err[order]
    contrasts = contrasts[order]
    contrasts_err = contrasts_err[order]
    rows = []
    for i, (fc, fc_err, C, C_err) in enumerate(
        zip(centers, centers_err, contrasts, contrasts_err), start=1
    ):
        x0 = fc - delta_hf
        x1 = fc
        x2 = fc + delta_hf
        rows.append({
            "dip_id": i,
            "dip_side": "left" if fc < D_GHz else "right",
            "center_GHz": fc,
            "center_err_MHz": fc_err * 1000.0,
            "x0_GHz": x0,
            "x1_GHz": x1,
            "x2_GHz": x2,
            "x0_MHz": x0 * 1000.0,
            "x1_MHz": x1 * 1000.0,
            "x2_MHz": x2 * 1000.0,
            "contrast": C,
            "contrast_err": C_err,
            "hf01_MHz": (x1 - x0) * 1000.0,
            "hf12_MHz": (x2 - x1) * 1000.0,
            "delta_hf_MHz": delta_hf * 1000.0,
            "delta_hf_err_MHz": delta_hf_err * 1000.0,
            "lorentz_fwhm_MHz": width_L * 1000.0,
            "lorentz_fwhm_err_MHz": width_L_err * 1000.0,
            "w_m1": w_m1,
            "w_0": w_0,
            "w_p1": w_p1,
            "global_fit_rms": rms,
        })
    results_df = pd.DataFrame(rows).sort_values("center_GHz").reset_index(drop=True)
    pairs_df, D_eff = pair_dips_by_symmetry(
        centers_ghz=centers,
        centers_err_ghz=centers_err,
        contrasts=contrasts,
        D_guess_GHz=D_GHz,
    )
    payload = {
        "f_hz": f_hz,
        "f_ghz": f_ghz,
        "y_raw": y_raw,
        "y_norm": y_norm,
        "y_smooth": y_smooth,
        "baseline_raw": baseline_raw,
        "baseline_coeffs": baseline_coeffs,
        "baseline_mask": baseline_mask,
        "y_fit": y_fit,
        "y_model": y_model,
        "residuals": residuals,
        "dip_centers_init": dip_centers_init,
        "dip_prominences": dip_prominences,
        "centers_fit": centers,
        "centers_err": centers_err,
        "contrasts_fit": contrasts,
        "contrasts_err": contrasts_err,
        "delta_hf_GHz": delta_hf,
        "delta_hf_err_GHz": delta_hf_err,
        "width_L_GHz": width_L,
        "width_L_err_GHz": width_L_err,
        "weights_fit": (w_m1, w_0, w_p1),
        "D_nominal_GHz": D_GHz,
        "D_eff_GHz": D_eff,
        "global_rms": rms,
        "popt": popt,
        "pcov": pcov,
        "perr": perr,
    }
    return results_df, pairs_df, payload
