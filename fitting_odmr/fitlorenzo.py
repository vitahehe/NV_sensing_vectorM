import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit


# ============================================================
# GLOBAL ODMR FIT FOR 8 HYPERFINE-SPLIT RESONANCES
#
# Literature-motivated design choices used here:
#
# (1) Global spectrum fit:
#     Fit the full ODMR spectrum at once as a sum of 8 hyperfine-split resonances,
#     rather than fitting each dip independently.
#     Motivation: recent NV ODMR work explicitly extracts the 8 resonant frequencies
#     by fitting 8 hyperfine-split Lorentzians to the full spectrum.
#
# (2) Baseline correction before resonance fitting:
#     Ensemble ODMR often sits on a slowly varying optical background.
#     So we first estimate a smooth polynomial baseline and divide the data by it.
#
# (3) Symmetric 14N hyperfine triplet:
#     For each ODMR dip i, we fit one center f_c,i and one shared hyperfine splitting Δ_hf,
#     imposing:
#           f_i,-1 = f_c,i - Δ_hf
#           f_i, 0 = f_c,i
#           f_i,+1 = f_c,i + Δ_hf
#     This is more physical than fitting x0, x1, x2 independently.
# ============================================================


# ============================================================
# 1. Line-shape model
# ============================================================

def lorentzian(x, x0, gamma):
    """
    Unit-height Lorentzian line:
        L(x) = 1 / (1 + 4 ((x - x0)/gamma)^2)

    gamma is the FWHM in GHz.
    """
    return 1.0 / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)


def global_odmr_model(x, *params):
    """
    Global model on baseline-corrected ODMR data.

    Model:
        y(x) = 1 - sum_{i=1}^8 C_i * T_i(x)

    where each triplet T_i is
        T_i(x) = w_{-1} L(x; f_c,i - Δ_hf, gamma)
               + w_0    L(x; f_c,i,        gamma)
               + w_{+1} L(x; f_c,i + Δ_hf, gamma)

    Parameters:
    ----------
    params = [f_c1, ..., f_c8, C1, ..., C8, delta_hf, gamma, w_m1, w_0, w_p1]

    Notes:
    ------
    - The baseline-corrected signal is modeled around 1.
    - One shared linewidth gamma across the full spectrum.
    - One shared hyperfine splitting delta_hf across the full spectrum.
    - Shared triplet weights across the full spectrum.
      These absorb unequal amplitudes between the three hyperfine lines.
    """
    n_dips = 8
    centers = np.array(params[:n_dips])
    contrasts = np.array(params[n_dips:2 * n_dips])
    delta_hf = params[2 * n_dips]
    gamma = params[2 * n_dips + 1]
    w_m1 = params[2 * n_dips + 2]
    w_0  = params[2 * n_dips + 3]
    w_p1 = params[2 * n_dips + 4]

    # Normalize weights internally so their scale is absorbed into contrast C_i
    wsum = w_m1 + w_0 + w_p1 + 1e-15
    w_m1 /= wsum
    w_0  /= wsum
    w_p1 /= wsum

    y = np.ones_like(x, dtype=float)

    for fc, C in zip(centers, contrasts):
        triplet = (
            w_m1 * lorentzian(x, fc - delta_hf, gamma)
            + w_0 * lorentzian(x, fc, gamma)
            + w_p1 * lorentzian(x, fc + delta_hf, gamma)
        )
        y -= C * triplet

    return y


# ============================================================
# 2. Basic preprocessing
# ============================================================

def normalize_signal(y):
    """
    Normalize raw signal to [0, 1].
    Used only for easier detection and plotting.
    """
    y = np.asarray(y, dtype=float)
    return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-15)


def smooth_signal(y, window=9, poly=3):
    """
    Mild Savitzky-Golay smoothing for dip detection only.
    """
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=poly)


# ============================================================
# 3. Detect 8 broad dip centers
# ============================================================

def detect_8_broad_dips(f_ghz, y_norm, D_GHz=2.87, fmin=2.79, fmax=2.96):
    """
    Detect 4 broad dips left of D and 4 broad dips right of D.

    Important:
    This step only finds the approximate broad ODMR dip centers.
    It does NOT try to identify the 3 hyperfine sub-lines yet.
    """
    y_smooth = smooth_signal(y_norm, window=9, poly=3)
    inv = 1.0 - y_smooth  # dips -> peaks

    left_mask = (f_ghz < D_GHz) & (f_ghz >= fmin)
    right_mask = (f_ghz > D_GHz) & (f_ghz <= fmax)

    def detect_side(mask):
        f_side = f_ghz[mask]
        inv_side = inv[mask]

        peaks, props = find_peaks(inv_side, prominence=0.01, distance=8)

        if len(peaks) == 0:
            return np.array([])

        prominences = props["prominences"]
        strongest = np.argsort(prominences)[::-1][:4]
        peak_freqs = f_side[peaks[strongest]]
        return np.sort(peak_freqs)

    left_centers = detect_side(left_mask)
    right_centers = detect_side(right_mask)

    if len(left_centers) != 4 or len(right_centers) != 4:
        raise RuntimeError(
            f"Expected 4 left and 4 right dips, got {len(left_centers)} left and {len(right_centers)} right."
        )

    dip_centers = np.sort(np.concatenate([left_centers, right_centers]))
    return dip_centers, y_smooth


# ============================================================
# 4. Baseline estimation from preprocessed data
# ============================================================

def fit_polynomial_baseline(f_ghz, y, dip_centers, mask_halfwidth_GHz=0.006, deg=2):
    """
    Estimate a slowly varying polynomial baseline using points away from ODMR dips.

    Logic:
    ------
    We mask out regions around the 8 detected ODMR dips, then fit a low-order polynomial
    to the remaining points.

    This gives a smooth optical background / baseline.
    """
    mask = np.ones_like(f_ghz, dtype=bool)

    for fc in dip_centers:
        mask &= np.abs(f_ghz - fc) > mask_halfwidth_GHz

    # fallback if too much was masked
    if np.sum(mask) < max(20, deg + 5):
        mask[:] = True

    coeffs = np.polyfit(f_ghz[mask], y[mask], deg=deg)
    baseline = np.polyval(coeffs, f_ghz)

    return baseline, coeffs, mask


# ============================================================
# 5. Global fitting workflow
# ============================================================

def fit_global_odmr(df, D_GHz=2.87, fmin=2.79, fmax=2.96):
    """
    Full global ODMR fitting workflow:

    1. Read data
    2. Restrict to the ODMR region of interest
    3. Detect the 8 broad dip centers
    4. Fit and remove a slowly varying polynomial baseline
    5. Fit the baseline-corrected spectrum globally as 8 symmetric 14N triplets
    """
    # ---------------------------
    # Read and crop data
    # ---------------------------
    f_hz = df.iloc[:, 0].to_numpy(dtype=float)
    y_raw = df.iloc[:, 1].to_numpy(dtype=float)
    f_ghz = f_hz / 1e9

    roi = (f_ghz >= fmin) & (f_ghz <= fmax)
    f_ghz = f_ghz[roi]
    y_raw = y_raw[roi]

    # ---------------------------
    # Broad dip detection on normalized trace
    # ---------------------------
    y_norm = normalize_signal(y_raw)
    dip_centers, y_smooth = detect_8_broad_dips(f_ghz, y_norm, D_GHz=D_GHz, fmin=fmin, fmax=fmax)

    # ---------------------------
    # Polynomial baseline correction
    # ---------------------------
    baseline_raw, baseline_coeffs, baseline_mask = fit_polynomial_baseline(
        f_ghz, y_raw, dip_centers, mask_halfwidth_GHz=0.006, deg=2
    )

    # divide by baseline so corrected trace lives near 1 away from ODMR dips
    y_corr = y_raw / (baseline_raw + 1e-15)

    # mild renormalization only for numerical convenience
    # keep the "near 1 with dips downward" structure
    median_top = np.median(np.sort(y_corr)[-max(10, len(y_corr)//10):])
    y_fit = y_corr / (median_top + 1e-15)

    # ---------------------------
    # Initial guess for global fit
    # ---------------------------
    n_dips = 8
    centers0 = dip_centers.copy()

    # local depth estimate per broad dip
    contrasts0 = []
    for fc in centers0:
        local = np.abs(f_ghz - fc) < 0.004
        if np.any(local):
            depth = 1.0 - np.min(y_fit[local])
            contrasts0.append(max(0.005, depth))
        else:
            contrasts0.append(0.02)
    contrasts0 = np.array(contrasts0)

    delta0 = 0.00216   # 14N hyperfine adjacent spacing in GHz
    gamma0 = 0.0010    # common linewidth guess in GHz
    w0 = [1.0, 1.0, 1.0]

    p0 = np.concatenate([centers0, contrasts0, [delta0, gamma0], w0])

    # ---------------------------
    # Bounds
    # ---------------------------
    lower_centers = centers0 - 0.004
    upper_centers = centers0 + 0.004

    lower_contrasts = np.full(n_dips, 0.0)
    upper_contrasts = np.full(n_dips, 0.5)

    lower = np.concatenate([
        lower_centers,
        lower_contrasts,
        [0.0015, 0.0002],   # delta_hf, gamma
        [0.0, 0.0, 0.0]     # weights
    ])

    upper = np.concatenate([
        upper_centers,
        upper_contrasts,
        [0.0030, 0.0040],
        [5.0, 5.0, 5.0]
    ])

    # ---------------------------
    # Global fit
    # ---------------------------
    popt, pcov = curve_fit(
        global_odmr_model,
        f_ghz,
        y_fit,
        p0=p0,
        bounds=(lower, upper),
        maxfev=50000
    )

    y_model = global_odmr_model(f_ghz, *popt)
    residuals = y_fit - y_model
    rms = float(np.sqrt(np.mean(residuals**2)))

    # ---------------------------
    # Unpack fitted parameters
    # ---------------------------
    centers = np.array(popt[:n_dips])
    contrasts = np.array(popt[n_dips:2*n_dips])
    delta_hf = float(popt[2*n_dips])
    gamma = float(popt[2*n_dips + 1])
    w_m1, w_0, w_p1 = popt[2*n_dips + 2:2*n_dips + 5]

    # normalize fitted weights for reporting
    wsum = w_m1 + w_0 + w_p1 + 1e-15
    w_m1, w_0, w_p1 = w_m1/wsum, w_0/wsum, w_p1/wsum

    order = np.argsort(centers)
    centers = centers[order]
    contrasts = contrasts[order]

    # ---------------------------
    # Build results table
    # ---------------------------
    rows = []
    for i, (fc, C) in enumerate(zip(centers, contrasts), start=1):
        x0 = fc - delta_hf
        x1 = fc
        x2 = fc + delta_hf

        rows.append({
            "dip_id": i,
            "dip_side": "left" if fc < D_GHz else "right",
            "center_GHz": fc,
            "x0_GHz": x0,
            "x1_GHz": x1,
            "x2_GHz": x2,
            "contrast": C,
            "hf01_MHz": (x1 - x0) * 1000,
            "hf12_MHz": (x2 - x1) * 1000,
            "gamma_MHz": gamma * 1000,
            "w_m1": w_m1,
            "w_0": w_0,
            "w_p1": w_p1,
            "global_fit_rms": rms,
        })

    results_df = pd.DataFrame(rows).sort_values("center_GHz").reset_index(drop=True)

    payload = {
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
        "dip_centers_init": dip_centers,
        "centers_fit": centers,
        "contrasts_fit": contrasts,
        "delta_hf_GHz": delta_hf,
        "gamma_GHz": gamma,
        "weights_fit": (w_m1, w_0, w_p1),
        "D_GHz": D_GHz,
        "global_rms": rms,
        "popt": popt,
        "pcov": pcov,
    }

    return results_df, payload


# ============================================================
# 6. Example usage
# ============================================================

df = pd.read_csv('../avareged_data/outputs/Variation Along X/0cm-0.0mm_averaged.csv', sep='\t')

results_df, payload = fit_global_odmr(df)

print("Global ODMR fit results:")
print(results_df.to_string(index=False))

# symmetry pairs summary
left = results_df[results_df.dip_side == 'left'].sort_values('center_GHz')
right = results_df[results_df.dip_side == 'right'].sort_values('center_GHz')

pairs = []
for i in range(min(len(left), len(right))):
    l = left.iloc[i]
    r = right.iloc[i]
    pairs.append({
        'pair_id': i + 1,
        'left_center_GHz': l['center_GHz'],
        'right_center_GHz': r['center_GHz'],
        'distance_GHz': r['center_GHz'] - l['center_GHz'],
        'avg_dist_from_D_GHz': ((2.87 - l['center_GHz']) + (r['center_GHz'] - 2.87)) / 2
    })

pair_df = pd.DataFrame(pairs)
print("\nSymmetry pairs summary:")
print(pair_df.to_string(index=False))

print("\nShared fitted parameters:")
print(f"delta_hf = {payload['delta_hf_GHz']*1000:.3f} MHz")
print(f"gamma    = {payload['gamma_GHz']*1000:.3f} MHz")
print(f"weights  = {payload['weights_fit']}")
print(f"global RMS = {payload['global_rms']:.6f}")


# ============================================================
# 7. Plotting
# ============================================================

# (a) raw data + estimated baseline
plt.figure(figsize=(12, 5))
plt.plot(payload["f_ghz"], payload["y_raw"], color="black", lw=1, label="raw data")
plt.plot(payload["f_ghz"], payload["baseline_raw"], color="tab:blue", lw=2, label="polynomial baseline")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Raw ODMR signal")
plt.title("Raw ODMR and estimated baseline")
plt.legend()
plt.tight_layout()
plt.savefig("odmr_baseline.png", dpi=300)
print("Saved: odmr_baseline.png")

# (b) baseline-corrected data + global fit
plt.figure(figsize=(12, 6))
plt.plot(payload["f_ghz"], payload["y_fit"], color="black", lw=1.2, label="baseline-corrected data")
plt.plot(payload["f_ghz"], payload["y_model"], color="tab:red", lw=2, label="global 8-triplet fit")
plt.axvline(payload["D_GHz"], color="red", linestyle=":", alpha=0.7, label="D = 2.87 GHz")

for fc in payload["centers_fit"]:
    plt.axvline(fc, color="gray", linestyle="--", alpha=0.25)

plt.xlim(2.79, 2.96)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Baseline-corrected ODMR")
plt.title("Global ODMR fit: 8 symmetric hyperfine triplets")
plt.legend()
plt.tight_layout()
plt.savefig("odmr_global_fit.png", dpi=300)
print("Saved: odmr_global_fit.png")

# (c) residuals
plt.figure(figsize=(12, 3.5))
plt.plot(payload["f_ghz"], payload["residuals"], color="tab:green", lw=1)
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.xlim(2.79, 2.96)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual")
plt.title("Global fit residuals")
plt.tight_layout()
plt.savefig("odmr_global_residuals.png", dpi=300)
print("Saved: odmr_global_residuals.png")