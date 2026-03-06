import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Configuration
output_dir = '/home/vita/projects/NV_sensing/fitting_ODMRs'
folders = ['x_averaged', 'y_averaged', 'z_averaged']
center_freq = 2.87e9  # Hz
hyperfine_split = 2.16e6  # Hz

# Lorentzian functions
def lorentzian(x, x0, gamma, A):
    return A * ((gamma/2)**2) / ((x - x0)**2 + (gamma/2)**2)

def triple_lorentzian(x, x0, x1, x2, gamma, A):
    return lorentzian(x, x0, gamma, A) + lorentzian(x, x1, gamma, A) + lorentzian(x, x2, gamma, A)

# Main
all_fits = {}

for folder in folders:
    folder_path = os.path.join(output_dir, folder)
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_averaged.csv')])
    
    fits_folder = os.path.join(output_dir, f'fits_{folder}')
    os.makedirs(fits_folder, exist_ok=True)
    
    for csv_file in csv_files:
        filepath = os.path.join(folder_path, csv_file)
        df = pd.read_csv(filepath, sep='\t')
        
        freq = df['Frequency_Hz'].values
        intensity = df['Intensity_Averaged'].values
        
        # Normalize
        intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        
        # Find dips
        inverted = -intensity_norm + np.max(intensity_norm)
        peaks, _ = find_peaks(inverted, height=0.02, distance=20, prominence=0.005)
        
        dip_freqs = freq[peaks]
        left_dips = dip_freqs[dip_freqs < center_freq]
        right_dips = dip_freqs[dip_freqs > center_freq]
        left_sorted = left_dips[np.argsort(np.abs(left_dips - center_freq))]
        right_sorted = right_dips[np.argsort(np.abs(right_dips - center_freq))]
        
        pairs = []
        for i in range(min(4, len(left_sorted), len(right_sorted))):
            pairs.append((left_sorted[i], right_sorted[i]))
        
        # Fit each dip
        fit_results = {}
        for pair_idx, (left_dip, right_dip) in enumerate(pairs):
            for side, dip_freq in [('left', left_dip), ('right', right_dip)]:
                window_mask = (freq >= dip_freq - 30e6) & (freq <= dip_freq + 30e6)
                freq_window = freq[window_mask]
                int_window = intensity_norm[window_mask]
                
                if len(freq_window) < 10:
                    continue
                
                x0_init = dip_freq - hyperfine_split
                x1_init = dip_freq
                x2_init = dip_freq + hyperfine_split
                gamma_init = 5e6
                A_init = 0.05
                p0 = [x0_init, x1_init, x2_init, gamma_init, A_init]
                
                try:
                    def model(x, x0, x1, x2, gamma, A):
                        return 1 - triple_lorentzian(x, x0, x1, x2, gamma, A)
                    
                    popt, _ = curve_fit(model, freq_window, int_window, p0=p0, maxfev=5000)
                    x0, x1, x2, gamma, A = popt
                    fit_results[f'Pair{pair_idx+1}_{side}'] = {
                        'x0_MHz': x0 / 1e6,
                        'x1_MHz': x1 / 1e6,
                        'x2_MHz': x2 / 1e6,
                        'gamma_MHz': gamma / 1e6,
                        'A': A
                    }
                except:
                    pass
        
        # Save fits
        fits_csv = os.path.join(fits_folder, f'fits_{csv_file}')
        fit_df = pd.DataFrame.from_dict(fit_results, orient='index')
        fit_df.to_csv(fits_csv)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(freq/1e6, intensity_norm, 'b-')
        plt.axvline(center_freq/1e6, color='k', linestyle='--')
        
        colors = ['red', 'green', 'orange', 'purple']
        for pair_idx, (left_dip, right_dip) in enumerate(pairs):
            color = colors[pair_idx % len(colors)]
            plt.plot(left_dip/1e6, intensity_norm[np.argmin(np.abs(freq - left_dip))], 'o', color=color)
            plt.plot(right_dip/1e6, intensity_norm[np.argmin(np.abs(freq - right_dip))], 's', color=color)
            
            for side, dip_freq in [('left', left_dip), ('right', right_dip)]:
                key = f'Pair{pair_idx+1}_{side}'
                if key in fit_results:
                    res = fit_results[key]
                    x0, x1, x2 = res['x0_MHz']*1e6, res['x1_MHz']*1e6, res['x2_MHz']*1e6
                    gamma, A = res['gamma_MHz']*1e6, res['A']
                    fit_curve = 1 - triple_lorentzian(freq_window, x0, x1, x2, gamma, A)
                    plt.plot(freq_window/1e6, fit_curve, '--', color=color)
        
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Normalized Intensity')
        plt.title(f'Fits: {csv_file}')
        plt.savefig(os.path.join(fits_folder, f'plot_{csv_file.replace(".csv", ".png")}'))
        plt.close()
        
        all_fits[csv_file] = fit_results

# Summary
summary = {}
for folder in folders:
    fits_folder = os.path.join(output_dir, f'fits_{folder}')
    summary[folder] = len([f for f in os.listdir(fits_folder) if f.startswith('fits_')])

print("Fitting complete. Summary:")
for k, v in summary.items():
    print(f"{k}: {v} fits")