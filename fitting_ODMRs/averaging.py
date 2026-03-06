import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Configuration
base_dir = '/home/vita/projects/NV_sensing/ODMR magnetic field reconstruction'
output_dir = '/home/vita/projects/NV_sensing/fitting_ODMRs'
variations = ['Variation Along X', 'Variation Along Y', 'Variation Along Z']
center_freq = 2.87e9  # Hz

os.makedirs(output_dir, exist_ok=True)

# Averaging function
def average_odmr_files(subdir_path, output_csv):
    csv_files = sorted([
        f for f in os.listdir(subdir_path)
        if f.startswith('ODMR_avg_') and f.endswith('.csv')
    ])
    
    if len(csv_files) == 0:
        return None
    
    curves = []
    freq = None
    
    for csv_file in csv_files:
        filepath = os.path.join(subdir_path, csv_file)
        df = pd.read_csv(filepath, sep='\t', header=0)
        
        if freq is None:
            freq = df.iloc[:, 0].values
        
        signal = df.iloc[:, 1].values
        curves.append(signal)
    
    curves = np.array(curves)
    signal_mean = np.mean(curves, axis=0)
    signal_std = np.std(curves, axis=0)
    
    # Noise analysis
    noise_single = np.nanmean(signal_std)
    noise_averaged = noise_single / np.sqrt(len(csv_files))
    snr_improvement = len(csv_files) ** 0.5
    
    # Save averaged
    df_avg = pd.DataFrame({
        'Frequency_Hz': freq,
        'Intensity_Averaged': signal_mean,
        'Intensity_StdDev': signal_std
    })
    df_avg.to_csv(output_csv, sep='\t', index=False)
    
    return {
        'num_files': len(csv_files),
        'noise_single': noise_single,
        'noise_averaged': noise_averaged,
        'snr_improvement': snr_improvement,
        'freq': freq,
        'intensity': signal_mean
    }

# Dip detection and pairing
def identify_dip_pairs(freq, intensity):
    # Normalize
    intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    
    # Find dips
    inverted = -intensity_norm + np.max(intensity_norm)
    peaks, _ = find_peaks(inverted, height=0.02, distance=20, prominence=0.005)
    
    dip_freqs = freq[peaks]
    
    # Separate left and right
    left_dips = dip_freqs[dip_freqs < center_freq]
    right_dips = dip_freqs[dip_freqs > center_freq]
    
    # Sort by distance to center
    left_sorted = left_dips[np.argsort(np.abs(left_dips - center_freq))]
    right_sorted = right_dips[np.argsort(np.abs(right_dips - center_freq))]
    
    pairs = []
    for i in range(min(4, len(left_sorted), len(right_sorted))):
        pairs.append((left_sorted[i], right_sorted[i]))
    
    return pairs

# Main
all_stats = []

for variation in variations:
    variation_path = os.path.join(base_dir, variation)
    var_short = variation.replace('Variation Along ', '')
    
    subdirs = sorted([
        d for d in os.listdir(variation_path)
        if os.path.isdir(os.path.join(variation_path, d)) and 'checkpoint' not in d.lower()
    ])
    
    for subdir in subdirs:
        subdir_path = os.path.join(variation_path, subdir)
        var_folder = os.path.join(output_dir, f'{var_short.lower()}_averaged')
        os.makedirs(var_folder, exist_ok=True)
        output_csv = os.path.join(var_folder, f'{var_short}_{subdir}_averaged.csv')
        
        stats = average_odmr_files(subdir_path, output_csv)
        if stats:
            pairs = identify_dip_pairs(stats['freq'], stats['intensity'])
            
            stat_row = {
                'Variation': var_short,
                'Position': subdir,
                'Num_Files': stats['num_files'],
                'Noise_Single': stats['noise_single'],
                'Noise_Averaged': stats['noise_averaged'],
                'SNR_Improvement': stats['snr_improvement'],
                'Num_Pairs': len(pairs)
            }
            all_stats.append(stat_row)
            
            print(f"✓ {variation}/{subdir}: {stats['num_files']} files → {len(pairs)} pairs")

# Save stats
stats_df = pd.DataFrame(all_stats)
stats_csv = os.path.join(output_dir, 'averaging_stats.csv')
stats_df.to_csv(stats_csv, index=False)

# Plot stats
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# SNR Improvement
for var in ['X', 'Y', 'Z']:
    data = stats_df[stats_df['Variation'] == var]
    ax1.plot(range(len(data)), data['SNR_Improvement'], 'o-', label=var)
ax1.set_xlabel('Position Index')
ax1.set_ylabel('SNR Improvement Factor')
ax1.set_title('Noise Reduction')
ax1.legend()
ax1.grid()

# Mean Error
for var in ['X', 'Y', 'Z']:
    data = stats_df[stats_df['Variation'] == var]
    ax2.plot(range(len(data)), data['Noise_Averaged'], 's-', label=var)
ax2.set_xlabel('Position Index')
ax2.set_ylabel('Mean Error (Noise_Averaged)')
ax2.set_title('Averaged Noise Level')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'averaging_stats.png'))
plt.close()

print(f"Stats saved to {stats_csv} and averaging_stats.png")