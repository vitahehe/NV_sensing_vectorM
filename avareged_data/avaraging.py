import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DATA_PATH = '/home/vita/projects/NV_sensing/ODMR magnetic field reconstruction'
OUTPUT_BASE_PATH = '/home/vita/projects/NV_sensing/avareged_data/outputs'
VARIATIONS = ['Variation Along X', 'Variation Along Y', 'Variation Along Z']


def average_csv_files(file_paths):
    """
    Average multiple CSV files by summing all values and dividing by count.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to average
        
    Returns:
    --------
    pd.DataFrame
        Averaged dataframe with same columns as input
    dict
        Statistics about the averaging operation
    """
    if not file_paths:
        raise ValueError("No files provided for averaging")
    
    # Read all files (tab-delimited)
    dataframes = [pd.read_csv(fp, sep='\t') for fp in file_paths]
    frequency = dataframes[0].iloc[:, 0]

    data_columns = dataframes[0].columns[1:]
    
    # Initialize sums
    data_sum = None
    
    for df in dataframes:
        if data_sum is None:
            data_sum = df.iloc[:, 1:].values.astype(float)
        else:
            data_sum += df.iloc[:, 1:].values.astype(float)
    
    # Calculate average
    num_files = len(file_paths)
    data_avg = data_sum / num_files
    
    # Create result dataframe
    result_df = pd.DataFrame(data_avg, columns=data_columns)
    result_df.insert(0, frequency.name, frequency.values)
    
    # Calculate statistics
    stats = {
        'num_files_averaged': num_files,
        'file_list': file_paths,
        'num_rows': len(result_df),
        'num_columns': len(data_columns),
        'mean_value': np.mean(data_avg),
        'std_value': np.std(data_avg),
        'min_value': np.min(data_avg),
        'max_value': np.max(data_avg)
    }
    
    return result_df, stats


def process_displacement_folder(folder_path):
    """
    Process a displacement folder by averaging all ODMR CSV files.
    
    Returns:
    --------
    tuple: (averaged_df, stats_dict, file_count)
    """
    csv_files = sorted([f for f in os.listdir(folder_path) 
                       if f.endswith('.csv') and f.startswith('ODMR_avg')])
    
    if not csv_files:
        return None, None, 0
    
    file_paths = [os.path.join(folder_path, f) for f in csv_files]
    avg_df, stats = average_csv_files(file_paths)
    
    return avg_df, stats, len(csv_files)


def create_output_directories():
    """Create output directory structure"""
    for variation in VARIATIONS:
        var_output = os.path.join(OUTPUT_BASE_PATH, variation)
        os.makedirs(var_output, exist_ok=True)


def process_variation(variation_name):
    """
    Process all displacement folders in a variation directory.
    
    Returns:
    --------
    dict: {displacement_name: {'averaged_df': df, 'stats': dict}}
    """
    variation_path = os.path.join(BASE_DATA_PATH, variation_name)
    output_path = os.path.join(OUTPUT_BASE_PATH, variation_name)
    
    results = {}
    displacement_folders = sorted([d for d in os.listdir(variation_path) 
                                  if os.path.isdir(os.path.join(variation_path, d)) 
                                  and d != '.ipynb_checkpoints'])
    
    print(f"\n{'='*60}")
    print(f"Processing {variation_name}")
    print(f"{'='*60}")
    
    for displacement in displacement_folders:
        displacement_path = os.path.join(variation_path, displacement)
        
        avg_df, stats, file_count = process_displacement_folder(displacement_path)
        
        if avg_df is None:
            print(f"  ⊘ {displacement}: No CSV files found")
            continue
        
        # Save averaged CSV (use tab delimiter to match input format)
        output_csv_path = os.path.join(output_path, f'{displacement}_averaged.csv')
        avg_df.to_csv(output_csv_path, sep='\t', index=False)
        
        # Store results
        results[displacement] = {
            'averaged_df': avg_df,
            'stats': stats,
            'output_path': output_csv_path
        }
        
        print(f"  ✓ {displacement}: Averaged {file_count} files")
    
    return results


def create_statistics_table(variation_results, variation_name):
    """
    Create a statistics table for a variation.
    
    Returns:
    --------
    pd.DataFrame with statistics
    """
    stats_data = []
    
    for displacement, data in sorted(variation_results.items()):
        stats = data['stats']
        stats_data.append({
            'Displacement': displacement,
            'Files Averaged': stats['num_files_averaged'],
            'Data Points': stats['num_rows'],
            'Columns': stats['num_columns'],
            'Mean Value': f"{stats['mean_value']:.6f}",
            'Std Dev': f"{stats['std_value']:.6f}",
            'Min Value': f"{stats['min_value']:.6f}",
            'Max Value': f"{stats['max_value']:.6f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Save statistics table
    output_path = os.path.join(OUTPUT_BASE_PATH, variation_name)
    stats_csv_path = os.path.join(output_path, 'averaging_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    
    return stats_df


def create_overlay_plot(variation_results, variation_name):
    """
    Create an overlay plot showing all averaged curves for a variation.
    """
    if not variation_results:
        print(f"  ⊘ No data to plot for {variation_name}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(variation_results)))
    
    for idx, (displacement, data) in enumerate(sorted(variation_results.items())):
        avg_df = data['averaged_df']
        freq = avg_df.iloc[:, 0]
        
        # Plot each pixel column
        for col in avg_df.columns[1:]:
            ax.plot(freq / 1e9, avg_df[col], label=f'{displacement}', 
                   color=colors[idx], alpha=0.7, linewidth=1.5)
            break  # Only plot first pixel column to avoid clutter
    
    ax.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ODMR Signal', fontsize=12, fontweight='bold')
    ax.set_title(f'ODMR Averaged Curves - {variation_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(OUTPUT_BASE_PATH, variation_name)
    plot_path = os.path.join(output_path, 'overlay_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Overlay plot saved: {plot_path}")


def create_individual_plots(variation_results, variation_name):
    """
    Create individual plots for each displacement.
    """
    output_path = os.path.join(OUTPUT_BASE_PATH, variation_name)
    plots_dir = os.path.join(output_path, 'individual_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for displacement, data in sorted(variation_results.items()):
        avg_df = data['averaged_df']
        freq = avg_df.iloc[:, 0]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in avg_df.columns[1:]:
            ax.plot(freq / 1e9, avg_df[col], label=col, linewidth=2)
        
        ax.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
        ax.set_ylabel('ODMR Signal', fontsize=11, fontweight='bold')
        ax.set_title(f'ODMR Averaged Curve - {displacement}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        plot_file = os.path.join(plots_dir, f'{displacement}_odmr.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Individual plots saved: {plots_dir}")


def main():
    """Main processing pipeline"""
    print("\n" + "="*60)
    print("ODMR MAGNETIC FIELD DATA AVERAGING PIPELINE")
    print("="*60)
    
    # Create output directories
    create_output_directories()
    
    all_stats = {}
    
    # Process each variation
    for variation in VARIATIONS:
        variation_results = process_variation(variation)
        
        if not variation_results:
            print(f"  No data found for {variation}")
            continue
        
        # Create statistics table
        print(f"\n  Creating statistics table...")
        stats_df = create_statistics_table(variation_results, variation)
        all_stats[variation] = stats_df
        print(f"  Statistics table:")
        print(f"  {stats_df.to_string(index=False)}")
        
        # Create overlay plot
        print(f"\n  Creating overlay plot...")
        create_overlay_plot(variation_results, variation)
        
        # Create individual plots
        print(f"\n  Creating individual plots...")
        create_individual_plots(variation_results, variation)
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_BASE_PATH}")
    print(f"\nGenerated files:")
    print(f"  - Averaged CSV files for each displacement")
    print(f"  - Averaging statistics tables (CSV)")
    print(f"  - Overlay plots for each variation")
    print(f"  - Individual plots for each displacement")
    print(f"{'='*60}\n")
    
    return all_stats


if __name__ == '__main__':
    all_stats = main()

