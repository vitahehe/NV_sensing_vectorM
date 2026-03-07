# ODMR Magnetic Field Data Averaging Pipeline

## Overview

This pipeline processes ODMR (Optically Detected Magnetic Resonance) data from the magnetic field reconstruction folder and produces averaged datasets, statistics tables, and visualization plots.

## What the Script Does

### 1. **Data Averaging Function** (`average_csv_files()`)
   - Reads all ODMR CSV files from a displacement folder
   - Sums all data values across files
   - Divides by the number of files to calculate the mean
   - Returns both the averaged dataframe and detailed statistics

### 2. **File Processing**
   - Processes all three variation types:
     - Variation Along X (5 displacement positions)
     - Variation Along Y (5 displacement positions)
     - Variation Along Z (6 displacement positions)
   - For each displacement folder, averages all ODMR measurements
   - Maintains the original tab-delimited CSV format

### 3. **Output Files Generated**

#### Per Variation Directory (`outputs/Variation Along X|Y|Z/`)
   - **`{displacement}_averaged.csv`**: Tab-delimited CSV with averaged ODMR data
     - Frequency column (GHz)
     - Averaged pixel intensity values
   
   - **`averaging_statistics.csv`**: Summary statistics table
     - Number of files averaged
     - Data points per file
     - Mean value across all frequencies
     - Standard deviation
     - Min/Max values

   - **`overlay_plot.png`**: Visualization showing all averaged curves
     - All displacement positions overlaid on one plot
     - Shows variation behavior across displacements

   - **`individual_plots/`** folder:
     - Individual ODMR plot for each displacement
     - High-resolution plots suitable for inclusion in reports

## Key Statistics Tracked

For each displacement averaging operation:
- **Files Averaged**: Number of ODMR files combined
- **Data Points**: Number of frequency measurements per file
- **Columns**: Number of pixel regions analyzed
- **Mean Value**: Average ODMR signal intensity
- **Std Dev**: Standard deviation of signal
- **Min/Max Values**: Signal range

## Results Summary

### Variation Along X
- **Displacements**: 0cm-0.0mm to 0cm-0.4mm (5 positions)
- **Files processed**: 5-7 per displacement
- **Total averaged**: 27 files

### Variation Along Y  
- **Displacements**: 0cm-0.0mm to 0cm-0.4mm (5 positions)
- **Files processed**: 4-10 per displacement
- **Total averaged**: 34 files

### Variation Along Z
- **Displacements**: 6.8cm-0.0mm to 6.8cm-0.5mm (6 positions)
- **Files processed**: 6-10 per displacement
- **Total averaged**: 46 files



## How to Use

### Running the Pipeline
```bash
cd /home/vita/projects/NV_sensing
python avareged_data/avaraging.py
```

### Using the Results
1. **For data analysis**: Load the averaged CSV files for further processing
2. **For visualization**: View PNG plots in the individual_plots folders
3. **For statistics**: Reference the averaging_statistics.csv tables for quantitative measures
4. **For presentations**: Use overlay_plot.png to show variation trends

### Accessing Averaged Data Programmatically
```python
import pandas as pd

# Load averaged data
df = pd.read_csv('avareged_data/outputs/Variation Along X/0cm-0.0mm_averaged.csv', sep='\t')

# Load statistics
stats = pd.read_csv('avareged_data/outputs/Variation Along X/averaging_statistics.csv')
```

## Implementation Details

- **Language**: Python 3
- **Dependencies**: pandas, numpy, matplotlib
- **File Format**: Input/output CSVs are tab-delimited to match original data
- **Error Handling**: Skips folders with missing CSV files, logs all operations
- **Plot Generation**: 300 DPI for publication-quality images

## Averaging Method

Simple arithmetic mean:
$$\text{Average}[i] = \frac{1}{N} \sum_{j=1}^{N} \text{Data}[i,j]$$

Where:
- $i$ = frequency index
- $j$ = file index (1 to N)
- $N$ = total number of files in displacement folder

## Notes

- All frequency values are preserved from the first input file (assumed identical across all files in a displacement)
- Pixel intensity values are averaged independently
- The script validates that all files in a displacement have the same structure
- Processing is sequential but very fast for this dataset size (~100 files total)
