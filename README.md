# NV Sensing Data and Analysis Workspace

This repository contains code and data for analyzing NV-center diamond ODMR and pulse experiments. Below is an overview of the data structure, file locations, and how to use the analysis tools.

## Data Structure

### 1. Single Averaged ODMR File
- **Location:** `data/ODMR_avg_99-6dBm_0mW.csv`
- **Description:** A single, fully averaged continuous-wave ODMR spectrum. Used for demonstration and single-spectrum fitting/visualization.

### 2. Batch Averaged ODMR Data (Displacement Series)
- **Location:** `avareged_data/outputs/Variation Along X|Y|Z/`
- **Files:**
  - `*_averaged.csv`: Averaged ODMR spectra for different displacements along X, Y, or Z.
  - `averaging_statistics.csv`: Statistics for each displacement.
  - `individual_plots/`, `overlay_plot.png`: Visualizations for each scan.
- **Processed Results:**
  - `fitting_odmr/batch_fit_outputs_lorentzian/Variation Along X|Y|Z/`: Contains fit results, plots, and summary tables for each displacement.

### 3. Pulse Data (Time Traces)
- **Location:** `data/Avg_npy_pulse/`
- **Files:** `Pulse_avg_d1.npy`, `Pulse_avg_d2.npy`, `Pulse_avg_d3.npy`, `Pulse_avg_d4.npy`
- **Description:** Each file contains a 2D array: first column is time (s), remaining columns are pixel voltages for a given NV orientation.

### 4. Magnetic Field Simulation and NV Frame Analysis
- **Location:** `B_field_simulation/`, `nv_to_lab_frame/`
- **Description:** Notebooks and scripts for simulating magnetic fields and transforming between NV and lab frames.

## Analysis Code

### Fitting Functions
- **Location:** `fitting_odmr/fitlorenzo_functions.py`
- **Description:** All fitting and preprocessing functions for ODMR analysis are here. Import this module in your notebooks/scripts:
  ```python
  from fitting_odmr.fitlorenzo_functions import fit_global_odmr, ...
  ```

### Notebooks
- **Single ODMR Fit:** `fitting_odmr/odmr_fit_single_file.ipynb`
  - Visualizes and fits the single averaged ODMR file. No results are saved; all outputs are displayed.
- **Batch ODMR Fit:** (see batch processing code in `fitlorenzo_literature_ready_corrected.ipynb` or convert to a notebook using `fitlorenzo_functions.py`)
- **Pulse Analysis:** `NV_pulse_analysis.ipynb`
  - Loads and visualizes pulse data from `.npy` files.

## How to Use
1. **Install dependencies:** Activate your Python environment and install required packages (see `.venv` or requirements in code cells).
2. **Run Notebooks:**
   - For single ODMR: open and run `fitting_odmr/odmr_fit_single_file.ipynb`.
   - For batch ODMR: adapt or run batch code using `fitlorenzo_functions.py`.
   - For pulse data: open and run `NV_pulse_analysis.ipynb`.
3. **Import Fitting Functions:**
   - Use `from fitting_odmr.fitlorenzo_functions import ...` in your scripts/notebooks for all fitting and preprocessing routines.

## Folder Overview
- `data/` — Raw and averaged data files
- `avareged_data/outputs/` — Batch-averaged ODMR data by displacement
- `fitting_odmr/` — Fitting code, results, and notebooks
- `B_field_simulation/`, `nv_to_lab_frame/` — Simulation and frame transformation notebooks

---
For further details, see code comments and notebook markdown cells. All analysis is modular and can be adapted for new data or additional scans.

