# NV Sensing Data and Analysis Workspace

This repository contains notebooks corresponding to The High-resolution measurement
of the magnetic field vector components project- The 5 sections from the report can be found in the following parts:

## In this repository

1. ** Reconstruction of the field-response matrix A- A_MATRIX folder**
   - `A_MATRIX/nv_hamiltonian.py` computes NV transition frequencies and numerical derivatives to build the sensitivity matrix `A`.
    - `A_MATRIX/fit_functions.py` fitting help functions.
   - `A_MATRIX/get_A_matrix.ipynb` shows how to calculate `A`. Input NV axis experimental asssignment

2. **Pulse-data magnetic field reconstruction  + simulations and wire geometry optimization**
   - `pulse_data_reconstruction_simulation' folder consist of both pulse data reconstruction and simulations+ geometry wire optimization

3. **Reconstruct current density**
   - `current_denity.ipynb` performs current density reconstruction from measured poulse data.
   - The reconstruction workflowrelies on the B-field component(s) measured by the NV sensor.

4. **Analyze experimental sensitivity**
   - `sensitivity.ipynb` measurement noice and estimation of sensitivity.



### 1. Install dependencies
Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2.The use of each file is contained in each notebooks separatly

## Recommended installation

```bash
python -m pip install -r requirements.txt
```

