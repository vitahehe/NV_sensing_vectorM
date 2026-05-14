import numpy as np


def spin1_matrices():
    """
    Spin-1 matrices in the {|+1>, |0>, |-1>} basis.
    Dimensionless operators.
    """
    Sx = (1 / np.sqrt(2)) * np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]],
        dtype=complex
    )

    Sy = (1 / np.sqrt(2)) * np.array(
        [[0, -1j, 0],
         [1j, 0, -1j],
         [0, 1j, 0]],
        dtype=complex
    )

    Sz = np.array(
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, -1]],
        dtype=complex
    )

    return Sx, Sy, Sz, Sz @ Sz


def make_local_frame(n_lab):
    """
    Construct a right-handed local NV frame.

    Input:
        n_lab: NV symmetry axis in the lab/diamond frame.

    Output:
        ex, ey, ez:
            local basis vectors expressed in the lab frame.
            ez is parallel to the NV symmetry axis.
    """
    ez = np.asarray(n_lab, dtype=float)
    ez = ez / np.linalg.norm(ez)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, ez)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    ex = np.cross(ref, ez)
    ex = ex / np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)

    return ex, ey, ez


def lab_to_local(B_lab_nT, local_frame):
    """
    Convert a magnetic field from lab coordinates into one NV-local frame.

    Units:
        B_lab_nT: nT
        output: nT
    """
    ex, ey, ez = local_frame
    B_lab_nT = np.asarray(B_lab_nT, dtype=float)

    return np.array([
        np.dot(B_lab_nT, ex),
        np.dot(B_lab_nT, ey),
        np.dot(B_lab_nT, ez),
    ], dtype=float)


def transition_frequencies_mhz(
    B_lab_nT,
    D_mhz,
    Mz_mhz,
    axis_name,
    nv_axes,
    gamma_e_mhz_per_nT,
):
    """
    Reduced spin-1 NV Hamiltonian.

    H/h = (D + Mz_i) Sz^2 + gamma_e * B_i . S

    Units:
        B_lab_nT: nT
        gamma_e_mhz_per_nT: MHz/nT
        D_mhz, Mz_mhz: MHz
        output transition frequencies: MHz

    Notes:
        - Mz_mhz is a phenomenological static axis-dependent shift.
        - Hyperfine splitting is not included here because the fitted center
          frequency of the hyperfine triplet should represent the electronic
          transition center.
    """
    Sx, Sy, Sz, Sz2 = spin1_matrices()

    local_frame = make_local_frame(nv_axes[axis_name])
    Bx, By, Bz = lab_to_local(B_lab_nT, local_frame)

    zeeman = gamma_e_mhz_per_nT * (Bx * Sx + By * Sy + Bz * Sz)
    H = (D_mhz + Mz_mhz) * Sz2 + zeeman

    evals = np.sort(np.real(np.linalg.eigvalsh(H)))

    f_lower = evals[1] - evals[0]
    f_upper = evals[2] - evals[0]

    return f_lower, f_upper


def compute_A_matrix_lower_transition(
    B0_nT,
    D_mhz,
    Mz_map_mhz,
    axes_order,
    nv_axes,
    gamma_e_mhz_per_nT,
    dB_step_nT=1000.0,
):
    """
    Numerically compute the linearized response matrix A for the lower transition.

    Definition:
        delta_nu = A @ delta_B

    Units:
        delta_nu: MHz
        delta_B: nT
        A: MHz/nT

    dB_step_nT:
        1000 nT = 1 microtesla.
        This is small relative to a mT-scale bias field but large enough
        to avoid numerical roundoff.
    """
    B0_nT = np.asarray(B0_nT, dtype=float)
    A = np.zeros((len(axes_order), 3), dtype=float)

    for j in range(3):
        step = np.zeros(3, dtype=float)
        step[j] = dB_step_nT

        B_plus = B0_nT + step
        B_minus = B0_nT - step

        for i, axis in enumerate(axes_order):
            f_plus, _ = transition_frequencies_mhz(
                B_plus,
                D_mhz,
                Mz_map_mhz[axis],
                axis,
                nv_axes,
                gamma_e_mhz_per_nT,
            )

            f_minus, _ = transition_frequencies_mhz(
                B_minus,
                D_mhz,
                Mz_map_mhz[axis],
                axis,
                nv_axes,
                gamma_e_mhz_per_nT,
            )

            A[i, j] = (f_plus - f_minus) / (2 * dB_step_nT)

    return A
    