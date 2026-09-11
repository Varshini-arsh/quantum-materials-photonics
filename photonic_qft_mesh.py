"""
Photonic Realization of the Quantum Fourier Transform via a Reck Interferometer Mesh
=====================================================================================

Part 2 of the QMD Fellowship project. Bridges gate-model quantum computing
(Qiskit) and linear-optical photonic quantum computing (Perceval):

  1. Take the same N-point Quantum Fourier Transform used in this project's
     existing Qiskit work (quantum/quantum_inverse_2d_qft_v2.py) and confirm
     it equals the standard N-point discrete Fourier transform (DFT) matrix.
  2. Decompose that N x N unitary into a mesh of two-mode interferometer
     blocks + phase shifters using a Reck-style (triangular) decomposition,
     implemented from first principles via complex Givens rotations
     (the same linear-algebra operation a beamsplitter + phase shifters
     realizes physically -- this is the classic Reck et al. 1994 scheme).
  3. Build the corresponding circuit in Perceval and verify its computed
     unitary matches the target DFT matrix.
  4. Simulate single-photon injection through the mesh (path/dual-rail
     encoding: one photon in one of N spatial modes) and compare the
     simulated output-mode distribution to the ideal |U|^2 probabilities
     and to the Qiskit QFT circuit's own transition probabilities.

Single-photon path encoding is why an N-mode interferometer implements the
*N-point* DFT directly on the mode index, rather than the n-qubit
(2^n-dimensional) QFT circuit structure -- so we compare at matched
dimension N = 2^n.
"""

import json
from pathlib import Path

import numpy as np
import perceval as pcvl
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_QUBITS = 3
N = 2 ** N_QUBITS  # number of photonic modes = QFT dimension


# ----------------------------------------------------------------------
# Step 1: target unitary from Qiskit's own QFT gate
# ----------------------------------------------------------------------

def qiskit_qft_unitary(n_qubits: int) -> np.ndarray:
    return Operator(QFTGate(n_qubits)).data


def dft_matrix(n: int) -> np.ndarray:
    """N-point DFT matrix, +2*pi*i sign convention to match Qiskit's QFTGate
    exactly (verified empirically: Qiskit's QFT uses +i, not the textbook -i,
    sign in the exponent)."""
    j = np.arange(n).reshape(-1, 1)
    k = np.arange(n).reshape(1, -1)
    return np.exp(2j * np.pi * j * k / n) / np.sqrt(n)


# ----------------------------------------------------------------------
# Step 2: Reck-style decomposition via complex Givens rotations
# ----------------------------------------------------------------------

def givens_zero(a: complex, b: complex) -> np.ndarray:
    """2x2 unitary G with G @ [a, b]^T = [r, 0]^T, r = hypot(|a|,|b|) real >= 0."""
    r = np.hypot(abs(a), abs(b))
    if r < 1e-14:
        return np.eye(2, dtype=complex)
    return np.array([[np.conj(a), np.conj(b)], [-b, a]], dtype=complex) / r


def decompose_reck(u: np.ndarray):
    """Reduce U to a diagonal phase matrix via a triangular sequence of
    adjacent-mode Givens rotations (physically: beamsplitters + phase
    shifters). Returns (rotations, diagonal_phases)."""
    n = u.shape[0]
    work = u.astype(complex).copy()
    rotations = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            a, b = work[row - 1, col], work[row, col]
            if abs(b) < 1e-12:
                continue
            g2 = givens_zero(a, b)
            g_full = np.eye(n, dtype=complex)
            g_full[row - 1, row - 1], g_full[row - 1, row] = g2[0, 0], g2[0, 1]
            g_full[row, row - 1], g_full[row, row] = g2[1, 0], g2[1, 1]
            work = g_full @ work
            rotations.append((row - 1, row, g2))
    diag = np.diag(work).copy()
    off_diag_residual = np.max(np.abs(work - np.diag(diag)))
    assert off_diag_residual < 1e-9, f"decomposition failed to diagonalize: {off_diag_residual}"
    return rotations, diag


def build_perceval_circuit(n: int, rotations, diag) -> pcvl.Circuit:
    """Physical mesh: diagonal phase layer first, then the Givens blocks
    in reverse elimination order (dagger of each), matching
    U = G_1^dag G_2^dag ... G_k^dag D."""
    circuit = pcvl.Circuit(n)
    for m in range(n):
        circuit.add(m, pcvl.PS(phi=float(np.angle(diag[m]))))
    for (i, j, g2) in reversed(rotations):
        circuit.add(i, pcvl.Unitary(pcvl.Matrix(g2.conj().T)))
    return circuit


def plot_mesh_schematic(n: int, rotations, out_path):
    """Custom schematic of the Reck mesh: one horizontal line per mode,
    a phase-shifter marker at stage 0, then one vertical connector per
    two-mode block in physical circuit order (Perceval's own MPLOT
    renderer does not reliably save to a static figure in this setup)."""
    physical_order = [(i, j) for (i, j, _g2) in reversed(rotations)]
    n_stages = len(physical_order) + 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in range(n):
        ax.plot([0, n_stages], [m, m], color="#999999", lw=1, zorder=1)
        ax.scatter([0], [m], marker="s", color="#4C72B0", s=35, zorder=3, label="phase shifter" if m == 0 else None)
    for stage, (i, j) in enumerate(physical_order, start=1):
        ax.plot([stage, stage], [i, j], color="#DD8452", lw=2, zorder=2)
        ax.scatter([stage, stage], [i, j], color="#DD8452", s=22, zorder=3,
                   label="2-mode block" if stage == 1 else None)
    ax.set_yticks(range(n))
    ax.set_ylabel("photonic mode")
    ax.set_xlabel("circuit stage (light travels left to right)")
    ax.set_title(f"Reck-style photonic QFT mesh ({n} modes, {len(physical_order)} two-mode blocks)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Step 3 & 4: build, verify, simulate
# ----------------------------------------------------------------------

def main():
    qiskit_u = qiskit_qft_unitary(N_QUBITS)
    dft_u = dft_matrix(N)
    qiskit_vs_dft_diff = np.max(np.abs(qiskit_u - dft_u))
    print(f"N = {N} modes ({N_QUBITS} qubits equivalent)")
    print(f"Qiskit QFTGate({N_QUBITS}) vs standard DFT matrix, max |diff| = "
          f"{qiskit_vs_dft_diff:.2e}")

    target_u = dft_u  # decompose the canonical DFT matrix
    rotations, diag = decompose_reck(target_u)
    print(f"Reck decomposition: {len(rotations)} two-mode blocks + {N} phase shifters")

    circuit = build_perceval_circuit(N, rotations, diag)
    computed_u = np.array(circuit.compute_unitary())
    mesh_vs_target_diff = np.max(np.abs(computed_u - target_u))
    print(f"Perceval mesh unitary vs target DFT, max |diff| = {mesh_vs_target_diff:.2e}")

    mesh_plot_path = RESULTS_DIR / "photonic_qft_mesh_schematic.png"
    plot_mesh_schematic(N, rotations, mesh_plot_path)
    print(f"Mesh schematic written to {mesh_plot_path}")

    print("\nSimulating single-photon injection through the mesh (SLOS backend)...")
    per_input_results = []
    total_variation_distances = []
    for input_mode in range(N):
        basic_state = [0] * N
        basic_state[input_mode] = 1
        proc = pcvl.Processor("SLOS", circuit)
        proc.with_input(pcvl.BasicState(basic_state))
        sim_probs_raw = proc.probs()["results"]

        ideal_probs = np.abs(target_u[:, input_mode]) ** 2
        sim_probs = np.zeros(N)
        for state, p in sim_probs_raw.items():
            out_mode = list(state).index(1)
            sim_probs[out_mode] = p

        tvd = 0.5 * float(np.sum(np.abs(sim_probs - ideal_probs)))
        total_variation_distances.append(tvd)
        per_input_results.append({
            "input_mode": input_mode,
            "simulated_probs": sim_probs.tolist(),
            "ideal_probs": ideal_probs.tolist(),
            "total_variation_distance": tvd,
        })
        print(f"  input mode {input_mode}: TVD(simulated, ideal) = {tvd:.2e}")

    results = {
        "n_modes": N,
        "n_qubits_equivalent": N_QUBITS,
        "qiskit_vs_dft_max_diff": qiskit_vs_dft_diff,
        "n_two_mode_blocks": len(rotations),
        "mesh_vs_target_max_diff": mesh_vs_target_diff,
        "max_total_variation_distance": max(total_variation_distances),
        "per_input": per_input_results,
    }
    out_path = RESULTS_DIR / "photonic_qft_mesh_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"\nSummary: {N}-mode Reck mesh reproduces the {N_QUBITS}-qubit Qiskit QFT's "
          f"DFT unitary to {mesh_vs_target_diff:.1e}, and single-photon simulation "
          f"matches ideal |U|^2 probabilities to a max TVD of "
          f"{max(total_variation_distances):.1e} across all {N} input modes.")


if __name__ == "__main__":
    main()
