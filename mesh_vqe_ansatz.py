"""
Mesh-VQE: A Photonic-Mesh-Structured, Symmetry-Preserving Ansatz
===================================================================

An original ansatz design for VQE materials simulation, combining the two
earlier pieces of this project rather than reproducing either:

  1. From `photonic_qft_mesh.py`: the triangular (Reck-style) mesh
     connectivity pattern used to lay out a photonic interferometer.
  2. From `vqe_materials_heisenberg.py`: the Heisenberg spin-chain VQE
     problem and its rigorous multi-trial comparison framework.

Design rationale (why this isn't just "a new ansatz for its own sake"):
  - The Heisenberg Hamiltonian H = J*sum(X_iX_j+Y_iY_j+Z_iZ_j) commutes
    with total magnetization (sum of Z_i), so its ground state lives in a
    single fixed-Hamming-weight symmetry sector. Standard hardware-efficient
    ansatze (EfficientSU2, RealAmplitudes) do NOT respect this -- they
    explore the full 2^n Hilbert space, wasting optimizer effort on
    symmetry-broken states the true ground state can never be in.
  - Qiskit's XXPlusYYGate is the qubit analog of a beamsplitter (identity
    on |00>,|11>, a rotation within {|01>,|10>}) -- it exactly preserves
    Hamming weight, gate by gate. Placing these gates in the same
    triangular connectivity pattern as the photonic mesh from part 2 gives
    an ansatz that is simultaneously:
      (a) structurally reusing this project's own photonic-mesh work,
      (b) provably confined to the correct symmetry sector throughout
          optimization -- a genuine, testable, structural advantage over
          the standard ansatze already benchmarked in part 1.

This script builds the ansatz, runs it through the exact same rigorous
(multi-trial, noiseless+noisy) comparison framework as
`vqe_materials_heisenberg.py`, and adds a new verification the standard
ansatze can't pass by design: confirming Hamming weight is exactly
conserved throughout the optimized circuit, contrasted against
EfficientSU2 where it is not.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate, efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SPINS = 4
J_COUPLING = 1.0
N_TRIALS = 5
MAXITER = 250


# ----------------------------------------------------------------------
# Hamiltonian (same as vqe_materials_heisenberg.py)
# ----------------------------------------------------------------------

def build_heisenberg_hamiltonian(n_spins: int, j: float) -> SparsePauliOp:
    terms, coeffs = [], []
    for i in range(n_spins - 1):
        for pauli in ("X", "Y", "Z"):
            label = ["I"] * n_spins
            label[i] = pauli
            label[i + 1] = pauli
            terms.append("".join(reversed(label)))
            coeffs.append(j)
    return SparsePauliOp(terms, coeffs)


def exact_ground_state_energy(hamiltonian: SparsePauliOp) -> float:
    return float(np.min(np.linalg.eigvalsh(hamiltonian.to_matrix())))


def make_noise_model(one_q_error: float = 0.001, two_q_error: float = 0.01) -> NoiseModel:
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(one_q_error, 1),
        ["sx", "x", "rz", "ry", "rx", "h", "s", "sdg", "sxdg"])
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(two_q_error, 2), ["cx", "cz", "ecr", "xx_plus_yy"])
    return noise_model


# ----------------------------------------------------------------------
# The mesh ansatz itself
# ----------------------------------------------------------------------

def reck_mesh_schedule(n: int):
    """Same triangular adjacent-pair connectivity pattern used to build
    the Reck photonic mesh in photonic_qft_mesh.py -- reused here purely
    as a *gate topology*, not to decompose any specific unitary."""
    elimination_order = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            elimination_order.append((row - 1, row))
    return list(reversed(elimination_order))  # "physical" left-to-right order


def mesh_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """Hamming-weight-preserving ansatz: XXPlusYYGate (qubit analog of a
    beamsplitter) placed on the photonic-mesh connectivity pattern, plus
    an RZ phase layer per rep (also Hamming-weight-preserving), repeated
    `reps` times. Reference state: alternating |0101...> (half-filling,
    the correct symmetry sector for the antiferromagnetic ground state)."""
    schedule = reck_mesh_schedule(n_qubits)
    n_theta = len(schedule) * reps
    n_phi = n_qubits * reps
    thetas = ParameterVector("theta", n_theta)
    phis = ParameterVector("phi", n_phi)

    qc = QuantumCircuit(n_qubits, name="mesh_ansatz")
    for q in range(0, n_qubits, 2):
        qc.x(q)  # reference state |0101...> (or |1010...> depending on parity)

    t_idx = 0
    p_idx = 0
    for _ in range(reps):
        for (i, j) in schedule:
            qc.append(XXPlusYYGate(thetas[t_idx]), [i, j])
            t_idx += 1
        for q in range(n_qubits):
            qc.rz(phis[p_idx], q)
            p_idx += 1
    return qc


def hamming_weight_expectation(state: Statevector) -> float:
    probs = state.probabilities_dict()
    return sum(p * bin(int(bits, 2)).count("1") for bits, p in probs.items())


# ----------------------------------------------------------------------
# VQE runner (same structure as vqe_materials_heisenberg.py)
# ----------------------------------------------------------------------

def run_vqe_trial(hamiltonian, ansatz, optimizer, estimator, seed):
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    history = {"evals": [], "values": []}

    def callback(eval_count, params, value, metadata):
        history["evals"].append(int(eval_count))
        history["values"].append(float(value))

    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point, callback=callback)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    return float(np.real(result.eigenvalue)), history, result.optimal_point


def run_config(hamiltonian, ansatz, estimator, label, n_trials=N_TRIALS):
    energies, histories, optimal_points = [], [], []
    for seed in range(n_trials):
        e, h, op = run_vqe_trial(hamiltonian, ansatz, COBYLA(maxiter=MAXITER), estimator, seed)
        energies.append(e)
        histories.append(h)
        optimal_points.append(op)
    energies = np.array(energies)
    best_idx = int(np.argmin(energies))
    result = {
        "label": label,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "best_energy": float(np.min(energies)),
        "all_energies": energies.tolist(),
        "representative_history": histories[best_idx],
        "best_optimal_point": optimal_points[best_idx].tolist(),
    }
    print(f"  [{label}] mean = {result['mean_energy']:.4f}  "
          f"std = {result['std_energy']:.4f}  best = {result['best_energy']:.4f}  "
          f"({n_trials} trials)")
    return result


def main():
    print(f"Building {N_SPINS}-site Heisenberg chain (J={J_COUPLING})...")
    hamiltonian = build_heisenberg_hamiltonian(N_SPINS, J_COUPLING)
    exact_energy = exact_ground_state_energy(hamiltonian)
    print(f"Exact ground-state energy: {exact_energy:.6f}\n")

    mesh = mesh_ansatz(N_SPINS, reps=2).decompose()  # standard-gate basis, needed
    # for the Aer noise model (xx_plus_yy isn't a recognized noise-model instruction)
    print(f"Mesh ansatz: {mesh.num_parameters} parameters, "
          f"{len(reck_mesh_schedule(N_SPINS))} XXPlusYY gates per rep x 2 reps")

    noiseless_estimator = StatevectorEstimator()
    noise_model = make_noise_model()
    noisy_estimator = AerEstimatorV2(options={"backend_options": {"noise_model": noise_model}})

    print("\n=== Mesh ansatz: noiseless ===")
    mesh_noiseless = run_config(hamiltonian, mesh, noiseless_estimator, "MeshAnsatz (noiseless)")
    print("\n=== Mesh ansatz: noisy (depolarizing: 0.1% 1Q / 1% 2Q) ===")
    mesh_noisy = run_config(hamiltonian, mesh, noisy_estimator, "MeshAnsatz (noisy)")

    # --- symmetry verification: Hamming weight (magnetization) conservation ---
    print("\n=== Symmetry verification: Hamming weight conservation ===")
    mesh_bound = mesh.assign_parameters(mesh_noiseless["best_optimal_point"])
    mesh_state = Statevector.from_instruction(mesh_bound)
    mesh_hw = hamming_weight_expectation(mesh_state)
    print(f"  MeshAnsatz optimized state: <Hamming weight> = {mesh_hw:.6f} "
          f"(expect exactly {N_SPINS // 2}.0 -- conserved by construction)")

    su2 = efficient_su2(N_SPINS, reps=3)
    rng = np.random.default_rng(0)
    su2_bound = su2.assign_parameters(rng.uniform(-np.pi, np.pi, su2.num_parameters))
    su2_state = Statevector.from_instruction(su2_bound)
    su2_hw = hamming_weight_expectation(su2_state)
    print(f"  EfficientSU2 (arbitrary point): <Hamming weight> = {su2_hw:.6f} "
          f"(not conserved -- explores full Hilbert space)")

    # --- load part-1 results for a direct side-by-side comparison ---
    part1_path = RESULTS_DIR / "vqe_heisenberg_results.json"
    comparison = None
    if part1_path.exists():
        with open(part1_path) as f:
            part1 = json.load(f)
        comparison = {
            "exact_energy": exact_energy,
            "mesh_noiseless_error": mesh_noiseless["mean_energy"] - exact_energy,
            "mesh_noisy_error": mesh_noisy["mean_energy"] - exact_energy,
            "part1_ansatze": [
                {"label": r["label"], "error": r["mean_energy"] - exact_energy}
                for r in part1["ansatz_comparison_noiseless"] + part1["ansatz_comparison_noisy"]
            ],
        }

    results = {
        "exact_energy": exact_energy,
        "mesh_n_parameters": mesh.num_parameters,
        "mesh_noiseless": mesh_noiseless,
        "mesh_noisy": mesh_noisy,
        "hamming_weight_mesh": mesh_hw,
        "hamming_weight_su2_uncontrolled": su2_hw,
        "comparison_vs_part1": comparison,
    }
    out_path = RESULTS_DIR / "mesh_vqe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # --- plots ---
    if comparison is not None:
        labels = [a["label"] for a in comparison["part1_ansatze"]] + \
                 ["MeshAnsatz (noiseless)", "MeshAnsatz (noisy)"]
        errors = [a["error"] for a in comparison["part1_ansatze"]] + \
                 [comparison["mesh_noiseless_error"], comparison["mesh_noisy_error"]]
        colors = ["#4C72B0"] * len(comparison["part1_ansatze"]) + ["#55A868", "#C44E52"]
        plt.figure(figsize=(9, 5))
        plt.bar(range(len(labels)), errors, color=colors)
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        plt.ylabel("mean energy error vs. exact")
        plt.title("Mesh ansatz vs. standard ansätze (green/red = new mesh ansatz)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "mesh_vs_standard_ansatze.png", dpi=150)
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(["MeshAnsatz\n(symmetry-preserving)", "EfficientSU2\n(unconstrained)"],
            [mesh_hw, su2_hw], color=["#55A868", "#C44E52"])
    plt.axhline(N_SPINS / 2, color="black", linestyle="--", label=f"correct sector ({N_SPINS//2})")
    plt.ylabel("<Hamming weight> (total excitation number)")
    plt.title("Symmetry conservation: mesh ansatz vs. standard ansatz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "symmetry_conservation.png", dpi=150)
    plt.close()

    h = mesh_noiseless["representative_history"]
    plt.figure(figsize=(7, 5))
    plt.plot(h["evals"], h["values"], color="#55A868", label="MeshAnsatz")
    plt.axhline(exact_energy, color="black", linestyle="--", label="exact")
    plt.xlabel("cost function evaluations")
    plt.ylabel("energy")
    plt.title("Mesh ansatz VQE convergence (best of 5 trials, noiseless)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mesh_convergence.png", dpi=150)
    plt.close()

    print(f"\nPlots written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
