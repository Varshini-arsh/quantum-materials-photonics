"""
Symmetry-Verification Error Mitigation for the Mesh Ansatz
=============================================================

Closes the loop on the finding in mesh_vqe_ansatz.py: the mesh ansatz is
more accurate and consistent than standard ansatze noiselessly, but its
accuracy collapses under noise *because* depolarizing noise breaks the
exact Hamming-weight (total magnetization) conservation the ansatz relies
on. This script fixes that using the same symmetry structure.

Method (standard symmetry verification / post-selection, e.g. McArdle,
Yuan & Benjamin, PRL 2019): rather than trusting the raw noisy state, we
measure the number operator (= a projector onto the correct Hamming-weight
subspace), discard the component of the state outside that subspace, and
compute the energy on the renormalized *surviving* state:

    rho_mitigated = P rho P / Tr(P rho P)
    E_mitigated   = Tr(H rho_mitigated)

Implemented exactly via Qiskit Aer's density-matrix simulator (not noisy
shot sampling with basis rotations, which would conflate the Pauli-term
measurement basis with the Hamming-weight post-selection basis and risks
a subtle correctness bug) -- this computes exactly what infinite-shot
symmetry verification converges to.
"""

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh_vqe_ansatz import (
    build_heisenberg_hamiltonian, exact_ground_state_energy,
    mesh_ansatz, N_SPINS, J_COUPLING,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_noise_model(one_q_error: float = 0.001, two_q_error: float = 0.01) -> NoiseModel:
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(one_q_error, 1),
        ["sx", "x", "rz", "ry", "rx", "h", "s", "sdg", "sxdg"])
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(two_q_error, 2), ["cx", "cz", "ecr"])
    return noise_model


def hamming_weight_projector(n_qubits: int, target_weight: int) -> np.ndarray:
    dim = 2 ** n_qubits
    diag = np.zeros(dim)
    for state in range(dim):
        if bin(state).count("1") == target_weight:
            diag[state] = 1.0
    return np.diag(diag)


def noisy_density_matrix(circuit: QuantumCircuit, noise_model: NoiseModel) -> DensityMatrix:
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    qc = circuit.copy()
    qc.save_density_matrix()
    result = sim.run(qc).result()
    return DensityMatrix(result.data(0)["density_matrix"])


def energy_from_density_matrix(rho: DensityMatrix, h_matrix: np.ndarray) -> float:
    return float(np.real(np.trace(h_matrix @ rho.data)))


def main():
    hamiltonian = build_heisenberg_hamiltonian(N_SPINS, J_COUPLING)
    h_matrix = hamiltonian.to_matrix()
    exact_energy = exact_ground_state_energy(hamiltonian)
    print(f"Exact ground-state energy: {exact_energy:.6f}")

    with open(RESULTS_DIR / "mesh_vqe_results.json") as f:
        mesh_results = json.load(f)
    best_point = np.array(mesh_results["mesh_noiseless"]["best_optimal_point"])

    mesh = mesh_ansatz(N_SPINS, reps=2).decompose()
    mesh_bound = mesh.assign_parameters(best_point)

    noise_model = make_noise_model()
    P = hamming_weight_projector(N_SPINS, N_SPINS // 2)

    print("\nRunning noisy density-matrix simulation of the mesh ansatz "
          "(same optimized parameters as the noiseless VQE result)...")
    rho_noisy = noisy_density_matrix(mesh_bound, noise_model)

    e_noisy_raw = energy_from_density_matrix(rho_noisy, h_matrix)

    survival_prob = float(np.real(np.trace(P @ rho_noisy.data @ P)))
    rho_mitigated_matrix = (P @ rho_noisy.data @ P) / survival_prob
    e_mitigated = float(np.real(np.trace(h_matrix @ rho_mitigated_matrix)))

    # noiseless reference for this exact circuit (sanity: should match VQE's
    # reported noiseless energy for this parameter point)
    sv = Statevector.from_instruction(mesh_bound)
    e_noiseless_this_circuit = float(np.real(sv.expectation_value(hamiltonian)))

    print(f"\nNoiseless energy (this circuit):      {e_noiseless_this_circuit:.4f}  "
          f"(error {e_noiseless_this_circuit - exact_energy:+.4f})")
    print(f"Noisy energy, unmitigated:             {e_noisy_raw:.4f}  "
          f"(error {e_noisy_raw - exact_energy:+.4f})")
    print(f"Survival probability (correct sector):  {survival_prob:.4f}")
    print(f"Noisy energy, symmetry-mitigated:       {e_mitigated:.4f}  "
          f"(error {e_mitigated - exact_energy:+.4f})")

    improvement = abs(e_noisy_raw - exact_energy) - abs(e_mitigated - exact_energy)
    print(f"\nError reduction from mitigation: {improvement:.4f} "
          f"({100*improvement/abs(e_noisy_raw - exact_energy):.1f}% of unmitigated error removed)")

    # --- same treatment for EfficientSU2, for comparison: does symmetry
    # post-selection help an ansatz that was never designed to respect
    # this symmetry in the first place? ---
    print("\n=== Comparison: same mitigation applied to EfficientSU2 ===")
    # part 1 didn't save optimal parameter points, only energies, so this
    # uses a fresh arbitrary point purely to illustrate the contrast: does
    # symmetry post-selection help an ansatz that was never confined to
    # this sector? Not a rigorous VQE-optimal comparison -- noted here and
    # in the results JSON.
    su2 = efficient_su2(N_SPINS, reps=3)
    rng = np.random.default_rng(0)
    su2_bound = su2.assign_parameters(rng.uniform(-np.pi, np.pi, su2.num_parameters))
    rho_su2_noisy = noisy_density_matrix(su2_bound, noise_model)
    e_su2_noisy_raw = energy_from_density_matrix(rho_su2_noisy, h_matrix)
    survival_su2 = float(np.real(np.trace(P @ rho_su2_noisy.data @ P)))
    rho_su2_mitigated = (P @ rho_su2_noisy.data @ P) / survival_su2
    e_su2_mitigated = float(np.real(np.trace(h_matrix @ rho_su2_mitigated)))
    print(f"EfficientSU2 (arbitrary point) survival probability: {survival_su2:.4f} "
          f"(much lower -- it was never confined to this sector)")
    print(f"EfficientSU2 noisy raw: {e_su2_noisy_raw:.4f}, mitigated: {e_su2_mitigated:.4f}")

    results = {
        "exact_energy": exact_energy,
        "mesh": {
            "noiseless_energy": e_noiseless_this_circuit,
            "noisy_raw_energy": e_noisy_raw,
            "survival_probability": survival_prob,
            "noisy_mitigated_energy": e_mitigated,
            "error_unmitigated": e_noisy_raw - exact_energy,
            "error_mitigated": e_mitigated - exact_energy,
        },
        "efficient_su2_comparison": {
            "survival_probability": survival_su2,
            "noisy_raw_energy": e_su2_noisy_raw,
            "noisy_mitigated_energy": e_su2_mitigated,
            "note": "arbitrary (non-VQE-optimal) parameter point, illustrative only",
        },
    }
    out_path = RESULTS_DIR / "symmetry_mitigation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # --- plot ---
    labels = ["Noiseless", "Noisy\n(unmitigated)", "Noisy\n(symmetry-mitigated)"]
    errors = [abs(e_noiseless_this_circuit - exact_energy),
              abs(e_noisy_raw - exact_energy),
              abs(e_mitigated - exact_energy)]
    colors = ["#55A868", "#C44E52", "#4C72B0"]
    plt.figure(figsize=(6, 5))
    plt.bar(labels, errors, color=colors)
    plt.ylabel("|energy error| vs. exact")
    plt.title("Symmetry-verification error mitigation (Mesh ansatz)")
    for i, e in enumerate(errors):
        plt.text(i, e + 0.02, f"{e:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "symmetry_mitigation.png", dpi=150)
    plt.close()
    print(f"Plot written to {RESULTS_DIR / 'symmetry_mitigation.png'}")


if __name__ == "__main__":
    main()
