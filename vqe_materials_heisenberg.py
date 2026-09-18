"""
VQE Ground-State Energy Estimation for a Heisenberg Spin Chain
================================================================

Part 1 of this project: a quantum-materials simulation
demonstration. Estimates the ground-state energy of a 1D antiferromagnetic
Heisenberg spin chain (a standard toy model for quantum magnetic materials)
using the Variational Quantum Eigensolver.

Two controlled comparisons (each varies exactly one factor at a time,
averaged over multiple random-seed trials so the conclusions aren't just
one lucky/unlucky run):

  A) Ansatz comparison -- optimizer and iteration budget held fixed
     (COBYLA, same maxiter), 3 ansatze compared, noiseless AND under a
     realistic depolarizing noise model. Tests whether ansatz depth trades
     off accuracy against noise resilience -- a "devices" relevant result,
     not just an algorithm result.

  B) Optimizer comparison -- ansatz held fixed (the best performer from A),
     COBYLA vs. SPSA compared, noiseless only.

Chain: N_SPINS sites, open boundary conditions, isotropic coupling J=1.
H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.circuit.library import efficient_su2, real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SPINS = 4
J_COUPLING = 1.0
N_TRIALS = 5
MAXITER = 250


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
        depolarizing_error(one_q_error, 1), ["sx", "x", "rz", "ry", "rx", "h"])
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(two_q_error, 2), ["cx", "cz", "ecr"])
    return noise_model


def run_vqe_trial(hamiltonian, ansatz, optimizer, estimator, seed):
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    history = {"evals": [], "values": []}

    def callback(eval_count, params, value, metadata):
        history["evals"].append(int(eval_count))
        history["values"].append(float(value))

    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point, callback=callback)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    return float(np.real(result.eigenvalue)), history


def run_config(hamiltonian, ansatz, optimizer_factory, estimator, label, n_trials=N_TRIALS):
    energies, histories = [], []
    for seed in range(n_trials):
        e, h = run_vqe_trial(hamiltonian, ansatz, optimizer_factory(), estimator, seed)
        energies.append(e)
        histories.append(h)
    energies = np.array(energies)
    best_idx = int(np.argmin(energies))
    result = {
        "label": label,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "best_energy": float(np.min(energies)),
        "all_energies": energies.tolist(),
        "representative_history": histories[best_idx],
    }
    print(f"  [{label}] mean = {result['mean_energy']:.4f}  "
          f"std = {result['std_energy']:.4f}  best = {result['best_energy']:.4f}  "
          f"({n_trials} trials)")
    return result


def plot_convergence(ansatz_results, exact_energy, out_path):
    plt.figure(figsize=(7, 5))
    for r in ansatz_results:
        h = r["representative_history"]
        plt.plot(h["evals"], h["values"], label=r["label"], alpha=0.85)
    plt.axhline(exact_energy, color="black", linestyle="--", label="exact (diagonalization)")
    plt.xlabel("cost function evaluations")
    plt.ylabel("energy")
    plt.title("VQE convergence (best of 5 trials per ansatz, noiseless)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_bars(ansatz_results_noiseless, ansatz_results_noisy, exact_energy, out_path):
    labels = [r["label"] for r in ansatz_results_noiseless]
    err_noiseless = [r["mean_energy"] - exact_energy for r in ansatz_results_noiseless]
    std_noiseless = [r["std_energy"] for r in ansatz_results_noiseless]
    err_noisy = [r["mean_energy"] - exact_energy for r in ansatz_results_noisy]
    std_noisy = [r["std_energy"] for r in ansatz_results_noisy]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, err_noiseless, width, yerr=std_noiseless, capsize=4, label="noiseless")
    plt.bar(x + width / 2, err_noisy, width, yerr=std_noisy, capsize=4, label="noisy")
    plt.xticks(x, labels, rotation=15, ha="right", fontsize=8)
    plt.ylabel("mean energy error vs. exact (+/- std over 5 trials)")
    plt.title("Ansatz comparison: accuracy vs. noise resilience")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_optimizer_comparison(opt_results, exact_energy, out_path):
    labels = [r["label"] for r in opt_results]
    err = [r["mean_energy"] - exact_energy for r in opt_results]
    std = [r["std_energy"] for r in opt_results]
    plt.figure(figsize=(5, 5))
    plt.bar(labels, err, yerr=std, capsize=4, color=["#4C72B0", "#DD8452"])
    plt.ylabel("mean energy error vs. exact (+/- std over 5 trials)")
    plt.title("Optimizer comparison (fixed ansatz, noiseless)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    print(f"Building {N_SPINS}-site open-boundary Heisenberg chain (J={J_COUPLING})...")
    hamiltonian = build_heisenberg_hamiltonian(N_SPINS, J_COUPLING)
    n_qubits = hamiltonian.num_qubits
    exact_energy = exact_ground_state_energy(hamiltonian)
    print(f"Exact ground-state energy (diagonalization): {exact_energy:.6f}\n")

    ansatze = {
        "EfficientSU2_reps1": efficient_su2(n_qubits, reps=1),
        "EfficientSU2_reps3": efficient_su2(n_qubits, reps=3),
        "RealAmplitudes_reps2": real_amplitudes(n_qubits, reps=2),
    }

    noiseless_estimator = StatevectorEstimator()
    noise_model = make_noise_model()
    noisy_estimator = AerEstimatorV2(options={"backend_options": {"noise_model": noise_model}})

    # --- A) Ansatz comparison: optimizer fixed at COBYLA(maxiter=MAXITER) ---
    print(f"=== A) Ansatz comparison (COBYLA, maxiter={MAXITER}, {N_TRIALS} trials each) ===")
    print("-- noiseless --")
    ansatz_noiseless = [
        run_config(hamiltonian, ansatz, lambda: COBYLA(maxiter=MAXITER), noiseless_estimator,
                   f"{name} (noiseless)")
        for name, ansatz in ansatze.items()
    ]
    print("-- noisy (depolarizing: 0.1% 1Q / 1% 2Q) --")
    ansatz_noisy = [
        run_config(hamiltonian, ansatz, lambda: COBYLA(maxiter=MAXITER), noisy_estimator,
                   f"{name} (noisy)")
        for name, ansatz in ansatze.items()
    ]

    best_ansatz_name = min(ansatze, key=lambda n: next(
        r["mean_energy"] for r in ansatz_noiseless if r["label"] == f"{n} (noiseless)"))
    print(f"\nBest-performing ansatz (noiseless, lowest mean energy): {best_ansatz_name}")

    # --- B) Optimizer comparison: ansatz fixed at the best performer ---
    print(f"\n=== B) Optimizer comparison (ansatz={best_ansatz_name}, noiseless, "
          f"{N_TRIALS} trials each) ===")
    best_ansatz = ansatze[best_ansatz_name]
    cobyla_result = next(r for r in ansatz_noiseless if r["label"] == f"{best_ansatz_name} (noiseless)")
    cobyla_result = dict(cobyla_result, label="COBYLA")
    spsa_result = run_config(hamiltonian, best_ansatz, lambda: SPSA(maxiter=MAXITER),
                              noiseless_estimator, "SPSA")
    optimizer_results = [cobyla_result, spsa_result]

    # --- plots ---
    plot_convergence(ansatz_noiseless, exact_energy, RESULTS_DIR / "vqe_convergence.png")
    plot_error_bars(ansatz_noiseless, ansatz_noisy, exact_energy,
                     RESULTS_DIR / "vqe_ansatz_noise_comparison.png")
    plot_optimizer_comparison(optimizer_results, exact_energy,
                               RESULTS_DIR / "vqe_optimizer_comparison.png")

    results = {
        "exact_energy": exact_energy,
        "n_qubits": n_qubits,
        "n_trials": N_TRIALS,
        "maxiter": MAXITER,
        "ansatz_comparison_noiseless": ansatz_noiseless,
        "ansatz_comparison_noisy": ansatz_noisy,
        "best_ansatz": best_ansatz_name,
        "optimizer_comparison": optimizer_results,
    }
    out_path = RESULTS_DIR / "vqe_heisenberg_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Plots written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
