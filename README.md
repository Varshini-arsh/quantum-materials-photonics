# QMD Fellowship Project — Quantum Materials & Photonic Devices

Three-part demonstration built for the QuantaRiSE Quantum Winter Fellowship application.

## 1. `vqe_materials_heisenberg.py` — Quantum materials simulation

Estimates the ground-state energy of a 4-site open-boundary antiferromagnetic
Heisenberg spin chain (a standard toy model for quantum magnetic materials)
using the Variational Quantum Eigensolver. Two controlled comparisons, each
averaged over 5 random-seed trials so conclusions aren't from one lucky run:

- **A) Ansatz comparison** — optimizer (COBYLA, maxiter=250) held fixed, 3 ansätze
  (EfficientSU2 reps=1/3, RealAmplitudes reps=2) compared against exact diagonalization,
  both noiseless and under a realistic depolarizing noise model (0.1% 1-qubit / 1% 2-qubit
  gate error).
- **B) Optimizer comparison** — ansatz held fixed at the best performer from A,
  COBYLA vs. SPSA compared, noiseless.

**Key finding:** EfficientSU2 reps=3 (the deepest, most-entangling ansatz) got closest to
the exact energy noiselessly (mean error 0.31, vs. 0.40 for reps=1 and 0.31 for
RealAmplitudes) — but under noise it degraded the *most* of the three (error grows to 0.85,
vs. 0.56 and 0.60 for the others), flipping it from best to worst performer. That's a
direct, device-relevant tradeoff (circuit depth vs. noise resilience) backed by
multi-trial statistics, not a single-run anecdote.

Run: `py -3.14 vqe_materials_heisenberg.py`
Results: `results/vqe_heisenberg_results.json`
Plots: `results/vqe_convergence.png`, `results/vqe_ansatz_noise_comparison.png`,
`results/vqe_optimizer_comparison.png`

## 2. `photonic_qft_mesh.py` — Photonic realization of the QFT

Bridges gate-model and photonic quantum computing:
1. Takes Qiskit's `QFTGate(3)` and confirms it equals the standard 8-point DFT matrix
   (to machine precision, once the sign convention is matched — Qiskit uses +2πi, not
   the textbook −2πi).
2. Decomposes that 8x8 unitary into a **Reck-style triangular interferometer mesh**
   (28 two-mode blocks + 8 phase shifters) via complex Givens rotations, implemented
   from first principles (not a library black box — Perceval's own `Circuit.decomposition`
   solver failed to converge on this matrix, so the decomposition math here is a working
   from-scratch implementation of the classic Reck et al. 1994 scheme).
3. Builds the mesh as an actual Perceval circuit and verifies its computed unitary
   matches the target DFT matrix to ~1e-15.
4. Simulates single-photon injection (path encoding) into each of the 8 input modes via
   Perceval's SLOS backend and confirms the output-mode probability distribution matches
   the ideal |U|^2 to ~1e-16 total variation distance, for every input mode.

Run: `py -3.14 photonic_qft_mesh.py`
Results: `results/photonic_qft_mesh_results.json`
Plot: `results/photonic_qft_mesh_schematic.png` (triangular mesh layout, 28 two-mode
blocks across 8 modes)

## 3. `mesh_vqe_ansatz.py` — An original ansatz: Mesh-VQE

This is not a reproduction of anything — it's a new ansatz design that combines
parts 1 and 2 of this project rather than either alone.

**Design rationale:** the Heisenberg Hamiltonian conserves total magnetization
(sum of Z), so its ground state lives in a single fixed-Hamming-weight symmetry
sector — but the standard ansätze benchmarked in part 1 (EfficientSU2,
RealAmplitudes) ignore this and explore the *full* Hilbert space, wasting
optimizer effort on states the true ground state can never be in. This ansatz
instead places Qiskit's `XXPlusYYGate` — the qubit-circuit analog of a
photonic beamsplitter (identity on \|00⟩/\|11⟩, a rotation within
\{\|01⟩,\|10⟩\}) — on the exact same triangular connectivity pattern used to
build the photonic interferometer mesh in part 2. Because this gate exactly
preserves Hamming weight, the ansatz is *structurally* confined to the correct
symmetry sector throughout optimization — a provable, testable advantage the
standard ansätze don't have.

**Result, run through the same rigorous 5-trial framework as part 1:**

| | mean energy error vs. exact |
|---|---|
| Mesh ansatz (noiseless) | **0.14** — best of all ansätze tested, and with std = 0.0003 (vs. 0.04–0.15 for the others) — dramatically more consistent convergence |
| Mesh ansatz (noisy) | 1.44 — worse than *every* standard ansatz under the same noise model |
| Symmetry check: ⟨Hamming weight⟩ | Mesh = exactly 2.000000 (as designed); EfficientSU2 = 1.93 (not conserved) |

**The real finding is the tradeoff, not just "which wins":** restricting the
ansatz to the correct symmetry sector makes noiseless optimization both more
accurate and far more reliable — but it also makes the ansatz *more* fragile
under noise, not less. Depolarizing noise doesn't respect the Hamming-weight
symmetry the ansatz relies on, so noise directly attacks the mechanism that
made it accurate in the first place. That's a genuine, non-obvious insight
about symmetry-preserving ansätze on NISQ hardware, not something copied from
a paper — it fell out of actually building and testing the idea.

Run: `py -3.14 mesh_vqe_ansatz.py` (run part 1 first — this script loads its
results JSON for the side-by-side comparison plot)
Results: `results/mesh_vqe_results.json`
Plots: `results/mesh_vs_standard_ansatze.png`, `results/symmetry_conservation.png`,
`results/mesh_convergence.png`

## Environment

All three scripts run under **Python 3.14** (`py -3.14`). Install pinned dependencies with:
```
py -3.14 -m pip install -r requirements.txt
```
No IBM Quantum account or real hardware access is required — everything above runs
on local simulators (Qiskit's `StatevectorEstimator`/Aer, Perceval's SLOS backend).

## Follow-on project

Part 2 above was extended into a separate project reproducing and cross-validating
arXiv:2608.09509 ("Gate-based emulation of boson sampling using photonic qubits",
IISc Bengaluru) — see the `boson-sampling-gate-emulation` repo. That project
implemented the paper's approach, found and documented a real physics limitation
in it via cross-validation, fixed it with a verified-correct alternative
construction, and extended the paper's 4-mode demonstration to N=5 and N=6 modes.

## IBM Quantum account status

An IBM Quantum Platform account and API key exist for this work
(`va12.sk2024@gmail.com`), created during this project but **not fully activated** —
IBM requires a credit card for identity verification to provision a compute
instance (debit cards were not accepted), which wasn't available at the time.
Nothing in either project currently depends on this; both run entirely on local
simulators. See the `boson-sampling-gate-emulation` repo's README
("Future directions") for what real-hardware access would be used for if this
gets resolved later (a genuine cross-platform validation: the original boson-sampling
paper used photonic-qubit hardware, running the same construction on IBM's
superconducting hardware would be a new comparison point).
