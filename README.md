# QMD Fellowship Project — Quantum Materials & Photonic Devices

Two-part demonstration built for the QuantaRiSE Quantum Winter Fellowship application.

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

## Environment

Both scripts run under **Python 3.14** (`py -3.14`). Install pinned dependencies with:
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
