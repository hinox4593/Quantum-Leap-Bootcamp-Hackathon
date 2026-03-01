
# Quantum Diagnostic Framework — Complete Code Guide

## Setup (One-Time Installation)
```bash
pip install qiskit qiskit-aer matplotlib numpy
```

---

## Files — Run in Order

| File | What It Does | Libraries Used |
|------|-------------|----------------|
| `part1_classical_noise.py` | Classical bit-flip noise, BER vs noise level | numpy, matplotlib |
| `part2_quantum_noise.py` | Qubit depolarizing noise, Bloch sphere visualization | qiskit, qiskit-aer, matplotlib |
| `part3_teleportation_noise.py` | Full teleportation circuit with noise injection | qiskit, qiskit-aer, matplotlib |
| `part4_swap_test.py` | Swap Test (quantum fingerprainting) circuit + results | qiskit, qiskit-aer, matplotlib |
| `part5_full_pipeline.py` | ⭐ Full dashboard — all metrics + QEC proposal | qiskit-aer, matplotlib |

> **Part 1 runs WITHOUT Qiskit** (pure numpy). Run this first to verify your environment.

---

## Quick Run (Just Part 5 — Full Dashboard)
```bash
python part5_full_pipeline.py
```
This does not use Qiskit circuits directly (uses analytical simulation), so it runs fastest.

---

## What Each Part Produces

### Part 1 — Classical Noise
- BER vs noise probability curve
- Side-by-side original vs received message heatmap
- Shows that classical noise = just bit flipping, predictable

### Part 2 — Quantum Noise (Bloch Sphere)
- 5 Bloch sphere snapshots showing state degradation
- Fidelity vs noise curve
- Bloch vector magnitude shrinkage (decoherence)
- **Key insight**: quantum noise rotates AND shrinks the state

### Part 3 — Quantum Teleportation
- Full 3-qubit teleportation circuit
- Noise injected on Bob's qubit (q2) via depolarizing channel
- Fidelity of received state vs original
- Density matrix before/after noise
- **Key insight**: noise breaks the entanglement that teleportation relies on

### Part 4 — Swap Test
- Swap Test circuit: ancilla + reference + received qubits
- P(ancilla=0) probability vs noise level
- Similarity score = 2·P(0) − 1
- **Key insight**: we compare states without measuring (destroying) them

### Part 5 — Full Dashboard
- Classical BER vs Quantum Fidelity overlay
- Swap Test noise detection zones (Safe / Warning / Danger)
- Complete metrics table at 6 noise levels
- Theoretical QEC proposal: Shor/Steane codes, Entanglement Purification, Quantum Repeaters

---

## Core Math Reference

### Depolarizing Channel
```
ρ_noisy = (1-p)·ρ + (p/3)·(X·ρ·X† + Y·ρ·Y† + Z·ρ·Z†)
```
- p=0   → No noise, state unchanged
- p=3/4 → Completely mixed state (no information)

### Fidelity (for pure reference state ρ₁)
```
F(ρ₁, ρ₂) = Tr(ρ₁ · ρ₂)
```
- F=1.0 → Identical states
- F=0.5 → Random noise threshold
- F=0.25 → Completely mixed state

### Swap Test
```
P(ancilla = |0⟩) = (1 + Tr(ρ₁·ρ₂)) / 2
Similarity = 2·P(0) - 1 = Tr(ρ₁·ρ₂)
```
- Similarity=1.0 → States identical
- Similarity=0.0 → States orthogonal (completely different)

### Bloch Vector (Decoherence indicator)
```
|v| = 1   → Pure state (no noise)
|v| = 0   → Completely mixed (maximum noise)
```

---

## Theoretical Proposed Solution (What to Present)

The code simulates and detects noise. For the solution proposal:

**1. Shor Code (9-qubit QEC)**
- Encodes 1 logical qubit into 9 physical qubits
- Corrects arbitrary single-qubit errors
- Uses syndrome measurement to detect errors without collapsing state

**2. Steane Code (7-qubit QEC)**
- More efficient — 7 physical qubits per logical qubit
- Based on classical Hamming [7,4,3] code
- Corrects all single-qubit Pauli errors

**3. Entanglement Purification**
- Takes N noisy Bell pairs → M < N high-fidelity pairs
- Uses LOCC (Local Operations + Classical Communication)
- Directly fixes the degraded Bell pairs that cause teleportation errors

**4. Quantum Repeater**
- Divides channel into short segments
- Refreshes entanglement at intermediate nodes
- Enables long-distance quantum communication

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: qiskit` | `pip install qiskit qiskit-aer` |
| Slow Part 4 (Swap Test) | Reduce `shots=8192` to `shots=1024` |
| Plot not showing | Add `plt.ion()` at top, or use `plt.savefig()` only |
| Part 3 if_else error | Use Qiskit ≥ 1.0: `pip install qiskit --upgrade` |
