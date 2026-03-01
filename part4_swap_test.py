"""
=============================================================
PART 4 — SWAP TEST (QUANTUM FINGERPRINTING)
=============================================================
Implements the Swap Test circuit to compare two quantum states.
Used to detect noise-induced degradation by comparing:
  - Reference state  (Alice's original)
  - Received state   (Bob's noisy teleported state)

Swap Test Circuit:
  ancilla ──[H]──────■──[H]──M
  ref     ──────[SWAP if ancilla=1]──
  target  ──────────────────────────

  P(0) = (1 + |⟨ref|target⟩|²) / 2
  Fidelity estimate = 2*P(0) - 1 = |⟨ref|target⟩|²

Install requirements:
    pip install qiskit qiskit-aer matplotlib numpy
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import (
    Statevector, DensityMatrix, state_fidelity
)
from qiskit_aer import AerSimulator

# ──────────────────────────────────────────────────────────
# 1. BUILD SWAP TEST CIRCUIT
# ──────────────────────────────────────────────────────────

def build_swap_test_circuit(state_ref, state_target):
    """
    Build the Swap Test circuit.

    Qubits:
      q0 = ancilla qubit (control)
      q1 = reference state
      q2 = target/received state

    state_ref    : list [alpha, beta]  — reference state amplitudes
    state_target : list [alpha, beta]  — received/noisy state amplitudes
    """
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)

    # ── Initialize reference state on q1 ──
    ref_norm = state_ref / np.linalg.norm(state_ref)
    qc.initialize(ref_norm.tolist(), qr[1])

    # ── Initialize target state on q2 ──
    tgt_norm = state_target / np.linalg.norm(state_target)
    qc.initialize(tgt_norm.tolist(), qr[2])

    # ── Swap Test ──
    qc.h(qr[0])              # Hadamard on ancilla
    qc.cswap(qr[0], qr[1], qr[2])   # Controlled-SWAP (Fredkin gate)
    qc.h(qr[0])              # Second Hadamard on ancilla
    qc.measure(qr[0], cr[0]) # Measure ancilla

    return qc


def run_swap_test(state_ref, state_target, shots=8192):
    """
    Run the Swap Test and return similarity score.

    Returns:
      P0        : probability of measuring |0⟩ on ancilla
      similarity: 2*P0 - 1  (= |⟨ref|target⟩|²)
    """
    qc = build_swap_test_circuit(state_ref, state_target)
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    count_0 = counts.get('0', 0)
    count_1 = counts.get('1', 0)
    total   = count_0 + count_1

    P0 = count_0 / total
    similarity = 2 * P0 - 1     # = |⟨ref|target⟩|²
    return P0, max(0, similarity), counts


# ──────────────────────────────────────────────────────────
# 2. APPLY DEPOLARIZING NOISE TO TARGET STATE
# ──────────────────────────────────────────────────────────

def apply_depolarizing(state_vec, p):
    """
    Apply depolarizing noise to a pure state vector.
    Returns a density matrix after noise.
    """
    sv   = np.array(state_vec, dtype=complex)
    sv   = sv / np.linalg.norm(sv)
    rho  = np.outer(sv, sv.conj())

    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)

    noisy  = (1-p) * rho
    noisy += (p/3) * (X @ rho @ X)
    noisy += (p/3) * (Y @ rho @ Y)
    noisy += (p/3) * (Z @ rho @ Z)
    return noisy


def dm_to_statevec_approx(dm):
    """
    Convert density matrix to closest pure state (for use in Swap Test circuit).
    Eigendecompose and take the dominant eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(dm)
    # Largest eigenvalue corresponds to the dominant pure component
    idx = np.argmax(eigenvalues)
    return eigenvectors[:, idx]


# ──────────────────────────────────────────────────────────
# 3. ANALYTICAL SWAP TEST (for verification + density matrices)
# ──────────────────────────────────────────────────────────

def analytical_swap_test(rho1, rho2):
    """
    Analytical version of Swap Test using density matrices.
    P(0) = (1 + Tr(ρ1 ρ2)) / 2
    Similarity = Tr(ρ1 ρ2)
    """
    trace_product = np.real(np.trace(rho1 @ rho2))
    P0 = (1 + trace_product) / 2
    return P0, trace_product


# ──────────────────────────────────────────────────────────
# 4. RUN FULL EXPERIMENT
# ──────────────────────────────────────────────────────────

# Reference state: |+⟩ = (|0⟩+|1⟩)/√2
theta = np.pi / 3
phi   = np.pi / 4
alpha = np.cos(theta / 2)
beta  = np.exp(1j * phi) * np.sin(theta / 2)
ref_state = np.array([alpha, beta], dtype=complex)
ref_dm = np.outer(ref_state, ref_state.conj())

print("=" * 60)
print("  SWAP TEST EXPERIMENT")
print("=" * 60)
print(f"  Reference state: α={alpha:.4f}, β={beta:.4f}")
print()

noise_levels    = np.linspace(0, 0.75, 80)
similarities_analytical = []
P0_values       = []
fidelities_true = []

for p in noise_levels:
    noisy_dm = apply_depolarizing(ref_state, p)
    P0, similarity = analytical_swap_test(ref_dm, noisy_dm)
    similarities_analytical.append(similarity)
    P0_values.append(P0)

    # True fidelity for comparison
    # F = Tr(√(√ρ1 ρ2 √ρ1)) ≈ Tr(ρ1 ρ2) for pure ρ1
    true_fid = float(np.real(np.trace(ref_dm @ noisy_dm)))
    fidelities_true.append(true_fid)

# Specific demo points
demo_probs   = [0.0, 0.1, 0.25, 0.5, 0.75]
demo_labels  = ['p=0\n(identical)', 'p=0.1\n(low noise)', 'p=0.25\n(medium)',
                'p=0.5\n(high)', 'p=0.75\n(max noise)']
demo_colors  = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
demo_sims    = []
demo_P0s     = []

print(f"  {'Noise p':<12} {'P(ancilla=0)':<18} {'Similarity':<18} {'Verdict'}")
print(f"  {'-'*65}")

for p in demo_probs:
    noisy_dm = apply_depolarizing(ref_state, p)
    P0, sim  = analytical_swap_test(ref_dm, noisy_dm)
    demo_P0s.append(P0)
    demo_sims.append(sim)

    verdict = "IDENTICAL" if sim > 0.95 else \
              "VERY SIMILAR" if sim > 0.7 else \
              "MODERATE NOISE" if sim > 0.4 else \
              "HIGH NOISE" if sim > 0.1 else "COMPLETELY DIFFERENT"

    print(f"  {p:<12.2f} {P0:<18.4f} {sim:<18.4f} {verdict}")

# ──────────────────────────────────────────────────────────
# 5. QISKIT CIRCUIT SWAP TEST (for pure states at p=0 and p=0.5)
# ──────────────────────────────────────────────────────────

print()
print("  Running Qiskit Swap Test circuits...")

# p=0: identical states
P0_qiskit_0, sim_qiskit_0, counts_0 = run_swap_test(ref_state, ref_state)
print(f"  p=0.0  → P(0)={P0_qiskit_0:.4f}, Similarity={sim_qiskit_0:.4f}  [Expected: 1.0]")

# p=0.5: noisy target (use dominant eigenvector of noisy DM as approx pure state)
noisy_dm_05 = apply_depolarizing(ref_state, 0.5)
noisy_approx_05 = dm_to_statevec_approx(noisy_dm_05)
P0_qiskit_5, sim_qiskit_5, counts_5 = run_swap_test(ref_state, noisy_approx_05)
print(f"  p=0.5  → P(0)={P0_qiskit_5:.4f}, Similarity={sim_qiskit_5:.4f}  [Expected: ~{demo_sims[3]:.3f}]")

# Completely orthogonal state (|1⟩ vs |0⟩)
state_0 = np.array([1, 0], dtype=complex)
state_1 = np.array([0, 1], dtype=complex)
P0_ortho, sim_ortho, counts_ortho = run_swap_test(state_0, state_1)
print(f"  |0⟩vs|1⟩ → P(0)={P0_ortho:.4f}, Similarity={sim_ortho:.4f}  [Expected: 0.0]")

# ──────────────────────────────────────────────────────────
# 6. PLOTTING
# ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Swap Test — Quantum Fingerprinting for Noise Detection\n"
             "Comparing Reference State vs Noisy Received State",
             fontsize=15, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: P(0) vs Noise Level ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(noise_levels, P0_values, color='steelblue', linewidth=2.5,
         label='P(ancilla=|0⟩) — Swap Test output')
ax1.fill_between(noise_levels, P0_values, 0.5, alpha=0.2, color='steelblue',
                 label='Detection region')
ax1.axhline(1.0, color='green', linestyle='--', linewidth=1.5,
            label='P=1.0 → States IDENTICAL')
ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.5,
            label='P=0.5 → States ORTHOGONAL (completely different)')
ax1.axhline(0.75, color='orange', linestyle=':', linewidth=1.5,
            label='P=0.75 → Midpoint (partial overlap)')

for p, P0, color in zip(demo_probs, demo_P0s, demo_colors):
    ax1.scatter([p], [P0], color=color, s=90, zorder=5)
    ax1.annotate(f"p={p}", xy=(p, P0), xytext=(p+0.01, P0+0.01),
                 fontsize=9, color=color)

# Qiskit circuit results
ax1.scatter([0.0], [P0_qiskit_0], marker='*', color='black', s=200,
            zorder=6, label=f'Qiskit: p=0.0 → P={P0_qiskit_0:.3f}')
ax1.scatter([0.5], [P0_qiskit_5], marker='*', color='brown', s=200,
            zorder=6, label=f'Qiskit: p=0.5 → P={P0_qiskit_5:.3f}')

ax1.set_xlabel("Noise Probability (p)", fontsize=12)
ax1.set_ylabel("P(ancilla = |0⟩)", fontsize=12)
ax1.set_title("Swap Test Output Probability vs Noise Level", fontsize=13)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 0.75)
ax1.set_ylim(0.45, 1.05)

# ── Plot 2: Measurement histogram from Qiskit (p=0 case) ──
ax2 = fig.add_subplot(gs[0, 2])
bars_0 = ax2.bar(['|0⟩ (Same)', '|1⟩ (Diff)'],
                  [counts_0.get('0', 0), counts_0.get('1', 0)],
                  color=['#27ae60', '#e74c3c'], edgecolor='black', alpha=0.85)
for bar in bars_0:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"{int(bar.get_height())}", ha='center', va='bottom',
             fontsize=11, fontweight='bold')
ax2.set_title("Qiskit Swap Test Counts\n(p=0: Identical States)", fontsize=12)
ax2.set_ylabel("Shot Count", fontsize=11)
ax2.grid(True, axis='y', alpha=0.3)

# ── Plot 3: Similarity score bar chart ──
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(range(len(demo_probs)), demo_sims,
                color=demo_colors, edgecolor='black', alpha=0.85)
ax3.set_xticks(range(len(demo_probs)))
ax3.set_xticklabels([f"p={p}" for p in demo_probs], fontsize=9)
ax3.axhline(0.9, color='green', linestyle='--', alpha=0.6, label='> 0.9 = High similarity')
ax3.axhline(0.5, color='orange', linestyle='--', alpha=0.6, label='< 0.5 = High noise')
for bar, sim in zip(bars3, demo_sims):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{sim:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylabel("Similarity Score", fontsize=11)
ax3.set_title("Swap Test Similarity Score\nat Different Noise Levels", fontsize=12)
ax3.legend(fontsize=9)
ax3.set_ylim(0, 1.1)
ax3.grid(True, axis='y', alpha=0.3)

# ── Plot 4: Similarity vs Fidelity comparison ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(noise_levels, similarities_analytical, color='steelblue', linewidth=2.5,
         label='Swap Test Similarity: Tr(ρ₁ρ₂)')
ax4.plot(noise_levels, fidelities_true, 'r--', linewidth=2,
         label='True Fidelity F(ρ₁,ρ₂)', alpha=0.8)
ax4.fill_between(noise_levels,
                  similarities_analytical,
                  fidelities_true,
                  alpha=0.15, color='purple',
                  label='Difference region')
ax4.set_xlabel("Noise Probability p", fontsize=11)
ax4.set_ylabel("Score", fontsize=11)
ax4.set_title("Swap Test Similarity vs True Fidelity\n(Comparison)", fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 0.75)

# ── Plot 5: Swap test circuit diagram ──
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
circuit_text = """
  SWAP TEST CIRCUIT
  ═══════════════════════════════

  ancilla ─[H]──────■──────[H]──M→0 or 1
                     │
  |ref⟩   ─────[CSWAP if anc=1]──
                     │
  |tgt⟩   ──────────┘

  P(measure 0) = (1 + |⟨ref|target⟩|²) / 2

  DECODING THE OUTPUT:
  ┌─────────────────────────────────────┐
  │ P(0) = 1.0  → States IDENTICAL     │
  │ P(0) = 0.75 → 50% overlap          │
  │ P(0) = 0.5  → States ORTHOGONAL    │
  └─────────────────────────────────────┘

  Similarity = 2·P(0) − 1 = |⟨ψ_ref|ψ_tgt⟩|²

  This is QUANTUM FINGERPRINTING:
  You can verify if two states match
  WITHOUT knowing what they are!
  (Uses the No-Cloning theorem cleverly)
"""
ax5.text(0.02, 0.98, circuit_text, transform=ax5.transAxes,
         fontsize=8.5, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax5.set_title("Swap Test — How It Works", fontsize=12)

plt.savefig("/mnt/user-data/outputs/part4_swap_test.png",
            dpi=150, bbox_inches='tight')
plt.show()

print()
print("=" * 60)
print("  SWAP TEST SUMMARY")
print("=" * 60)
print("  The Swap Test is our quantum noise DETECTOR.")
print()
print("  How to interpret results:")
print("  ┌────────────────────────────────────────────────┐")
print("  │ Similarity > 0.9  → State intact, no noise    │")
print("  │ Similarity 0.7-0.9→ Minor noise detected      │")
print("  │ Similarity 0.4-0.7→ Moderate noise — verify   │")
print("  │ Similarity < 0.4  → Heavy noise — retransmit  │")
print("  └────────────────────────────────────────────────┘")
print()
print("  Key insight: We never measured the actual state.")
print("  We only compared it — protecting quantum information!")
print("=" * 60)
