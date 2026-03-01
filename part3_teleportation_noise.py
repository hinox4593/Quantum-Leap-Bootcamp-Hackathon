"""
=============================================================
PART 3 — QUANTUM TELEPORTATION WITH DEPOLARIZING NOISE
=============================================================
Implements full quantum teleportation protocol and injects
depolarizing noise to show how it degrades the teleported state.

Circuit structure:
  q0 = Alice's message qubit (state to teleport)
  q1 = Alice's half of Bell pair
  q2 = Bob's half of Bell pair  ← noise injected here
  c0, c1 = classical bits (measurement results)

Install requirements:
    pip install qiskit qiskit-aer matplotlib numpy
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import (
    Statevector, DensityMatrix, state_fidelity, partial_trace
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ──────────────────────────────────────────────────────────
# 1. BUILD TELEPORTATION CIRCUIT (no noise — ideal)
# ──────────────────────────────────────────────────────────

def build_teleportation_circuit(theta=np.pi/3, phi=np.pi/4, noise_prob=0.0):
    """
    Full quantum teleportation circuit.

    Alice wants to teleport state: cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

    Steps:
      1. Prepare Alice's state on q0
      2. Create Bell pair on (q1, q2)
      3. [NOISE] Apply depolarizing noise on q2 (channel degradation)
      4. Alice performs Bell measurement on (q0, q1)
      5. Bob applies corrections on q2 based on classical bits
    """
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)

    # ── Step 1: Prepare message state on q0 ──
    qc.ry(theta, qr[0])          # rotate by theta
    qc.rz(phi,   qr[0])          # phase by phi
    qc.barrier(label='prepare')

    # ── Step 2: Create Bell pair (entanglement) on q1, q2 ──
    qc.h(qr[1])                  # Hadamard on q1
    qc.cx(qr[1], qr[2])          # CNOT: q1→q2  (creates |Φ+⟩)
    qc.barrier(label='entangle')

    # ── Step 3: (Noise is injected via noise model — see below) ──
    # We mark q2 as the noisy qubit with an identity gate
    # The noise model will attach depolarizing noise to this gate
    qc.id(qr[2])                 # identity on q2 — noise attached here
    qc.barrier(label='noise')

    # ── Step 4: Alice's Bell Measurement ──
    qc.cx(qr[0], qr[1])         # CNOT: q0→q1
    qc.h(qr[0])                  # Hadamard on q0
    qc.barrier(label='bell_meas')
    qc.measure(qr[0], cr[0])    # measure q0 → c0
    qc.measure(qr[1], cr[1])    # measure q1 → c1
    qc.barrier(label='measure')

    # ── Step 5: Bob's Corrections ──
    with qc.if_else((cr[1], 1),  # if c1 == 1
                    lambda qc, qr, cr: qc.x(qr[2]),   # apply X gate
                    None, [qr[2]], []):
        pass
    with qc.if_else((cr[0], 1),  # if c0 == 1
                    lambda qc, qr, cr: qc.z(qr[2]),   # apply Z gate
                    None, [qr[2]], []):
        pass

    return qc


def build_teleportation_circuit_v2(theta=np.pi/3, phi=np.pi/4):
    """
    Simplified teleportation — use statevector simulation.
    We manually simulate the protocol step by step to track the state.
    This avoids if_else issues with some Qiskit versions.
    """
    # We'll simulate it manually using DensityMatrix evolution
    pass


# ──────────────────────────────────────────────────────────
# 2. MANUAL TELEPORTATION SIMULATION WITH NOISE
#    (More transparent — we track each step explicitly)
# ──────────────────────────────────────────────────────────

def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def hadamard():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def cnot():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

def state_to_dm(state_vec):
    """State vector → density matrix."""
    sv = np.array(state_vec, dtype=complex)
    sv = sv / np.linalg.norm(sv)
    return np.outer(sv, sv.conj())

def apply_gate_to_qubit(dm_3qubit, gate_2x2, qubit_idx, n_qubits=3):
    """Apply a single-qubit gate to qubit_idx of an n-qubit density matrix."""
    gates = [np.eye(2, dtype=complex)] * n_qubits
    gates[qubit_idx] = gate_2x2
    full_gate = gates[0]
    for g in gates[1:]:
        full_gate = np.kron(full_gate, g)
    return full_gate @ dm_3qubit @ full_gate.conj().T

def apply_cnot_2q(dm_2qubit):
    """Apply CNOT to a 2-qubit density matrix."""
    C = cnot()
    return C @ dm_2qubit @ C.conj().T

def apply_depolarizing_single(dm_1qubit, p):
    """Apply depolarizing channel to a single-qubit density matrix."""
    I = np.eye(2, dtype=complex)
    X, Y, Z = pauli_x(), pauli_y(), pauli_z()
    result  = (1 - p) * dm_1qubit
    result += (p / 3) * (X @ dm_1qubit @ X)
    result += (p / 3) * (Y @ dm_1qubit @ Y)
    result += (p / 3) * (Z @ dm_1qubit @ Z)
    return result

def teleport_with_noise(theta, phi, noise_prob):
    """
    Simulate quantum teleportation with depolarizing noise on the channel.
    Returns fidelity of Bob's received state vs Alice's original state.
    """
    # ── Alice's message state ──
    alpha = np.cos(theta / 2)
    beta  = np.exp(1j * phi) * np.sin(theta / 2)
    msg_state = np.array([alpha, beta], dtype=complex)
    msg_dm    = state_to_dm(msg_state)   # 2x2

    # ── Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2 ──
    bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    bell_dm    = state_to_dm(bell_state)   # 4x4

    # ── 3-qubit system: q0 ⊗ (q1,q2) ──
    full_dm = np.kron(msg_dm, bell_dm)  # 8x8

    # ── NOISE: Apply depolarizing on q2 (Bob's qubit) ──
    # Trace out q0 and q1 to get q2's reduced state
    # Then apply noise, then reconstruct
    # Simpler: apply noise directly on q2 in full 3-qubit space
    # Using: ρ_noisy = (1-p)ρ + (p/3)(I⊗I⊗X ρ I⊗I⊗X + ...)
    I2 = np.eye(2, dtype=complex)
    I4 = np.eye(4, dtype=complex)
    X, Y, Z = pauli_x(), pauli_y(), pauli_z()

    def noise_op(single_pauli):
        return np.kron(I4, single_pauli)

    full_dm_noisy  = (1 - noise_prob) * full_dm
    full_dm_noisy += (noise_prob / 3) * (noise_op(X) @ full_dm @ noise_op(X))
    full_dm_noisy += (noise_prob / 3) * (noise_op(Y) @ full_dm @ noise_op(Y))
    full_dm_noisy += (noise_prob / 3) * (noise_op(Z) @ full_dm @ noise_op(Z))

    # ── Alice's Bell measurement on (q0, q1) ──
    # Bell basis states
    bell_basis = [
        np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),   # |Φ+⟩
        np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # |Φ-⟩
        np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),   # |Ψ+⟩
        np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # |Ψ-⟩
    ]

    # Bob's corrections for each Bell measurement outcome
    corrections = [
        np.eye(2, dtype=complex),    # |Φ+⟩ → no correction
        np.array([[1,0],[0,-1]], dtype=complex),  # |Φ-⟩ → Z
        np.array([[0,1],[1,0]], dtype=complex),   # |Ψ+⟩ → X
        np.array([[0,-1j],[1j,0]], dtype=complex) # |Ψ-⟩ → iY
    ]

    # Perform projective measurement + correction
    bob_dm_final = np.zeros((2, 2), dtype=complex)

    for bell_vec, correction in zip(bell_basis, corrections):
        # Projection operator on Alice's qubits (q0, q1)
        proj_alice = np.outer(bell_vec, bell_vec.conj())  # 4x4
        proj_full  = np.kron(proj_alice, np.eye(2, dtype=complex))  # 8x8

        # Post-measurement state (unnormalized)
        projected = proj_full @ full_dm_noisy @ proj_full.conj().T

        # Probability of this outcome
        prob = np.real(np.trace(projected))
        if prob < 1e-10:
            continue

        # Bob's state = partial trace over Alice's qubits
        # Reshape 8x8 → trace out first 4 dimensions
        proj_normalized = projected / prob
        # Bob's reduced state
        bob_state = np.zeros((2, 2), dtype=complex)
        for i in range(4):
            for j in range(4):
                bob_state += proj_normalized[i*2:(i+1)*2, j*2:(j+1)*2] * (1 if i == j else 0)

        # Apply Bob's correction
        bob_corrected = correction @ bob_state @ correction.conj().T

        # Weight by probability
        bob_dm_final += prob * bob_corrected

    # ── Fidelity: Bob's state vs Alice's original ──
    fidelity = np.real(np.trace(
        np.linalg.matrix_power(
            np.linalg.cholesky(msg_dm + 1e-10 * np.eye(2)) @ bob_dm_final @
            np.linalg.cholesky(msg_dm + 1e-10 * np.eye(2)).conj().T,
            1  # approximate
        )
    ))

    # Simpler fidelity: F = Tr(ρ_orig @ ρ_bob) for pure original state
    fidelity_simple = np.real(np.trace(msg_dm @ bob_dm_final))
    return float(fidelity_simple), bob_dm_final, msg_dm


# ──────────────────────────────────────────────────────────
# 3. RUN TELEPORTATION AT MULTIPLE NOISE LEVELS
# ──────────────────────────────────────────────────────────

theta = np.pi / 3     # message state angle
phi   = np.pi / 4     # message state phase

noise_probs   = np.linspace(0, 0.75, 80)
fid_results   = []

print("Running teleportation simulation...")
print(f"Message state: θ={theta:.3f}, φ={phi:.3f}")
print(f"  α = {np.cos(theta/2):.4f}")
print(f"  β = {np.exp(1j*phi)*np.sin(theta/2):.4f}")
print()

for p in noise_probs:
    fid, _, _ = teleport_with_noise(theta, phi, p)
    fid_results.append(fid)

# Specific snapshots
snap_probs  = [0.0, 0.15, 0.30, 0.50, 0.75]
snap_labels = ['No Noise', 'Low (15%)', 'Medium (30%)', 'High (50%)', 'Max (75%)']
snap_colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
snap_fids   = []
snap_bob_dms = []

for p in snap_probs:
    fid, bob_dm, orig_dm = teleport_with_noise(theta, phi, p)
    snap_fids.append(fid)
    snap_bob_dms.append(bob_dm)
    print(f"  Noise={p:.2f}  |  Fidelity={fid:.4f}  |  "
          f"Bob's diagonal: [{bob_dm[0,0].real:.3f}, {bob_dm[1,1].real:.3f}]")

# ──────────────────────────────────────────────────────────
# 4. PLOTTING
# ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Quantum Teleportation with Depolarizing Noise\n"
             "Tracking State Degradation from Alice to Bob",
             fontsize=15, fontweight='bold')

# ── Plot 1: Fidelity vs Noise (main result) ──
ax = axes[0, 0]
ax.plot(noise_probs, fid_results, color='steelblue', linewidth=2.5,
        label='Teleportation Fidelity')
ax.fill_between(noise_probs, fid_results, alpha=0.15, color='steelblue')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5,
           label='Random guess threshold (0.5)')
ax.axhline(1.0, color='green', linestyle=':', linewidth=1.0, alpha=0.5,
           label='Perfect fidelity (1.0)')
for p, fid, color in zip(snap_probs, snap_fids, snap_colors):
    ax.scatter([p], [fid], color=color, s=80, zorder=5)
ax.set_xlabel("Noise Probability p", fontsize=11)
ax.set_ylabel("Fidelity F(Alice, Bob)", fontsize=11)
ax.set_title("Teleportation Fidelity vs Channel Noise", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.75)
ax.set_ylim(0.2, 1.05)

# ── Plot 2: Original state density matrix ──
_, _, orig_dm = teleport_with_noise(theta, phi, 0.0)
ax = axes[0, 1]
dm_display = np.abs(orig_dm)
im = ax.imshow(dm_display, cmap='Blues', vmin=0, vmax=0.6)
ax.set_title("Alice's Original State\nDensity Matrix |ρ|", fontsize=12)
ax.set_xticks([0, 1]); ax.set_xticklabels(['|0⟩', '|1⟩'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['|0⟩', '|1⟩'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{orig_dm[i,j].real:.3f}",
                ha='center', va='center', fontsize=12,
                color='white' if dm_display[i,j] > 0.3 else 'black',
                fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# ── Plot 3: Bob's received state at max noise ──
_, bob_max_noise, _ = teleport_with_noise(theta, phi, 0.75)
ax = axes[0, 2]
dm_display_bob = np.abs(bob_max_noise)
im2 = ax.imshow(dm_display_bob, cmap='Reds', vmin=0, vmax=0.6)
ax.set_title("Bob's Received State (p=0.75)\nDensity Matrix |ρ_noisy|", fontsize=12)
ax.set_xticks([0, 1]); ax.set_xticklabels(['|0⟩', '|1⟩'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['|0⟩', '|1⟩'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{bob_max_noise[i,j].real:.3f}",
                ha='center', va='center', fontsize=12,
                color='white' if dm_display_bob[i,j] > 0.3 else 'black',
                fontweight='bold')
plt.colorbar(im2, ax=ax, fraction=0.046)

# ── Plot 4: Fidelity snapshots bar chart ──
ax = axes[1, 0]
bars = ax.bar(snap_labels, snap_fids, color=snap_colors, edgecolor='black',
              linewidth=1.2, alpha=0.85)
ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
for bar, fid in zip(bars, snap_fids):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{fid:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel("Fidelity", fontsize=11)
ax.set_title("Fidelity at Different Noise Levels", fontsize=12)
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
ax.tick_params(axis='x', labelsize=9)

# ── Plot 5: State coherence (off-diagonal) vs noise ──
ax = axes[1, 1]
coherences = []
for p in noise_probs:
    _, bob_dm, _ = teleport_with_noise(theta, phi, p)
    coherences.append(abs(bob_dm[0, 1]))   # |ρ_01| = coherence

ax.plot(noise_probs, coherences, color='purple', linewidth=2.5,
        label='|ρ₀₁| Coherence (Bob\'s state)')
ax.fill_between(noise_probs, coherences, alpha=0.15, color='purple')
orig_coherence = abs(orig_dm[0, 1])
ax.axhline(orig_coherence, color='green', linestyle='--', linewidth=1.5,
           label=f'Original coherence = {orig_coherence:.3f}')
ax.set_xlabel("Noise Probability p", fontsize=11)
ax.set_ylabel("|ρ₀₁| (Off-diagonal element)", fontsize=11)
ax.set_title("Quantum Coherence Degradation\n(Decoherence Effect)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 6: Teleportation circuit diagram (text representation) ──
ax = axes[1, 2]
ax.axis('off')
circuit_text = """
QUANTUM TELEPORTATION CIRCUIT
══════════════════════════════════════════

  q0 ─[Ry(θ)]─[Rz(φ)]─────────■──[H]──M ─────
                                │              │
  q1 ────────────────[H]──■────⊕──────M  │    │
                           │          │  │    │
  q2 ──────────────────────⊕──[NOISE]─┼──┼────[X?]─[Z?]→ Bob
                                       │  │
  c1 ────────────────────────────────[M]─│────────────
  c0 ──────────────────────────────────[M]────────────

  [NOISE] = Depolarizing Channel
         ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)

  Bob's Corrections:
  ├─ if c0=1 → apply Z gate
  └─ if c1=1 → apply X gate
"""
ax.text(0.05, 0.95, circuit_text, transform=ax.transAxes,
        fontsize=8, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title("Circuit Overview", fontsize=12)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/part3_teleportation_noise.png",
            dpi=150, bbox_inches='tight')
plt.show()

print()
print("=" * 60)
print("  TELEPORTATION SIMULATION SUMMARY")
print("=" * 60)
print("  Perfect teleportation (p=0): Fidelity = 1.0")
print("  At p=0.75 noise: State is nearly completely mixed")
print()
print("  The off-diagonal density matrix elements (coherence)")
print("  shrink to zero as noise increases — this is DECOHERENCE,")
print("  the main enemy of quantum communication.")
print("=" * 60)
