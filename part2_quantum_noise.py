"""
=============================================================
PART 2 — QUANTUM NOISE SIMULATION (Depolarizing Channel)
=============================================================
Simulates how quantum noise affects a qubit state.
- Visualizes qubit states on the Bloch sphere
- Shows how depolarizing noise shrinks the Bloch vector
- Compares fidelity of noisy states vs the original
- Plots state degradation as noise increases

Install requirements:
    pip install qiskit qiskit-aer matplotlib numpy
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector, DensityMatrix, state_fidelity, random_statevector
)
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

# ──────────────────────────────────────────────────────────
# HELPER: Bloch Vector from Density Matrix
# ──────────────────────────────────────────────────────────

def density_matrix_to_bloch(rho):
    """
    Extract Bloch vector (x, y, z) from a single-qubit density matrix.
    rho = (I + x*X + y*Y + z*Z) / 2
    """
    rho_arr = np.array(rho.data)
    x = 2 * np.real(rho_arr[0, 1])
    y = 2 * np.imag(rho_arr[1, 0])
    z = np.real(rho_arr[0, 0] - rho_arr[1, 1])
    return np.array([x, y, z])


# ──────────────────────────────────────────────────────────
# HELPER: Draw Bloch Sphere
# ──────────────────────────────────────────────────────────

def draw_bloch_sphere(ax, title=""):
    """Draw a clean Bloch sphere on a 3D matplotlib axis."""
    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.05, color='lightblue')
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray', linewidth=0.4)

    # Axes
    ax.quiver(0, 0, 0, 1.4, 0, 0, color='gray', alpha=0.4, arrow_length_ratio=0.1, linewidth=0.8)
    ax.quiver(0, 0, 0, 0, 1.4, 0, color='gray', alpha=0.4, arrow_length_ratio=0.1, linewidth=0.8)
    ax.quiver(0, 0, 0, 0, 0, 1.4, color='gray', alpha=0.4, arrow_length_ratio=0.1, linewidth=0.8)

    # Labels
    ax.text(1.6, 0, 0, 'X', fontsize=9, color='gray')
    ax.text(0, 1.6, 0, 'Y', fontsize=9, color='gray')
    ax.text(0, 0, 1.6, '|0⟩', fontsize=10, color='black', fontweight='bold')
    ax.text(0, 0, -1.7, '|1⟩', fontsize=10, color='black', fontweight='bold')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=4)


# ──────────────────────────────────────────────────────────
# 1. DEFINE ORIGINAL STATE
# ──────────────────────────────────────────────────────────

np.random.seed(42)

# |+⟩ state = (|0⟩ + |1⟩) / √2  — a superposition state on equator of Bloch sphere
original_state = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
original_dm    = DensityMatrix(original_state)
original_bloch = density_matrix_to_bloch(original_dm)

print("Original state: |+⟩ = (|0⟩ + |1⟩) / √2")
print(f"Original Bloch vector: {original_bloch.round(4)}")
print()

# ──────────────────────────────────────────────────────────
# 2. APPLY DEPOLARIZING NOISE MANUALLY AT VARIOUS LEVELS
# ──────────────────────────────────────────────────────────
# Depolarizing channel: ρ → (1-p)*ρ + (p/3)*(XρX + YρY + ZρZ)
# This shrinks the Bloch vector by factor (1 - 4p/3)

def apply_depolarizing(rho, p):
    """Apply depolarizing noise to density matrix rho with probability p."""
    rho_arr = np.array(rho.data, dtype=complex)

    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    noisy = (1 - p) * rho_arr
    noisy += (p / 3) * (X @ rho_arr @ X)
    noisy += (p / 3) * (Y @ rho_arr @ Y)
    noisy += (p / 3) * (Z @ rho_arr @ Z)

    return DensityMatrix(noisy)


noise_levels = np.linspace(0, 0.75, 200)
fidelities   = []
bloch_magnitudes = []

for p in noise_levels:
    noisy_dm = apply_depolarizing(original_dm, p)
    fid = state_fidelity(original_dm, noisy_dm)
    bloch = density_matrix_to_bloch(noisy_dm)
    fidelities.append(float(fid))
    bloch_magnitudes.append(np.linalg.norm(bloch))

# ──────────────────────────────────────────────────────────
# 3. SPECIFIC NOISE SNAPSHOTS FOR BLOCH SPHERE VISUALIZATION
# ──────────────────────────────────────────────────────────

snapshot_levels = [0.0, 0.1, 0.3, 0.5, 0.75]
snapshot_states = []
snapshot_fids   = []
snapshot_bloch  = []

for p in snapshot_levels:
    dm   = apply_depolarizing(original_dm, p)
    fid  = float(state_fidelity(original_dm, dm))
    blch = density_matrix_to_bloch(dm)
    snapshot_states.append(dm)
    snapshot_fids.append(fid)
    snapshot_bloch.append(blch)
    print(f"  p = {p:.2f}  |  Fidelity = {fid:.4f}  |  Bloch = {blch.round(3)}")

# ──────────────────────────────────────────────────────────
# 4. PLOTTING
# ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 12))
fig.suptitle("Quantum Noise Simulation — Depolarizing Channel on a Qubit",
             fontsize=16, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.1)

# ── Top Row: Bloch Sphere Snapshots ──
colors_snap = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

for idx, (p, blch, fid, color) in enumerate(
        zip(snapshot_levels, snapshot_bloch, snapshot_fids, colors_snap)):
    ax = fig.add_subplot(gs[0, idx], projection='3d')
    draw_bloch_sphere(ax, title=f"p = {p:.2f}\nFidelity = {fid:.3f}")

    # Original state vector (gray reference)
    ox, oy, oz = original_bloch
    ax.quiver(0, 0, 0, ox, oy, oz,
              color='gray', alpha=0.4, arrow_length_ratio=0.15,
              linewidth=1.5, linestyle='dashed')

    # Noisy state vector
    nx, ny, nz = blch
    if np.linalg.norm([nx, ny, nz]) > 1e-6:
        ax.quiver(0, 0, 0, nx, ny, nz,
                  color=color, alpha=0.9, arrow_length_ratio=0.2,
                  linewidth=3)

    # Mark endpoint
    ax.scatter([nx], [ny], [nz], color=color, s=60, zorder=5)

# ── Bottom Left: Fidelity vs Noise ──
ax_fid = fig.add_subplot(gs[1, :3])
ax_fid.plot(noise_levels, fidelities, color='steelblue', linewidth=2.5,
            label='Quantum Fidelity F(ρ_orig, ρ_noisy)')

# Theoretical: F = (1 + (1-4p/3)²·||Bloch||²) / something — approximate overlay
theoretical_fid = [(1 + (1 - 4*p/3)**2) / 2 for p in noise_levels]
ax_fid.plot(noise_levels, theoretical_fid, 'r--', linewidth=1.5,
            alpha=0.7, label='Theoretical F ≈ (1+(1-4p/3)²)/2')

ax_fid.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5,
               label='F=0.5 (random noise threshold)')
ax_fid.axhline(y=1.0, color='green', linestyle=':', linewidth=1.0, alpha=0.5)

# Mark snapshots
for p, fid, color in zip(snapshot_levels, snapshot_fids, colors_snap):
    ax_fid.scatter([p], [fid], color=color, s=80, zorder=5)
    ax_fid.annotate(f"p={p}", xy=(p, fid), xytext=(p+0.01, fid+0.03),
                    fontsize=8, color=color)

ax_fid.set_xlabel("Noise Probability (p)", fontsize=12)
ax_fid.set_ylabel("Fidelity", fontsize=12)
ax_fid.set_title("Quantum State Fidelity vs Depolarizing Noise Level", fontsize=13)
ax_fid.legend(fontsize=10)
ax_fid.grid(True, alpha=0.3)
ax_fid.set_xlim(0, 0.75)
ax_fid.set_ylim(0, 1.05)

# ── Bottom Right: Bloch Vector Magnitude vs Noise ──
ax_bloch = fig.add_subplot(gs[1, 3:])
ax_bloch.plot(noise_levels, bloch_magnitudes, color='purple', linewidth=2.5,
              label='|Bloch vector|')
theoretical_shrink = [abs(1 - 4*p/3) for p in noise_levels]
ax_bloch.plot(noise_levels, theoretical_shrink, 'r--', linewidth=1.5,
              alpha=0.7, label='Theoretical: |1-4p/3|')
ax_bloch.axhline(y=0, color='orange', linestyle=':', linewidth=1.5)
ax_bloch.set_xlabel("Noise Probability (p)", fontsize=12)
ax_bloch.set_ylabel("Bloch Vector Magnitude", fontsize=12)
ax_bloch.set_title("Bloch Vector Shrinkage\n(Purity Loss due to Noise)", fontsize=12)
ax_bloch.legend(fontsize=10)
ax_bloch.grid(True, alpha=0.3)
ax_bloch.set_xlim(0, 0.75)
ax_bloch.set_ylim(-0.05, 1.05)
ax_bloch.fill_between(noise_levels, bloch_magnitudes, alpha=0.15, color='purple')

plt.savefig("/mnt/user-data/outputs/part2_quantum_noise.png",
            dpi=150, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────
# 5. PRINT KEY INSIGHT
# ──────────────────────────────────────────────────────────

print()
print("=" * 60)
print("  QUANTUM NOISE SIMULATION — KEY INSIGHTS")
print("=" * 60)
print("  Quantum noise is fundamentally different from classical:")
print()
print("  1. Classical noise: flips discrete bits (0→1 or 1→0)")
print("     Measurable, detectable, correctable with parity checks.")
print()
print("  2. Quantum noise (Depolarizing): SHRINKS the Bloch vector.")
print("     The qubit state rotates AND loses purity (coherence).")
print("     You CANNOT simply read the qubit to check for errors")
print("     because measurement collapses the superposition!")
print()
print("  3. At p = 0.75, the Bloch vector → 0 (completely mixed state)")
print("     This means the qubit carries NO information — maximum noise.")
print()
print("  Fidelity = 1.0  →  Perfect state (no noise)")
print("  Fidelity = 0.5  →  Random noise threshold")
print("  Fidelity = 0.25 →  Completely mixed / destroyed state")
print("=" * 60)
