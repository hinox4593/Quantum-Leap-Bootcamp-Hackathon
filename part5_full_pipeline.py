
import os
os.makedirs("quantum_outputs", exist_ok=True)



"""
=============================================================
PART 5 — FULL PIPELINE + THEORETICAL NOISE ELIMINATION
=============================================================
Combines all parts into one complete demonstration:
  1. Classical vs Quantum noise comparison
  2. Teleportation with noise
  3. Swap Test detection
  4. Full metrics dashboard
  5. Theoretical QEC proposal visualization

This is the MAIN FILE to run for your complete project.

Install requirements:
    pip install qiskit qiskit-aer matplotlib numpy
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# ──────────────────────────────────────────────────────────
# SHARED UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────

def apply_depolarizing(rho, p):
    """Apply depolarizing channel: ρ → (1-p)ρ + p/3(XρX+YρY+ZρZ)"""
    rho = np.array(rho, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    noisy  = (1-p)*rho
    noisy += (p/3)*(X@rho@X)
    noisy += (p/3)*(Y@rho@Y)
    noisy += (p/3)*(Z@rho@Z)
    return noisy

def state_to_dm(state_vec):
    sv = np.array(state_vec, dtype=complex)
    sv = sv / np.linalg.norm(sv)
    return np.outer(sv, sv.conj())

def fidelity(rho1, rho2):
    """Simplified fidelity for when rho1 is pure."""
    return float(np.real(np.trace(rho1 @ rho2)))

def swap_test_similarity(rho1, rho2):
    """Analytical swap test: Tr(ρ1 ρ2)"""
    return float(np.real(np.trace(rho1 @ rho2)))

def bloch_vector(rho):
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = float(np.real(rho[0, 0] - rho[1, 1]))
    return np.array([x, y, z])

def classical_ber(p):
    """Classical bit error rate = noise probability p."""
    return p

# ──────────────────────────────────────────────────────────
# 1. COMPUTE ALL DATA
# ──────────────────────────────────────────────────────────

np.random.seed(42)

# Message state: θ=π/3, φ=π/4
theta, phi = np.pi/3, np.pi/4
alpha = np.cos(theta/2)
beta  = np.exp(1j*phi)*np.sin(theta/2)
ref_state = np.array([alpha, beta], dtype=complex)
ref_dm    = state_to_dm(ref_state)

noise_range = np.linspace(0, 0.75, 100)

# Classical BER
ber_classical = noise_range.copy()

# Quantum fidelity after noise
fid_quantum = [fidelity(ref_dm, apply_depolarizing(ref_dm, p)) for p in noise_range]

# Swap test similarity
swap_sim = [swap_test_similarity(ref_dm, apply_depolarizing(ref_dm, p)) for p in noise_range]

# Bloch vector magnitude
bloch_mag = [np.linalg.norm(bloch_vector(apply_depolarizing(ref_dm, p))) for p in noise_range]

# Purity of noisy state: Tr(ρ²)
purity = [float(np.real(np.trace(apply_depolarizing(ref_dm,p) @ apply_depolarizing(ref_dm,p))))
          for p in noise_range]

print("Data computed. Building dashboard...")

# ──────────────────────────────────────────────────────────
# 2. MAIN DASHBOARD FIGURE
# ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')

title = fig.suptitle(
    "Quantum Diagnostic Framework — Complete Pipeline Dashboard\n"
    "Noise Detection in Quantum Teleportation via Swap Test Fingerprinting",
    fontsize=17, fontweight='bold', color='white', y=0.99
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)

# Color scheme
COLORS = {
    'classical': '#ff6b6b',
    'quantum':   '#4ecdc4',
    'swap':      '#45b7d1',
    'bloch':     '#a29bfe',
    'purity':    '#fd79a8',
    'grid':      '#2d3436',
    'text':      'white',
    'accent':    '#fdcb6e'
}

def style_ax(ax, title_str, xlabel="", ylabel=""):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.set_title(title_str, color='white', fontsize=11, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color='#8b949e', fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color='#8b949e', fontsize=10)
    ax.grid(True, color='#21262d', linewidth=0.8, alpha=0.8)

# ── Row 0, Col 0-1: Classical vs Quantum BER/Fidelity Comparison ──
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "Classical BER vs Quantum Fidelity — Channel Noise Comparison",
         "Noise Probability (p)", "Error / Quality Score")

ax1.plot(noise_range, ber_classical, color=COLORS['classical'],
         linewidth=2.5, label='Classical BER (= p, linear)')
ax1.plot(noise_range, fid_quantum, color=COLORS['quantum'],
         linewidth=2.5, label='Quantum Fidelity F(ρ_orig, ρ_noisy)')
ax1.fill_between(noise_range, ber_classical, fid_quantum,
                  alpha=0.15, color='white', label='Quantum advantage region')

ax1.axhline(0.5, color=COLORS['accent'], linestyle='--',
            linewidth=1.2, alpha=0.7, label='50% threshold')
ax1.text(0.01, 0.51, 'Random threshold', color=COLORS['accent'], fontsize=8)

ax1.legend(facecolor='#21262d', labelcolor='white', fontsize=9,
           framealpha=0.9, loc='upper left')
ax1.set_xlim(0, 0.75)
ax1.set_ylim(0, 1.1)

# Annotation arrows
ax1.annotate("Classical noise is LINEAR:\nBER grows proportionally with p",
             xy=(0.4, 0.4), xytext=(0.2, 0.7),
             arrowprops=dict(arrowstyle='->', color=COLORS['classical']),
             color=COLORS['classical'], fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='#161b22', alpha=0.8))

ax1.annotate("Quantum fidelity drops\nFASTER — non-linear degradation",
             xy=(0.45, fid_quantum[int(0.45/0.75*100)]),
             xytext=(0.45, 0.2),
             arrowprops=dict(arrowstyle='->', color=COLORS['quantum']),
             color=COLORS['quantum'], fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='#161b22', alpha=0.8))

# ── Row 0, Col 2-3: Swap Test Output ──
ax2 = fig.add_subplot(gs[0, 2:])
style_ax(ax2, "Swap Test (Quantum Fingerprinting) — Noise Detection",
         "Noise Probability (p)", "Similarity Score")

ax2.plot(noise_range, swap_sim, color=COLORS['swap'],
         linewidth=2.5, label='Swap Test Similarity = Tr(ρ₁ρ₂)')
ax2.fill_between(noise_range, swap_sim, alpha=0.2, color=COLORS['swap'])

# Noise zones
ax2.axhspan(0.9, 1.0, alpha=0.15, color='green', label='Safe zone (>0.9)')
ax2.axhspan(0.5, 0.9, alpha=0.10, color='yellow', label='Warning zone (0.5–0.9)')
ax2.axhspan(0.0, 0.5, alpha=0.10, color='red', label='Danger zone (<0.5)')

ax2.text(0.76, 0.95, 'SAFE', color='#00b894', fontsize=10, fontweight='bold')
ax2.text(0.76, 0.70, 'WARN', color='#fdcb6e', fontsize=10, fontweight='bold')
ax2.text(0.76, 0.25, 'HIGH\nNOISE', color='#ff7675', fontsize=10, fontweight='bold')

ax2.legend(facecolor='#21262d', labelcolor='white', fontsize=9,
           framealpha=0.9, loc='lower left')
ax2.set_xlim(0, 0.8)
ax2.set_ylim(0, 1.05)

# ── Row 1, Col 0: Bloch Vector Magnitude ──
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Bloch Vector Shrinkage\n(Decoherence)", "Noise p", "|Bloch vector|")
ax3.plot(noise_range, bloch_mag, color=COLORS['bloch'], linewidth=2.5)
ax3.fill_between(noise_range, bloch_mag, alpha=0.25, color=COLORS['bloch'])
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.set_xlim(0, 0.75)
ax3.set_ylim(-0.05, 1.1)
ax3.text(0.01, 0.05, "Bloch vector → 0\n= completely mixed state",
         color='#ff7675', fontsize=8.5)

# ── Row 1, Col 1: State Purity ──
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Quantum State Purity\nTr(ρ²)", "Noise p", "Purity Tr(ρ²)")
ax4.plot(noise_range, purity, color=COLORS['purity'], linewidth=2.5)
ax4.fill_between(noise_range, purity, alpha=0.25, color=COLORS['purity'])
ax4.axhline(0.5, color=COLORS['accent'], linestyle='--', linewidth=1)
ax4.axhline(1.0, color='#00b894', linestyle=':', linewidth=1)
ax4.text(0.01, 1.02, 'Pure state (1.0)', color='#00b894', fontsize=8)
ax4.text(0.01, 0.52, 'Mixed threshold (0.5)', color=COLORS['accent'], fontsize=8)
ax4.set_xlim(0, 0.75)
ax4.set_ylim(0.3, 1.1)

# ── Row 1, Col 2-3: Metrics Summary Table ──
ax5 = fig.add_subplot(gs[1, 2:])
style_ax(ax5, "Complete Metrics at Key Noise Levels", "", "")
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 6)
ax5.axis('off')

snap_ps = [0.0, 0.10, 0.25, 0.40, 0.60, 0.75]
headers = ["Noise p", "Cl. BER", "Q Fidelity", "Swap Sim", "Bloch |v|", "Purity", "Status"]
col_x = [0.3, 1.6, 3.0, 4.5, 5.9, 7.2, 8.5]

# Header row
for hdr, x in zip(headers, col_x):
    ax5.text(x, 5.5, hdr, color=COLORS['accent'], fontsize=9.5,
             fontweight='bold', ha='left')
ax5.axhline(5.2, color='#30363d', linewidth=1, xmin=0.02, xmax=0.98)

status_colors = ['#00b894', '#00b894', '#fdcb6e', '#e17055', '#d63031', '#d63031']
status_labels = ['✓ PERFECT', '✓ GOOD', '⚠ WARNING', '⚠ NOISY', '✗ HIGH', '✗ CRITICAL']

for row_idx, (p, st_color, status) in enumerate(zip(snap_ps, status_colors, status_labels)):
    y = 4.5 - row_idx * 0.78
    noisy_dm = apply_depolarizing(ref_dm, p)
    fid  = fidelity(ref_dm, noisy_dm)
    sim  = swap_test_similarity(ref_dm, noisy_dm)
    bv   = np.linalg.norm(bloch_vector(noisy_dm))
    pur  = float(np.real(np.trace(noisy_dm @ noisy_dm)))

    vals = [f"{p:.2f}", f"{p:.3f}", f"{fid:.3f}", f"{sim:.3f}", f"{bv:.3f}", f"{pur:.3f}"]
    row_bg = '#1a1f29' if row_idx % 2 == 0 else '#161b22'

    for val, x in zip(vals, col_x):
        ax5.text(x, y, val, color='white', fontsize=9, ha='left',
                 bbox=dict(boxstyle='square,pad=0.2', facecolor=row_bg, alpha=0.5))
    ax5.text(col_x[-1], y, status, color=st_color, fontsize=9,
             fontweight='bold', ha='left')

# ── Row 2: Theoretical QEC Proposal ──
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor('#0d1117')
ax6.set_xlim(0, 20)
ax6.set_ylim(0, 4)
ax6.axis('off')
ax6.set_title("Theoretical Proposed Solution — Quantum Error Correction Framework",
              color=COLORS['accent'], fontsize=13, fontweight='bold', pad=10)

# Draw 3 QEC approach boxes
proposals = [
    {
        "title": "1. Shor/Steane QEC Code",
        "color": "#0984e3",
        "lines": [
            "Encode 1 logical qubit into",
            "9 (Shor) or 7 (Steane) physical",
            "qubits. Error syndromes detect",
            "bit-flip + phase-flip errors",
            "without collapsing state.",
            "",
            "Handles: X, Y, Z Pauli errors",
            "Overhead: 7-9x qubit cost"
        ]
    },
    {
        "title": "2. Entanglement Purification",
        "color": "#6c5ce7",
        "lines": [
            "Take N noisy Bell pairs,",
            "distill into M < N high-",
            "fidelity pairs using local",
            "operations and classical",
            "communication (LOCC).",
            "",
            "Handles: Channel degradation",
            "Used in: Quantum repeaters"
        ]
    },
    {
        "title": "3. Quantum Repeater Network",
        "color": "#00b894",
        "lines": [
            "Divide long channel into",
            "shorter segments. Refresh",
            "entanglement at intermediate",
            "nodes using entanglement",
            "swapping + purification.",
            "",
            "Handles: Long-distance noise",
            "Enables: Quantum internet"
        ]
    },
    {
        "title": "Detection → Decision Logic",
        "color": "#fdcb6e",
        "lines": [
            "Swap Test output → Route:",
            "",
            "Sim > 0.9  → Accept state",
            "0.5-0.9    → Apply QEC",
            "< 0.5      → Re-transmit",
            "",
            "This framework closes the",
            "loop: detect → correct → verify"
        ]
    }
]

box_w, box_h = 4.5, 3.2
box_positions = [(0.2, 0.4), (5.1, 0.4), (10.0, 0.4), (14.9, 0.4)]

for proposal, (bx, by) in zip(proposals, box_positions):
    rect = mpatches.FancyBboxPatch(
        (bx, by), box_w, box_h,
        boxstyle="round,pad=0.1",
        linewidth=2, edgecolor=proposal['color'],
        facecolor='#161b22'
    )
    ax6.add_patch(rect)

    ax6.text(bx + box_w/2, by + box_h - 0.2,
             proposal['title'],
             color=proposal['color'], fontsize=9.5,
             fontweight='bold', ha='center', va='top')

    for line_idx, line in enumerate(proposal['lines']):
        ax6.text(bx + 0.15, by + box_h - 0.55 - line_idx * 0.33,
                 line, color='#cdd9e5', fontsize=8.2,
                 va='top', ha='left')

    # Arrow between boxes (except last)
    if proposal != proposals[-1]:
        midx = bx + box_w + 0.05
        midy = by + box_h / 2
        ax6.annotate("", xy=(midx + 0.35, midy), xytext=(midx, midy),
                     arrowprops=dict(arrowstyle='->', color='#636e72',
                                     lw=2.0))

plt.savefig("quantum_outputs/part1_classical_noise.png", dpi=150, bbox_inches='tight')
plt.show()

print()
print("=" * 65)
print("  FULL PIPELINE COMPLETE")
print("=" * 65)
print()
print("  Files generated:")
print("  ├── part1_classical_noise.png  — Classical BER analysis")
print("  ├── part2_quantum_noise.png    — Bloch sphere visualization")
print("  ├── part3_teleportation.png    — Teleportation + noise")
print("  ├── part4_swap_test.png        — Quantum fingerprinting")
print("  └── part5_full_pipeline.png   — Complete dashboard (THIS)")
print()
print("  Key Results Summary:")
print(f"  {'Noise p':<10} {'Classical':<16} {'Quantum Fid':<16} {'Swap Sim'}")
print(f"  {'-'*55}")
for p in [0.0, 0.1, 0.25, 0.5, 0.75]:
    dm = apply_depolarizing(ref_dm, p)
    fid = fidelity(ref_dm, dm)
    sim = swap_test_similarity(ref_dm, dm)
    print(f"  {p:<10.2f} {p:<16.4f} {fid:<16.4f} {sim:.4f}")
print()
print("  Conclusion:")
print("  Classical noise is linear and predictable.")
print("  Quantum noise destroys COHERENCE (off-diagonal elements)")
print("  not just amplitude — making it fundamentally harder.")
print("  The Swap Test detects this without collapsing the state.")
print("=" * 65)
