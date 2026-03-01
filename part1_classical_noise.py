"""
=============================================================
PART 1 — CLASSICAL NOISE SIMULATION
=============================================================
Simulates bit-flip noise in a classical communication channel.
- Sends a binary message through a noisy channel
- Measures Bit Error Rate (BER) at various noise levels
- Plots BER vs noise probability
- Plots original vs received signal comparison

Install requirements:
    pip install numpy matplotlib
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────
# 1. CLASSICAL BIT-FLIP CHANNEL
# ──────────────────────────────────────────────────────────

def generate_message(length=64):
    """Generate a random binary message."""
    return np.random.randint(0, 2, size=length)


def classical_noisy_channel(bits, noise_prob):
    """
    Simulate a Binary Symmetric Channel (BSC).
    Each bit is flipped independently with probability noise_prob.
    """
    noise_mask = np.random.random(len(bits)) < noise_prob
    received = np.bitwise_xor(bits, noise_mask.astype(int))
    return received, noise_mask


def bit_error_rate(original, received):
    """Calculate the fraction of bits that were flipped."""
    errors = np.sum(original != received)
    return errors / len(original)


# ──────────────────────────────────────────────────────────
# 2. SIMULATE OVER RANGE OF NOISE PROBABILITIES
# ──────────────────────────────────────────────────────────

np.random.seed(42)
message_length = 256
noise_levels   = np.linspace(0, 0.5, 100)   # 0% to 50% noise
trials         = 50                          # average over multiple trials

original_message = generate_message(message_length)

ber_results = []
for p in noise_levels:
    trial_bers = []
    for _ in range(trials):
        received, _ = classical_noisy_channel(original_message, p)
        trial_bers.append(bit_error_rate(original_message, received))
    ber_results.append(np.mean(trial_bers))

# ──────────────────────────────────────────────────────────
# 3. SPECIFIC EXAMPLE — SHOW DISTORTION IN A SHORT MESSAGE
# ──────────────────────────────────────────────────────────

demo_message = np.array([1,0,1,1,0,0,1,0,
                          1,1,0,1,0,1,1,0,
                          0,0,1,1,0,1,0,1,
                          1,0,0,1,1,1,0,0])

demo_received_low,  _  = classical_noisy_channel(demo_message, 0.05)
demo_received_med,  _  = classical_noisy_channel(demo_message, 0.20)
demo_received_high, _  = classical_noisy_channel(demo_message, 0.45)

ber_low  = bit_error_rate(demo_message, demo_received_low)
ber_med  = bit_error_rate(demo_message, demo_received_med)
ber_high = bit_error_rate(demo_message, demo_received_high)

# ──────────────────────────────────────────────────────────
# 4. PLOTTING
# ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Classical Noise Simulation — Binary Symmetric Channel",
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: BER vs Noise Probability ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(noise_levels, ber_results, color='crimson', linewidth=2.5, label='Simulated BER')
ax1.plot(noise_levels, noise_levels, 'k--', linewidth=1.5, alpha=0.6, label='Theoretical BER (= p)')
ax1.fill_between(noise_levels, ber_results, alpha=0.15, color='crimson')
ax1.set_xlabel("Noise Probability (p)", fontsize=12)
ax1.set_ylabel("Bit Error Rate (BER)", fontsize=12)
ax1.set_title("BER vs Noise Probability — Classical Channel", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 0.55)
ax1.annotate("At p=0.5, channel is\ncompletely random (BER=0.5)",
             xy=(0.5, 0.5), xytext=(0.35, 0.35),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, color='black')

# ── Plot 2: Original Message ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(demo_message.reshape(4, 8), cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax2.set_title(f"Original Message (32 bits)", fontsize=11)
ax2.set_xlabel("Bit Position")
ax2.set_ylabel("Row")
for i in range(4):
    for j in range(8):
        ax2.text(j, i, str(demo_message[i*8+j]),
                 ha='center', va='center', fontsize=9, color='black', fontweight='bold')

# ── Plot 3: Received Messages at different noise levels ──
ax3 = fig.add_subplot(gs[1, 1])

# Stack all 4 messages vertically for comparison
comparison = np.vstack([
    demo_message,
    demo_received_low,
    demo_received_med,
    demo_received_high
]).reshape(4, 32)

# Highlight flipped bits
flip_map_low  = (demo_message != demo_received_low).astype(int)
flip_map_med  = (demo_message != demo_received_med).astype(int)
flip_map_high = (demo_message != demo_received_high).astype(int)

labels = [
    f"Original",
    f"Received p=0.05  (BER={ber_low:.2f})",
    f"Received p=0.20  (BER={ber_med:.2f})",
    f"Received p=0.45  (BER={ber_high:.2f})"
]

colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']

bar_x = np.arange(32)
for row_idx, (row_data, label, color) in enumerate(zip(comparison, labels, colors)):
    offset = row_idx * 1.5
    ax3.bar(bar_x, row_data + offset, bottom=offset,
            color=color, alpha=0.8, width=0.9, label=label)

ax3.set_title("Message Comparison at Different Noise Levels", fontsize=11)
ax3.set_xlabel("Bit Index")
ax3.set_yticks([])
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(True, axis='x', alpha=0.3)

plt.savefig("quantum_outputs/part1_classical_noise.png", dpi=150, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────
# 5. PRINT SUMMARY
# ──────────────────────────────────────────────────────────

print("=" * 55)
print("  CLASSICAL NOISE SIMULATION — SUMMARY")
print("=" * 55)
print(f"  Message Length   : {message_length} bits")
print(f"  Trials per level : {trials}")
print()
print(f"  {'Noise (p)':<15} {'BER':<15} {'Errors / 32 bits'}")
print(f"  {'-'*45}")
for p, received, label in [
    (0.05, demo_received_low,  "Low"),
    (0.20, demo_received_med,  "Medium"),
    (0.45, demo_received_high, "High")
]:
    ber = bit_error_rate(demo_message, received)
    errs = int(ber * 32)
    print(f"  {p:<15} {ber:<15.4f} ~{errs} bits flipped  ({label})")
print()
print("  Key Insight: Classical BER = noise probability p")
print("  This is predictable and correctable with Hamming/CRC codes.")
print("=" * 55)
