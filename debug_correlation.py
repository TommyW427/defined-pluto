#!/usr/bin/env python3
"""Debug script to visualize correlation and understand the data"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
received_symbols = np.fromfile('received_symbols.bin', dtype=np.complex64)
transmitted_bits = np.fromfile('bit_tests/bits_000.bin', dtype=np.uint8)

print(f"Received: {len(received_symbols)} symbols")
print(f"Transmitted: {len(transmitted_bits)} bits")

# Use pilot for correlation
pilot_bits = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
pilot_bpsk = (2 * np.array(pilot_bits, dtype=np.complex64) - 1)

# Correlate
print("\nCorrelating...")
n_fft = len(received_symbols) + len(pilot_bpsk) - 1
Y = np.fft.fft(received_symbols, n=n_fft)
P = np.fft.fft(np.conj(pilot_bpsk[::-1]), n=n_fft)
corr_full = np.fft.ifft(Y * P)
corr = corr_full[len(pilot_bpsk) - 1: len(pilot_bpsk) - 1 + len(received_symbols) - len(pilot_bpsk) + 1]

mag = np.abs(corr)

print(f"Correlation length: {len(mag)}")
print(f"Max correlation: {np.max(mag):.2f} at position {np.argmax(mag)}")

# Look for peaks above threshold
threshold = np.max(mag) * 0.7
peaks = np.where(mag > threshold)[0]
print(f"\nPositions above {0.7} * max_corr: {len(peaks)}")
print(f"Positions: {peaks[:20].tolist()}{'...' if len(peaks) > 20 else ''}")

# Check if there are periodic patterns
if len(peaks) > 1:
    diffs = np.diff(peaks)
    print(f"\nDifferences between consecutive peaks:")
    print(f"  {diffs[:20].tolist()}{'...' if len(diffs) > 20 else ''}")

# Plot correlation
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Full correlation
axes[0].plot(mag)
axes[0].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({0.7} * max)')
axes[0].set_xlabel('Position (symbols)')
axes[0].set_ylabel('Correlation magnitude')
axes[0].set_title('Full Correlation Output')
axes[0].legend()
axes[0].grid(True)

# Zoom around the peak
peak_pos = np.argmax(mag)
window = 1000000
axes[1].plot(mag[max(0, peak_pos-window):min(len(mag), peak_pos+window)])
axes[1].axhline(y=threshold, color='r', linestyle='--')
axes[1].set_xlabel(f'Position relative to {max(0, peak_pos-window)}')
axes[1].set_ylabel('Correlation magnitude')
axes[1].set_title(f'Zoomed: ±{window} symbols around peak at {peak_pos}')
axes[1].grid(True)

# Histogram of correlation values
axes[2].hist(mag, bins=100, log=True)
axes[2].axvline(x=threshold, color='r', linestyle='--', label='Threshold')
axes[2].set_xlabel('Correlation magnitude')
axes[2].set_ylabel('Count (log scale)')
axes[2].set_title('Histogram of Correlation Values')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('correlation_debug.png', dpi=150)
print(f"\nSaved correlation_debug.png")

# Check for periodicity in the correlation at expected interval
payload_len = len(transmitted_bits)
print(f"\nChecking for periodicity at expected interval ({payload_len} symbols):")
for i in range(10):
    pos = peak_pos + i * payload_len
    if 0 <= pos < len(mag):
        print(f"  Position {pos:7d} (peak + {i}*{payload_len}): corr_mag = {mag[pos]:8.2f}")
