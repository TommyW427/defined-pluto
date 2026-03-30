#!/usr/bin/env python3
"""Check which bit file was actually transmitted"""

import numpy as np
import glob

# Load received symbols
received_symbols = np.fromfile('received_symbols.bin', dtype=np.complex64)
print(f"Received: {len(received_symbols)} symbols\n")

# Test each bit file
bit_files = sorted(glob.glob('bit_tests/bits_*.bin'))
pilot_bits = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
pilot_bpsk = (2 * np.array(pilot_bits, dtype=np.complex64) - 1)

results = []

for bit_file in bit_files:
    transmitted_bits = np.fromfile(bit_file, dtype=np.uint8)

    # Correlate pilot
    n_fft = len(received_symbols) + len(pilot_bpsk) - 1
    Y = np.fft.fft(received_symbols, n=n_fft)
    P = np.fft.fft(np.conj(pilot_bpsk[::-1]), n=n_fft)
    corr_full = np.fft.ifft(Y * P)
    corr = corr_full[len(pilot_bpsk) - 1: len(pilot_bpsk) - 1 + len(received_symbols) - len(pilot_bpsk) + 1]

    mag = np.abs(corr)
    max_corr = np.max(mag)
    max_pos = np.argmax(mag)

    # Count peaks above threshold
    threshold = max_corr * 0.7
    n_peaks = np.sum(mag > threshold)

    # Check periodicity
    payload_len = len(transmitted_bits)
    corr_at_intervals = []
    for i in range(1, min(10, len(mag) // payload_len)):
        pos = max_pos + i * payload_len
        if 0 <= pos < len(mag):
            corr_at_intervals.append(mag[pos])

    avg_periodic_corr = np.mean(corr_at_intervals) if corr_at_intervals else 0

    results.append({
        'file': bit_file,
        'max_corr': max_corr,
        'max_pos': max_pos,
        'n_peaks': n_peaks,
        'avg_periodic': avg_periodic_corr,
        'payload_len': payload_len
    })

    print(f"{bit_file:30s}: max_corr={max_corr:7.2f} @ pos {max_pos:7d}, "
          f"peaks={n_peaks:3d}, avg_periodic_corr={avg_periodic_corr:6.2f}")

# Find best match
best = max(results, key=lambda x: x['avg_periodic'])
print(f"\n{'='*70}")
print(f"BEST MATCH (by periodic correlation): {best['file']}")
print(f"  This file shows the strongest evidence of periodic repetition")
print(f"  Avg correlation at periodic intervals: {best['avg_periodic']:.2f}")
