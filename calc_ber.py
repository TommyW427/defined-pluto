#!/usr/bin/env python3
"""Calculate BER from raw IQ"""
import numpy as np

# Load
raw_iq = np.frombuffer(open('received_iq_raw.bin','rb').read(), dtype=np.complex64)
bits_ref = np.frombuffer(open('bit_tests/bits_000.bin','rb').read(), dtype=np.uint8)

# Decimate by 25
rx = raw_iq[::25]

# Detect (inverted due to 180deg phase lock)
rx_bits = (rx.real < 0).astype(np.uint8)

# Find pilot
pilot = bits_ref[2000:2064]
best_pos = 0
best_err = 64

for pos in range(0, len(rx_bits)-64, 10):
    errs = np.sum(pilot != rx_bits[pos:pos+64])
    if errs < best_err:
        best_err = errs
        best_pos = pos

print(f"Pilot: position {best_pos}, {best_err} errors")

# Calculate BER
data_start = best_pos + 64
ref_data = bits_ref[2064:2064+17936]
rx_data = rx_bits[data_start:data_start+17936]

errors = np.sum(ref_data != rx_data)
ber = errors / len(ref_data)

print(f"BER: {ber:.6f} ({errors}/{len(ref_data)} errors)")
