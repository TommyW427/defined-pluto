#!/usr/bin/env python3
"""Compare transmitted vs received bits"""
import numpy as np

raw_iq = np.frombuffer(open('received_iq_raw.bin','rb').read(), dtype=np.complex64)
bits_ref = np.frombuffer(open('bit_tests/bits_000.bin','rb').read(), dtype=np.uint8)

rx = raw_iq[::25]
rx_bits = (rx.real < 0).astype(np.uint8)

# Find pilot
pilot = bits_ref[2000:2064]
for pos in range(0, len(rx_bits)-64, 10):
    if np.sum(pilot != rx_bits[pos:pos+64]) == 0:
        pilot_pos = pos
        break

print(f"Pilot at: {pilot_pos}")
print(f"REF: {''.join(str(b) for b in pilot[:32])}...")
print(f"RX:  {''.join(str(b) for b in rx_bits[pilot_pos:pilot_pos+32])}...")
print()

# Data
data_start = pilot_pos + 64
ref = bits_ref[2064:2064+200]
rx_data = rx_bits[data_start:data_start+200]

print("First 200 data bits:")
for i in range(0, 200, 100):
    print(f"\nBits {i}-{i+100}:")
    print(f"TX: {''.join(str(b) for b in ref[i:i+100])}")
    print(f"RX: {''.join(str(b) for b in rx_data[i:i+100])}")
    errs = np.sum(ref[i:i+100] != rx_data[i:i+100])
    print(f"    {errs} errors")
