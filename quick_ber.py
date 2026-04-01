#!/usr/bin/env python3
"""Quick and dirty BER calculation"""

import numpy as np

# Load files
bits = np.frombuffer(open('DEFINED-SDR/bit_tests/bits_000.bin','rb').read(), dtype=np.uint8)
pilot = np.frombuffer(open('DEFINED-SDR/bit_tests/pilot.bin','rb').read(), dtype=np.uint8)

# Find pilot
pilot_pos = 2000  # we know this from inspection

# Extract data after pilot
data_start_bit = pilot_pos + len(pilot)  # 2000 + 64 = 2064
num_symbols = 17936  # This is SYMBOLS not bits
tx_bits_start = data_start_bit

print(f"Pilot at bit: {pilot_pos}")
print(f"Data starts at bit: {data_start_bit}")

# Now load received and detect bits
# First need to load symbols and detect
symbols = np.frombuffer(open('DEFINED-SDR/received_symbols.bin','rb').read(), dtype=np.complex64)

print(f"Total symbols: {len(symbols)}")

# Map bit position to symbol position
# Ratio is ~3.54, try 4 samples/bit
samp_per_bit = 4

rx_symbol_start = data_start_bit * samp_per_bit
rx_data_symbols = symbols[rx_symbol_start:rx_symbol_start + num_symbols]

print(f"RX symbol start: {rx_symbol_start}")
print(f"RX symbols extracted: {len(rx_data_symbols)} (want {num_symbols})")

# Calculate how many bits this represents
num_bits = num_symbols // samp_per_bit
tx_data = bits[tx_bits_start:tx_bits_start + num_bits]
print(f"This is {num_bits} bits of data")

# Detect bits - simple BPSK detection
# Downsample by taking every 4th sample (or average)
rx_symbols_decimated = rx_data_symbols[::samp_per_bit][:num_bits]

# BPSK detection: real part > 0 -> 1, else 0
rx_bits = (rx_symbols_decimated.real > 0).astype(np.uint8)

print(f"RX bits detected: {len(rx_bits)}")

# Match lengths
min_len = min(len(tx_data), len(rx_bits))
tx_data = tx_data[:min_len]
rx_bits = rx_bits[:min_len]
print(f"Using {min_len} bits for comparison")

# Calculate BER
errors = np.sum(tx_data != rx_bits)
ber = errors / min_len

print(f"\n{'='*60}")
print(f"BIT ERROR RATE ANALYSIS")
print(f"{'='*60}")
print(f"TX bits: {len(tx_data)}")
print(f"RX bits: {len(rx_bits)}")
print(f"Errors: {errors}")
print(f"BER: {ber:.6f} ({ber*100:.4f}%)")
print(f"{'='*60}")

# Show first 100 bits comparison
print(f"\nFirst 100 bits comparison:")
print(f"TX: {''.join(str(b) for b in tx_data[:100])}")
print(f"RX: {''.join(str(b) for b in rx_bits[:100])}")
print(f"    {''.join('X' if tx_data[i]!=rx_bits[i] else ' ' for i in range(100))}")

# Try other sampling rates if BER is bad
if ber > 0.1:
    print(f"\n⚠ High BER! Trying other sampling rates...")
    for rate in [1, 2, 3, 4, 8]:
        rx_start = data_start_bit * rate
        if rx_start + num_symbols > len(symbols):
            print(f"  Rate {rate}: skipped (not enough symbols)")
            continue
        rx_syms = symbols[rx_start:rx_start + num_symbols]
        nbits = num_symbols // rate
        rx_dec = rx_syms[::rate][:nbits]
        rx_b = (rx_dec.real > 0).astype(np.uint8)
        tx_test = bits[tx_bits_start:tx_bits_start + nbits]
        mlen = min(len(tx_test), len(rx_b))
        errs = np.sum(tx_test[:mlen] != rx_b[:mlen])
        print(f"  Rate {rate}: BER = {errs/mlen:.6f} ({errs}/{mlen} errors)")

# Try phase offsets
print(f"\nTrying phase corrections (at rate {samp_per_bit})...")
for phase_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
    phase = np.deg2rad(phase_deg)
    corrected = rx_symbols_decimated * np.exp(-1j*phase)
    rx_b = (corrected.real > 0).astype(np.uint8)
    mlen = min(len(tx_data), len(rx_b))
    errs = np.sum(tx_data[:mlen] != rx_b[:mlen])
    print(f"  Phase {phase_deg:3d}°: BER = {errs/mlen:.6f} ({errs}/{mlen} errors)")

# Try timing offsets (maybe not starting at exact symbol)
print(f"\nTrying timing offsets (different start samples)...")
for offset in range(8):
    rx_start_offset = rx_symbol_start + offset
    rx_syms_offset = symbols[rx_start_offset:rx_start_offset + num_symbols]
    nbits = num_symbols // samp_per_bit
    rx_dec = rx_syms_offset[::samp_per_bit][:nbits]
    rx_b = (rx_dec.real > 0).astype(np.uint8)
    tx_test = bits[tx_bits_start:tx_bits_start + nbits]
    mlen = min(len(tx_test), len(rx_b))
    errs = np.sum(tx_test[:mlen] != rx_b[:mlen])
    print(f"  Offset +{offset}: BER = {errs/mlen:.6f} ({errs}/{mlen} errors)")