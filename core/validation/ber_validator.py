#!/usr/bin/env python3
"""
BER/SER Validation
==================

Simple validation for received frames.
"""

import numpy as np
from pathlib import Path


def validate_frame(tx_frame_file, rx_symbols_file, sps=1):
    """
    Validate BER and SER for received frame

    Args:
        tx_frame_file: Transmitted frame bits (e.g., 'full_frames/frame_0000.bin')
        rx_symbols_file: Received symbols from SPV.py ('received_symbols.bin')
        sps: Additional decimation (default 1, since symbol_sync already outputs at symbol rate)

    Returns:
        dict with BER, SER, and detailed results
    """
    # Load transmitted bits
    tx_bits = np.fromfile(tx_frame_file, dtype=np.uint8)

    # Load received symbols (already at symbol rate from GNU Radio symbol_sync)
    rx_symbols = np.fromfile(rx_symbols_file, dtype=np.complex64)

    # No decimation needed - symbol_sync already did it
    # But keep variable for compatibility
    rx_symbols_decimated = rx_symbols if sps == 1 else rx_symbols[::sps]

    # Remove DC offset by centering around mean
    mean_real = np.mean(rx_symbols_decimated.real)
    rx_symbols_centered = rx_symbols_decimated.real - mean_real

    # Hard decision: convert symbols to bits (BPSK: centered real > 0 => 1, else 0)
    rx_bits = (rx_symbols_centered > 0).astype(np.uint8)

    # Frame structure
    PREAMBLE_LEN = 2000
    PILOT_LEN = 64
    DATA_LEN = 64

    print("="*70)
    print("FRAME VALIDATION")
    print("="*70)
    print(f"TX bits: {len(tx_bits)}")
    print(f"RX symbols (raw): {len(rx_symbols)}")
    print(f"RX symbols (decimated): {len(rx_symbols_decimated)}")
    print(f"RX bits: {len(rx_bits)}")
    print(f"SPS: {sps}")
    print()

    # Find where strong signal starts (skip silence at beginning)
    strong_symbols = np.where(np.abs(rx_symbols_centered) > 0.1)[0]
    if len(strong_symbols) > 0:
        signal_start = strong_symbols[0]
        print(f"Strong signal detected starting at symbol: {signal_start}")
    else:
        signal_start = 0
        print("No strong signal detected, searching from beginning")
    print()

    # Find pilot using correlation
    pilot_ref = tx_bits[PREAMBLE_LEN:PREAMBLE_LEN+PILOT_LEN]

    # Search for pilot (try both normal and inverted) starting from where signal begins
    best = {'pos': 0, 'errors': PILOT_LEN, 'inverted': False}

    search_start = max(0, signal_start - 100)  # Start a bit before strong signal
    search_end = min(len(rx_bits) - PILOT_LEN, signal_start + 10000)

    for pos in range(search_start, search_end, 10):
        # Normal
        errs = np.sum(pilot_ref != rx_bits[pos:pos+PILOT_LEN])
        if errs < best['errors']:
            best = {'pos': pos, 'errors': errs, 'inverted': False}

        # Inverted (180° phase)
        errs_inv = np.sum(pilot_ref != (1 - rx_bits[pos:pos+PILOT_LEN]))
        if errs_inv < best['errors']:
            best = {'pos': pos, 'errors': errs_inv, 'inverted': True}

    print(f"Pilot found at position: {best['pos']}")
    print(f"Pilot errors: {best['errors']}/{PILOT_LEN}")
    print(f"Phase inverted: {best['inverted']}")
    print()

    # Apply phase correction
    if best['inverted']:
        rx_bits_corrected = 1 - rx_bits
    else:
        rx_bits_corrected = rx_bits

    # Extract data section
    data_start = best['pos'] + PILOT_LEN
    tx_data = tx_bits[PREAMBLE_LEN + PILOT_LEN:PREAMBLE_LEN + PILOT_LEN + DATA_LEN]
    rx_data = rx_bits_corrected[data_start:data_start+DATA_LEN]

    # Calculate BER
    bit_errors = np.sum(tx_data != rx_data)
    ber = bit_errors / DATA_LEN

    # Calculate SER (symbol errors)
    # For BPSK, 1 symbol = 1 bit, so SER = BER
    ser = ber

    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Data bits compared: {DATA_LEN}")
    print(f"Bit errors: {bit_errors}")
    print(f"BER: {ber:.6f} ({ber*100:.4f}%)")
    print(f"SER: {ser:.6f} ({ser*100:.4f}%)")

    if ber == 0:
        print("✓✓✓ PERFECT! Zero errors!")
    elif ber < 0.01:
        print("✓✓ EXCELLENT!")
    elif ber < 0.05:
        print("✓ GOOD")
    else:
        print("⚠ Poor link quality")

    print("="*70)

    return {
        'ber': ber,
        'ser': ser,
        'bit_errors': bit_errors,
        'total_bits': DATA_LEN,
        'pilot_errors': best['errors'],
        'pilot_position': best['pos'],
        'phase_inverted': best['inverted']
    }
