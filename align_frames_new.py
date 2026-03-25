#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Payload Alignment for Periodic Repetitions
===================================================

This script aligns a received symbol stream (already time/freq synchronized)
with a known transmitted payload that repeats periodically.

Scenario:
  - Transmitter loops a payload continuously: [payload][payload][payload]...
  - Receiver captures starting at arbitrary point: [...partial][full][full]...
  - Both are already synchronized (time, frequency, phase recovery done)
  - We need to find where each complete payload starts in the received stream

This is pure digital correlation - no RF, no peak detection, just alignment.

Usage:
  python3 align_frames_new.py --symbols received_symbols.bin --bits bit_tests/bits_000.bin
"""

import numpy as np
import argparse
import sys


# ═══════════════════════════════════════════════════════════════
#  BPSK MAPPING
# ═══════════════════════════════════════════════════════════════

def bits_to_bpsk(bits):
    """Convert {0, 1} bits to BPSK symbols {-1+0j, +1+0j}."""
    return (2 * bits.astype(np.complex64) - 1)


def bpsk_to_bits(symbols):
    """Convert BPSK symbols back to bits. Positive real → 1, negative → 0."""
    return (np.real(symbols) > 0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
#  CORRELATION-BASED ALIGNMENT
# ═══════════════════════════════════════════════════════════════

def correlate_sequences(received_symbols, reference_bpsk):
    """
    Cross-correlate received symbols with reference BPSK sequence.

    This tells us how well the reference aligns at each position
    in the received stream.

    Parameters
    ----------
    received_symbols : (N,) complex64 — received symbol stream
    reference_bpsk   : (L,) complex64 — known reference sequence

    Returns
    -------
    corr : (N - L + 1,) complex128 — correlation at each valid position
    """
    L = len(reference_bpsk)
    N = len(received_symbols)
    n_out = N - L + 1

    if n_out <= 0:
        return np.array([], dtype=np.complex128)

    # FFT-based correlation for speed
    n_fft = N + L - 1
    Y = np.fft.fft(received_symbols, n=n_fft)
    R = np.fft.fft(np.conj(reference_bpsk[::-1]), n=n_fft)
    corr_full = np.fft.ifft(Y * R)

    # Extract valid correlation outputs
    corr = corr_full[L - 1: L - 1 + n_out]

    return corr


def find_repetition_starts(corr, payload_len, threshold_factor=0.7):
    """
    Find all positions where the payload starts in the received stream.

    For a periodically repeating payload, correlation peaks occur at
    regular intervals of ~payload_len. We find all such positions.

    Parameters
    ----------
    corr            : (M,) complex — correlation output
    payload_len     : int — length of one complete payload
    threshold_factor: float — accept correlations >= this * max_correlation

    Returns
    -------
    starts : list of int — positions where payload repetitions start
    """
    mag = np.abs(corr)

    if len(mag) == 0:
        return []

    max_mag = np.max(mag)
    threshold = max_mag * threshold_factor

    # Find the strongest correlation (guaranteed to be a valid alignment)
    best_pos = int(np.argmax(mag))
    starts = []

    # Allow up to 10% period variation (clock drift, timing jitter)
    search_window = max(int(payload_len * 0.1), 100)

    # Search backwards from best position
    expected_pos = best_pos
    while expected_pos >= 0:
        window_start = max(0, expected_pos - search_window)
        window_end = min(len(mag), expected_pos + search_window + 1)

        if window_end <= window_start:
            break

        window = mag[window_start:window_end]
        local_max_idx = window_start + int(np.argmax(window))
        local_max_mag = mag[local_max_idx]

        if local_max_mag >= threshold:
            starts.insert(0, local_max_idx)
            expected_pos = local_max_idx - payload_len
        else:
            break

    # Search forwards from best position
    expected_pos = best_pos + payload_len
    while expected_pos < len(mag):
        window_start = max(0, expected_pos - search_window)
        window_end = min(len(mag), expected_pos + search_window + 1)

        if window_end <= window_start:
            break

        window = mag[window_start:window_end]
        local_max_idx = window_start + int(np.argmax(window))
        local_max_mag = mag[local_max_idx]

        if local_max_mag >= threshold:
            starts.append(local_max_idx)
            expected_pos = local_max_idx + payload_len
        else:
            break

    return starts


def estimate_channel(received_pilots, reference_pilots_bpsk):
    """
    Estimate channel coefficient from pilot symbols.

    For flat fading: y = h*x + n
    Least-squares estimate: h_hat = sum(y * conj(x)) / sum(|x|^2)
    """
    h_hat = np.sum(received_pilots * np.conj(reference_pilots_bpsk)) / len(reference_pilots_bpsk)
    return h_hat


def correct_phase(symbols, h_hat):
    """Apply zero-forcing equalization: multiply by conj(h) / |h|^2"""
    if abs(h_hat) < 1e-12:
        return symbols
    return symbols * np.conj(h_hat) / (np.abs(h_hat) ** 2)


# ═══════════════════════════════════════════════════════════════
#  MAIN ALIGNMENT ROUTINE
# ═══════════════════════════════════════════════════════════════

def align_periodic_payload(received_symbols, transmitted_bits, pilot_bits=None):
    """
    Align received symbols with transmitted payload and extract all repetitions.

    Parameters
    ----------
    received_symbols : (N,) complex64 — received symbol stream (already synchronized)
    transmitted_bits : (L,) uint8 — transmitted payload (one complete copy)
    pilot_bits       : list or None — pilot sequence at start of payload

    Returns
    -------
    dict with:
      - all_frames: list of dicts, one per complete payload found
      - h_hat: channel estimate
      - phase_deg: channel phase in degrees
      - n_frames: number of complete payloads found
      - period: measured period between payloads
    """
    # Use first 16 bits as pilot if not specified
    if pilot_bits is None:
        pilot_bits = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]

    pilot_len = len(pilot_bits)
    payload_len = len(transmitted_bits)

    print(f"Digital Payload Alignment")
    print(f"=" * 60)
    print(f"Received symbols : {len(received_symbols)}")
    print(f"Payload length   : {payload_len} symbols")
    print(f"Pilot length     : {pilot_len} symbols")
    print(f"Expected reps    : ~{len(received_symbols) / payload_len:.1f}")
    print()

    # Convert pilot to BPSK for correlation
    pilot_bpsk = bits_to_bpsk(np.array(pilot_bits, dtype=np.uint8))

    # Step 1: Correlate pilot against received stream
    print(f"Step 1: Correlating pilot with received symbols...")
    corr = correlate_sequences(received_symbols, pilot_bpsk)

    if len(corr) == 0:
        print("ERROR: Received stream shorter than pilot")
        return None

    print(f"  Correlation output: {len(corr)} positions")
    print(f"  Max correlation: {np.max(np.abs(corr)):.2f}")
    print()

    # Step 2: Find all payload repetition starts
    print(f"Step 2: Finding all payload repetitions...")
    starts = find_repetition_starts(corr, payload_len, threshold_factor=0.7)

    if len(starts) == 0:
        print("ERROR: No payload repetitions found")
        return None

    print(f"  Found {len(starts)} payload repetitions")
    print(f"  Positions: {starts[:10]}{'...' if len(starts) > 10 else ''}")

    # Measure period
    if len(starts) > 1:
        periods = np.diff(starts)
        median_period = int(np.median(periods))
        print(f"  Measured periods: {periods[:10].tolist()}{'...' if len(periods) > 10 else ''}")
        print(f"  Median period: {median_period} (expected: {payload_len})")
        print(f"  Period error: {median_period - payload_len:+d} symbols")
    else:
        median_period = payload_len
    print()

    # Step 3: Estimate channel from best pilot match
    print(f"Step 3: Estimating channel from pilots...")
    best_start = starts[0]
    best_h = 0
    best_score = -1

    for start in starts[:min(10, len(starts))]:  # Check first 10
        if start + pilot_len > len(received_symbols):
            continue

        rx_pilot = received_symbols[start:start + pilot_len]
        h_cand = estimate_channel(rx_pilot, pilot_bpsk)

        # Test this h by correcting and detecting
        rx_corrected = correct_phase(rx_pilot, h_cand)
        detected = bpsk_to_bits(rx_corrected)
        matches = int(np.sum(detected == np.array(pilot_bits, dtype=np.uint8)))

        print(f"  Position {start:6d}: h={h_cand:.3f}, |h|={abs(h_cand):.3f}, "
              f"∠h={np.degrees(np.angle(h_cand)):+6.1f}°, pilot_match={matches}/{pilot_len}")

        if matches > best_score:
            best_score = matches
            best_start = start
            best_h = h_cand

    # Check for 180° phase ambiguity
    if best_score < pilot_len * 0.6:
        print(f"\n  Checking 180° phase ambiguity...")
        h_flipped = -best_h
        rx_pilot = received_symbols[best_start:best_start + pilot_len]
        rx_corrected = correct_phase(rx_pilot, h_flipped)
        detected = bpsk_to_bits(rx_corrected)
        matches = int(np.sum(detected == np.array(pilot_bits, dtype=np.uint8)))

        if matches > best_score:
            print(f"    Using flipped phase: {matches}/{pilot_len} matches (was {best_score})")
            best_h = h_flipped
            best_score = matches

    phase_deg = np.degrees(np.angle(best_h))
    print(f"\n  Using h = {best_h:.4f} (|h|={abs(best_h):.3f}, ∠h={phase_deg:+.1f}°)")
    print()

    # Step 4: Extract all complete payloads
    print(f"Step 4: Extracting all complete payloads...")
    all_frames = []

    for i, start in enumerate(starts):
        if start + payload_len > len(received_symbols):
            print(f"  Frame {i+1} at {start:6d}: INCOMPLETE (truncated)")
            continue

        # Extract and correct this payload
        rx_payload = received_symbols[start:start + payload_len]
        rx_corrected = correct_phase(rx_payload, best_h)

        # Split into pilot and data
        rx_pilots = rx_corrected[:pilot_len]
        rx_data = rx_corrected[pilot_len:]

        # Check pilot
        pilot_detected = bpsk_to_bits(rx_pilots)
        pilot_errors = int(np.sum(pilot_detected != np.array(pilot_bits, dtype=np.uint8)))

        # Data and ground truth
        data_bits_true = transmitted_bits[pilot_len:].copy()

        # Quick SER
        data_detected = bpsk_to_bits(rx_data)
        data_errors = int(np.sum(data_detected != data_bits_true))
        ser = data_errors / len(data_bits_true) if len(data_bits_true) > 0 else 0

        print(f"  Frame {i+1:2d} at {start:6d}: pilot_err={pilot_errors}/{pilot_len}, "
              f"data_err={data_errors}/{len(data_bits_true)}, SER={ser:.4f}")

        all_frames.append({
            'rx_payload': rx_corrected,
            'rx_pilots': rx_pilots,
            'rx_data': rx_data,
            'data_bits_true': data_bits_true,
            'pilot_errors': pilot_errors,
            'data_errors': data_errors,
            'ser': ser,
            'start': start
        })

    print()
    return {
        'all_frames': all_frames,
        'h_hat': best_h,
        'phase_deg': phase_deg,
        'n_frames': len(all_frames),
        'period': median_period,
        'pilot_bits': np.array(pilot_bits, dtype=np.uint8)
    }


# ═══════════════════════════════════════════════════════════════
#  COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Align received symbols with periodically repeating payload"
    )
    parser.add_argument("--symbols", required=True, help="Path to received_symbols.bin")
    parser.add_argument("--bits", required=True, help="Path to transmitted bit file")
    args = parser.parse_args()

    # Load data
    print(f"Loading data...")
    received_symbols = np.fromfile(args.symbols, dtype=np.complex64)
    transmitted_bits = np.fromfile(args.bits, dtype=np.uint8)
    print(f"  Received: {len(received_symbols)} symbols from {args.symbols}")
    print(f"  Transmitted: {len(transmitted_bits)} bits from {args.bits}")
    print()

    # Run alignment
    result = align_periodic_payload(received_symbols, transmitted_bits)

    if result is None:
        print("Alignment failed")
        sys.exit(1)

    # Summary
    print(f"=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Frames found     : {result['n_frames']}")
    print(f"Channel h        : {result['h_hat']:.4f}")
    print(f"Channel |h|      : {abs(result['h_hat']):.3f}")
    print(f"Channel phase    : {result['phase_deg']:+.1f}°")
    print(f"Measured period  : {result['period']}")
    print()

    # Overall statistics
    total_data_errors = sum(f['data_errors'] for f in result['all_frames'])
    total_data_symbols = sum(len(f['data_bits_true']) for f in result['all_frames'])
    overall_ser = total_data_errors / total_data_symbols if total_data_symbols > 0 else 0

    print(f"Overall SER      : {overall_ser:.4f} ({overall_ser*100:.2f}%)")
    print(f"Total errors     : {total_data_errors} / {total_data_symbols} data symbols")
    print()

    # Save results
    output_path = args.bits.replace(".bin", "_aligned.npz")

    np.savez(
        output_path,
        all_rx_data=np.array([f['rx_data'] for f in result['all_frames']], dtype=object),
        all_data_bits_true=np.array([f['data_bits_true'] for f in result['all_frames']], dtype=object),
        frame_starts=np.array([f['start'] for f in result['all_frames']], dtype=np.int32),
        pilot_errors=np.array([f['pilot_errors'] for f in result['all_frames']], dtype=np.int32),
        data_errors=np.array([f['data_errors'] for f in result['all_frames']], dtype=np.int32),
        ser_per_frame=np.array([f['ser'] for f in result['all_frames']], dtype=np.float32),
        h_hat=result['h_hat'],
        phase_deg=result['phase_deg'],
        period=result['period'],
        n_frames=result['n_frames'],
        pilot_bits=result['pilot_bits']
    )

    print(f"Saved: {output_path}")
    print(f"Load with: data = np.load('{output_path}', allow_pickle=True)")


if __name__ == "__main__":
    main()
