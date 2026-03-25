#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Alignment: received_symbols.bin  ↔  bits_000.bin
======================================================

The receiver's clock recovery block (Mueller & Mueller) outputs
one complex sample per symbol into received_symbols.bin.  The
transmitter was looping bits_000.bin continuously.

This script finds exactly where in the received stream each
repetition of the bit file begins, so that received symbol N
can be compared to transmitted bit N.

Key complications this handles:

  1. UNKNOWN START OFFSET
     The receiver started capturing at an arbitrary point in the
     transmitter's repeating stream.  The first received symbol
     could correspond to any bit in the file.

  2. BPSK 180° PHASE AMBIGUITY
     The Costas loop can lock with a 180° phase offset.  When this
     happens, every symbol is flipped: +1 looks like -1 and vice
     versa.  We detect this by checking the pilot correlation sign
     and correct for it.

  3. MULTIPLE FRAME REPETITIONS
     If the capture is long enough, it contains several copies of
     the bit file back to back.  We find all of them and can
     extract multiple independent frames for averaging SER.

Usage:
  # As a script — prints alignment info and saves results
  python3 align_frames.py --symbols received_symbols.bin --bits bit_tests/bits_000.bin

  # As a module — import the functions you need
  from align_frames import align_frame, load_and_align

Output:
  An .npz file containing the aligned received symbols, the
  corresponding ground-truth bits, the estimated phase, and
  the frame boundaries.  Feed this directly to mmse_detect_ser.py
  or the DEFINED pipeline.
"""

import numpy as np
import argparse
import sys


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION — must match the transmitter
# ═══════════════════════════════════════════════════════════════

# The pilot bit pattern at the start of every bit file.
# These are the first k bits of bits_000.bin (and bits_001.bin, etc).
# The transmitter maps each bit to a BPSK symbol (0→-1, 1→+1),
# holds it for sps=25 samples, and the clock recovery collapses
# it back to 1 sample/symbol.  So these 16 bits become 16 complex
# values in the received stream.
PILOT_BITS = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]


# ═══════════════════════════════════════════════════════════════
#  BPSK MAPPING
# ═══════════════════════════════════════════════════════════════

def bits_to_bpsk(bits):
    """
    Convert {0, 1} bits to BPSK symbols {-1+0j, +1+0j}.

    Matches the transmitter's chunks_to_symbols block:
      symbol_table = [-1+0j, 1+0j]
      bit 0  →  -1 + 0j
      bit 1  →  +1 + 0j
    """
    return (2 * bits.astype(np.complex64) - 1)


def bpsk_to_bits(symbols):
    """
    Convert BPSK symbols back to bits.
    Positive real part → 1, negative → 0.
    """
    return (np.real(symbols) > 0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
#  CROSS-CORRELATION
# ═══════════════════════════════════════════════════════════════

def correlate_pilot(y_symbols, pilot_bpsk):
    """
    Slide the known pilot sequence across the received symbol stream
    and compute the complex correlation at every position.

    We use COMPLEX correlation (not magnitude) because we need to
    know the phase of the match, not just the strength.  For BPSK,
    the Costas loop can lock with 0° or 180° offset.  A positive
    real peak means 0° (no flip), a negative real peak means 180°
    (all symbols are inverted).

    Uses FFT-based cross-correlation for speed.

    Parameters
    ----------
    y_symbols  : (N,) complex64 — received symbol stream
    pilot_bpsk : (k,) complex64 — known pilot as BPSK symbols

    Returns
    -------
    corr : (N - k + 1,) complex128 — complex correlation at each lag.
           The magnitude tells you how well the pilot matches.
           The phase tells you the channel phase at that point.
    """
    k = len(pilot_bpsk)
    n_out = len(y_symbols) - k + 1

    if n_out <= 0:
        return np.array([], dtype=np.complex128)

    # Cross-correlation via FFT:
    #   corr[lag] = sum_i  y[lag + i] * conj(pilot[i])
    # This is equivalent to matched filtering the received stream
    # with the known pilot, which is the optimal detection strategy
    # in AWGN for finding a known sequence.
    n_fft = len(y_symbols) + k - 1
    Y = np.fft.fft(y_symbols, n=n_fft)
    P = np.fft.fft(np.conj(pilot_bpsk[::-1]), n=n_fft)
    corr_full = np.fft.ifft(Y * P)

    # Extract the valid portion (full overlap between pilot and y)
    corr = corr_full[k - 1: k - 1 + n_out]

    return corr


def find_pilot_peaks(corr, pilot_len, frame_len, threshold_factor=0.6):
    """
    Find all positions in the correlation output where the pilot
    pattern appears strongly.

    Since the transmitter loops the file, the pilot appears once
    every frame_len symbols.  We look for all peaks that are at
    least threshold_factor * max_peak in magnitude.

    Parameters
    ----------
    corr             : (M,) complex — output of correlate_pilot
    pilot_len        : int — length of the pilot sequence (k)
    frame_len        : int — total bits in the file (pilot + data)
    threshold_factor : float — minimum peak height relative to the
                       strongest peak (0.6 = at least 60% as strong)

    Returns
    -------
    peaks : list of int — indices where the pilot starts in y_symbols
    """
    mag = np.abs(corr)

    if len(mag) == 0:
        return []

    max_mag = np.max(mag)
    threshold = max_mag * threshold_factor

    # Find all local maxima above threshold.
    # A peak must be the highest point within ±(frame_len/2) to
    # avoid picking up sidelobes of the same frame's correlation.
    peaks = []
    min_spacing = frame_len // 2

    # Sort all above-threshold indices by magnitude (strongest first)
    candidates = np.where(mag > threshold)[0]
    if len(candidates) == 0:
        return []

    # Greedy selection: pick the strongest, exclude its neighborhood,
    # pick the next strongest, and so on.
    order = np.argsort(mag[candidates])[::-1]
    used = np.zeros(len(mag), dtype=bool)

    for idx in candidates[order]:
        if used[idx]:
            continue
        peaks.append(int(idx))
        # Mark the neighborhood as used
        lo = max(0, idx - min_spacing)
        hi = min(len(mag), idx + min_spacing + 1)
        used[lo:hi] = True

    # Sort by position (chronological order in the received stream)
    peaks.sort()

    return peaks


# ═══════════════════════════════════════════════════════════════
#  PHASE ESTIMATION AND CORRECTION
# ═══════════════════════════════════════════════════════════════

def estimate_phase_from_pilots(y_pilots, pilot_bpsk):
    """
    Estimate the complex channel coefficient (phase + amplitude)
    from the received pilot symbols and the known transmitted pilots.

    For BPSK with a flat fading channel:
      y_pilot[i] = h * x_pilot[i] + noise[i]

    The least-squares estimate of h is:
      h_hat = sum(y_pilot * conj(x_pilot)) / sum(|x_pilot|^2)

    Since BPSK symbols have |x|=1, the denominator is just k.

    The phase of h_hat tells us the channel rotation.
    For BPSK, if angle(h_hat) is near 180°, the Costas loop locked
    with inverted polarity and all symbols are flipped.

    Parameters
    ----------
    y_pilots   : (k,) complex — received samples at pilot positions
    pilot_bpsk : (k,) complex — known pilot symbols

    Returns
    -------
    h_hat : complex — channel estimate
    """
    k = len(pilot_bpsk)
    h_hat = np.sum(y_pilots * np.conj(pilot_bpsk)) / k
    return h_hat


def correct_phase(y_symbols, h_hat):
    """
    Remove the channel phase from the received symbols.

    Multiply by conj(h_hat) / |h_hat| to rotate the constellation
    so that the BPSK points sit on the real axis at ±|h|.

    Then divide by |h_hat| to normalize amplitude to ±1.

    Parameters
    ----------
    y_symbols : (N,) complex — received symbols with channel phase
    h_hat     : complex — channel estimate

    Returns
    -------
    y_corrected : (N,) complex — phase-corrected symbols,
                  real part should be near ±1 for BPSK
    """
    # conj(h) / |h|^2 = 1/h  (the zero-forcing equalizer)
    # This rotates and scales so that h*x → x
    if abs(h_hat) < 1e-12:
        # Channel estimate is essentially zero — can't correct
        return y_symbols
    y_corrected = y_symbols * np.conj(h_hat) / (np.abs(h_hat) ** 2)
    return y_corrected


# ═══════════════════════════════════════════════════════════════
#  MAIN ALIGNMENT FUNCTION
# ═══════════════════════════════════════════════════════════════

def align_frame(y_symbols, tx_bits, pilot_bits=None):
    """
    Find all repetitions of the transmitted payload in the received
    symbol stream, estimate and correct the channel phase, and
    return ALL aligned frames with their ground-truth bits.

    The transmitter loops the bit file continuously, so the received
    symbols contain: [payload, payload, payload, ...] where each
    payload is the complete tx_bits sequence.

    Parameters
    ----------
    y_symbols  : (N,) complex64 — received symbol stream from
                 received_symbols.bin (1 sample per symbol,
                 output of clock recovery)
    tx_bits    : (L,) uint8 — the transmitted bit file contents,
                 starting with the pilot
    pilot_bits : list of int or None — the pilot pattern.
                 If None, uses the module-level PILOT_BITS.

    Returns
    -------
    dict with keys:
      frame_starts     : list of int — where each complete frame starts in y_symbols
      h_hat            : complex — channel estimate from the best pilot match
      phase_deg        : float — channel phase in degrees
      is_inverted      : bool — True if the Costas loop locked at 180°
      all_frames       : list of dict — one entry per complete frame found, each with:
                           - y_frame: (frame_len,) complex — phase-corrected frame
                           - y_pilots: (k,) complex — pilot portion
                           - y_data: (frame_len-k,) complex — data portion
                           - data_bits_true: (frame_len-k,) uint8 — ground truth
                           - pilot_errors: int — pilot bit mismatches
                           - frame_start: int — start position in y_symbols
      n_frames_found   : int — total number of complete frame repetitions
      frame_period     : int — estimated period between frames (should ≈ frame_len)
    """
    if pilot_bits is None:
        pilot_bits = PILOT_BITS

    k = len(pilot_bits)
    pilot_bpsk = bits_to_bpsk(np.array(pilot_bits, dtype=np.uint8))
    frame_len = len(tx_bits)
    data_bits_all = tx_bits[k:]

    print(f"Aligning received symbols to transmitted bits (PERIODIC MODE)...")
    print(f"  Received symbols : {len(y_symbols)}")
    print(f"  Payload length   : {frame_len} symbols (pilot: {k}, data: {frame_len - k})")
    print(f"  Expected frames  : ~{len(y_symbols) / frame_len:.1f} repetitions")
    print()

    # ── Step 1: Correlate pilot against received stream ─────────
    print(f"  Step 1: Cross-correlating pilot ({k} symbols) against "
          f"received stream ({len(y_symbols)} symbols)...")
    corr = correlate_pilot(y_symbols, pilot_bpsk)

    if len(corr) == 0:
        print("  ERROR: Received stream is shorter than the pilot.")
        return None

    # ── Step 2: Find all correlation peaks ──────────────────────
    print(f"  Step 2: Finding all periodic correlation peaks...")
    frame_starts = find_pilot_peaks(corr, k, frame_len)
    print(f"    Found {len(frame_starts)} candidate frame start(s)")

    if len(frame_starts) == 0:
        print("  ERROR: No pilot correlation peaks found.")
        print("    The pilot pattern was not detected in the received data.")
        print("    Possible causes:")
        print("      - Wrong bit file (transmitter was sending a different file)")
        print("      - Signal too weak or not present")
        print("      - Clock recovery not locked (check constellation display)")
        return None

    # ── Step 3: Estimate the frame period from detected peaks ───
    if len(frame_starts) > 1:
        periods = np.diff(frame_starts)
        median_period = int(np.median(periods))
        print(f"    Detected periods between frames: {periods.tolist()}")
        print(f"    Median period: {median_period} symbols (expected: {frame_len})")
        print(f"    Period error: {median_period - frame_len:+d} symbols")
    else:
        median_period = frame_len
        print(f"    Only one frame found, using expected period: {frame_len}")

    # ── Step 4: Pick the best pilot match for phase estimation ──
    print(f"\n  Step 4: Scoring each candidate frame start...")
    best_start = frame_starts[0]
    best_score = -1
    best_h = 0

    for start in frame_starts:
        # Check we have enough symbols for at least the pilot
        if start + k > len(y_symbols):
            continue

        # Extract candidate pilot region
        y_pilot_cand = y_symbols[start:start + k]

        # Estimate channel from this pilot region
        h_cand = estimate_phase_from_pilots(y_pilot_cand, pilot_bpsk)

        # Phase-correct and hard-detect the pilot
        y_corrected = correct_phase(y_pilot_cand, h_cand)
        detected_bits = bpsk_to_bits(y_corrected)

        # Count matching pilot bits
        matches = int(np.sum(detected_bits == np.array(pilot_bits, dtype=np.uint8)))

        print(f"    Frame at symbol {start:6d}: "
              f"|h|={abs(h_cand):.3f}, "
              f"angle={np.degrees(np.angle(h_cand)):+.1f}°, "
              f"pilot match={matches}/{k}")

        if matches > best_score:
            best_score = matches
            best_start = start
            best_h = h_cand

    print(f"\n  Using frame at symbol {best_start} for phase estimate "
          f"({best_score}/{k} pilot matches)")

    # ── Step 5: Check for 180° phase ambiguity ──────────────────
    phase_deg = np.degrees(np.angle(best_h))
    is_inverted = False

    if best_score < k * 0.6:
        # Poor match — try the opposite phase
        h_flipped = -best_h
        y_pilot_test = y_symbols[best_start:best_start + k]
        y_test_corrected = correct_phase(y_pilot_test, h_flipped)
        detected_flipped = bpsk_to_bits(y_test_corrected)
        matches_flipped = int(np.sum(
            detected_flipped == np.array(pilot_bits, dtype=np.uint8)
        ))

        if matches_flipped > best_score:
            print(f"\n  180° phase ambiguity detected!")
            print(f"    Original phase: {phase_deg:+.1f}° → {best_score}/{k} matches")
            print(f"    Flipped phase:  {np.degrees(np.angle(h_flipped)):+.1f}° "
                  f"→ {matches_flipped}/{k} matches")
            print(f"    Using flipped phase.")
            best_h = h_flipped
            best_score = matches_flipped
            is_inverted = True
            phase_deg = np.degrees(np.angle(best_h))

    # ── Step 6: Extract ALL complete frames using the estimated h ─
    print(f"\n  Step 6: Extracting all complete frames with phase correction")
    print(f"    Channel estimate: h = {best_h:.4f}")
    print(f"    Channel phase: {phase_deg:+.1f}°")
    print()

    all_frames = []
    total_pilot_errors = 0

    for start in frame_starts:
        # Check if we have a complete frame
        if start + frame_len > len(y_symbols):
            print(f"    Frame at {start:6d}: INCOMPLETE (only {len(y_symbols) - start} symbols available)")
            continue

        # Extract and phase-correct the complete frame
        y_frame_raw = y_symbols[start:start + frame_len]
        y_frame_corrected = correct_phase(y_frame_raw, best_h)

        # Split into pilot and data
        y_pilots = y_frame_corrected[:k]
        y_data = y_frame_corrected[k:]

        # Verify pilot
        pilot_detected = bpsk_to_bits(y_pilots)
        pilot_errors = int(np.sum(
            pilot_detected != np.array(pilot_bits, dtype=np.uint8)
        ))
        total_pilot_errors += pilot_errors

        # Ground truth for the data portion
        data_bits_true = data_bits_all.copy()

        print(f"    Frame at {start:6d}: pilot errors = {pilot_errors}/{k}")

        # Store this frame
        all_frames.append({
            "y_frame": y_frame_corrected,
            "y_pilots": y_pilots,
            "y_data": y_data,
            "data_bits_true": data_bits_true,
            "pilot_errors": pilot_errors,
            "frame_start": start,
        })

    print(f"\n  Extracted {len(all_frames)} complete frames")
    print(f"  Total pilot errors: {total_pilot_errors} / {len(all_frames) * k}")
    print(f"  Average pilot errors per frame: {total_pilot_errors / len(all_frames):.2f}")
    print()

    return {
        "frame_starts": frame_starts,
        "h_hat": best_h,
        "phase_deg": phase_deg,
        "is_inverted": is_inverted,
        "all_frames": all_frames,
        "n_frames_found": len(all_frames),
        "frame_period": median_period,
    }


# ═══════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER: load files and align in one call
# ═══════════════════════════════════════════════════════════════

def load_and_align(symbols_path, bits_path, pilot_bits=None):
    """
    Load received_symbols.bin and bits_000.bin, run alignment,
    return the result dict.

    Parameters
    ----------
    symbols_path : str — path to received_symbols.bin
                   (complex64, 1 sample/symbol, from clock recovery)
    bits_path    : str — path to the transmitted bit file
                   (uint8, each byte is 0 or 1)
    pilot_bits   : list or None — pilot pattern (default: PILOT_BITS)

    Returns
    -------
    result dict from align_frame(), or None if alignment failed
    """
    # Load received symbols
    # These are complex64: pairs of float32 (I, Q, I, Q, ...)
    # written by GNURadio's file_sink block after clock recovery.
    y_symbols = np.fromfile(symbols_path, dtype=np.complex64)
    print(f"Loaded {len(y_symbols)} symbols from {symbols_path}")

    # Load transmitted bits
    # These are uint8: one byte per bit, each either 0 or 1.
    # Written by make_bit_files.py.
    tx_bits = np.fromfile(bits_path, dtype=np.uint8)
    print(f"Loaded {len(tx_bits)} bits from {bits_path}")
    print()

    # Run alignment
    result = align_frame(y_symbols, tx_bits, pilot_bits)

    return result


# ═══════════════════════════════════════════════════════════════
#  COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Align received symbols with transmitted bit file "
                    "using pilot correlation"
    )
    parser.add_argument(
        "--symbols", type=str, required=True,
        help="Path to received_symbols.bin (complex64, 1 sample/symbol)"
    )
    parser.add_argument(
        "--bits", type=str, required=True,
        help="Path to transmitted bit file (uint8, 0s and 1s)"
    )
    args = parser.parse_args()

    # Run alignment
    result = load_and_align(args.symbols, args.bits)

    if result is None:
        print("Alignment failed.")
        sys.exit(1)

    # Print summary
    print(f"═══ Alignment summary ═══")
    print(f"  Frames found      : {result['n_frames_found']}")
    print(f"  Frame period      : {result['frame_period']} symbols")
    print(f"  Channel estimate  : h = {result['h_hat']:.4f}")
    print(f"  Channel magnitude : |h| = {abs(result['h_hat']):.4f}")
    print(f"  Channel phase     : {result['phase_deg']:+.1f}°")
    print(f"  Phase inverted    : {'Yes (180° Costas ambiguity corrected)' if result['is_inverted'] else 'No'}")
    print()

    # Quick SER estimate using simple hard detection across all frames
    # (this is NOT the MMSE baseline — just a quick sanity check)
    if result['n_frames_found'] > 0:
        total_errors = 0
        total_symbols = 0

        for i, frame in enumerate(result['all_frames']):
            y_data = frame['y_data']
            true_bits = frame['data_bits_true']
            n = min(len(y_data), len(true_bits))

            detected = bpsk_to_bits(y_data[:n])
            errors = int(np.sum(detected != true_bits[:n]))
            total_errors += errors
            total_symbols += n

            frame_ser = errors / n if n > 0 else 0
            print(f"  Frame {i+1}: {errors}/{n} errors, SER = {frame_ser:.4f} ({frame_ser*100:.2f}%)")

        avg_ser = total_errors / total_symbols if total_symbols > 0 else 0
        print(f"\n  Overall SER (hard detection, {result['n_frames_found']} frames):")
        print(f"    {total_errors} errors in {total_symbols} symbols → SER = {avg_ser:.4f} ({avg_ser*100:.2f}%)")
        print()

    # Save aligned data for downstream processing
    output_path = args.bits.replace(".bin", "_aligned.npz")

    # Collect all frame data into arrays
    all_y_data = []
    all_y_pilots = []
    all_data_bits_true = []
    all_pilot_errors = []
    all_frame_starts = []

    for frame in result['all_frames']:
        all_y_data.append(frame['y_data'])
        all_y_pilots.append(frame['y_pilots'])
        all_data_bits_true.append(frame['data_bits_true'])
        all_pilot_errors.append(frame['pilot_errors'])
        all_frame_starts.append(frame['frame_start'])

    np.savez(
        output_path,
        # All frames' data (list of arrays)
        all_y_data=np.array(all_y_data, dtype=object),
        all_y_pilots=np.array(all_y_pilots, dtype=object),
        all_data_bits_true=np.array(all_data_bits_true, dtype=object),
        # Channel estimate (use this as the starting point for MMSE)
        h_hat=result['h_hat'],
        # Alignment metadata
        frame_starts=np.array(all_frame_starts, dtype=np.int32),
        frame_period=result['frame_period'],
        phase_deg=result['phase_deg'],
        is_inverted=result['is_inverted'],
        pilot_errors=np.array(all_pilot_errors, dtype=np.int32),
        pilot_bits=np.array(PILOT_BITS, dtype=np.uint8),
        n_frames=result['n_frames_found'],
    )
    print(f"Saved aligned data to: {output_path}")
    print(f"  Load with:")
    print(f"    data = np.load('{output_path}', allow_pickle=True)")
    print(f"    all_y_data = data['all_y_data']           # list of phase-corrected data frames")
    print(f"    all_data_bits_true = data['all_data_bits_true']  # ground truth for each frame")
    print(f"    h_hat = data['h_hat']                     # channel estimate from pilots")
    print(f"    n_frames = data['n_frames']               # number of frames extracted")


if __name__ == "__main__":
    main()
