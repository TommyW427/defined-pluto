#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMSE Baseline — Real Captured IQ Data Only
==========================================

This script takes real IQ samples captured from a PlutoSDR receiver
and compares detected symbols against the known transmitted bit file
to compute the Symbol Error Rate (SER).

There is NO synthetic simulation in this script. Both --iq and --bits
are required arguments that must correspond to an actual transmission.

Signal chain in the transmitter (Send_Signal.grc / Send_Signal.py):
-------------------------------------------------------------------
  1. File source reads bytes from disk.  Each byte is 0 or 1.
     Example: [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, ...]
              |<-------- 16 pilot bits -------->|<--- random data --->

  2. chunks_to_symbols maps each byte to a complex BPSK symbol:
       0  →  -1 + 0j
       1  →  +1 + 0j

  3. repeat(sps=25) holds each symbol for 25 consecutive samples.
     So one bit in the file becomes 25 IDENTICAL complex samples
     in the baseband waveform.  The 16 pilot bits occupy
     16 × 25 = 400 samples.  The full 20,000-bit file produces
     20,000 × 25 = 500,000 samples before the file loops.

  4. multiply_const(0.5) scales everything by 0.5 to avoid
     clipping in the PlutoSDR's DAC.

  5. The PlutoSDR upconverts to 915 MHz and transmits.

  6. The file source has repeat=True, so after all 20,000 bits
     are sent (500,000 samples), it wraps around and starts
     transmitting from bit 0 again.  The receiver capture
     starts at an arbitrary point in this repeating stream.

What this script does:
----------------------
  1. Load the transmitted bit file (the SAME file the transmitter used).
  2. Load the raw IQ capture from the receiver.
  3. Find the correct alignment:
     a. Try all 25 possible sample offsets for downsampling
        (because we don't know which of the 25 samples within
        a symbol period the capture started on).
     b. For each offset, downsample by averaging every 25 samples
        to get one complex value per symbol period.
     c. Cross-correlate the downsampled stream against the full
        known transmitted BPSK sequence to find where in the
        repeating stream the capture landed.
     d. Fine-tune by checking that the pilot bits at the candidate
        alignment actually decode correctly.
  4. Estimate the channel h from the pilot symbols using MMSE.
  5. Detect all remaining data symbols using h.
  6. Compare detected bits against the known transmitted bits → SER.

Usage:
  python3 mmse_detect_ser.py --iq received_iq.npy --bits bit_tests/bits_000.bin
"""

import numpy as np
import argparse
import sys


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION — must match the transmitter exactly
# ═══════════════════════════════════════════════════════════════════

# The pilot bit pattern prepended to every bit file by make_bit_files.py.
# These are the known symbols the receiver uses to estimate the channel.
# In the paper's notation, k = len(PILOT_BITS) = 16.
PILOT_BITS = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]

# Samples per symbol.  The transmitter's repeat block holds each
# BPSK symbol for this many consecutive samples.
# sps = sample_rate / symbol_rate = 250,000 / 10,000 = 25
SPS = 25


# ═══════════════════════════════════════════════════════════════════
#  BPSK MAPPING
# ═══════════════════════════════════════════════════════════════════

def bits_to_bpsk(bits):
    """
    Convert an array of {0, 1} bits to BPSK symbols {-1+0j, +1+0j}.

    This matches the transmitter's chunks_to_symbols block:
      symbol_table = [-1+0j, 1+0j]
      bit 0 → index 0 → -1+0j
      bit 1 → index 1 → +1+0j

    Parameters
    ----------
    bits : numpy array of uint8, values 0 or 1

    Returns
    -------
    numpy array of complex64, values -1+0j or +1+0j
    """
    return (2 * bits.astype(np.complex64) - 1)


def bpsk_to_bits(symbols):
    """
    Convert BPSK symbols back to bits.

    If the real part is positive → bit 1
    If the real part is negative → bit 0

    Parameters
    ----------
    symbols : numpy array of complex or float, values near -1 or +1

    Returns
    -------
    numpy array of uint8, values 0 or 1
    """
    return (np.real(symbols) > 0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
#  DOWNSAMPLING: raw IQ (250 kHz) → symbol rate (10 kHz)
# ═══════════════════════════════════════════════════════════════════

def downsample_to_symbols(iq_samples, sps, offset=0):
    """
    Downsample oversampled IQ to one complex value per symbol.

    The transmitter sends each BPSK symbol as `sps` identical
    consecutive samples (via the repeat block).  So the optimal
    "matched filter" for rectangular pulses is to average each
    block of `sps` samples.  This gives maximum SNR.

    The `offset` parameter controls which sample within the first
    symbol period we start averaging from.  Since the receiver
    capture starts at an arbitrary time, we don't know the correct
    offset a priori — we try all sps=25 possible values and pick
    the one that gives the best pilot correlation.

    Example with sps=4 and offset=0:
      raw samples: [a a a a  b b b b  c c c c ...]
      symbols:     [   a        b        c     ...]

    Example with sps=4 and offset=2:
      raw samples: [x x | a a a a | b b b b | ...]
                    skip   avg       avg
      The first 2 samples are discarded, then we average in
      blocks of 4.

    Parameters
    ----------
    iq_samples : (N,) complex64 array — raw IQ at sample rate (250 kHz)
    sps        : int — samples per symbol (25)
    offset     : int — sample offset for alignment (0 to sps-1)

    Returns
    -------
    (M,) complex64 array — one complex value per symbol period
    where M = (N - offset) // sps
    """
    # Discard the first `offset` samples to align to symbol boundaries
    iq = iq_samples[offset:]

    # Figure out how many complete symbols we can form
    n_symbols = len(iq) // sps

    # Trim to exact multiple of sps, reshape into (n_symbols, sps),
    # and average each row to get one value per symbol
    iq_trimmed = iq[:n_symbols * sps]
    symbols = iq_trimmed.reshape(n_symbols, sps).mean(axis=1)

    return symbols


# ═══════════════════════════════════════════════════════════════════
#  FRAME ALIGNMENT
# ═══════════════════════════════════════════════════════════════════

def cross_correlate_magnitude(y_symbols, reference_bpsk):
    """
    Compute the magnitude of the sliding cross-correlation between
    the received symbol stream and a known reference sequence.

    This tells us: at each possible starting position in y_symbols,
    how well does the received data match the reference?  The position
    with the highest correlation is where the transmitted frame starts.

    Uses FFT-based correlation for speed (O(N log N) instead of O(N*M)).

    Parameters
    ----------
    y_symbols      : (N,) complex array — received symbols
    reference_bpsk : (M,) complex array — known transmitted BPSK sequence

    Returns
    -------
    (N - M + 1,) float array — |correlation| at each offset
    """
    n_ref = len(reference_bpsk)
    n_out = len(y_symbols) - n_ref + 1

    if n_out <= 0:
        # Received stream is shorter than one frame — can't correlate
        return np.array([])

    # Standard FFT cross-correlation:
    # corr[k] = sum_i y[k+i] * conj(ref[i])
    # Implemented as ifft(fft(y) * fft(conj(ref_reversed)))
    n_fft = len(y_symbols) + n_ref - 1
    Y = np.fft.fft(y_symbols, n=n_fft)
    R = np.fft.fft(np.conj(reference_bpsk[::-1]), n=n_fft)
    corr_full = np.fft.ifft(Y * R)

    # Extract the valid portion (where the full reference overlaps y)
    corr = corr_full[n_ref - 1: n_ref - 1 + n_out]

    return np.abs(corr)


def score_alignment(y_symbols, frame_start, pilot_bpsk):
    """
    Test a candidate frame alignment by checking how many pilot
    symbols decode correctly.

    At the candidate `frame_start`, the first k=16 symbols should
    correspond to the known pilot.  We estimate a rough channel h
    from those symbols and count how many hard-detect correctly.

    A correct alignment with reasonable SNR should give 16/16 or
    close to it.  A wrong alignment gives roughly 8/16 (random).

    Parameters
    ----------
    y_symbols   : (N,) complex — the full downsampled received stream
    frame_start : int — candidate index where the frame begins
    pilot_bpsk  : (k,) complex — known pilot as BPSK symbols

    Returns
    -------
    matches : int — number of correctly decoded pilot symbols (0 to k)
    h_rough : complex — the rough channel estimate at this alignment
    """
    k = len(pilot_bpsk)

    # Check we have enough symbols after frame_start
    if frame_start + k > len(y_symbols):
        return 0, 0.0

    # Extract the candidate pilot region
    y_pilot_candidate = y_symbols[frame_start:frame_start + k]

    # Rough channel estimate: just the sample correlation divided by k
    # (ignoring the sigma^2 term — this is only for scoring, not final use)
    h_rough = np.sum(y_pilot_candidate * np.conj(pilot_bpsk)) / k

    # If h is essentially zero, the signal isn't there
    if abs(h_rough) < 1e-12:
        return 0, h_rough

    # Hard-detect the pilot symbols using this rough h
    # x_hat = sign(Re(conj(h) * y))
    detected = np.sign(np.real(np.conj(h_rough) * y_pilot_candidate))

    # Count how many match the known pilot
    matches = int(np.sum(detected == np.real(pilot_bpsk)))

    return matches, h_rough


def find_best_alignment(iq_raw, sps, tx_bpsk, pilot_bpsk):
    """
    Search over all possible sample offsets and frame positions
    to find where the transmitted frame starts in the received IQ.

    The search has two levels:
      Outer loop: sample offset (0 to sps-1)
        — controls which of the 25 samples within a symbol period
          we treat as the boundary.
      Inner loop: frame position within the downsampled stream
        — the transmitter loops the file, so the first symbol of
          the file could appear anywhere in the received stream.

    For each (offset, position) pair, we score it by counting how
    many pilot symbols decode correctly.  The pair with the most
    correct pilots wins.

    Parameters
    ----------
    iq_raw     : (N,) complex64 — raw IQ samples from the receiver
    sps        : int — samples per symbol (25)
    tx_bpsk    : (L,) complex — full transmitted frame as BPSK symbols
    pilot_bpsk : (k,) complex — just the pilot portion as BPSK symbols

    Returns
    -------
    best_offset      : int — best sample offset (0 to sps-1)
    best_frame_start : int — best frame start index in the downsampled stream
    best_y           : (M,) complex — downsampled symbols at best offset
    best_pilot_match : int — number of pilot symbols that decoded correctly
    """
    k = len(pilot_bpsk)
    n_tx = len(tx_bpsk)

    # Track the best alignment found so far
    best_offset = 0
    best_frame_start = 0
    best_pilot_match = 0
    best_corr_peak = 0.0
    best_y = None

    print(f"Searching all {sps} sample offsets for frame alignment...")
    print(f"  Each offset produces a different set of symbol-rate samples.")
    print(f"  For each, we cross-correlate against the {n_tx}-symbol")
    print(f"  transmitted frame to find where it starts.")
    print()

    for offset in range(sps):
        # Downsample at this offset: average every `sps` samples
        # starting from sample number `offset`
        y_test = downsample_to_symbols(iq_raw, sps, offset)

        # Need at least one full frame plus some margin
        if len(y_test) < n_tx + 10:
            continue

        # Cross-correlate against the full transmitted frame.
        # We search up to 3 frame lengths in case the capture
        # spans multiple repetitions of the file.
        search_len = min(len(y_test), n_tx * 3)
        corr = cross_correlate_magnitude(y_test[:search_len], tx_bpsk)

        if len(corr) == 0:
            continue

        # The correlation has peaks wherever a new copy of the
        # transmitted frame begins.  Get the top few peaks.
        n_peaks = min(5, len(corr))
        peak_indices = np.argsort(corr)[-n_peaks:]

        # Fine-tune: for each coarse peak, try ±5 symbol shifts.
        # This accounts for small errors in the FFT correlation peak
        # (e.g., due to noise or partial symbol overlap).
        for peak_idx in peak_indices:
            search_lo = max(0, peak_idx - 5)
            search_hi = min(len(y_test) - n_tx, peak_idx + 6)

            for shift in range(search_lo, search_hi):
                # Score: how many of the 16 pilot symbols decode correctly?
                matches, _ = score_alignment(y_test, shift, pilot_bpsk)
                peak_val = corr[min(peak_idx, len(corr) - 1)]

                # Keep this alignment if it has more pilot matches,
                # or same matches but stronger correlation
                if matches > best_pilot_match or (
                    matches == best_pilot_match and peak_val > best_corr_peak
                ):
                    best_pilot_match = matches
                    best_corr_peak = peak_val
                    best_offset = offset
                    best_frame_start = shift
                    best_y = y_test

    print(f"Best alignment found:")
    print(f"  Sample offset   : {best_offset} / {sps}")
    print(f"    (we start averaging from sample #{best_offset} within")
    print(f"     each symbol period of {sps} samples)")
    print(f"  Frame start     : symbol index {best_frame_start}")
    print(f"    (the transmitted file's bit 0 corresponds to this")
    print(f"     position in the downsampled received stream)")
    print(f"  Pilot match     : {best_pilot_match} / {k}")
    print(f"    ({best_pilot_match} of the {k} known pilot symbols")
    print(f"     decoded correctly at this alignment)")
    print()

    return best_offset, best_frame_start, best_y, best_pilot_match


# ═══════════════════════════════════════════════════════════════════
#  MMSE CHANNEL ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def mmse_estimate_h(y_pilots, x_pilots, sigma2):
    """
    MMSE channel estimate for a SISO Rayleigh fading channel.

    Model:  y_t = h * x_t + z_t
    Prior:  h ~ CN(0, 1)  (Rayleigh fading, unit variance)
    Noise:  z_t ~ CN(0, sigma^2)

    The MMSE estimate given k pilot pairs {(y_i, x_i)} is:

        h_hat = sum_i(y_i * conj(x_i)) / (k + sigma^2)

    This is Eq. (5) from the DEFINED paper, specialized to the
    SISO case (Nr = Nt = 1).

    The denominator (k + sigma^2) comes from the MMSE formula:
      h_hat = (X^H X + sigma^2 I)^{-1} X^H Y
    For SISO with unit-power BPSK symbols, X^H X = k (scalar),
    so it simplifies to the formula above.

    Parameters
    ----------
    y_pilots : (k,) complex — received samples during pilot period
    x_pilots : (k,) complex — known transmitted pilot symbols
    sigma2   : float — noise variance (sigma^2 = 1 / SNR_linear)

    Returns
    -------
    h_hat : complex — scalar channel estimate
    """
    k = len(y_pilots)
    h_hat = np.sum(y_pilots * np.conj(x_pilots)) / (k + sigma2)
    return h_hat


def estimate_sigma2_from_pilots(y_pilots, x_pilots, h_hat):
    """
    Estimate the noise variance from the residuals between
    what we received and what we would have received with
    perfect channel knowledge.

    residual_i = y_i - h_hat * x_i

    If h_hat is close to the true h, these residuals are
    approximately the noise samples z_i, so their average
    power estimates sigma^2.

    Parameters
    ----------
    y_pilots : (k,) complex — received pilot samples
    x_pilots : (k,) complex — known pilot symbols
    h_hat    : complex — channel estimate

    Returns
    -------
    sigma2 : float — estimated noise variance
    """
    residuals = y_pilots - h_hat * x_pilots
    sigma2 = np.mean(np.abs(residuals) ** 2)
    return sigma2


# ═══════════════════════════════════════════════════════════════════
#  BPSK DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_bpsk(y, h_hat):
    """
    Maximum-likelihood BPSK detection given a channel estimate.

    For each received sample y_t, we decide which BPSK symbol
    was transmitted:

        x_hat = argmin_{x ∈ {-1, +1}}  |h_hat * x - y_t|^2

    This simplifies to:

        x_hat = sign( Re( conj(h_hat) * y_t ) )

    Intuition: conj(h_hat) * y_t rotates the received signal
    so that the channel phase is removed.  Then we just check
    whether the real part is positive (+1) or negative (-1).

    Parameters
    ----------
    y     : (n,) complex array — received samples
    h_hat : complex — channel estimate

    Returns
    -------
    x_hat : (n,) float array — detected symbols, each -1.0 or +1.0
    """
    # Rotate by conjugate of h to undo channel phase, take real part
    projections = np.real(np.conj(h_hat) * y)

    # Decide: positive → +1, negative → -1
    x_hat = np.sign(projections)

    # numpy sign(0) = 0, but that shouldn't happen with real signals;
    # map to +1 by convention just in case
    x_hat[x_hat == 0] = 1.0

    return x_hat


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MMSE baseline: estimate h from pilots in real captured IQ, "
                    "detect BPSK data, compute SER against known transmitted bits"
    )
    parser.add_argument(
        "--iq", type=str, required=True,
        help="Path to received IQ file (.npy or .bin) captured from PlutoSDR"
    )
    parser.add_argument(
        "--bits", type=str, required=True,
        help="Path to transmitted bit file (must be the SAME file the "
             "transmitter was sending during the capture)"
    )
    parser.add_argument(
        "--sps", type=int, default=SPS,
        help=f"Samples per symbol (default: {SPS}). Must match the "
             f"transmitter's repeat block interpolation factor."
    )
    parser.add_argument(
        "--sigma2", type=float, default=None,
        help="Known noise variance. If omitted, estimated from pilot "
             "residuals (which requires a reasonable initial h estimate)."
    )
    args = parser.parse_args()

    # ═════════════════════════════════════════════════════════════
    #  Step 1: Load the transmitted bit file
    # ═════════════════════════════════════════════════════════════
    # This file contains uint8 bytes, each either 0 or 1.
    # The first k bytes are the pilot pattern, the rest are data.
    # The transmitter reads this file, maps each byte to a BPSK
    # symbol, holds each symbol for sps=25 samples, scales by 0.5,
    # and sends it out the radio.  When the file ends, it loops.

    tx_bits = np.fromfile(args.bits, dtype=np.uint8)
    k = len(PILOT_BITS)                                # number of pilot symbols
    pilot_bits = np.array(PILOT_BITS, dtype=np.uint8)  # known pilot as bits
    pilot_bpsk = bits_to_bpsk(pilot_bits)              # known pilot as BPSK symbols
    tx_bpsk = bits_to_bpsk(tx_bits)                    # full file as BPSK symbols
    data_bits = tx_bits[k:]                             # everything after the pilot

    print(f"═══ Transmitted bit file ═══")
    print(f"  File         : {args.bits}")
    print(f"  Total bits   : {len(tx_bits)}")
    print(f"  Pilot bits   : {k}  {PILOT_BITS}")
    print(f"  Data bits    : {len(data_bits)}")
    print(f"  As BPSK, each bit becomes 1 symbol held for {args.sps} samples.")
    print(f"  Total samples per file loop: {len(tx_bits)} × {args.sps} "
          f"= {len(tx_bits) * args.sps}")
    print()

    # ═════════════════════════════════════════════════════════════
    #  Step 2: Load the received IQ capture
    # ═════════════════════════════════════════════════════════════
    # This is a raw array of complex64 values at the sample rate
    # (250 kHz).  Each complex value has a real (I) and imaginary (Q)
    # component, representing the baseband signal after the PlutoSDR's
    # analog-to-digital converter and downconversion from 915 MHz.

    if args.iq.endswith(".npy"):
        iq_raw = np.load(args.iq)
    else:
        # .bin files from GRC's file_sink are raw interleaved float32: I,Q,I,Q,...
        # numpy reads these as complex64 (each complex = two float32)
        iq_raw = np.fromfile(args.iq, dtype=np.complex64)

    print(f"═══ Received IQ capture ═══")
    print(f"  File         : {args.iq}")
    print(f"  Raw samples  : {len(iq_raw)}")
    print(f"  Duration     : {len(iq_raw) / (args.sps * 10000):.2f} seconds "
          f"(at {args.sps * 10000} samples/sec)")
    power_db = 10 * np.log10(np.mean(np.abs(iq_raw) ** 2) + 1e-12)
    print(f"  Mean power   : {power_db:.1f} dB")
    max_symbols = len(iq_raw) // args.sps
    print(f"  Max symbols after downsampling: {max_symbols}")
    print()

    # ═════════════════════════════════════════════════════════════
    #  Step 3: Find frame alignment
    # ═════════════════════════════════════════════════════════════
    # The receiver started capturing at an unknown point in the
    # transmitter's repeating stream.  We need to find:
    #   a) The correct sample offset (0 to 24) for downsampling
    #   b) Which symbol in the downsampled stream corresponds to
    #      bit 0 of the transmitted file
    #
    # If either is wrong, every comparison will be against the
    # wrong ground truth, giving ~50% SER (random on BPSK).

    sample_offset, frame_start, y_symbols, pilot_match = \
        find_best_alignment(iq_raw, args.sps, tx_bpsk, pilot_bpsk)

    # Check that we actually found a valid alignment
    if y_symbols is None:
        print("ERROR: Could not find any alignment.")
        print("  Is there actually a signal in the capture?")
        print("  Check the spectrum in the GRC receiver.")
        sys.exit(1)

    if pilot_match < k * 0.6:
        print(f"WARNING: Only {pilot_match}/{k} pilot symbols matched.")
        print(f"  This is barely better than random (expect ~{k//2} by chance).")
        print(f"  Possible causes:")
        print(f"    - Wrong --bits file (transmitter was sending a different file)")
        print(f"    - No signal in the capture (transmitter off or wrong frequency)")
        print(f"    - SNR too low for reliable alignment")
        print(f"  Proceeding anyway, but results may be meaningless.")
        print()

    # ═════════════════════════════════════════════════════════════
    #  Step 4: Extract the aligned frame
    # ═════════════════════════════════════════════════════════════
    # Now we know that y_symbols[frame_start] corresponds to bit 0
    # of the transmitted file.  So:
    #   y_symbols[frame_start]          ↔ tx_bits[0]  (first pilot bit)
    #   y_symbols[frame_start + 1]      ↔ tx_bits[1]  (second pilot bit)
    #   ...
    #   y_symbols[frame_start + k - 1]  ↔ tx_bits[k-1] (last pilot bit)
    #   y_symbols[frame_start + k]      ↔ tx_bits[k]   (first data bit)
    #   ...

    # Don't go past the end of either the received stream or the bit file
    frame_len = min(len(tx_bits), len(y_symbols) - frame_start)

    if frame_len < k + 1:
        print("ERROR: Not enough symbols after alignment to extract even")
        print("       one data symbol.  The capture is too short or")
        print("       alignment landed near the end of the stream.")
        sys.exit(1)

    # Slice out the aligned frame
    y_frame = y_symbols[frame_start:frame_start + frame_len]

    # Split into pilot and data portions
    y_pilots = y_frame[:k]       # received samples during known pilot
    y_data = y_frame[k:]         # received samples during unknown data

    # Trim data to match available ground truth
    n_data = min(len(y_data), len(data_bits))
    y_data = y_data[:n_data]
    data_bits_trimmed = data_bits[:n_data]

    print(f"═══ Aligned frame ═══")
    print(f"  Frame starts at symbol index {frame_start} in the received stream")
    print(f"  Frame length : {frame_len} symbols")
    print(f"  Pilot region : symbols [0:{k}]  ({k} symbols)")
    print(f"  Data region  : symbols [{k}:{k + n_data}]  ({n_data} symbols)")
    print()

    # ═════════════════════════════════════════════════════════════
    #  Step 5: Estimate the channel h from pilot symbols
    # ═════════════════════════════════════════════════════════════
    # The received pilot samples are:
    #   y_pilots[i] = h * pilot_bpsk[i] + noise[i]
    #
    # We use MMSE to estimate h.  If sigma^2 is not known,
    # we estimate it from the pilot residuals in a two-step process:
    #   1. Get a rough h (ignoring sigma^2 in the denominator)
    #   2. Compute residuals to estimate sigma^2
    #   3. Re-estimate h with the proper MMSE formula

    if args.sigma2 is not None:
        # User provided the noise variance (e.g., from a known SNR)
        sigma2 = args.sigma2
    else:
        # Step 5a: rough h estimate (no sigma^2 correction)
        h_rough = np.sum(y_pilots * np.conj(pilot_bpsk)) / k

        # Step 5b: estimate sigma^2 from pilot residuals
        sigma2 = estimate_sigma2_from_pilots(y_pilots, pilot_bpsk, h_rough)

    # Step 5c: proper MMSE estimate using sigma^2
    h_hat = mmse_estimate_h(y_pilots, pilot_bpsk, sigma2)

    # Compute estimated SNR for diagnostics
    # SNR = |h|^2 / sigma^2  (signal power / noise power)
    snr_est = abs(h_hat) ** 2 / max(sigma2, 1e-12)
    snr_db = 10 * np.log10(max(snr_est, 1e-12))

    print(f"═══ Channel estimate (MMSE, {k} pilots) ═══")
    print(f"  h_hat        = {h_hat:.6f}")
    print(f"  |h_hat|      = {abs(h_hat):.6f}")
    print(f"  angle(h_hat) = {np.degrees(np.angle(h_hat)):.1f}°")
    print(f"  sigma^2      = {sigma2:.6f}")
    print(f"  Est. SNR     = {snr_db:.1f} dB")
    print()

    # ═════════════════════════════════════════════════════════════
    #  Step 6: Verify alignment by re-detecting the pilot
    # ═════════════════════════════════════════════════════════════
    # This is a sanity check.  If the alignment is correct and
    # SNR is reasonable, we should decode most/all pilot symbols
    # correctly using our estimated h.

    x_hat_pilots = detect_bpsk(y_pilots, h_hat)
    pilot_detected_bits = bpsk_to_bits(x_hat_pilots).tolist()
    pilot_errors = int(np.sum(x_hat_pilots != np.real(pilot_bpsk)))

    print(f"═══ Pilot verification ═══")
    print(f"  Expected : {PILOT_BITS}")
    print(f"  Detected : {pilot_detected_bits}")
    print(f"  Errors   : {pilot_errors} / {k}")

    if pilot_errors == 0:
        print(f"  ✓ All {k} pilot symbols correct — alignment confirmed")
    elif pilot_errors <= 2:
        print(f"  ~ Mostly correct — alignment likely good, some noise")
    else:
        print(f"  ✗ {pilot_errors} pilot errors — alignment is probably WRONG")
        print(f"    The SER below will be unreliable.")
    print()

    # ═════════════════════════════════════════════════════════════
    #  Step 7: Detect all data symbols
    # ═════════════════════════════════════════════════════════════
    # Apply the same detection rule to every data symbol:
    #   x_hat = sign(Re(conj(h_hat) * y))
    # This uses the single h_hat estimated from pilots — it does
    # NOT update h as we go (that would be MMSE-DF, a different baseline).

    x_hat_data = detect_bpsk(y_data, h_hat)
    x_hat_bits = bpsk_to_bits(x_hat_data)

    # ═════════════════════════════════════════════════════════════
    #  Step 8: Compute Symbol Error Rate
    # ═════════════════════════════════════════════════════════════
    # Compare each detected symbol against the known transmitted symbol.
    # An error occurs when x_hat != x_true.

    x_true_bpsk = bits_to_bpsk(data_bits_trimmed)
    errors = int(np.sum(x_true_bpsk != x_hat_data))
    ser = errors / n_data

    print(f"═══ RESULTS ═══")
    print(f"  Data symbols compared : {n_data}")
    print(f"  Symbol errors         : {errors}")
    print(f"  SER                   : {ser:.6f}  ({ser * 100:.3f}%)")
    print()

    # Flag suspicious results
    if ser > 0.40:
        print(f"  ⚠  SER near 50% — this means the detections are essentially")
        print(f"     random.  Almost certainly one of:")
        print(f"       1. Wrong --bits file (not what the transmitter was sending)")
        print(f"       2. No signal in the capture")
        print(f"       3. Frame alignment failed")
        print()

    # ═════════════════════════════════════════════════════════════
    #  Step 9: Show SER distribution across the frame
    # ═════════════════════════════════════════════════════════════
    # If the channel is truly block-fading (constant during the frame),
    # the SER should be roughly uniform.  If it spikes partway through,
    # possible causes:
    #   - The channel changed (block-fading assumption violated)
    #   - Clock drift between TX and RX shifted the symbol alignment
    #   - The frame wrapped around and we're comparing against the
    #     wrong part of the bit file

    window = min(200, n_data // 5)
    if window > 0:
        print(f"SER across frame (sliding window of {window} symbols):")
        steps = max(n_data // 10, 1)
        for start in range(0, n_data - window + 1, steps):
            chunk_true = x_true_bpsk[start:start + window]
            chunk_hat = x_hat_data[start:start + window]
            chunk_ser = np.sum(chunk_true != chunk_hat) / window
            # Visual bar: each █ represents 2% SER
            bar = "█" * int(chunk_ser * 50)
            print(f"  [{start:6d}:{start + window:6d}]  "
                  f"SER={chunk_ser:.4f}  {bar}")
        print()

        # Check for non-uniform errors (suggests channel change or drift)
        block_sers = []
        for start in range(0, n_data - window + 1, window):
            chunk_true = x_true_bpsk[start:start + window]
            chunk_hat = x_hat_data[start:start + window]
            block_sers.append(np.sum(chunk_true != chunk_hat) / window)

        if len(block_sers) > 2:
            ser_std = np.std(block_sers)
            if ser_std > 0.05:
                print(f"  Note: SER varies significantly across the frame")
                print(f"  (std = {ser_std:.3f}).  Possible causes:")
                print(f"    - Channel changed during the capture")
                print(f"    - Clock drift shifted symbol alignment over time")
                print(f"    - Frame wrapped and bits no longer line up")
                print()

    # ═════════════════════════════════════════════════════════════
    #  Step 10: Save results for comparison with DEFINED
    # ═════════════════════════════════════════════════════════════
    # Save everything needed to compare MMSE against DEFINED later:
    #   - The channel estimate and noise variance
    #   - The detected bits (so you can diff against DEFINED's detections)
    #   - The SER
    #   - The alignment parameters (so DEFINED can use the same alignment)

    output_file = args.bits.replace(".bin", "_mmse_detected.npz")
    np.savez(
        output_file,
        # Channel estimation results
        h_hat=h_hat,
        sigma2=sigma2,
        # Detection results
        x_hat_bits=x_hat_bits,             # detected bits as {0, 1}
        x_hat_bpsk=x_hat_data,             # detected symbols as {-1, +1}
        true_data_bits=data_bits_trimmed,   # ground truth data bits
        ser=ser,
        # Alignment info (reuse for DEFINED pipeline)
        pilot_bits=pilot_bits,
        frame_start_symbol=frame_start,
        sample_offset=sample_offset,
        pilot_errors=pilot_errors,
    )
    print(f"Results saved to: {output_file}")
    print(f"  Load with: data = np.load('{output_file}')")
    print(f"  Access:    data['ser'], data['h_hat'], data['x_hat_bits'], etc.")


if __name__ == "__main__":
    main()
