#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMSE Baseline for BPSK Symbol Detection
========================================

Implements the two classical baselines from the DEFINED paper (Section V-A):

  1. MMSE    — Estimate h from k pilot pairs, detect all remaining symbols
               using that fixed estimate.

  2. MMSE-DF — Same initial estimate from k pilots, but after each detection
               the decided symbol is folded back in as if it were a new pilot,
               and h is re-estimated from the growing set.

Both operate at the *symbol rate* (one complex sample per symbol), which is
the abstraction layer used in the paper.  A helper is included to downsample
raw IQ captured at the sample rate.

System model (SISO, narrowband block-fading):

    y_t = h * x_t + z_t

    h   ~ CN(0, 1)   (Rayleigh fading, unit variance)
    x_t in {-1, +1}  (BPSK constellation)
    z_t ~ CN(0, sigma^2)

MMSE channel estimate from k pilot pairs:

    h_hat = (sum_i y_i * x_i^*) / (k + sigma^2)

BPSK detection given h_hat:

    x_hat = sign( Re( h_hat^* * y_t ) )

    which is equivalent to  argmin_{x in {-1,+1}} | h_hat * x - y_t |^2
"""

import numpy as np
from typing import Tuple, Optional

# ── Pilot sequence (must match transmitter) ─────────────────────
PILOT_BITS = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
# ────────────────────────────────────────────────────────────────


# ================================================================
#  Core functions
# ================================================================

def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """Map {0, 1} bits to BPSK symbols {-1+0j, +1+0j}."""
    return (2 * bits.astype(np.complex64) - 1)


def mmse_estimate_h(y_pilots: np.ndarray,
                    x_pilots: np.ndarray,
                    sigma2: float) -> complex:
    """
    MMSE channel estimate for a SISO Rayleigh fading channel.

    h_hat = (1 / (k + sigma^2)) * sum_i  y_i * conj(x_i)

    This is Eq. (5) in the paper specialised to SISO (Nr=Nt=1)
    with the Rayleigh prior h ~ CN(0,1).

    Parameters
    ----------
    y_pilots : (k,) complex array — received pilot samples
    x_pilots : (k,) complex array — known transmitted pilot symbols
    sigma2   : float — noise variance  (sigma^2 = 1/SNR_linear)

    Returns
    -------
    h_hat : complex — scalar channel estimate
    """
    k = len(y_pilots)
    h_hat = np.sum(y_pilots * np.conj(x_pilots)) / (k + sigma2)
    return h_hat


def detect_bpsk(y: np.ndarray, h_hat: complex) -> np.ndarray:
    """
    ML detection of BPSK symbols given a channel estimate.

    x_hat = sign( Re( conj(h_hat) * y ) )

    Parameters
    ----------
    y     : (n,) complex array — received samples
    h_hat : complex — channel estimate

    Returns
    -------
    x_hat : (n,) array of {-1, +1}
    """
    projections = np.real(np.conj(h_hat) * y)
    x_hat = np.sign(projections)
    # sign(0) = 0 in numpy; map to +1 by convention
    x_hat[x_hat == 0] = 1.0
    return x_hat


def compute_ser(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Symbol error rate: fraction of symbols where x_hat != x_true."""
    return np.mean(x_true != x_hat)


# ================================================================
#  MMSE baseline (fixed pilot estimate)
# ================================================================

def mmse_detect(y_all: np.ndarray,
                x_pilots_bpsk: np.ndarray,
                sigma2: float,
                k: Optional[int] = None
                ) -> Tuple[np.ndarray, complex, float]:
    """
    Standard MMSE baseline: estimate h from k pilots, detect the rest.

    Parameters
    ----------
    y_all         : (T,) complex — full frame of received symbols
    x_pilots_bpsk : (k,) complex — known pilot symbols in {-1, +1}
    sigma2        : float — noise variance
    k             : int, optional — number of pilots (default: len(x_pilots_bpsk))

    Returns
    -------
    x_hat_data : (T-k,) array of detected symbols {-1, +1}
    h_hat      : complex — the channel estimate used
    ser        : float — SER over the data portion (NaN if no ground truth)
    """
    if k is None:
        k = len(x_pilots_bpsk)

    y_pilots = y_all[:k]
    y_data = y_all[k:]

    h_hat = mmse_estimate_h(y_pilots, x_pilots_bpsk, sigma2)
    x_hat_data = detect_bpsk(y_data, h_hat)

    return x_hat_data, h_hat, None


# ================================================================
#  MMSE-DF baseline (decision feedback, paper Section V-A-3)
# ================================================================

def mmse_df_detect(y_all: np.ndarray,
                   x_pilots_bpsk: np.ndarray,
                   sigma2: float,
                   k: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MMSE with decision feedback: after each detection, fold the decided
    symbol back in and re-estimate h from the expanded set.

    This matches the MMSE-DF baseline in the paper.

    Parameters
    ----------
    y_all         : (T,) complex — full frame of received symbols
    x_pilots_bpsk : (k,) complex — known pilot symbols
    sigma2        : float — noise variance

    Returns
    -------
    x_hat_data   : (T-k,) detected symbols {-1, +1}
    h_hat_trace  : (T-k,) channel estimates after each feedback step
    ser_trace    : (T-k,) cumulative SER after each detection
    """
    if k is None:
        k = len(x_pilots_bpsk)

    T = len(y_all)
    y_pilots = y_all[:k]

    # Running accumulators for the MMSE numerator and denominator
    # h_hat = numerator / denominator
    numerator = np.sum(y_pilots * np.conj(x_pilots_bpsk))
    count = k  # number of pairs used so far

    x_hat_list = []
    h_hat_trace = []

    for t in range(k, T):
        # Current estimate
        h_hat = numerator / (count + sigma2)
        h_hat_trace.append(h_hat)

        # Detect this symbol
        x_hat_t = detect_bpsk(y_all[t:t+1], h_hat)[0]
        x_hat_list.append(x_hat_t)

        # Fold decision back in as a noisy pilot
        numerator += y_all[t] * np.conj(x_hat_t)
        count += 1

    x_hat_data = np.array(x_hat_list)
    h_hat_trace = np.array(h_hat_trace)

    return x_hat_data, h_hat_trace, None


# ================================================================
#  Full pipeline: raw IQ  →  symbol-rate  →  detect  →  SER
# ================================================================

def downsample_to_symbols(iq_samples: np.ndarray,
                          sps: int,
                          offset: int = 0) -> np.ndarray:
    """
    Downsample oversampled IQ to one sample per symbol by averaging
    each block of `sps` samples (matched filter for rectangular pulses).

    Parameters
    ----------
    iq_samples : (N,) complex — raw IQ at sample rate
    sps        : int — samples per symbol
    offset     : int — sample offset for symbol alignment (0..sps-1)

    Returns
    -------
    symbols : (N//sps,) complex — one complex value per symbol
    """
    # Apply offset
    iq = iq_samples[offset:]
    # Trim to an integer number of symbols
    n_symbols = len(iq) // sps
    iq = iq[:n_symbols * sps]
    # Reshape and average each symbol period (optimal for rect pulses)
    symbols = iq.reshape(n_symbols, sps).mean(axis=1)
    return symbols


def estimate_snr(y_pilots: np.ndarray,
                 x_pilots: np.ndarray,
                 h_hat: complex) -> Tuple[float, float]:
    """
    Rough SNR estimate from pilot residuals.

    Returns (snr_linear, sigma2).
    """
    residuals = y_pilots - h_hat * x_pilots
    sigma2_est = np.mean(np.abs(residuals) ** 2)
    signal_power = np.abs(h_hat) ** 2
    snr_linear = signal_power / max(sigma2_est, 1e-12)
    return snr_linear, sigma2_est


def run_baseline(y_symbols: np.ndarray,
                 pilot_bits: list = PILOT_BITS,
                 sigma2: Optional[float] = None,
                 true_data_bits: Optional[np.ndarray] = None,
                 use_df: bool = False) -> dict:
    """
    Run the full MMSE (or MMSE-DF) baseline on symbol-rate data.

    Parameters
    ----------
    y_symbols      : (T,) complex — received symbols (1 per symbol period)
    pilot_bits     : list of {0,1} — known pilot bit pattern
    sigma2         : float or None — noise variance; if None, estimated from pilots
    true_data_bits : (T-k,) array of {0,1} or None — ground truth for SER
    use_df         : bool — if True, use MMSE-DF instead of plain MMSE

    Returns
    -------
    dict with keys:
        x_hat_bits   — (T-k,) detected bits {0, 1}
        x_hat_bpsk   — (T-k,) detected symbols {-1, +1}
        h_hat        — channel estimate (scalar or trace for DF)
        sigma2       — noise variance used
        snr_db       — estimated SNR in dB
        ser          — SER against ground truth (None if no truth given)
    """
    k = len(pilot_bits)
    x_pilots = bits_to_bpsk(np.array(pilot_bits))

    # If sigma2 unknown, do a rough 2-pass estimate
    if sigma2 is None:
        h_rough = mmse_estimate_h(y_symbols[:k], x_pilots, sigma2=0.0)
        _, sigma2 = estimate_snr(y_symbols[:k], x_pilots, h_rough)
        # Re-estimate with the sigma2 we just got
        # (one iteration is usually enough for a rough baseline)

    if use_df:
        x_hat_bpsk, h_hat, _ = mmse_df_detect(y_symbols, x_pilots, sigma2)
    else:
        x_hat_bpsk, h_hat, _ = mmse_detect(y_symbols, x_pilots, sigma2)

    # Convert detected symbols back to bits: -1 -> 0, +1 -> 1
    x_hat_bits = ((x_hat_bpsk + 1) / 2).astype(np.uint8)

    # Compute SER if ground truth available
    ser = None
    if true_data_bits is not None:
        x_true_bpsk = bits_to_bpsk(np.array(true_data_bits[:len(x_hat_bpsk)]))
        ser = compute_ser(x_true_bpsk, x_hat_bpsk)

    # SNR estimate
    if not use_df:
        snr_lin, _ = estimate_snr(y_symbols[:k], x_pilots, h_hat)
    else:
        snr_lin, _ = estimate_snr(y_symbols[:k], x_pilots, h_hat[0])
    snr_db = 10 * np.log10(max(snr_lin, 1e-12))

    return {
        "x_hat_bits": x_hat_bits,
        "x_hat_bpsk": x_hat_bpsk,
        "h_hat": h_hat,
        "sigma2": sigma2,
        "snr_db": snr_db,
        "ser": ser,
    }


# ================================================================
#  Simulation (no hardware needed — validates the math)
# ================================================================

def simulate_and_test(snr_db: float = 10.0,
                      T: int = 200,
                      k: int = 16,
                      seed: int = 42):
    """
    Synthetic test: generate a block-fading SISO BPSK frame,
    run both MMSE and MMSE-DF, print results.
    """
    rng = np.random.default_rng(seed)

    # Channel and noise
    h = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    sigma2 = 10 ** (-snr_db / 10)
    sigma = np.sqrt(sigma2)

    # Transmitted bits and BPSK symbols
    bits = rng.integers(0, 2, size=T, dtype=np.uint8)
    # Force first k bits to be our pilot pattern (repeated if needed)
    pilot = np.array(PILOT_BITS, dtype=np.uint8)
    pilot_extended = np.tile(pilot, (k // len(pilot)) + 1)[:k]
    bits[:k] = pilot_extended

    x = bits_to_bpsk(bits)

    # Received signal
    noise = (rng.standard_normal(T) + 1j * rng.standard_normal(T)) * sigma / np.sqrt(2)
    y = h * x + noise

    print(f"=== Simulation: SNR={snr_db} dB, T={T}, k={k} pilots ===")
    print(f"True h = {h:.4f}  (|h| = {abs(h):.4f})")
    print(f"sigma^2 = {sigma2:.6f}")
    print()

    # MMSE baseline
    result_mmse = run_baseline(y, pilot_bits=list(pilot_extended),
                               sigma2=sigma2,
                               true_data_bits=bits[k:],
                               use_df=False)
    print(f"MMSE     :  h_hat = {result_mmse['h_hat']:.4f}  "
          f"|h_hat| = {abs(result_mmse['h_hat']):.4f}  "
          f"SER = {result_mmse['ser']:.6f}  "
          f"SNR_est = {result_mmse['snr_db']:.1f} dB")

    # MMSE-DF baseline
    result_df = run_baseline(y, pilot_bits=list(pilot_extended),
                             sigma2=sigma2,
                             true_data_bits=bits[k:],
                             use_df=True)
    print(f"MMSE-DF  :  h_hat[0] = {result_df['h_hat'][0]:.4f}  "
          f"h_hat[-1] = {result_df['h_hat'][-1]:.4f}  "
          f"SER = {result_df['ser']:.6f}  "
          f"SNR_est = {result_df['snr_db']:.1f} dB")

    # Compare to perfect CSI (oracle bound)
    x_hat_oracle = detect_bpsk(y[k:], h)
    ser_oracle = compute_ser(bits_to_bpsk(bits[k:]), x_hat_oracle)
    print(f"Oracle   :  (perfect h)  SER = {ser_oracle:.6f}")
    print()

    # Show how MMSE-DF SER evolves with context length
    print("MMSE-DF SER vs context length (first 20 steps):")
    x_true_data = bits_to_bpsk(bits[k:])
    for i in [1, 2, 5, 10, 20, 50, min(100, T-k)]:
        if i <= len(result_df['x_hat_bpsk']):
            partial_ser = compute_ser(x_true_data[:i], result_df['x_hat_bpsk'][:i])
            print(f"  context={k}+{i:3d} = {k+i:3d} total  →  SER = {partial_ser:.4f}")


if __name__ == "__main__":
    # Run at a few SNR points to see the trend
    for snr in [0, 5, 10, 15, 20]:
        simulate_and_test(snr_db=snr, T=500, k=len(PILOT_BITS))
        print("-" * 60)
