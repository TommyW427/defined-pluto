#!/usr/bin/env python3
"""
Configuration for DEFINED ML Pipeline
Matches the PlutoSDR testbed parameters
"""

import dataclasses
from typing import Tuple, List
import numpy as np


@dataclasses.dataclass
class SDRConfig:
    """PlutoSDR hardware configuration - must match transmitter/receiver"""
    center_freq: float = 915e6          # 915 MHz
    sample_rate: float = 250000         # 250 kHz
    symbol_rate: float = 10000          # 10 kHz
    samples_per_symbol: int = 25        # sps = sample_rate / symbol_rate

    # Frame structure from make_bit_files.py
    preamble_bits: int = 2000           # Synchronization sequence
    pilot_bits: int = 64                # Original pilot length
    data_bits: int = 17936              # Payload


@dataclasses.dataclass
class DEFINEDConfig:
    """DEFINED model and experiment configuration"""

    # === Modulation ===
    modulation: str = "BPSK"            # Start with BPSK, then QPSK, 16QAM
    constellation_size: int = 2         # 2 for BPSK, 4 for QPSK, 16 for 16QAM

    # === Frame structure (for ML training/inference) ===
    # Keep T modest for first experiments (16-64 symbols)
    # This is separate from the full SDR frame (20k bits)
    frame_length: int = 64              # T = total symbols in one inference frame
    min_pilots: int = 4                 # Minimum pilot symbols to test
    max_pilots: int = 32                # Maximum pilot symbols to test

    # === Transformer architecture ===
    d_model: int = 128                  # Embedding dimension
    n_heads: int = 4                    # Attention heads
    n_layers: int = 4                   # Transformer layers
    d_ff: int = 512                     # Feedforward dimension
    dropout: float = 0.1                # Dropout rate
    max_seq_len: int = 256              # Maximum sequence length for pos encoding

    # === Training ===
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs_pretrain: int = 100      # Pre-training on synthetic data
    num_epochs_finetune: int = 50       # Fine-tuning on real data
    weight_decay: float = 1e-5

    # === Dataset generation (synthetic) ===
    num_train_samples: int = 50000      # Training examples
    num_val_samples: int = 5000         # Validation examples
    num_test_samples: int = 10000       # Test examples

    # SNR range for training (dB)
    snr_train_min: float = -5.0
    snr_train_max: float = 20.0
    snr_test_points: List[float] = dataclasses.field(
        default_factory=lambda: [-5, 0, 5, 10, 15, 20]
    )

    # === Channel model (for synthetic pre-training) ===
    channel_type: str = "rayleigh"      # rayleigh, rician, awgn
    doppler_hz: float = 10.0            # Doppler spread (for time-varying channels)
    rician_k: float = 3.0               # K-factor for Rician fading (LOS component)

    # === Prompt construction ===
    # Vanilla ICL: use only clean pilots
    # DEFINED: add decision feedback from previous detections
    use_decision_feedback: bool = True  # False = vanilla ICL, True = DEFINED
    feedback_window: int = 16           # How many previous decisions to include

    # === Evaluation ===
    ser_threshold: float = 0.01         # Target SER (1%)

    # === Real-world testbed ===
    sdr_data_dir: str = "real_captures"
    results_dir: str = "results"
    model_dir: str = "models"


def get_constellation(modulation: str) -> np.ndarray:
    """Return constellation points for a given modulation scheme"""
    if modulation == "BPSK":
        # Maps bits: 0 -> -1, 1 -> +1
        return np.array([-1+0j, 1+0j], dtype=np.complex64)

    elif modulation == "QPSK":
        # Gray-coded QPSK
        # 00 -> π/4, 01 -> 3π/4, 11 -> 5π/4, 10 -> 7π/4
        return np.array([
            np.exp(1j * np.pi / 4),      # 00
            np.exp(1j * 3 * np.pi / 4),  # 01
            np.exp(1j * 5 * np.pi / 4),  # 11
            np.exp(1j * 7 * np.pi / 4),  # 10
        ], dtype=np.complex64) / np.sqrt(2)

    elif modulation == "16QAM":
        # 16-QAM constellation (normalized to unit average power)
        points = []
        for i in range(4):
            for q in range(4):
                # Map to [-3, -1, 1, 3] for both I and Q
                real = 2 * i - 3
                imag = 2 * q - 3
                points.append(complex(real, imag))
        constellation = np.array(points, dtype=np.complex64)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        return constellation / np.sqrt(avg_power)

    else:
        raise ValueError(f"Unknown modulation: {modulation}")


def bits_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    Convert bit array to complex symbols.

    Args:
        bits: array of bits (0s and 1s)
        modulation: "BPSK", "QPSK", or "16QAM"

    Returns:
        Complex symbol array
    """
    constellation = get_constellation(modulation)

    if modulation == "BPSK":
        # 1 bit per symbol
        return constellation[bits]

    elif modulation == "QPSK":
        # 2 bits per symbol
        bits = bits.reshape(-1, 2)
        indices = bits[:, 0] * 2 + bits[:, 1]
        return constellation[indices]

    elif modulation == "16QAM":
        # 4 bits per symbol
        bits = bits.reshape(-1, 4)
        indices = (bits[:, 0] * 8 + bits[:, 1] * 4 +
                   bits[:, 2] * 2 + bits[:, 3])
        return constellation[indices]

    else:
        raise ValueError(f"Unknown modulation: {modulation}")


def symbols_to_bits(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """
    Perform hard decisions: map received symbols to nearest constellation point,
    then convert to bits.

    Args:
        symbols: received complex symbols (potentially noisy)
        modulation: "BPSK", "QPSK", or "16QAM"

    Returns:
        Detected bit array
    """
    constellation = get_constellation(modulation)

    # Find nearest constellation point for each symbol
    # Shape: (num_symbols, num_constellation_points)
    distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])
    detected_indices = np.argmin(distances, axis=1)

    if modulation == "BPSK":
        # 1 bit per symbol
        return detected_indices.astype(np.uint8)

    elif modulation == "QPSK":
        # 2 bits per symbol
        bits = np.zeros((len(symbols), 2), dtype=np.uint8)
        bits[:, 0] = (detected_indices >> 1) & 1
        bits[:, 1] = detected_indices & 1
        return bits.flatten()

    elif modulation == "16QAM":
        # 4 bits per symbol
        bits = np.zeros((len(symbols), 4), dtype=np.uint8)
        bits[:, 0] = (detected_indices >> 3) & 1
        bits[:, 1] = (detected_indices >> 2) & 1
        bits[:, 2] = (detected_indices >> 1) & 1
        bits[:, 3] = detected_indices & 1
        return bits.flatten()

    else:
        raise ValueError(f"Unknown modulation: {modulation}")


# === Global config instances ===
sdr_config = SDRConfig()
defined_config = DEFINEDConfig()


if __name__ == "__main__":
    # Test constellation generation
    print("=== BPSK Constellation ===")
    print(get_constellation("BPSK"))

    print("\n=== QPSK Constellation ===")
    print(get_constellation("QPSK"))

    print("\n=== 16QAM Constellation ===")
    qam16 = get_constellation("16QAM")
    print(f"Shape: {qam16.shape}")
    print(f"Average power: {np.mean(np.abs(qam16)**2):.4f} (should be ~1.0)")

    # Test bit mapping
    print("\n=== Bit-to-Symbol Test (BPSK) ===")
    test_bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    test_syms = bits_to_symbols(test_bits, "BPSK")
    print(f"Bits: {test_bits}")
    print(f"Symbols: {test_syms}")

    # Test hard decision
    noisy_syms = test_syms + 0.1 * (np.random.randn(len(test_syms)) +
                                     1j * np.random.randn(len(test_syms)))
    detected = symbols_to_bits(noisy_syms, "BPSK")
    print(f"Detected: {detected}")
    print(f"Errors: {np.sum(test_bits != detected)}")
