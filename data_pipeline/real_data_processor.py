#!/usr/bin/env python3
"""
Real PlutoSDR Data Processor

Processes raw IQ data from PlutoSDR receiver using the proven approach from calc_ber.py:
- Load received_iq_raw.bin (raw IQ at 250 kHz sample rate)
- Decimate by 25 to get symbols at 10 kHz symbol rate
- Handle 180° phase inversion from Costas loop
- Work with bits directly

This avoids the variable symbol rate issues from Mueller & Muller clock recovery.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import argparse


class RealDataProcessor:
    """
    Process real captured SDR data into ML-ready format.

    Based on the proven calc_ber.py approach:
    1. Load raw IQ (250 kHz sample rate, 25 samples/symbol)
    2. Decimate by 25 to get symbols
    3. Detect bits with phase correction (handle 180° inversion)
    4. Work with bits throughout the pipeline
    """

    def __init__(self):
        # Hardware constants from SDR configuration
        self.sample_rate = 250000      # 250 kHz (from PlutoSDR)
        self.symbol_rate = 10000       # 10 kHz
        self.samples_per_symbol = 25   # Exactly 25 samples per symbol

        # Frame structure from make_bit_files.py
        self.preamble_bits = 2000
        self.pilot_bits = 64
        self.data_bits = 17936
        self.total_bits = 20000

        # BPSK constellation
        self.constellation = np.array([-1+0j, 1+0j], dtype=np.complex64)

    def load_raw_iq(self, filepath: str) -> np.ndarray:
        """
        Load raw IQ from PlutoSDR receiver.

        File: received_iq_raw.bin (output of Receive_Signal.py)
        Format: complex64, 250 kHz sample rate, 25 samples per symbol

        Returns:
            Complex IQ samples
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Raw IQ file not found: {filepath}")

        raw_iq = np.fromfile(filepath, dtype=np.complex64)

        # Validate length
        if len(raw_iq) == 0:
            raise ValueError(f"Empty raw IQ file: {filepath}")

        print(f"✓ Loaded {len(raw_iq):,} raw IQ samples from {filepath}")
        print(f"  Sample rate: {self.sample_rate/1e3:.0f} kHz")
        print(f"  Duration: {len(raw_iq)/self.sample_rate:.2f} seconds")

        return raw_iq

    def decimate_to_symbols(self, raw_iq: np.ndarray) -> np.ndarray:
        """
        Decimate raw IQ to symbol rate by taking every 25th sample.

        This is the proven approach from calc_ber.py that works reliably.

        Args:
            raw_iq: Raw IQ samples at 250 kHz

        Returns:
            Symbol-rate samples (one per symbol)
        """
        # Simple decimation: take every Nth sample
        symbols = raw_iq[::self.samples_per_symbol]

        print(f"✓ Decimated to {len(symbols):,} symbols (sps={self.samples_per_symbol})")
        print(f"  Approximate symbol duration: {len(symbols)/self.symbol_rate:.2f} seconds")

        return symbols

    def detect_bits_bpsk(
        self,
        symbols: np.ndarray,
        assume_inverted: bool = False
    ) -> np.ndarray:
        """
        Detect BPSK bits from symbols.

        Uses the proven detection from calc_ber.py:
        - Normal: bit = (symbol.real > 0)
        - Inverted (180° phase): bit = (symbol.real < 0)

        Args:
            symbols: Complex symbols
            assume_inverted: If True, assume 180° phase inversion

        Returns:
            Detected bits (0 or 1)
        """
        if assume_inverted:
            # Inverted detection (Costas loop locked at 180°)
            bits = (symbols.real < 0).astype(np.uint8)
        else:
            # Normal detection
            bits = (symbols.real > 0).astype(np.uint8)

        return bits

    def detect_frame_start(
        self,
        symbols: np.ndarray,
        threshold_factor: float = 3.0,
        window_size: int = 100
    ) -> int:
        """
        Detect start of transmission using received signal energy.

        Method: Look for sudden increase in signal power above noise floor.

        Args:
            symbols: Received symbols
            threshold_factor: Multiplier above noise floor (default 3x)
            window_size: Window for power estimation

        Returns:
            Index of transmission start
        """
        # Compute moving average power
        power = np.abs(symbols) ** 2

        # Estimate noise floor from first samples (assume no signal yet)
        noise_floor = np.mean(power[:window_size])

        # Find where power exceeds threshold
        threshold = noise_floor * threshold_factor

        # Look for sustained power above threshold
        above_threshold = power > threshold

        # Find first point where signal stays above threshold
        for i in range(len(above_threshold) - window_size):
            if np.mean(above_threshold[i:i + window_size]) > 0.8:  # 80% of window
                print(f"✓ Frame start detected at sample {i}")
                print(f"  Noise floor: {10*np.log10(noise_floor + 1e-12):.2f} dB")
                print(f"  Threshold: {10*np.log10(threshold + 1e-12):.2f} dB")
                return i

        # If no clear start found, assume it starts immediately
        print("⚠ No clear frame start detected, assuming start at 0")
        return 0

    def find_pilot_position(
        self,
        rx_bits: np.ndarray,
        pilot_ref: np.ndarray,
        search_start: int = 1900,
        search_end: int = 2200,
        check_inversion: bool = True
    ) -> Tuple[int, bool]:
        """
        Find pilot sequence position and detect phase inversion.

        Searches around the expected position (after 2000-bit preamble).

        Args:
            rx_bits: Received bits
            pilot_ref: Reference pilot bits (64 bits)
            search_start: Start of search window
            search_end: End of search window
            check_inversion: Whether to check for bit inversion

        Returns:
            Tuple of (pilot_position, is_inverted)
        """
        best_pos = search_start
        min_errors = len(pilot_ref)
        is_inverted = False

        # Clamp search window to available data
        search_end = min(search_end, len(rx_bits) - len(pilot_ref))

        for pos in range(search_start, search_end, 10):
            rx_segment = rx_bits[pos:pos + len(pilot_ref)]

            # Try normal comparison
            errors_normal = np.sum(rx_segment != pilot_ref)

            if errors_normal < min_errors:
                min_errors = errors_normal
                best_pos = pos
                is_inverted = False

            # Try inverted comparison
            if check_inversion:
                errors_inverted = np.sum(rx_segment == pilot_ref)  # Inverted
                if errors_inverted < min_errors:
                    min_errors = errors_inverted
                    best_pos = pos
                    is_inverted = True

        print(f"✓ Pilot found at position {best_pos}")
        print(f"  Expected position: {self.preamble_bits}")
        print(f"  Offset: {best_pos - self.preamble_bits:+d} bits")
        print(f"  Phase inverted: {is_inverted}")
        print(f"  Pilot errors: {min_errors}/{len(pilot_ref)}")

        return best_pos, is_inverted

    def extract_frame(
        self,
        raw_iq_file: str,
        tx_bits_file: str,
        pilot_bits_file: str
    ) -> Dict:
        """
        Extract and validate a complete frame from raw IQ.

        This is the main processing function that:
        1. Loads raw IQ
        2. Decimates to symbols
        3. Detects bits
        4. Finds pilot
        5. Corrects phase inversion
        6. Extracts data bits
        7. Computes BER

        Args:
            raw_iq_file: Path to received_iq_raw.bin
            tx_bits_file: Path to reference transmitted bits (e.g., bits_000.bin)
            pilot_bits_file: Path to reference pilot bits (pilot.bin)

        Returns:
            Dictionary with all extracted data
        """
        print("="*70)
        print("FRAME EXTRACTION")
        print("="*70)

        # Load raw IQ
        raw_iq = self.load_raw_iq(raw_iq_file)

        # Decimate to symbols
        symbols = self.decimate_to_symbols(raw_iq)

        # Load reference data
        tx_bits_ref = np.fromfile(tx_bits_file, dtype=np.uint8)
        pilot_ref = np.fromfile(pilot_bits_file, dtype=np.uint8)

        print(f"✓ Loaded reference data:")
        print(f"  TX bits: {len(tx_bits_ref)}")
        print(f"  Pilot bits: {len(pilot_ref)}")

        # Detect frame start using energy detection
        frame_start = self.detect_frame_start(symbols)

        # Detect bits (try normal first)
        rx_bits = self.detect_bits_bpsk(symbols, assume_inverted=False)

        # Pilot should be at frame_start + preamble_bits
        expected_pilot_pos = frame_start + self.preamble_bits

        # Find pilot and check for inversion (search around expected position)
        pilot_pos, is_inverted = self.find_pilot_position(
            rx_bits,
            pilot_ref,
            search_start=max(0, expected_pilot_pos - 100),
            search_end=min(len(rx_bits) - len(pilot_ref), expected_pilot_pos + 100)
        )

        # If inverted, re-detect bits with inversion
        was_inverted = is_inverted  # Save for return value
        if is_inverted:
            print("✓ Re-detecting bits with phase correction...")
            rx_bits = self.detect_bits_bpsk(symbols, assume_inverted=True)
            # Verify pilot again
            pilot_pos, _ = self.find_pilot_position(
                rx_bits,
                pilot_ref,
                search_start=pilot_pos - 10,
                search_end=pilot_pos + 10,
                check_inversion=False  # Already corrected
            )

        # Extract data section (after pilot)
        data_start_rx = pilot_pos + len(pilot_ref)
        data_start_ref = self.preamble_bits + len(pilot_ref)

        # Get data bits (align lengths)
        rx_data_bits = rx_bits[data_start_rx:data_start_rx + self.data_bits]
        tx_data_bits = tx_bits_ref[data_start_ref:data_start_ref + self.data_bits]

        min_len = min(len(rx_data_bits), len(tx_data_bits))
        rx_data_bits = rx_data_bits[:min_len]
        tx_data_bits = tx_data_bits[:min_len]

        # Compute BER
        errors = np.sum(rx_data_bits != tx_data_bits)
        ber = errors / min_len

        print(f"\n{'='*70}")
        print(f"BIT ERROR RATE")
        print(f"{'='*70}")
        print(f"  Data bits compared: {min_len:,}")
        print(f"  Bit errors: {errors:,}")
        print(f"  BER: {ber:.6f} ({ber*100:.4f}%)")
        print(f"{'='*70}")

        # Extract pilot symbols for reference
        pilot_symbols = symbols[pilot_pos:pilot_pos + len(pilot_ref)]

        # Extract data symbols
        data_symbols = symbols[data_start_rx:data_start_rx + min_len]

        # Estimate SNR from data symbols
        snr_db = self.estimate_snr_bpsk(data_symbols)
        print(f"  Estimated SNR: {snr_db:.2f} dB")

        return {
            "rx_data_bits": rx_data_bits,
            "tx_data_bits": tx_data_bits,
            "rx_pilot_symbols": pilot_symbols.astype(np.complex64),
            "tx_pilot_symbols": self.constellation[pilot_ref].astype(np.complex64),
            "rx_data_symbols": data_symbols.astype(np.complex64),
            "frame_start": frame_start,
            "pilot_position": pilot_pos,
            "is_inverted": was_inverted,
            "ber": ber,
            "num_errors": errors,
            "num_bits": min_len,
            "snr_db": snr_db,
        }

    def estimate_snr_bpsk(self, symbols: np.ndarray) -> float:
        """
        Estimate SNR from BPSK symbols.

        Method: assumes symbols should be at ±1, noise is the deviation.

        Args:
            symbols: Complex BPSK symbols

        Returns:
            SNR in dB
        """
        # Hard decisions: map to nearest constellation point
        hard_decisions = np.sign(symbols.real).astype(np.complex64)

        # Signal power (should be ~1.0 for normalized BPSK)
        signal_power = np.mean(np.abs(hard_decisions) ** 2)

        # Noise is the residual
        noise = symbols - hard_decisions
        noise_power = np.mean(np.abs(noise) ** 2)

        # SNR in dB
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))

        return snr_db


def process_and_save(
    raw_iq_file: str,
    tx_bits_file: str,
    pilot_bits_file: str,
    output_file: str
):
    """
    Process one capture and save results.
    """
    processor = RealDataProcessor()

    frame = processor.extract_frame(
        raw_iq_file,
        tx_bits_file,
        pilot_bits_file
    )

    # Save to .npz
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **frame
    )

    print(f"\n✓ Saved processed frame to {output_file}")

    # Print file size
    file_size_kb = output_path.stat().st_size / 1024
    print(f"  File size: {file_size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Process real PlutoSDR raw IQ captures"
    )
    parser.add_argument(
        "--raw-iq",
        type=str,
        default="received_iq_raw.bin",
        help="Path to raw IQ file (received_iq_raw.bin)"
    )
    parser.add_argument(
        "--tx-bits",
        type=str,
        default="bit_tests/bits_000.bin",
        help="Path to reference TX bits"
    )
    parser.add_argument(
        "--pilot",
        type=str,
        default="bit_tests/pilot.bin",
        help="Path to reference pilot bits"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="real_captures/processed_frame.npz",
        help="Output file for processed frame"
    )

    args = parser.parse_args()

    process_and_save(
        args.raw_iq,
        args.tx_bits,
        args.pilot,
        args.output
    )


if __name__ == "__main__":
    main()
