#!/usr/bin/env python3
"""
Unit Tests for Real Data Processor

Comprehensive tests to ensure data processing is correct.
"""

import numpy as np
import unittest
import tempfile
import os
from pathlib import Path

from real_data_processor import RealDataProcessor


class TestRealDataProcessor(unittest.TestCase):
    """Test suite for RealDataProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.processor = RealDataProcessor()
        self.rng = np.random.default_rng(42)

    def test_constants(self):
        """Test that constants match the known frame structure"""
        # From make_bit_files.py
        self.assertEqual(self.processor.preamble_bits, 2000)
        self.assertEqual(self.processor.pilot_bits, 64)
        self.assertEqual(self.processor.data_bits, 17936)
        self.assertEqual(self.processor.total_bits, 20000)

        # From SDR config
        self.assertEqual(self.processor.sample_rate, 250000)
        self.assertEqual(self.processor.symbol_rate, 10000)
        self.assertEqual(self.processor.samples_per_symbol, 25)

    def test_decimation(self):
        """Test that decimation by 25 works correctly"""
        # Create test signal: 250 samples (10 symbols at 25 sps)
        num_symbols = 10
        raw_iq = self.rng.standard_normal(num_symbols * 25) + \
                 1j * self.rng.standard_normal(num_symbols * 25)
        raw_iq = raw_iq.astype(np.complex64)

        # Decimate
        symbols = self.processor.decimate_to_symbols(raw_iq)

        # Check length
        self.assertEqual(len(symbols), num_symbols)

        # Check that decimation takes every 25th sample
        for i in range(num_symbols):
            self.assertEqual(symbols[i], raw_iq[i * 25])

    def test_bpsk_detection_normal(self):
        """Test BPSK bit detection (no inversion)"""
        # Create known BPSK symbols: -1 for bit 0, +1 for bit 1
        tx_bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        tx_symbols = self.processor.constellation[tx_bits]

        # Add small noise
        noise = 0.01 * (self.rng.standard_normal(len(tx_symbols)) +
                       1j * self.rng.standard_normal(len(tx_symbols)))
        rx_symbols = tx_symbols + noise

        # Detect bits
        rx_bits = self.processor.detect_bits_bpsk(rx_symbols, assume_inverted=False)

        # Should match perfectly with low noise
        np.testing.assert_array_equal(rx_bits, tx_bits)

    def test_bpsk_detection_inverted(self):
        """Test BPSK bit detection with 180° phase inversion"""
        # Create known BPSK symbols
        tx_bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        tx_symbols = self.processor.constellation[tx_bits]

        # Apply 180° phase shift (inversion)
        rx_symbols = -tx_symbols

        # Add small noise
        noise = 0.01 * (self.rng.standard_normal(len(rx_symbols)) +
                       1j * self.rng.standard_normal(len(rx_symbols)))
        rx_symbols = rx_symbols + noise

        # Detect with inversion flag
        rx_bits = self.processor.detect_bits_bpsk(rx_symbols, assume_inverted=True)

        # Should match original bits
        np.testing.assert_array_equal(rx_bits, tx_bits)

    def test_pilot_detection_exact_match(self):
        """Test pilot detection when pilot is at exact expected position"""
        # Create a frame: [preamble][pilot][data]
        preamble = self.rng.integers(0, 2, self.processor.preamble_bits, dtype=np.uint8)
        pilot = self.rng.integers(0, 2, self.processor.pilot_bits, dtype=np.uint8)
        data = self.rng.integers(0, 2, 1000, dtype=np.uint8)

        frame_bits = np.concatenate([preamble, pilot, data])

        # Find pilot
        pilot_pos, is_inverted = self.processor.find_pilot_position(
            frame_bits,
            pilot,
            search_start=1900,
            search_end=2100,
            check_inversion=False
        )

        # Should find at exactly bit 2000
        self.assertEqual(pilot_pos, self.processor.preamble_bits)
        self.assertFalse(is_inverted)

    def test_pilot_detection_with_inversion(self):
        """Test pilot detection when bits are inverted"""
        # Create pilot
        pilot = self.rng.integers(0, 2, self.processor.pilot_bits, dtype=np.uint8)

        # Create frame with inverted bits
        preamble = self.rng.integers(0, 2, self.processor.preamble_bits, dtype=np.uint8)
        pilot_inverted = 1 - pilot  # Bit inversion
        data = self.rng.integers(0, 2, 1000, dtype=np.uint8)

        frame_bits = np.concatenate([preamble, pilot_inverted, data])

        # Find pilot with inversion check
        pilot_pos, is_inverted = self.processor.find_pilot_position(
            frame_bits,
            pilot,
            search_start=1900,
            search_end=2100,
            check_inversion=True
        )

        # Should find at bit 2000 and detect inversion
        self.assertEqual(pilot_pos, self.processor.preamble_bits)
        self.assertTrue(is_inverted)

    def test_pilot_detection_with_offset(self):
        """Test pilot detection when pilot is slightly offset"""
        # Create pilot
        pilot = self.rng.integers(0, 2, self.processor.pilot_bits, dtype=np.uint8)

        # Create frame with pilot at offset position
        offset = 20  # 20 bits late (use multiple of 10 for search step)
        preamble = self.rng.integers(0, 2, self.processor.preamble_bits + offset, dtype=np.uint8)
        data = self.rng.integers(0, 2, 1000, dtype=np.uint8)

        frame_bits = np.concatenate([preamble, pilot, data])

        # Find pilot with finer search around expected position
        expected_pos = self.processor.preamble_bits + offset
        pilot_pos, is_inverted = self.processor.find_pilot_position(
            frame_bits,
            pilot,
            search_start=expected_pos - 50,
            search_end=expected_pos + 50,
            check_inversion=False
        )

        # Should find at offset position (within search granularity of 10)
        self.assertAlmostEqual(pilot_pos, expected_pos, delta=10)

    def test_snr_estimation(self):
        """Test SNR estimation for BPSK symbols"""
        # Create clean BPSK symbols
        tx_bits = self.rng.integers(0, 2, 1000, dtype=np.uint8)
        tx_symbols = self.processor.constellation[tx_bits]

        # Add noise with known SNR (10 dB)
        signal_power = np.mean(np.abs(tx_symbols) ** 2)
        snr_linear = 10 ** (10 / 10)  # 10 dB
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)

        noise = noise_std * (self.rng.standard_normal(len(tx_symbols)) +
                            1j * self.rng.standard_normal(len(tx_symbols)))
        rx_symbols = tx_symbols + noise

        # Estimate SNR
        estimated_snr = self.processor.estimate_snr_bpsk(rx_symbols)

        # Should be close to 10 dB (within 1 dB tolerance)
        self.assertAlmostEqual(estimated_snr, 10.0, delta=1.0)

    def test_end_to_end_with_simulated_data(self):
        """
        End-to-end test with simulated SDR capture.
        Creates a complete frame and processes it.
        """
        # Create a complete frame matching make_bit_files.py format
        rng = np.random.default_rng(1234)  # Match make_bit_files seed

        # Generate frame bits (same as make_bit_files.py would)
        preamble = rng.integers(0, 2, self.processor.preamble_bits, dtype=np.uint8)
        pilot = rng.integers(0, 2, self.processor.pilot_bits, dtype=np.uint8)
        data = rng.integers(0, 2, self.processor.data_bits, dtype=np.uint8)

        tx_bits = np.concatenate([preamble, pilot, data])

        # Convert to BPSK symbols
        tx_symbols = self.processor.constellation[tx_bits]

        # Create raw IQ by repeating each symbol 25 times (simulated oversampling)
        raw_iq = np.repeat(tx_symbols, self.processor.samples_per_symbol)

        # Add noise (SNR = 15 dB)
        signal_power = np.mean(np.abs(raw_iq) ** 2)
        snr_linear = 10 ** (15 / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)

        noise = noise_std * (rng.standard_normal(len(raw_iq)) +
                            1j * rng.standard_normal(len(raw_iq)))
        raw_iq_noisy = raw_iq + noise

        # Apply 180° phase inversion (simulate Costas loop lock at wrong phase)
        raw_iq_noisy = -raw_iq_noisy

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_iq_file = os.path.join(tmpdir, "raw_iq.bin")
            tx_bits_file = os.path.join(tmpdir, "tx_bits.bin")
            pilot_file = os.path.join(tmpdir, "pilot.bin")

            # Write files
            raw_iq_noisy.astype(np.complex64).tofile(raw_iq_file)
            tx_bits.tofile(tx_bits_file)
            pilot.tofile(pilot_file)

            # Process frame
            frame = self.processor.extract_frame(
                raw_iq_file,
                tx_bits_file,
                pilot_file
            )

            # Validate results
            self.assertEqual(frame["pilot_position"], self.processor.preamble_bits)
            self.assertTrue(frame["is_inverted"])  # Should detect inversion

            # BER should be low (< 1%) with 15 dB SNR
            self.assertLess(frame["ber"], 0.01)

            # SNR estimate should be reasonable
            self.assertGreater(frame["snr_db"], 10.0)
            self.assertLess(frame["snr_db"], 20.0)

            # Data lengths should match
            self.assertEqual(len(frame["rx_data_bits"]), len(frame["tx_data_bits"]))

            print(f"\n✓ End-to-end test passed:")
            print(f"  Pilot position: {frame['pilot_position']}")
            print(f"  Phase inverted: {frame['is_inverted']}")
            print(f"  BER: {frame['ber']:.6f}")
            print(f"  SNR: {frame['snr_db']:.2f} dB")

    def test_frame_structure_validation(self):
        """Test that frame structure constants add up correctly"""
        total = (self.processor.preamble_bits +
                self.processor.pilot_bits +
                self.processor.data_bits)

        self.assertEqual(total, self.processor.total_bits)
        self.assertEqual(total, 20000)


class TestFileOperations(unittest.TestCase):
    """Test file I/O operations"""

    def setUp(self):
        self.processor = RealDataProcessor()

    def test_load_raw_iq(self):
        """Test loading raw IQ from file"""
        # Create temporary raw IQ file
        test_data = np.random.randn(1000) + 1j * np.random.randn(1000)
        test_data = test_data.astype(np.complex64)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_file = f.name
            test_data.tofile(f)

        try:
            # Load
            loaded_data = self.processor.load_raw_iq(temp_file)

            # Verify
            np.testing.assert_array_equal(loaded_data, test_data)
        finally:
            os.unlink(temp_file)

    def test_load_raw_iq_missing_file(self):
        """Test error handling for missing file"""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_raw_iq("nonexistent_file.bin")

    def test_load_raw_iq_empty_file(self):
        """Test error handling for empty file"""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_file = f.name
            # Write nothing

        try:
            with self.assertRaises(ValueError):
                self.processor.load_raw_iq(temp_file)
        finally:
            os.unlink(temp_file)


def run_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestRealDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestFileOperations))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
