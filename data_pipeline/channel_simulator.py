#!/usr/bin/env python3
"""
Wireless Channel Simulator for DEFINED Pre-training

Generates realistic channel conditions:
- AWGN (baseline)
- Rayleigh fading (no LOS)
- Rician fading (with LOS component)
- Block fading (channel constant over blocks, changes between blocks)
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal


class ChannelSimulator:
    """
    Simulates wireless channel effects for training the DEFINED model.

    The received signal model is:
        y_t = h_t · x_t + z_t

    where:
        x_t = transmitted symbol
        h_t = complex channel coefficient (fading)
        z_t = additive white Gaussian noise
        y_t = received symbol
    """

    def __init__(
        self,
        channel_type: str = "rayleigh",
        snr_db: float = 10.0,
        doppler_hz: float = 10.0,
        sample_rate_hz: float = 10000,  # Symbol rate for the ML model
        rician_k: float = 3.0,
        seed: Optional[int] = None
    ):
        """
        Args:
            channel_type: "awgn", "rayleigh", or "rician"
            snr_db: Signal-to-noise ratio in dB
            doppler_hz: Maximum Doppler shift (for time-varying fading)
            sample_rate_hz: Symbol rate (symbols per second)
            rician_k: K-factor for Rician fading (LOS power / scattered power)
            seed: Random seed for reproducibility
        """
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.doppler_hz = doppler_hz
        self.sample_rate_hz = sample_rate_hz
        self.rician_k = rician_k

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def generate_fading_coefficients(
        self,
        num_symbols: int,
        block_size: int = 16
    ) -> np.ndarray:
        """
        Generate complex channel coefficients (h_t).

        For block fading: the channel stays constant for `block_size` symbols,
        then changes to a new independent realization. This models slow fading
        where the channel is approximately constant over short time scales.

        Args:
            num_symbols: Total number of symbols
            block_size: Number of symbols per coherence block

        Returns:
            Array of complex channel coefficients, shape (num_symbols,)
        """
        if self.channel_type == "awgn":
            # No fading, h = 1 for all symbols
            return np.ones(num_symbols, dtype=np.complex64)

        # Number of independent channel realizations needed
        num_blocks = int(np.ceil(num_symbols / block_size))

        if self.channel_type == "rayleigh":
            # Rayleigh fading: |h| ~ Rayleigh, angle ~ Uniform[0, 2π)
            # Equivalent to h = CN(0, 1), where real and imag parts are N(0, 0.5)
            h_blocks = (
                self.rng.normal(0, 1/np.sqrt(2), num_blocks) +
                1j * self.rng.normal(0, 1/np.sqrt(2), num_blocks)
            )

        elif self.channel_type == "rician":
            # Rician fading: h = h_LOS + h_scattered
            # h_LOS = sqrt(K/(K+1)) * exp(j*theta)  (deterministic LOS)
            # h_scattered = sqrt(1/(K+1)) * CN(0, 1)  (random scatter)
            los_amplitude = np.sqrt(self.rician_k / (self.rician_k + 1))
            scatter_power = 1 / (self.rician_k + 1)

            # LOS component (constant angle, you can randomize if desired)
            h_los = los_amplitude * np.exp(1j * 0)  # or random phase

            # Scattered component (Rayleigh-like)
            h_scattered = (
                self.rng.normal(0, np.sqrt(scatter_power/2), num_blocks) +
                1j * self.rng.normal(0, np.sqrt(scatter_power/2), num_blocks)
            )

            h_blocks = h_los + h_scattered

        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

        # Expand blocks: repeat each coefficient for block_size symbols
        h = np.repeat(h_blocks, block_size)[:num_symbols]

        return h.astype(np.complex64)

    def add_awgn(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Add additive white Gaussian noise to achieve target SNR.

        SNR is defined as:
            SNR = E[|x|^2] / E[|z|^2]

        Args:
            signal: Clean transmitted symbols (x_t)

        Returns:
            Tuple of (noisy_signal, noise_power)
        """
        # Measure signal power
        signal_power = np.mean(np.abs(signal) ** 2)

        # Compute noise power from SNR
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate complex Gaussian noise: z ~ CN(0, noise_power)
        noise_std = np.sqrt(noise_power / 2)  # Per real/imag component
        noise = (
            self.rng.normal(0, noise_std, len(signal)) +
            1j * self.rng.normal(0, noise_std, len(signal))
        )

        noisy_signal = signal + noise

        return noisy_signal.astype(np.complex64), noise_power

    def apply_channel(
        self,
        tx_symbols: np.ndarray,
        block_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full channel model: fading + AWGN.

        Model:
            y_t = h_t · x_t + z_t

        Args:
            tx_symbols: Transmitted symbols (x_t), shape (num_symbols,)
            block_size: Coherence block size (symbols)

        Returns:
            Tuple of:
                - rx_symbols: Received symbols (y_t), shape (num_symbols,)
                - h: Channel coefficients (h_t), shape (num_symbols,)
        """
        num_symbols = len(tx_symbols)

        # Generate fading coefficients
        h = self.generate_fading_coefficients(num_symbols, block_size)

        # Apply fading: y = h * x
        faded_signal = h * tx_symbols

        # Add noise: y = h * x + z
        rx_symbols, _ = self.add_awgn(faded_signal)

        return rx_symbols, h

    def set_snr(self, snr_db: float):
        """Update SNR for this channel"""
        self.snr_db = snr_db


class MultiSNRChannelSimulator:
    """
    Helper class to generate data across multiple SNR points.
    Useful for training with SNR diversity.
    """

    def __init__(
        self,
        snr_min_db: float = -5.0,
        snr_max_db: float = 20.0,
        channel_type: str = "rayleigh",
        doppler_hz: float = 10.0,
        rician_k: float = 3.0,
        seed: Optional[int] = None
    ):
        self.snr_min = snr_min_db
        self.snr_max = snr_max_db
        self.channel_type = channel_type
        self.doppler_hz = doppler_hz
        self.rician_k = rician_k

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def apply_random_snr_channel(
        self,
        tx_symbols: np.ndarray,
        block_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply channel with random SNR drawn uniformly from [snr_min, snr_max].

        Returns:
            Tuple of (rx_symbols, h, snr_db)
        """
        # Draw random SNR
        snr_db = self.rng.uniform(self.snr_min, self.snr_max)

        # Create channel simulator with this SNR
        channel = ChannelSimulator(
            channel_type=self.channel_type,
            snr_db=snr_db,
            doppler_hz=self.doppler_hz,
            rician_k=self.rician_k,
            seed=int(self.rng.integers(0, 2**31))
        )

        # Apply channel
        rx_symbols, h = channel.apply_channel(tx_symbols, block_size)

        return rx_symbols, h, snr_db


def test_channel_simulator():
    """Test the channel simulator with different configurations"""
    print("=== Channel Simulator Test ===\n")

    # Generate test signal: BPSK symbols
    from config import get_constellation
    constellation = get_constellation("BPSK")
    rng = np.random.default_rng(42)
    tx_bits = rng.integers(0, 2, 100)
    tx_symbols = constellation[tx_bits]

    print(f"TX signal power: {np.mean(np.abs(tx_symbols)**2):.4f}")

    # Test 1: AWGN channel
    print("\n=== AWGN Channel (SNR=10dB) ===")
    ch_awgn = ChannelSimulator(channel_type="awgn", snr_db=10.0, seed=42)
    rx_awgn, h_awgn = ch_awgn.apply_channel(tx_symbols)
    print(f"RX signal power: {np.mean(np.abs(rx_awgn)**2):.4f}")
    print(f"Channel coefficients (first 10): {h_awgn[:10]}")
    print(f"All ones (AWGN): {np.allclose(h_awgn, 1.0)}")

    # Verify SNR
    noise = rx_awgn - tx_symbols
    measured_snr = 10 * np.log10(np.mean(np.abs(tx_symbols)**2) /
                                  np.mean(np.abs(noise)**2))
    print(f"Measured SNR: {measured_snr:.2f} dB (target: 10.00 dB)")

    # Test 2: Rayleigh fading
    print("\n=== Rayleigh Fading (SNR=10dB, block_size=16) ===")
    ch_rayleigh = ChannelSimulator(channel_type="rayleigh", snr_db=10.0, seed=42)
    rx_rayleigh, h_rayleigh = ch_rayleigh.apply_channel(tx_symbols, block_size=16)
    print(f"Channel power: {np.mean(np.abs(h_rayleigh)**2):.4f} (should be ~1.0)")
    print(f"Channel coefficients (first 32):")
    print(h_rayleigh[:32])
    print(f"Block structure visible: {h_rayleigh[0] == h_rayleigh[15]}")
    print(f"Blocks are different: {h_rayleigh[15] != h_rayleigh[16]}")

    # Test 3: Rician fading
    print("\n=== Rician Fading (K=3dB, SNR=10dB) ===")
    ch_rician = ChannelSimulator(
        channel_type="rician",
        snr_db=10.0,
        rician_k=3.0,
        seed=42
    )
    rx_rician, h_rician = ch_rician.apply_channel(tx_symbols, block_size=16)
    print(f"Channel power: {np.mean(np.abs(h_rician)**2):.4f}")
    print(f"Channel coefficients (first 32):")
    print(h_rician[:32])

    # Test 4: Multi-SNR simulator
    print("\n=== Multi-SNR Random Sampling ===")
    multi_snr = MultiSNRChannelSimulator(
        snr_min_db=-5.0,
        snr_max_db=20.0,
        channel_type="rayleigh",
        seed=42
    )

    snr_samples = []
    for _ in range(10):
        rx, h, snr = multi_snr.apply_random_snr_channel(tx_symbols, block_size=16)
        snr_samples.append(snr)

    print(f"Random SNR samples: {snr_samples}")
    print(f"Min: {min(snr_samples):.2f}, Max: {max(snr_samples):.2f}")


if __name__ == "__main__":
    test_channel_simulator()
