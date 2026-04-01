#!/usr/bin/env python3
"""
Visualization Tools for DEFINED Project

Creates plots for:
- Constellation diagrams
- BER vs SNR curves
- BER vs pilot count
- Pilot correlation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional, List, Tuple

plt.style.use('seaborn-v0_8-darkgrid')


class Visualizer:
    """Create publication-quality plots for DEFINED project"""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_constellation(
        self,
        symbols: np.ndarray,
        title: str = "BPSK Constellation",
        reference_points: Optional[np.ndarray] = None,
        save_name: str = "constellation.png"
    ):
        """
        Plot constellation diagram.

        Args:
            symbols: Complex symbols to plot
            title: Plot title
            reference_points: Ideal constellation points (e.g., ±1 for BPSK)
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot received symbols
        ax.plot(symbols.real, symbols.imag, 'b.', alpha=0.3, ms=2,
                label='Received symbols')

        # Plot reference constellation if provided
        if reference_points is not None:
            ax.plot(reference_points.real, reference_points.imag,
                   'rx', ms=15, mew=3, label='Ideal constellation')

        # Formatting
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('In-Phase (I)', fontsize=12)
        ax.set_ylabel('Quadrature (Q)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Set symmetric limits
        max_val = max(np.max(np.abs(symbols.real)), np.max(np.abs(symbols.imag)))
        lim = max_val * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved constellation plot to {save_path}")
        plt.close()

    def plot_ber_vs_snr(
        self,
        snr_db: np.ndarray,
        ber: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "BER vs SNR",
        save_name: str = "ber_vs_snr.png"
    ):
        """
        Plot BER vs SNR curves.

        Args:
            snr_db: SNR values in dB (can be 1D or 2D for multiple curves)
            ber: BER values (same shape as snr_db)
            labels: Labels for each curve
            title: Plot title
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Handle both single and multiple curves
        if snr_db.ndim == 1:
            snr_db = snr_db[np.newaxis, :]
            ber = ber[np.newaxis, :]

        colors = plt.cm.tab10(np.linspace(0, 1, len(snr_db)))
        markers = ['o', 's', '^', 'v', 'D', 'p']

        for i, (snr, ber_vals) in enumerate(zip(snr_db, ber)):
            label = labels[i] if labels and i < len(labels) else f'Curve {i+1}'
            marker = markers[i % len(markers)]

            ax.semilogy(snr, ber_vals, marker=marker, linewidth=2,
                       markersize=6, label=label, color=colors[i])

        # Reference lines
        ax.axhline(1e-2, color='r', linestyle='--', alpha=0.5,
                  linewidth=1, label='1% BER')
        ax.axhline(1e-3, color='g', linestyle='--', alpha=0.5,
                  linewidth=1, label='0.1% BER')

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1e-5, 1)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved BER vs SNR plot to {save_path}")
        plt.close()

    def plot_ber_vs_pilots(
        self,
        num_pilots: np.ndarray,
        ber: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "BER vs Number of Pilots",
        save_name: str = "ber_vs_pilots.png"
    ):
        """
        Plot BER vs number of pilot symbols.

        This is a key plot for the DEFINED paper:
        comparing vanilla ICL vs DEFINED at different pilot counts.

        Args:
            num_pilots: Number of pilot symbols (can be 1D or 2D for multiple curves)
            ber: BER values (same shape as num_pilots)
            labels: Labels for each curve (e.g., ["Vanilla ICL", "DEFINED"])
            title: Plot title
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Handle both single and multiple curves
        if num_pilots.ndim == 1:
            num_pilots = num_pilots[np.newaxis, :]
            ber = ber[np.newaxis, :]

        colors = plt.cm.tab10(np.linspace(0, 1, len(num_pilots)))
        markers = ['o', 's', '^', 'v', 'D', 'p']

        for i, (k, ber_vals) in enumerate(zip(num_pilots, ber)):
            label = labels[i] if labels and i < len(labels) else f'Method {i+1}'
            marker = markers[i % len(markers)]

            ax.semilogy(k, ber_vals, marker=marker, linewidth=2,
                       markersize=8, label=label, color=colors[i])

        # Reference line
        ax.axhline(1e-2, color='r', linestyle='--', alpha=0.5,
                  linewidth=1, label='1% BER target')

        ax.set_xlabel('Number of Pilot Symbols (k)', fontsize=12)
        ax.set_ylabel('Symbol Error Rate (SER)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1e-4, 1)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved BER vs pilots plot to {save_path}")
        plt.close()

    def plot_time_series(
        self,
        symbols: np.ndarray,
        sample_rate: float = 10000,
        title: str = "Symbol Time Series",
        save_name: str = "time_series.png",
        max_samples: int = 500
    ):
        """
        Plot time series of symbol real and imaginary parts.

        Args:
            symbols: Complex symbols
            sample_rate: Symbol rate in Hz
            title: Plot title
            save_name: Output filename
            max_samples: Maximum number of samples to plot
        """
        # Limit to max_samples for readability
        symbols = symbols[:max_samples]
        time = np.arange(len(symbols)) / sample_rate * 1000  # Time in ms

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot real part (I)
        ax1.plot(time, symbols.real, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('In-Phase (I)', fontsize=11)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot imaginary part (Q)
        ax2.plot(time, symbols.imag, 'r-', linewidth=1, alpha=0.7)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (ms)', fontsize=11)
        ax2.set_ylabel('Quadrature (Q)', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved time series plot to {save_path}")
        plt.close()

    def plot_pilot_correlation(
        self,
        correlation: np.ndarray,
        pilot_position: int,
        title: str = "Pilot Correlation",
        save_name: str = "pilot_correlation.png"
    ):
        """
        Plot pilot correlation for synchronization.

        Args:
            correlation: Correlation values
            pilot_position: Detected pilot position
            title: Plot title
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot correlation
        ax.plot(correlation, 'b-', linewidth=1, alpha=0.7, label='Correlation')

        # Mark detected pilot position
        ax.axvline(pilot_position, color='r', linestyle='--', linewidth=2,
                  label=f'Pilot @ {pilot_position}')

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Correlation Magnitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved pilot correlation plot to {save_path}")
        plt.close()


def demo_plots():
    """Generate demo plots with synthetic data"""
    print("="*70)
    print("DEFINED Visualization Demo")
    print("="*70)

    viz = Visualizer(output_dir="demo_plots")

    # 1. Constellation plot (BPSK with noise)
    print("\n1. Creating constellation plot...")
    rng = np.random.default_rng(42)
    tx_bits = rng.integers(0, 2, 1000)
    constellation = np.array([-1+0j, 1+0j])
    tx_symbols = constellation[tx_bits]

    # Add noise (SNR = 10 dB)
    signal_power = np.mean(np.abs(tx_symbols) ** 2)
    snr_linear = 10 ** (10 / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(tx_symbols)) +
                                       1j * rng.standard_normal(len(tx_symbols)))
    rx_symbols = tx_symbols + noise

    viz.plot_constellation(
        rx_symbols,
        title="BPSK Constellation (SNR=10dB)",
        reference_points=constellation,
        save_name="demo_constellation.png"
    )

    # 2. BER vs SNR curves
    print("2. Creating BER vs SNR plot...")
    snr_db_range = np.arange(-5, 21, 2)
    ber_baseline = 0.5 * np.exp(-0.7 * (10 ** (snr_db_range / 10)))  # Dummy curve
    ber_vanilla = 0.3 * np.exp(-0.9 * (10 ** (snr_db_range / 10)))
    ber_defined = 0.1 * np.exp(-1.2 * (10 ** (snr_db_range / 10)))

    viz.plot_ber_vs_snr(
        np.array([snr_db_range, snr_db_range, snr_db_range]),
        np.array([ber_baseline, ber_vanilla, ber_defined]),
        labels=["Baseline", "Vanilla ICL", "DEFINED"],
        title="BER vs SNR (Simulated)",
        save_name="demo_ber_vs_snr.png"
    )

    # 3. BER vs pilot count
    print("3. Creating BER vs pilot count plot...")
    pilot_counts = np.array([4, 8, 16, 32, 64])
    ser_vanilla = np.array([0.15, 0.08, 0.04, 0.02, 0.01])
    ser_defined = np.array([0.08, 0.03, 0.012, 0.005, 0.002])

    viz.plot_ber_vs_pilots(
        np.array([pilot_counts, pilot_counts]),
        np.array([ser_vanilla, ser_defined]),
        labels=["Vanilla ICL (pilots only)", "DEFINED (pilots + feedback)"],
        title="DEFINED Reduces Pilot Overhead",
        save_name="demo_ber_vs_pilots.png"
    )

    # 4. Time series
    print("4. Creating time series plot...")
    viz.plot_time_series(
        rx_symbols,
        sample_rate=10000,
        title="BPSK Symbol Time Series",
        save_name="demo_time_series.png"
    )

    print("\n" + "="*70)
    print(f"Demo plots saved to: demo_plots/")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualization tools for DEFINED project"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo plots with synthetic data"
    )

    args = parser.parse_args()

    if args.demo:
        demo_plots()
    else:
        print("Use --demo to generate demonstration plots")
        print("Or import this module to use the Visualizer class in your own scripts")


if __name__ == "__main__":
    main()
