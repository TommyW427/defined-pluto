#!/usr/bin/env python3
"""
Generate Synthetic Training Dataset for DEFINED

Creates train/val/test splits with:
- Various pilot lengths (k)
- Different SNR conditions
- Block-fading channels
- Frame structure: [k pilots] + [T-k data symbols]
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

from config import (
    DEFINEDConfig,
    get_constellation,
    bits_to_symbols,
)
from channel_simulator import MultiSNRChannelSimulator


class SyntheticDatasetGenerator:
    """
    Generate synthetic dataset for DEFINED pre-training.

    Each sample consists of:
        - tx_symbols: transmitted symbols (ground truth), shape (T,)
        - rx_symbols: received symbols (noisy), shape (T,)
        - pilot_symbols: known pilot symbols, shape (k,)
        - pilot_positions: indices of pilot symbols [0, 1, ..., k-1]
        - channel_coeff: true channel coefficients, shape (T,)
        - snr_db: SNR for this sample
        - num_pilots: k (number of pilots)
    """

    def __init__(self, config: DEFINEDConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.constellation = get_constellation(config.modulation)

        # Create multi-SNR channel simulator
        self.channel_sim = MultiSNRChannelSimulator(
            snr_min_db=config.snr_train_min,
            snr_max_db=config.snr_train_max,
            channel_type=config.channel_type,
            doppler_hz=config.doppler_hz,
            rician_k=config.rician_k,
            seed=seed
        )

    def generate_frame(
        self,
        num_pilots: int,
        block_size: int = 16
    ) -> dict:
        """
        Generate one training sample (frame).

        Frame structure:
            [pilot_0, pilot_1, ..., pilot_{k-1}, data_k, data_{k+1}, ..., data_{T-1}]

        Args:
            num_pilots: k, number of pilot symbols (first k positions)
            block_size: coherence block size for fading

        Returns:
            Dictionary with all relevant data for this frame
        """
        T = self.config.frame_length

        if num_pilots >= T:
            raise ValueError(f"num_pilots ({num_pilots}) must be < frame_length ({T})")

        # Generate random transmitted symbols
        if self.config.modulation == "BPSK":
            bits_per_symbol = 1
        elif self.config.modulation == "QPSK":
            bits_per_symbol = 2
        elif self.config.modulation == "16QAM":
            bits_per_symbol = 4
        else:
            raise ValueError(f"Unknown modulation: {self.config.modulation}")

        # Generate random bits and map to symbols
        num_bits = T * bits_per_symbol
        tx_bits = self.rng.integers(0, 2, num_bits, dtype=np.uint8)
        tx_symbols = bits_to_symbols(tx_bits, self.config.modulation)

        # Apply channel
        rx_symbols, channel_coeff, snr_db = self.channel_sim.apply_random_snr_channel(
            tx_symbols,
            block_size=block_size
        )

        # Extract pilot symbols (first k symbols are pilots)
        pilot_symbols = tx_symbols[:num_pilots].copy()
        pilot_positions = np.arange(num_pilots, dtype=np.int32)

        return {
            "tx_symbols": tx_symbols.astype(np.complex64),
            "rx_symbols": rx_symbols.astype(np.complex64),
            "pilot_symbols": pilot_symbols.astype(np.complex64),
            "pilot_positions": pilot_positions,
            "channel_coeff": channel_coeff.astype(np.complex64),
            "snr_db": np.float32(snr_db),
            "num_pilots": np.int32(num_pilots),
            "tx_bits": tx_bits,
        }

    def generate_dataset(
        self,
        num_samples: int,
        pilot_range: tuple = None,
        desc: str = "Generating"
    ) -> dict:
        """
        Generate a dataset with multiple samples.

        Args:
            num_samples: Number of frames to generate
            pilot_range: (min_pilots, max_pilots) or None to use config defaults
            desc: Description for progress bar

        Returns:
            Dictionary of arrays, each shape (num_samples, ...)
        """
        if pilot_range is None:
            pilot_range = (self.config.min_pilots, self.config.max_pilots)

        min_pilots, max_pilots = pilot_range

        # Pre-allocate arrays
        dataset = {
            "tx_symbols": np.zeros((num_samples, self.config.frame_length), dtype=np.complex64),
            "rx_symbols": np.zeros((num_samples, self.config.frame_length), dtype=np.complex64),
            "channel_coeff": np.zeros((num_samples, self.config.frame_length), dtype=np.complex64),
            "snr_db": np.zeros(num_samples, dtype=np.float32),
            "num_pilots": np.zeros(num_samples, dtype=np.int32),
            # Variable-length fields stored as lists
            "pilot_symbols": [],
            "pilot_positions": [],
            "tx_bits": [],
        }

        for i in tqdm(range(num_samples), desc=desc):
            # Randomly choose number of pilots
            k = self.rng.integers(min_pilots, max_pilots + 1)

            # Generate frame
            frame = self.generate_frame(num_pilots=k)

            # Store in dataset
            dataset["tx_symbols"][i] = frame["tx_symbols"]
            dataset["rx_symbols"][i] = frame["rx_symbols"]
            dataset["channel_coeff"][i] = frame["channel_coeff"]
            dataset["snr_db"][i] = frame["snr_db"]
            dataset["num_pilots"][i] = frame["num_pilots"]
            dataset["pilot_symbols"].append(frame["pilot_symbols"])
            dataset["pilot_positions"].append(frame["pilot_positions"])
            dataset["tx_bits"].append(frame["tx_bits"])

        return dataset

    def save_dataset(self, dataset: dict, output_path: str):
        """Save dataset to .npz file"""
        # Convert lists to object arrays for variable-length data
        dataset_to_save = dataset.copy()
        dataset_to_save["pilot_symbols"] = np.array(dataset["pilot_symbols"], dtype=object)
        dataset_to_save["pilot_positions"] = np.array(dataset["pilot_positions"], dtype=object)
        dataset_to_save["tx_bits"] = np.array(dataset["tx_bits"], dtype=object)

        np.savez_compressed(output_path, **dataset_to_save)
        print(f"Saved dataset to {output_path}")

        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"  Num samples: {len(dataset['tx_symbols'])}")
        print(f"  Frame length: {self.config.frame_length}")
        print(f"  Modulation: {self.config.modulation}")
        print(f"  SNR range: [{dataset['snr_db'].min():.2f}, {dataset['snr_db'].max():.2f}] dB")
        print(f"  Pilot range: [{dataset['num_pilots'].min()}, {dataset['num_pilots'].max()}]")
        print(f"  Channel type: {self.config.channel_type}")

        # File size
        file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"  File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for DEFINED pre-training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/synthetic",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--modulation",
        type=str,
        default="BPSK",
        choices=["BPSK", "QPSK", "16QAM"],
        help="Modulation scheme"
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=64,
        help="Frame length T (total symbols)"
    )
    parser.add_argument(
        "--channel-type",
        type=str,
        default="rayleigh",
        choices=["awgn", "rayleigh", "rician"],
        help="Channel type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Create config
    config = DEFINEDConfig()
    config.modulation = args.modulation
    config.frame_length = args.frame_length
    config.channel_type = args.channel_type

    if args.modulation == "BPSK":
        config.constellation_size = 2
    elif args.modulation == "QPSK":
        config.constellation_size = 4
    elif args.modulation == "16QAM":
        config.constellation_size = 16

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = SyntheticDatasetGenerator(config, seed=args.seed)

    print("="*70)
    print("DEFINED Synthetic Dataset Generation")
    print("="*70)
    print(f"Modulation: {config.modulation}")
    print(f"Frame length: {config.frame_length}")
    print(f"Channel: {config.channel_type}")
    print(f"SNR range: [{config.snr_train_min}, {config.snr_train_max}] dB")
    print(f"Pilot range: [{config.min_pilots}, {config.max_pilots}]")
    print("="*70)

    # Generate train set
    print("\n=== Generating Training Set ===")
    train_dataset = generator.generate_dataset(
        num_samples=config.num_train_samples,
        desc="Train"
    )
    train_path = output_dir / f"train_{args.modulation}_{args.channel_type}.npz"
    generator.save_dataset(train_dataset, str(train_path))

    # Generate validation set
    print("\n=== Generating Validation Set ===")
    val_dataset = generator.generate_dataset(
        num_samples=config.num_val_samples,
        desc="Val"
    )
    val_path = output_dir / f"val_{args.modulation}_{args.channel_type}.npz"
    generator.save_dataset(val_dataset, str(val_path))

    # Generate test set
    print("\n=== Generating Test Set ===")
    test_dataset = generator.generate_dataset(
        num_samples=config.num_test_samples,
        desc="Test"
    )
    test_path = output_dir / f"test_{args.modulation}_{args.channel_type}.npz"
    generator.save_dataset(test_dataset, str(test_path))

    print("\n" + "="*70)
    print("Dataset generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
