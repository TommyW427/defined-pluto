#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BPSK PlutoSDR Receiver - Raw IQ Collection
# Collects received samples and saves to file for offline processing
# (e.g., DEFINED Transformer-based symbol detection pipeline)
#
# Matches transmitter settings:
#   - Center freq: 915 MHz
#   - Sample rate: 250 kHz
#   - Symbol rate: 10 kHz  (sps = 25)
#   - Modulation:  BPSK [-1, +1]

import numpy as np
import argparse
import time
import sys

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
except ImportError:
    print("SoapySDR not found. Install with: apt install python3-soapysdr soapysdr-module-plutosdr")
    sys.exit(1)


def collect_samples(args):
    """Collect raw IQ samples from PlutoSDR and save to disk."""

    print(f"=== BPSK Receiver - IQ Collection ===")
    print(f"Center frequency : {args.freq / 1e6:.1f} MHz")
    print(f"Sample rate      : {args.rate / 1e3:.0f} kHz")
    print(f"Symbol rate      : {args.symbol_rate / 1e3:.0f} kHz")
    print(f"Samples/symbol   : {int(args.rate // args.symbol_rate)}")
    print(f"Gain             : {args.gain} dB")
    print(f"Duration         : {args.duration:.1f} s")
    print(f"Output file      : {args.output}")
    print()

    # Open PlutoSDR
    sdr = SoapySDR.Device({"driver": "plutosdr"})

    # Configure RX channel
    sdr.setSampleRate(SOAPY_SDR_RX, 0, args.rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, args.freq)
    sdr.setBandwidth(SOAPY_SDR_RX, 0, args.rate)
    sdr.setGain(SOAPY_SDR_RX, 0, args.gain)

    # Setup receive stream
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)

    # Calculate total samples to collect
    total_samples = int(args.rate * args.duration)
    chunk_size = 4096
    collected = 0
    all_samples = []

    print(f"Collecting {total_samples} samples...")

    try:
        while collected < total_samples:
            # Read a chunk
            n_read = min(chunk_size, total_samples - collected)
            buf = np.zeros(n_read, dtype=np.complex64)
            status = sdr.readStream(rx_stream, [buf], n_read)

            if status.ret > 0:
                all_samples.append(buf[:status.ret].copy())
                collected += status.ret

                # Progress update every ~0.5s worth of samples
                if collected % (int(args.rate * 0.5)) < chunk_size:
                    elapsed_pct = collected / total_samples * 100
                    print(f"  {elapsed_pct:5.1f}%  ({collected}/{total_samples} samples)")
            elif status.ret < 0:
                print(f"  Stream error: {status.ret}, retrying...")
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")

    finally:
        sdr.deactivateStream(rx_stream)
        sdr.closeStream(rx_stream)

    # Concatenate all chunks
    samples = np.concatenate(all_samples) if all_samples else np.array([], dtype=np.complex64)
    print(f"\nCollected {len(samples)} raw IQ samples total.")

    # Save raw IQ samples
    if args.output.endswith(".npy"):
        np.save(args.output, samples)
    elif args.output.endswith(".bin") or args.output.endswith(".raw"):
        # Save as interleaved float32 (GNURadio-compatible complex float)
        samples.tofile(args.output)
    else:
        np.save(args.output, samples)

    print(f"Saved to: {args.output}")

    # Also save a metadata sidecar so the processing pipeline knows the params
    meta = {
        "center_freq_hz": args.freq,
        "sample_rate_hz": args.rate,
        "symbol_rate_hz": args.symbol_rate,
        "samples_per_symbol": int(args.rate // args.symbol_rate),
        "gain_db": args.gain,
        "duration_s": args.duration,
        "num_samples": len(samples),
        "dtype": "complex64",
        "format_note": "numpy .npy or raw interleaved float32 (I,Q,I,Q,...)",
    }
    meta_path = args.output.rsplit(".", 1)[0] + "_meta.npy"
    np.save(meta_path, meta)
    print(f"Metadata saved to: {meta_path}")

    # Print quick stats for sanity check
    if len(samples) > 0:
        power_db = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-12)
        print(f"\nQuick stats:")
        print(f"  Mean power     : {power_db:.1f} dB")
        print(f"  Peak |sample|  : {np.max(np.abs(samples)):.4f}")
        print(f"  Symbols approx : {len(samples) // int(args.rate // args.symbol_rate)}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect raw IQ samples from PlutoSDR for offline BPSK processing"
    )
    parser.add_argument(
        "-f", "--freq", type=float, default=915e6,
        help="Center frequency in Hz (default: 915 MHz)"
    )
    parser.add_argument(
        "-r", "--rate", type=float, default=250000,
        help="Sample rate in Hz (default: 250000)"
    )
    parser.add_argument(
        "--symbol-rate", type=float, default=10000,
        help="Symbol rate in Hz (default: 10000)"
    )
    parser.add_argument(
        "-g", "--gain", type=float, default=40,
        help="RX gain in dB (default: 40)"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0,
        help="Collection duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="received_iq.npy",
        help="Output file path (default: received_iq.npy)"
    )

    args = parser.parse_args()
    collect_samples(args)


if __name__ == "__main__":
    main()
