#!/usr/bin/env python3
import os
import numpy as np

# ── Configuration ──────────────────────────────────────────────
PREAMBLE_LEN = 2000
PILOT_LEN = 64
TOTAL_BITS = 20000   # total frame size

SEED = 1234  # ensures TX/RX consistency
# ───────────────────────────────────────────────────────────────


def make_sequences(rng):
    """
    Generate deterministic preamble and pilot sequences.
    These MUST be identical on TX and RX.
    """
    preamble = rng.integers(0, 2, size=PREAMBLE_LEN, dtype=np.uint8)
    pilot = rng.integers(0, 2, size=PILOT_LEN, dtype=np.uint8)
    return preamble, pilot


def make_bit_files(output_dir="bit_tests", num_files=10):
    rng = np.random.default_rng(SEED)
    os.makedirs(output_dir, exist_ok=True)

    preamble, pilot = make_sequences(rng)

    num_data_bits = TOTAL_BITS - PREAMBLE_LEN - PILOT_LEN

    if num_data_bits < 0:
        raise ValueError(
            f"TOTAL_BITS ({TOTAL_BITS}) must be >= preamble + pilot "
            f"({PREAMBLE_LEN + PILOT_LEN})"
        )

    print(f"\nFrame structure:")
    print(f"  Preamble: {PREAMBLE_LEN} bits")
    print(f"  Pilot:    {PILOT_LEN} bits")
    print(f"  Data:     {num_data_bits} bits")
    print(f"  Total:    {TOTAL_BITS} bits\n")

    for k in range(num_files):
        data = rng.integers(0, 2, size=num_data_bits, dtype=np.uint8)

        # Final frame
        frame = np.concatenate([preamble, pilot, data])

        filename = os.path.join(output_dir, f"bits_{k:03d}.bin")
        frame.tofile(filename)

        print(f"Wrote {filename}")

    # Save reference sequences for receiver use
    preamble.tofile(os.path.join(output_dir, "preamble.bin"))
    pilot.tofile(os.path.join(output_dir, "pilot.bin"))

    print("\nSaved reference sequences:")
    print("  preamble.bin")
    print("  pilot.bin")


if __name__ == "__main__":
    make_bit_files()