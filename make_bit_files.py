#!/usr/bin/env python3
import os
import numpy as np
 
# ── Pilot sequence ──────────────────────────────────────────────
# This MUST match the receiver side so DEFINED knows which
# symbols are ground-truth pilots.  Length k = number of pilot
# symbols in the paper's notation.
PILOT_BITS = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
# ────────────────────────────────────────────────────────────────
 
def make_bit_files(output_dir="bit_tests", num_files=10,
                   bits_per_file=20000, seed=1234):
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
 
    pilot = np.array(PILOT_BITS, dtype=np.uint8)
    num_data_bits = bits_per_file - len(pilot)
 
    if num_data_bits < 0:
        raise ValueError(
            f"bits_per_file ({bits_per_file}) must be >= pilot length ({len(pilot)})"
        )
 
    for k in range(num_files):
        data = rng.integers(0, 2, size=num_data_bits, dtype=np.uint8)
        frame = np.concatenate([pilot, data])
 
        filename = os.path.join(output_dir, f"bits_{k:03d}.bin")
        frame.tofile(filename)
        print(f"Wrote {filename}  "
              f"(pilot: {len(pilot)} bits | data: {num_data_bits} bits | "
              f"total: {bits_per_file} bits)")
 
if __name__ == "__main__":
    make_bit_files()
 