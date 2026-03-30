#!/usr/bin/env python3
"""Quick constellation plot"""
import numpy as np
import matplotlib.pyplot as plt

raw_iq = np.frombuffer(open('received_iq_raw.bin','rb').read(), dtype=np.complex64)
rx = raw_iq[::25]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Raw IQ
ax1.plot(raw_iq[::100].real[:5000], raw_iq[::100].imag[:5000], 'b.', alpha=0.3, ms=1)
ax1.set_title('Raw IQ')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# After decimation
ax2.plot(rx[::10].real[:5000], rx[::10].imag[:5000], 'r.', alpha=0.5, ms=2)
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Decimated (BPSK)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('constellation.png', dpi=100)
print("Saved constellation.png")
