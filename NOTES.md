# SDR Wireless Bitstream Reception Notes (ADALM-PLUTO)

## 1. Experimental Setup

### 1.1 Hardware

- **Transmitter/Receiver:** ADALM-PLUTO SDR
- **Frequency:** 915 MHz
- **Sample Rate:** 250 kHz (complex baseband IQ)
- **Symbol Rate:** 10 kHz -> 25 samples/symbol
- **Antennas:** Used for wireless transmission; can move closer/farther to observe channel effects

### 1.2 Test Configurations

| Setup | Description |
|-------|-------------|
| **Ideal** | Full wireless link with antennas; compare three receiver variations: DEFINED, Basic ICL, Existing SDR |
| **Next Best** | Receiver without DEFINED (vanilla ICL); test using direct cable hookup |
| **Needs** | Receiver requires transformer for signal conditioning |
| **Offline** | Data can be collected and processed offline for debugging/analysis |

---

## 2. Receiver Workflow

The receiver chain reverses the transmitter processing. Key steps:

### 2.1 Automatic Gain Control (AGC)

- Normalizes amplitude to ~1.0 for consistent processing
- Prevents downstream blocks (Costas loop, clock recovery) from being affected by varying signal strength
- Adaptation rate: 1e-4 (slow averaging over ~10,000 samples)

### 2.2 Frequency & Phase Synchronization (Costas Loop)

- Corrects **carrier frequency offset** and **phase offset**
- BPSK-specific (order=2)
- Loop bandwidth example: 0.028-0.0628 rad/sample
- After lock: constellation points sit on the real axis at +/-1

### 2.3 Clock Recovery / Symbol Synchronization (Mueller & Mueller TED)

- Finds optimal sampling instant for each symbol
- Converts input 25 samples/symbol -> output 1 sample/symbol
- TED error:

```text
e(n) = Re{ x^(n-1) * y(n) - x^(n) * y(n-1) }
```

- Uses MMSE 8-tap interpolator to sample fractional positions between input samples

### 2.4 Frame Alignment

- Detects start of frame using preamble (not pilots)
- Ensures symbols are correctly mapped to frame positions before demodulation

### 2.5 Channel Estimation / Equalization

- Uses pilot symbols to measure amplitude/phase distortion of the wireless channel
- Corrects per-symbol distortions due to multipath, fading, or attenuation
- Needed even if synchronization is perfect; synchronization only fixes timing and frequency offsets

### 2.6 Bit Estimation / Demodulation

- Converts equalized symbols -> bits
- Soft or hard decisions can be applied (e.g., LLR for FEC)
- Any error in synchronization or channel estimation propagates to bit errors

## 3. Observations / Notes

### 3.1 Effects of Imperfect Processing

| Error Type | Symptom | Result |
|-----------|---------|--------|
| Sync error | Symbols misaligned in time or spinning in phase | Lost information; many bit errors |
| Channel estimate error | Symbols offset in amplitude/phase | Some symbols decoded incorrectly |

### 3.2 Visual Indicators on PlutoSDR

Constellation plot:

- Two tight clusters on real axis -> BPSK properly synchronized and equalized
- Spinning -> Costas loop not locked
- Smeared -> clock recovery not locked

Time-domain plot:

- Rectangular wave jumping +/-1 -> ideal BPSK after clock recovery
- Variations in amplitude -> channel distortion

## 4. Implementation Details (Mar 18th)

- Clock recovery: Mueller & Mueller
- Loop BW = 0.045
- Phase/Frequency recovery: Costas loop
- Order = 2
- Loop BW = 0.028
- Output: `received_symbols.bin` (1 complex value per symbol)
- Optional: `received_iq_raw.bin` for offline debugging

### 4.1 Python/GNU Radio Setup

- Blocks: PlutoSDR source -> AGC -> Costas Loop -> Symbol Sync -> File Sink
- Visualization: Constellation sink, time-domain sink, spectrum sink
- GUI sliders: Adjust RX gain, Costas BW, Clock BW during runtime

## 5. Next Steps / To Do

- Ensure proper synchronization and alignment for all test scenarios
- Implement DEFINED processing after verifying basic symbol recovery
- Collect data in multiple setups (wireless, cable, varying antenna distances)
- Evaluate BER across setups for comparison of DEFINED vs Basic ICL vs Existing SDR
