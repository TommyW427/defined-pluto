# DEFINED: Decision-Feedback In-Context Detection for SDR

**Real-world validation of Transformer-based wireless symbol detection with decision feedback**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Frame Structure](#frame-structure)
4. [Quick Start](#quick-start)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Unit Tests](#unit-tests)
7. [Visualization Tools](#visualization-tools)
8. [Key Discoveries](#key-discoveries)
9. [File Structure](#file-structure)

---

## Project Overview

This project brings **DEFINED** (Decision-Feedback In-Context Detection) from simulation to real hardware using PlutoSDR transceivers.

###Core Question

> **Can decision-feedback in-context learning reduce pilot overhead when facing real hardware impairments, clock offsets, and multipath—not just idealized simulated channels?**

### Two Detection Modes

1. **Vanilla ICL**: Uses only clean pilot symbols as reference
   - Prompt: `[pilot₁, pilot₂, ..., pilotₖ]` → detect data symbols

2. **DEFINED**: Adds decision feedback from previous detections
   - Prompt: `[pilot₁, ..., pilotₖ, feedback₁, ..., feedbackₙ]` → detect next symbol
   - Feedback comes from imperfect real-world detections (not ground truth)

### Key Metrics

- **BER vs Pilot Count**: How many pilots needed to reach target BER?
- **BER vs SNR**: Performance across different noise levels
- **Latency**: Inference time per frame
- **Robustness**: Performance with domain shift (different rooms, distances, etc.)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSMITTER (PlutoSDR)                       │
│  [Preamble: 2000 bits] + [Pilot: 64 bits] + [Data: 17936 bits] │
│                           │                                     │
│                           ▼ Over-the-air at 915 MHz            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼ Hardware impairments:
                            │ • Frequency offset
                            │ • Phase noise
                            │ • Multipath
                            │ • Clock drift
                            │ • Nonlinearities
┌───────────────────────────┴─────────────────────────────────────┐
│                    RECEIVER (PlutoSDR)                          │
│  AGC → Costas Loop → Mueller & Muller Clock Recovery           │
│           │                                                     │
│           └──→ received_iq_raw.bin (250 kHz, 25 sps)           │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────┴───────────────────────────────┐
│              DATA PROCESSING (Workstation)                      │
│  1. Load raw IQ (250 kHz sample rate)                           │
│  2. Decimate by 25 → symbols (10 kHz symbol rate)              │
│  3. Detect pilot → find frame alignment                         │
│  4. Correct 180° phase inversion (Costas ambiguity)            │
│  5. Extract data bits → compute BER                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────┴───────────────────────────────┐
│           TRANSFORMER INFERENCE (Future Work)                   │
│  • Load pre-trained model (simulated data)                      │
│  • Fine-tune on real captures                                   │
│  • Run vanilla ICL vs DEFINED detection                         │
│  • Log SER, latency, pilot requirements                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Frame Structure

### Transmitted Frame (20,000 bits)

```
┌──────────────┬──────────┬────────────────────────────────┐
│  Preamble    │  Pilot   │           Data                  │
│  2000 bits   │  64 bits │        17,936 bits              │
└──────────────┴──────────┴────────────────────────────────┘
      │             │                   │
      │             │                   └─→ Random payload
      │             └──────────────────────→ Known sync sequence
      └────────────────────────────────────→ For timing/freq sync
```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Center Frequency | 915 MHz | ISM band |
| Sample Rate | 250 kHz | PlutoSDR baseband |
| Symbol Rate | 10 kHz | BPSK modulation |
| Samples/Symbol | 25 | Exactly! (not 3.54) |
| Modulation | BPSK | -1 for bit 0, +1 for bit 1 |
| Frame Length | 20,000 bits | Total transmission |

### Critical Discovery: Inverted Bits

The Costas loop can lock at **0° or 180° phase**, causing bit inversion:
- Normal: `bit = (symbol.real > 0)`
- Inverted: `bit = (symbol.real < 0)` ✓ **Use this!**

The pilot detection automatically finds and corrects this phase ambiguity.

---

## Quick Start

### 1. Process Real SDR Capture

```bash
# Basic processing: extract frame and compute BER
python3 real_data_processor.py \
    --raw-iq ../received_iq_raw.bin \
    --tx-bits ../bit_tests/bits_000.bin \
    --pilot ../bit_tests/pilot.bin \
    --output real_captures/frame_001.npz
```

**Output:**
```
✓ Loaded 500,000 raw IQ samples
✓ Decimated to 20,000 symbols (sps=25)
✓ Pilot found at position 2000
  Phase inverted: True
✓ BER: 0.001234 (0.1234%)
  Estimated SNR: 12.45 dB
✓ Saved to real_captures/frame_001.npz
```

### 2. Run Unit Tests

```bash
# Comprehensive test suite (13 tests)
python3 test_real_data_processor.py
```

**Output:**
```
======================================================================
TEST SUMMARY
======================================================================
Tests run: 13
Successes: 13
Failures: 0
Errors: 0
======================================================================
```

### 3. Generate Synthetic Training Data

```bash
# Create pre-training dataset (BPSK, Rayleigh fading)
python3 generate_synthetic_data.py \
    --modulation BPSK \
    --frame-length 64 \
    --channel-type rayleigh \
    --output-dir datasets/synthetic
```

**Output:**
```
✓ Train: 50,000 samples (150 MB)
✓ Val: 5,000 samples (15 MB)
✓ Test: 10,000 samples (30 MB)
```

### 4. Create Visualizations

```bash
# Generate demo plots
python3 visualize.py --demo
```

Creates:
- `demo_constellation.png` - BPSK constellation diagram
- `demo_ber_vs_snr.png` - BER vs SNR curves
- `demo_ber_vs_pilots.png` - **Key plot:** pilot overhead comparison
- `demo_time_series.png` - Symbol time series

---

## Data Processing Pipeline

### Step-by-Step Breakdown

#### 1. Load Raw IQ

```python
from real_data_processor import RealDataProcessor

processor = RealDataProcessor()
raw_iq = processor.load_raw_iq("received_iq_raw.bin")
# → Complex array, 250 kHz sample rate, 25 samples per symbol
```

#### 2. Decimate to Symbol Rate

```python
symbols = processor.decimate_to_symbols(raw_iq)
# Takes every 25th sample → 10 kHz symbol rate
```

**Why this works:**
- Raw IQ: 250,000 samples/sec
- Symbol rate: 10,000 symbols/sec
- Samples per symbol: 250,000 / 10,000 = **25 exactly**
- Simple decimation: `symbols = raw_iq[::25]`

#### 3. Detect Bits (Handle Phase Inversion)

```python
# Try normal detection first
rx_bits = processor.detect_bits_bpsk(symbols, assume_inverted=False)

# Find pilot and check for inversion
pilot_pos, is_inverted = processor.find_pilot_position(
    rx_bits, pilot_ref
)

# If inverted, re-detect with correction
if is_inverted:
    rx_bits = processor.detect_bits_bpsk(symbols, assume_inverted=True)
```

#### 4. Extract Data and Compute BER

```python
# Data starts after pilot
data_start = pilot_pos + 64  # 64-bit pilot

rx_data = rx_bits[data_start:data_start + 17936]
tx_data = tx_bits_ref[2064:2064 + 17936]  # Skip preamble+pilot

errors = np.sum(rx_data != tx_data)
ber = errors / len(tx_data)
```

---

## Unit Tests

### Test Coverage

| Test | Description | Validates |
|------|-------------|-----------|
| `test_constants` | Frame structure constants | 2000 + 64 + 17936 = 20000 ✓ |
| `test_decimation` | Every 25th sample taken | Correct symbol rate |
| `test_bpsk_detection_normal` | Normal phase detection | Maps -1→0, +1→1 |
| `test_bpsk_detection_inverted` | 180° phase detection | Maps +1→0, -1→1 |
| `test_pilot_detection_exact` | Pilot at position 2000 | Frame alignment |
| `test_pilot_detection_inversion` | Inverted pilot detection | Phase ambiguity |
| `test_pilot_detection_offset` | Pilot with timing offset | Robust search |
| `test_snr_estimation` | SNR from symbol noise | Accurate within 1 dB |
| `test_end_to_end` | Complete frame processing | All steps work together |
| `test_frame_structure` | Bit counts sum correctly | No off-by-one errors |
| `test_load_raw_iq` | File I/O works | Correct data loading |
| `test_load_missing_file` | Error handling | Fails gracefully |
| `test_load_empty_file` | Empty file detection | Validates data |

**Run tests:**
```bash
python3 test_real_data_processor.py -v
```

---

## Visualization Tools

### Available Plots

#### 1. Constellation Diagram
```python
from visualize import Visualizer

viz = Visualizer(output_dir="plots")
viz.plot_constellation(
    symbols=rx_symbols,
    reference_points=np.array([-1+0j, 1+0j]),
    title="BPSK Constellation (SNR=10dB)"
)
```

**Shows:**
- Received symbol scatter (I/Q plane)
- Ideal constellation points
- Noise spread around each point

#### 2. BER vs SNR
```python
viz.plot_ber_vs_snr(
    snr_db=np.array([snr_range, snr_range]),
    ber=np.array([ber_vanilla, ber_defined]),
    labels=["Vanilla ICL", "DEFINED"]
)
```

**Compares:**
- Multiple detection methods
- Performance across SNR range
- Reference BER targets (1%, 0.1%)

#### 3. BER vs Pilot Count ⭐ **KEY PLOT**
```python
viz.plot_ber_vs_pilots(
    num_pilots=np.array([4, 8, 16, 32, 64]),
    ber=np.array([ber_vanilla, ber_defined]),
    labels=["Vanilla ICL", "DEFINED"]
)
```

**This plot answers the core question:**
- How many pilots does vanilla ICL need?
- How many pilots does DEFINED need?
- What is the pilot overhead reduction?

**Example result (expected):**
```
Target: SER < 1%
Vanilla ICL: requires 32 pilots
DEFINED: requires 8 pilots
→ 4× reduction in pilot overhead!
```

#### 4. Time Series
```python
viz.plot_time_series(
    symbols=rx_symbols,
    sample_rate=10000,
    max_samples=500
)
```

**Shows:**
- Symbol I/Q components over time
- BPSK transitions between ±1
- Timing and sync quality

---

## Key Discoveries

### 1. ⚠️ Avoid `received_symbols.bin` from Clock Recovery

**Problem:** Mueller & Muller clock recovery outputs **variable sample rate**
- Expected: 10,000 symbols/sec (1 sample per symbol)
- Actual: ~3.54 samples per symbol (variable)
- Result: Frame misalignment, incorrect BER

**Solution:** Use `received_iq_raw.bin` directly
- Fixed 250 kHz sample rate
- Decimate by exactly 25 → correct symbol rate
- Reliable frame alignment

### 2. ✓ 180° Phase Inversion is Normal

**Cause:** Costas loop phase ambiguity
- BPSK has 2-fold symmetry
- Costas loop can lock at 0° or 180°
- 180° lock inverts all bits

**Detection:**
```python
# Compare pilot against reference
pilot_errors_normal = count_errors(rx_pilot, pilot_ref)
pilot_errors_inverted = count_errors(rx_pilot, ~pilot_ref)

is_inverted = (pilot_errors_inverted < pilot_errors_normal)
```

**Correction:**
```python
if is_inverted:
    rx_bits = 1 - rx_bits  # Flip all bits
    # Or equivalently: detect_bits_bpsk(symbols, assume_inverted=True)
```

### 3. ✓ Decimation by 25 is Exact

**Math:**
```
Sample rate:     250,000 Hz
Symbol rate:      10,000 Hz
Samples/symbol:  250,000 / 10,000 = 25 exactly
```

**Implementation:**
```python
symbols = raw_iq[::25]  # Simple, exact, works!
```

No interpolation needed. No fractional resampling. Just take every 25th sample.

---

## File Structure

```
DEFINED-SDR/
├── ml_pipeline/                    ← You are here
│   ├── README.md                   ← This file
│   │
│   ├── config.py                   ← Configuration and constants
│   ├── channel_simulator.py       ← Synthetic channel models
│   ├── generate_synthetic_data.py ← Pre-training dataset generation
│   │
│   ├── real_data_processor.py     ← Real SDR data processing ⭐
│   ├── test_real_data_processor.py← Unit tests (13 tests)
│   │
│   ├── visualize.py                ← Plotting tools
│   │
│   └── datasets/                   ← Generated datasets
│       ├── synthetic/              ← Pre-training data
│       └── real_captures/          ← Processed SDR captures
│
├── bit_tests/                      ← Test bit sequences
│   ├── bits_000.bin ... bits_009.bin  ← 10 test frames
│   ├── preamble.bin                   ← Sync preamble
│   └── pilot.bin                      ← Pilot sequence
│
├── make_bit_files.py               ← Generate test frames
├── Receive_Signal.py               ← GNU Radio receiver
├── Receive_Signal_2.py             ← Receiver (annotated)
├── Send_Signal.py                  ← GNU Radio transmitter
│
├── received_iq_raw.bin             ← Raw IQ capture (250 kHz)
├── received_symbols.bin            ← After clock recovery (AVOID!)
│
├── calc_ber.py                     ← Quick BER calculation
├── check_bits.py                   ← Bit comparison utility
└── plot_constellation.py           ← Constellation plot
```

---

## Next Steps

### Phase 1: Data Collection ✓ (Current)
- [x] Real data processor with unit tests
- [x] Synthetic data generator
- [x] Visualization tools
- [x] Frame structure validation

### Phase 2: ML Model (Future)
- [ ] Implement DEFINED Transformer architecture
- [ ] Train on synthetic data (pre-training)
- [ ] Fine-tune on real captures
- [ ] Inference service for real-time detection

### Phase 3: Evaluation
- [ ] Collect real captures at different SNRs
- [ ] Run vanilla ICL vs DEFINED comparison
- [ ] Plot BER vs pilot count (key result!)
- [ ] Ablation: sim-only, real-only, sim+real fine-tune
- [ ] Test robustness (different rooms, distances, etc.)

---

## Usage Examples

### Example 1: Process a Single Capture

```bash
# Transmit bits_000.bin over the air
# Capture to received_iq_raw.bin
# Process:

python3 real_data_processor.py \
    --raw-iq ../received_iq_raw.bin \
    --tx-bits ../bit_tests/bits_000.bin \
    --pilot ../bit_tests/pilot.bin \
    --output results/capture_001.npz

# Analyze results:
python3 -c "
import numpy as np
data = np.load('results/capture_001.npz')
print(f'BER: {data[\"ber\"]:.6f}')
print(f'SNR: {data[\"snr_db\"]:.2f} dB')
print(f'Inverted: {data[\"is_inverted\"]}')
"
```

### Example 2: Batch Process Multiple Captures

```bash
# Process all 10 test files
for i in {0..9}; do
    python3 real_data_processor.py \
        --raw-iq ../received_iq_raw_$i.bin \
        --tx-bits ../bit_tests/bits_00$i.bin \
        --pilot ../bit_tests/pilot.bin \
        --output results/capture_00$i.npz
done
```

### Example 3: Generate Training Data

```bash
# BPSK, Rayleigh fading
python3 generate_synthetic_data.py \
    --modulation BPSK \
    --channel-type rayleigh \
    --frame-length 64 \
    --output-dir datasets/bpsk_rayleigh

# QPSK, Rician fading
python3 generate_synthetic_data.py \
    --modulation QPSK \
    --channel-type rician \
    --frame-length 64 \
    --output-dir datasets/qpsk_rician
```

---

## Troubleshooting

### Q: BER is very high (>10%)

**Check:**
1. Phase inversion detected? Look for `is_inverted: True`
2. Pilot position correct? Should be near 2000
3. SNR reasonable? Should be >5 dB for low BER

**Debug:**
```python
processor = RealDataProcessor()
frame = processor.extract_frame(...)

print(f"Pilot position: {frame['pilot_position']}")
print(f"Expected: {processor.preamble_bits} (2000)")
print(f"Offset: {frame['pilot_position'] - 2000}")
print(f"Phase inverted: {frame['is_inverted']}")
print(f"SNR: {frame['snr_db']:.2f} dB")
```

### Q: Unit tests fail

**Check:**
1. NumPy version: `pip install numpy --upgrade`
2. Python version: Requires Python 3.8+

**Run verbose:**
```bash
python3 test_real_data_processor.py -v
```

### Q: Visualizations don't appear

**Check:**
1. Matplotlib backend: May need `export MPLBACKEND=Agg` on servers
2. Output directory created: Check `plots/` or `demo_plots/`

---

## Citation

If you use this code, please cite the DEFINED paper:

```bibtex
@article{defined2024,
  title={DEFINED: Decision-Feedback Inference for Wireless Symbol Detection},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## License

GPL-3.0 (matching GNU Radio components)

---

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

**Happy wireless ML! 📡🤖**
