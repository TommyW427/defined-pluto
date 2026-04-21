# PaperMMSE: MMSE vs MMSE-DF Comparison

Source script: [paper_verification/PaperMMSE.py](../paper_verification/PaperMMSE.py)

## Quick References

### A) Core functions and roles

| Function | Role | Main inputs | Main output |
|---|---|---|---|
| generate_frame | Generate synthetic block-fading data | batch_size, T, snr_db | y, x, h, sigma2 |
| mmse_channel_estimate | Estimate scalar channel from pilot/context pairs | y_p, x_p, sigma2 | h_hat |
| mmse_detect | Coherent BPSK slicing | y, h_hat | x_hat in {-1, +1} |
| mmse_eval | Baseline MMSE (no feedback) | y, x, sigma2, k | Average SER |
| mmse_df_eval | MMSE with full decision feedback | y, x, sigma2, k | Average SER |
| mmse_df_eval_fixed_feedback_window | MMSE-DF with limited feedback memory | y, x, sigma2, k_initial, L | Average SER |
| evaluate_fixed_pilot_sliding_feedback | MC sweep over feedback window L | snr_db, k_initial, L list | Fixed MMSE SER + DF curve |
| evaluate_snr | MC sweep over pilot count k | snr_db, k_values | MMSE and DF curves |

### B) Key equations used in code

- Channel model: $y_t = h x_t + n_t$
- MMSE-style SISO estimate:
  $$\hat{h} = \frac{\sum_i y_i x_i^*}{\sum_i |x_i|^2 + \sigma^2}$$
- BPSK coherent decision:
  $$\hat{x}_t = \operatorname{sign}(\Re\{\hat{h}^* y_t\})$$

### C) Fixed parameters in this script

- Frame length: $T=31$
- Modulation: BPSK
- Batch size: 512
- Monte Carlo trials: 200
- Default run block: fixed pilot $k=1$, SNR = 5 dB, sweep feedback window $L$ from 0 to $T-k-1$

## Compare and Contrast: MMSE vs MMSE-DF (Code Structure)

## 1) Shared setup blocks

Both methods share the same:
- Data generator: generate_frame
- Channel estimator: mmse_channel_estimate
- Symbol slicer: mmse_detect
- Error aggregation style: collect per-time errors then average

Why this matters:
- The comparison is fair because only the feedback mechanism changes, not the signal model or detector primitive.

## 2) Structural difference in context used for channel estimation

### MMSE (no feedback)

Location: mmse_eval

Code structure:
- Slice pilots once: y[:, :k], x[:, :k]
- Estimate h_hat once before loop
- Loop t from k to T-1 with fixed h_hat

Interpretation:
- Static channel estimate from pilot-only context.
- No update from model decisions.

### MMSE-DF (full feedback)

Location: mmse_df_eval

Code structure:
- Start with x_feedback = x.clone()
- For each t from k to T-1:
  - Build context y[:, :t], x_feedback[:, :t]
  - Re-estimate h_hat
  - Detect x_hat at t
  - Write back decision into x_feedback[:, t]

Interpretation:
- Dynamic channel estimate that grows with decisions.
- Uses pilots plus all prior decisions as pseudo-pilots.

Bottom line:
- MMSE: estimate once, detect many.
- MMSE-DF: estimate-detect-update at every step.

## 3) Loop topology and computational cost

### MMSE

- One estimator call per frame.
- One detection call per data symbol.
- Lower compute and simpler dependency graph.

### MMSE-DF

- One estimator call per detected symbol.
- Sequential dependency: t-th decision affects all later steps.
- Higher compute and more sensitivity to early decision errors.

Practical implication:
- MMSE is cheaper and stable.
- MMSE-DF can outperform MMSE when feedback quality is good, but may suffer from error propagation.

## 4) Error propagation behavior in code

### MMSE

- No writes to context during detection.
- Wrong decisions do not alter future channel estimates.

### MMSE-DF

- Writes each decision back into x_feedback.
- Wrong early decisions can distort later estimates.

Why this is important:
- This is the core tradeoff behind MMSE-DF gains versus instability.

## 5) Additional DF variant: fixed sliding feedback window

Location: mmse_df_eval_fixed_feedback_window

What changes versus full DF:
- Always keep initial k pilots.
- Instead of all prior decisions, include only most recent L decisions.

Code-level effect:
- Context for estimator is concatenation of:
  - fixed pilot block
  - rolling feedback block

Interpretation:
- L=0 collapses to pilot-only estimate behavior.
- Larger L approaches full-history DF.
- Useful to study memory depth versus robustness.

## 6) Experiment driver contrast

### Fixed-pilot, sweep-L experiment

Location: evaluate_fixed_pilot_sliding_feedback

Compares:
- A constant MMSE reference at one pilot budget.
- A DF curve as L varies.

Use case:
- Isolate the effect of feedback memory size.

### Sweep-k experiment

Location: evaluate_snr

Compares:
- MMSE and MMSE-DF across pilot budgets k.

Use case:
- Figure-4-style pilot-length trend analysis.

## 7) Side-by-side summary table

| Aspect | MMSE | MMSE-DF |
|---|---|---|
| Estimation context | First k pilots only | Pilots + previous decisions |
| Channel estimate updates | Once per frame | Every symbol |
| Sequential dependency | Low | High |
| Error propagation risk | Low | Medium to high |
| Compute cost | Lower | Higher |
| Potential gain in low pilot regime | Limited | Higher potential |

## 8) Quick interpretation guide for your plots

- If DF curve is below MMSE baseline, feedback is helping.
- If DF improves up to moderate L then saturates, only recent feedback is useful.
- If DF worsens at large L, long-memory error accumulation is likely.
- If MMSE and DF are close, either pilots are already sufficient or SNR is too low for reliable feedback.

