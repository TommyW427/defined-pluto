# DEFINED vs PaperICL: Block-by-Block Notes

## Quick Reference (Most Important Differences)

| Topic | PaperDEFINED | PaperICL |
|---|---|---|
| Core idea | Two-phase method (ICL pretrain + DF finetune) | Vanilla ICL baseline |
| Training phases | 2 phases (`icl_step`, `defined_step`) | 1 phase (`train_step`) |
| Context source during training | True labels + self-feedback labels | True labels only |
| Inference style | Sequential decision feedback loop | Clean-context prompting |
| Main objective | Blended loss: `0.7 * loss_df + 0.3 * loss_icl` | Cross-entropy only |
| Training context lengths | Samples `k` during ICL pretraining and DF finetuning | Samples `k` during training |
| Exposure-bias handling | Explicitly addressed in DF finetuning | Not explicitly addressed |
| Evaluation extras | Single DEFINED SER report | SER and optional SER-vs-`k` sweep/plot |
| Expected strength | Better robustness when using model-generated history | Strong clean-context baseline |

Use this table as the fast summary, then read the sections below for block-level details.

## 1) Header and Goal

### PaperDEFINED
- Labeled as `DEFINED (PAPER METHOD)`.
- Purpose is not just baseline ICL detection; it adds decision-feedback training and sequential feedback inference.

### PaperICL
- Labeled as `VANILLA ICL BASELINE (PAPER STYLE)`.
- Pure in-context learning baseline with clean context only (no self-feedback loop).

### Comparison
- PaperICL is the reference baseline.
- PaperDEFINED is a two-phase extension of that baseline intended to handle mismatch between clean training context and self-generated context at inference.

---

## 2) Imports and Device

### PaperDEFINED
- Imports: `torch`, `torch.nn`, `torch.nn.functional as F`, `math`.
- Device selection: `"cuda" if torch.cuda.is_available() else "cpu"`.

### PaperICL
- Same imports and same device logic.

### Comparison
- Identical setup for compute and dependencies.
- Fair comparison because infrastructure is unchanged.

---

## 3) Channel Model (`generate_batch`)

### PaperDEFINED
- Uses Rayleigh block fading channel:
	- `h ~ CN(0, 1)` per sample.
- Uses fixed `snr_db = 10` and properly normalized complex Gaussian noise.
- Returns `y` and `x_idx` (bit labels 0/1).

### PaperICL
- Same channel generation code and same return format.

### Comparison
- This block is effectively identical.
- Any performance difference should come from training/inference strategy, not data generation.

---

## 4) Model Definition (`Transformer`)

### PaperDEFINED
- Input projection: `Linear(3, d_model)`.
- Transformer encoder: 4 layers, 4 heads.
- Output projection: `Linear(d_model, 2)` logits for BPSK class prediction.

### PaperICL
- Same architecture and dimensions.

### Comparison
- Same capacity and architecture for fairness.
- Keeps ablation clean: method difference, not model-size difference.

---

## 5) Optimizer Setup

### PaperDEFINED
- `Adam` with `lr=1e-4`.

### PaperICL
- Same optimizer and learning rate.

### Comparison
- Optimization hyperparameters are matched.
- Good experimental control.

---

## 6) Prompt Builder (`build_prompt`)

### PaperDEFINED
- `build_prompt(y, x_ctx, k, t)` where `x_ctx` can be either:
	- true labels (`x`) during clean ICL,
	- or self-feedback labels (`feedback`) during DEFINED.
- Feature tensor shape: `[B, k+1, 3]`.
- Context positions `:k` include `[Re(y), Im(y), x_ctx]`.
- Query position `k` includes only `[Re(y_t), Im(y_t)]`.

### PaperICL
- `build_prompt(y, x, k, t)` always uses true context labels.

### Comparison
- Core feature format is the same.
- Key DEFINED change is the flexible context source (`x_ctx`) enabling train-time exposure to model-generated context.

---

## 7) Training Phase 1 (`icl_step`) in PaperDEFINED

### What it does
- Runs standard ICL objective first.
- Samples `k` from the training range.
- For each target time `t` from `k` to `T-1`, builds clean prompt with true context labels.
- Computes average cross-entropy over all targets.

### Compared to PaperICL train step
- PaperICL does the same style objective, but uses a single training phase and samples `k` during training.

### Why this matters
- DEFINED starts by learning a clean detector first (similar to baseline behavior).
- Variable context lengths make the model less dependent on a single training prompt length and better aligned with the sweep.

---

## 8) Training Phase 2 (`defined_step`) in PaperDEFINED

### What it does
- Generates a fresh batch.
- Initializes `feedback = x.clone()`.
- Computes two losses:
	- `loss_icl`: clean-context loss (true labels in context).
	- `loss_df`: decision-feedback loss (context progressively replaced with model predictions).
- In DF loop:
	- Predict at time `t` from current `feedback` context.
	- Set `feedback[:, t] = pred`.
	- Accumulate CE against ground truth target.
- Final blended loss: `0.7 * loss_df + 0.3 * loss_icl`.

### Compared to PaperICL
- PaperICL has no feedback loop and no blended objective.
- PaperICL only optimizes clean-context CE.

### Why this matters
- This addresses exposure bias: at inference, model conditions on its own past decisions, not true labels.
- DEFINED explicitly trains for that scenario, which can improve robustness under error propagation.

---

## 9) Training Schedule

### PaperDEFINED
- Stage A: `1500` steps of `icl_step()` pretraining.
- Stage B: `1500` steps of `defined_step()` finetuning.

### PaperICL
- Single stage: `2000` steps of clean `train_step()`.

### Comparison
- PaperDEFINED uses curriculum/two-stage optimization.
- PaperICL uses one-stage baseline optimization.
- Total updates: DEFINED uses more total steps (3000 vs 2000), so gains may be due to both method and longer training unless controlled.

---

## 10) Inference / Evaluation

### PaperDEFINED (`evaluate_defined`)
- Uses sequential feedback at test time.
- Uses the provided evaluation `k` value, with the default set to `MIN_CONTEXT_K`.
- For each `t >= k`:
	- Predict symbol from prompt built with current `feedback`.
	- Write prediction back into `feedback[:, t]`.
	- Continue sequentially.
- Reports SER over all evaluated symbols.

### PaperICL (`evaluate`)
- Uses clean context labels directly (no self-feedback state update).
- For each `t >= k`, predicts from prompt containing true context labels.
- Reports SER.

### Comparison
- PaperICL evaluation is cleaner/easier but optimistic if deployment requires self-feedback.
- PaperDEFINED evaluation is more deployment-like for decision-feedback receivers.

---

## 11) Extra Analysis in PaperICL (SER vs `k`)

### PaperICL
- Sweeps `k` from 1 to 29.
- Computes and plots SER vs context length.

### PaperDEFINED
- No context-length sweep implemented in this script.

### Comparison
- PaperICL gives a useful context-sensitivity curve.
- Adding the same sweep to DEFINED would make comparison more complete (DEFINED SER vs `k`).

---

## 12) Practical Interpretation

## Similarities
- Same channel simulator.
- Same transformer backbone.
- Same optimizer and CE-style symbol objective.
- Same SER-style reporting.

## Main Differences
- PaperICL: single-phase, clean-context-only training and evaluation.
- PaperDEFINED: two-phase training with explicit decision-feedback fine-tuning and sequential self-feedback inference.
- PaperICL and PaperDEFINED both now train on variable context lengths; they still differ because PaperDEFINED adds decision-feedback finetuning and sequential self-feedback inference.

## Expected Behavior
- PaperICL can look strong in clean-context evaluation.
- PaperDEFINED should be more robust when real inference depends on model-generated prior decisions.
- If feedback errors cascade, DEFINED training is designed to reduce that mismatch.

---

## 13) Suggested Fair-Comparison Checklist

- Match total optimization steps (or report both wall-clock and step-normalized comparisons).
- Evaluate both methods under the same `k` values.
- Report both clean-context SER and self-feedback SER.
- Add SER vs `k` plots for DEFINED as done in PaperICL.
- Keep identical random seeds and channel batches when possible.

## 14) Updated Interpretation After PaperICL Change

- PaperICL is no longer a fixed-`k` training baseline if you keep the new variable-`k` setup.
- Its SER-vs-`k` plot is now closer to a true context-length robustness test.
- DEFINED still differs by adding self-feedback finetuning and sequential inference, but the context-length comparison is now fairer on both sides.
