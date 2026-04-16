# ============================================================
# DEFINED (PAPER METHOD)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SAME CHANNEL MODEL
# ============================================================
def generate_batch(batch_size=256, T=31):
    x_idx = torch.randint(0, 2, (batch_size, T), device=device)
    x = 2 * x_idx.float() - 1

    h = (torch.randn(batch_size, 1, device=device) +
         1j * torch.randn(batch_size, 1, device=device)) / math.sqrt(2)

    snr_db = 10
    snr = 10 ** (snr_db / 10)
    noise_sigma = math.sqrt(1 / snr)

    noise = noise_sigma / math.sqrt(2) * (
        torch.randn(batch_size, T, device=device) +
        1j * torch.randn(batch_size, T, device=device)
    )

    y = h * x + noise
    return y, x_idx


# ============================================================
# MODEL (same as ICL for fairness)
# ============================================================
class Transformer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.inp = nn.Linear(3, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=4)

        self.out = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.inp(x)
        x = self.tr(x)
        return self.out(x)


model = Transformer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)


# ============================================================
# PROMPT BUILDER (USED FOR BOTH ICL + DF)
# ============================================================
def build_prompt(y, x_ctx, k, t):
    B = y.shape[0]
    feat = torch.zeros(B, k + 1, 3, device=device)

    feat[:, :k, 0] = y[:, :k].real
    feat[:, :k, 1] = y[:, :k].imag
    feat[:, :k, 2] = x_ctx[:, :k].float()

    feat[:, k, 0] = y[:, t].real
    feat[:, k, 1] = y[:, t].imag

    return feat


# ============================================================
# PHASE 1 — ICL PRETRAINING
# ============================================================
def icl_step():
    y, x = generate_batch()

    k = 1
    loss = 0

    for t in range(k, x.shape[1]):
        prompt = build_prompt(y, x, k, t)
        logits = model(prompt)[:, -1, :]
        loss = loss + F.cross_entropy(logits, x[:, t])

    loss = loss / (x.shape[1] - k)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


# ============================================================
# PHASE 2 — DEFINED FINETUNING (decision feedback)
# ============================================================
def defined_step():
    y, x = generate_batch()

    k = 1
    feedback = x.clone()

    loss_icl = 0
    loss_df = 0

    # ICL loss (clean)
    for t in range(k, x.shape[1]):
        prompt = build_prompt(y, x, k, t)
        logits = model(prompt)[:, -1, :]
        loss_icl += F.cross_entropy(logits, x[:, t])

    loss_icl /= (x.shape[1] - k)

    # DF loss (with self-feedback)
    for t in range(k, x.shape[1]):
        prompt = build_prompt(y, feedback, k, t)
        logits = model(prompt)[:, -1, :]

        pred = logits.argmax(-1)
        feedback[:, t] = pred

        loss_df += F.cross_entropy(logits, x[:, t])

    loss_df /= (x.shape[1] - k)

    loss = 0.7 * loss_df + 0.3 * loss_icl

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


# ============================================================
# TRAINING
# ============================================================
for i in range(1500):
    l = icl_step()
    if i % 200 == 0:
        print("ICL pretrain", i, l)

for i in range(1500):
    l = defined_step()
    if i % 200 == 0:
        print("DEFINED finetune", i, l)


# ============================================================
# DEFINED INFERENCE (sequential feedback loop)
# ============================================================
@torch.no_grad()
def evaluate_defined():
    y, x = generate_batch(batch_size=512)

    k = 1
    feedback = x.clone()

    errors = 0
    total = 0

    for t in range(k, x.shape[1]):
        prompt = build_prompt(y, feedback, k, t)
        logits = model(prompt)[:, -1, :]
        pred = logits.argmax(-1)

        feedback[:, t] = pred

        errors += (pred != x[:, t]).sum().item()
        total += x.size(0)

    return errors / total


print("DEFINED SER:", evaluate_defined())