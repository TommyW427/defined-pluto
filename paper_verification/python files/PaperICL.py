# ============================================================
# VANILLA ICL BASELINE (PAPER STYLE)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CONTEXT LENGTH SETUP
# ============================================================
MIN_CONTEXT_K = 1
MAX_CONTEXT_K = 30


def sample_context_length(min_k=MIN_CONTEXT_K, max_k=MAX_CONTEXT_K):
    return int(torch.randint(min_k, max_k + 1, (1,), device=device).item())


# ============================================================
# CONSTELLATION (BPSK default)
# ============================================================
def get_constellation():
    return torch.tensor([-1.0 + 0j, 1.0 + 0j], dtype=torch.complex64)


# ============================================================
# CHANNEL MODEL (Rayleigh block fading)
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
# PROMPT BUILDER (ICL only, clean context)
# ============================================================
def build_prompt(y, x, k, t):
    # features: [Re(y), Im(y), x_onehot]
    B = y.shape[0]
    feat = torch.zeros(B, k + 1, 3, device=device)

    feat[:, :k, 0] = y[:, :k].real
    feat[:, :k, 1] = y[:, :k].imag
    feat[:, :k, 2] = x[:, :k].float()

    feat[:, k, 0] = y[:, t].real
    feat[:, k, 1] = y[:, t].imag

    return feat


# ============================================================
# MODEL (decoder-style via causal mask)
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

        self.out = nn.Linear(d_model, 2)  # BPSK logits

    def forward(self, x):
        x = self.inp(x)
        x = self.tr(x)
        return self.out(x)


# ============================================================
# TRAINING (ICL only)
# ============================================================
model = Transformer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_step():
    y, x = generate_batch()

    k = sample_context_length()
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
# TRAIN LOOP
# ============================================================
for i in range(2000):
    l = train_step()
    if i % 200 == 0:
        print("ICL step", i, "loss", l)


# ============================================================
# EVALUATION (ICL baseline)
# ============================================================
@torch.no_grad()
def evaluate(k_val=4):
    y, x = generate_batch(batch_size=512)

    k = k_val
    errors = 0
    total = 0

    for t in range(k, x.shape[1]):
        prompt = build_prompt(y, x, k, t)
        pred = model(prompt)[:, -1].argmax(-1)

        errors += (pred != x[:, t]).sum().item()
        total += x.size(0)

    return errors / total


print("ICL SER:", evaluate())

# ============================================================
## Plotting SER vs. Context Length `k`
# ============================================================

import matplotlib.pyplot as plt

k_values = list(range(1, 31))  # Test k from 1 to 30
ser_results = []

print("Evaluating SER for different k values...")
for k_val in k_values:
    ser = evaluate(k_val=k_val)
    ser_results.append(ser)
    print(f"k={k_val}, SER={ser:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, ser_results, marker='o', linestyle='-', color='b')
plt.title('Symbol Error Rate (SER) vs. Context Length (k)')
plt.xlabel('Context Length (k)')
plt.ylabel('Symbol Error Rate (SER)')
plt.xticks(k_values)
plt.grid(True)
plt.show()