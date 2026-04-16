import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CONSTELLATION (BPSK default)
# ============================================================
def get_constellation():
    return torch.tensor([-1.0, 1.0], device=device)


# ============================================================
# CHANNEL MODEL (SISO block fading)
# y_t = h x_t + n_t
# ============================================================
def generate_batch(batch_size=512, T=31, snr_db=10):
    x_idx = torch.randint(0, 2, (batch_size, T), device=device)
    x = 2 * x_idx.float() - 1

    # Rayleigh channel
    h = (torch.randn(batch_size, 1, device=device) +
         1j * torch.randn(batch_size, 1, device=device)) / math.sqrt(2)

    snr = 10 ** (snr_db / 10)
    sigma = math.sqrt(1 / snr)

    noise = sigma / math.sqrt(2) * (
        torch.randn(batch_size, T, device=device) +
        1j * torch.randn(batch_size, T, device=device)
    )

    y = h * x + noise
    return y, x, h


# ============================================================
# MMSE CHANNEL ESTIMATION (SISO)
# ============================================================
def mmse_channel_estimate(y_p, x_p, sigma2):
    """
    SISO MMSE reduces to LS estimator:
        h_hat = sum(y_p * x_p*) / (sum(|x_p|^2) + sigma^2)
    """
    num = torch.sum(y_p * x_p.conj(), dim=1, keepdim=True)
    den = torch.sum((x_p * x_p.conj()).real, dim=1, keepdim=True) + sigma2
    h_hat = num / den
    return h_hat


# ============================================================
# MMSE DETECTION
# ============================================================
def mmse_detect(y, h_hat):
    """
    BPSK detection:
        x_hat = sign(Re(h* y))
    """
    z = (h_hat.conj() * y).real
    return torch.sign(z)


# ============================================================
# MMSE BASELINE (fixed k pilots)
# ============================================================
def mmse_eval(y, x, k, sigma2):
    B, T = y.shape

    y_p = y[:, :k]
    x_p = x[:, :k]

    h_hat = mmse_channel_estimate(y_p, x_p, sigma2)

    errors = []
    for t in range(k, T):
        x_hat = mmse_detect(y[:, t], h_hat)
        errors.append((x_hat != x[:, t]).float())

    return torch.stack(errors).mean().item()


# ============================================================
# MMSE-DF (Decision Feedback version)
# ============================================================
def mmse_df_eval(y, x, k, sigma2):
    B, T = y.shape

    x_feedback = x.clone()

    errors_per_t = []

    for t in range(k, T):

        # use current feedback as "augmented pilots"
        y_p = y[:, :t]
        x_p = x_feedback[:, :t]

        h_hat = mmse_channel_estimate(y_p, x_p, sigma2)

        x_hat = mmse_detect(y[:, t], h_hat)

        errors_per_t.append((x_hat != x[:, t]).float())

        # update feedback
        x_feedback[:, t] = x_hat

    return torch.stack(errors_per_t).mean().item()


# ============================================================
# FIGURE 4 STYLE EVALUATION
# SER vs context length (k → T-1)
# ============================================================
def run_fig4(snr_db=10, T=31):
    batch_size = 512

    y, x, _ = generate_batch(batch_size, T, snr_db)

    snr = 10 ** (snr_db / 10)
    sigma2 = 1 / snr

    ks = list(range(1, 10))  # pilot sweep like Fig. 4

    mmse_ser = []
    mmse_df_ser = []

    for k in ks:
        mmse_ser.append(mmse_eval(y, x, k, sigma2))
        mmse_df_ser.append(mmse_df_eval(y, x, k, sigma2))

        print(f"k={k:2d} | MMSE={mmse_ser[-1]:.4f} | MMSE-DF={mmse_df_ser[-1]:.4f}")

    return ks, mmse_ser, mmse_df_ser


# ============================================================
# RUN
# ============================================================
ks, mmse_ser, mmse_df_ser = run_fig4(snr_db=10, T=31)


# ============================================================
# PLOT (optional)
# ============================================================
import matplotlib.pyplot as plt

plt.plot(ks, mmse_ser, marker='o', label="MMSE")
plt.plot(ks, mmse_df_ser, marker='o', label="MMSE-DF")
plt.xlabel("Number of Pilots (k)")
plt.ylabel("SER")
plt.title("MMSE vs MMSE-DF (Fig. 4 style)")
plt.grid()
plt.legend()
plt.show()