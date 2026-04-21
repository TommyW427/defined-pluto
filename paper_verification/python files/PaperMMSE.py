# MMSE + MMSE-DF BASELINE (FIGURE 4 STYLE)

import torch
import math
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# SYSTEM PARAMETERS
T = 31
MOD = "BPSK"
BATCH_SIZE = 512
MC_TRIALS = 200  # Monte Carlo averaging (important for Fig. 4)

# CONSTELLATION
def get_constellation():
    return torch.tensor([-1.0, 1.0], device=device)

# CHANNEL + SIGNAL MODEL
def generate_frame(batch_size, T, snr_db):
    x = torch.randint(0, 2, (batch_size, T), device=device).float()
    x = 2 * x - 1  # BPSK

    # Rayleigh block fading channel
    h = (torch.randn(batch_size, 1, device=device) +
         1j * torch.randn(batch_size, 1, device=device)) / math.sqrt(2)

    snr = 10 ** (snr_db / 10)
    sigma = math.sqrt(1 / snr)

    noise = sigma / math.sqrt(2) * (
        torch.randn(batch_size, T, device=device) +
        1j * torch.randn(batch_size, T, device=device)
    )

    y = h * x + noise
    return y, x, h, sigma**2

# MMSE CHANNEL ESTIMATION (SISO)
def mmse_channel_estimate(y_p, x_p, sigma2):
    num = torch.sum(y_p * x_p.conj(), dim=1, keepdim=True)
    den = torch.sum((x_p * x_p.conj()).real, dim=1, keepdim=True) + sigma2
    return num / den

# MMSE DETECTOR (coherent BPSK slicing)
def mmse_detect(y, h_hat):
    y = y.unsqueeze(-1) if y.dim() == 1 else y
    metric = (h_hat.conj() * y).squeeze(-1).real
    return torch.sign(metric)

# MMSE (NO FEEDBACK)
def mmse_eval(y, x, sigma2, k):
    """
    Pilot-based MMSE:
    - estimate channel using first k pilots
    - detect remaining symbols independently
    """
    y_p = y[:, :k]
    x_p = x[:, :k]

    h_hat = mmse_channel_estimate(y_p, x_p, sigma2)

    errors = []

    for t in range(k, T):
        x_hat = mmse_detect(y[:, t], h_hat)
        errors.append((x_hat != x[:, t]).float())

    return torch.stack(errors).mean().item()

# MMSE-DF (Decision Feedback)
def mmse_df_eval(y, x, sigma2, k):
    """
    Decision Feedback MMSE:
    - start with k pilots
    - recursively update channel estimate using detected symbols
    """

    x_feedback = x.clone()

    errors = []

    for t in range(k, T):

        # use all past (pilots + decisions)
        y_p = y[:, :t]
        x_p = x_feedback[:, :t]

        h_hat = mmse_channel_estimate(y_p, x_p, sigma2)

        x_hat = mmse_detect(y[:, t], h_hat)

        errors.append((x_hat != x[:, t]).float())

        # update feedback
        x_feedback[:, t] = x_hat.view(-1)

    return torch.stack(errors).mean().item()

def mmse_df_eval_fixed_feedback_window(y, x, sigma2, k_initial, L_feedback_window_size):
    """
    Decision Feedback MMSE with a fixed-size sliding window for feedback symbols:
    - Starts with k_initial pilots.
    - Recursively updates channel estimate using k_initial pilots + L_feedback_window_size
      most recent detected symbols (or fewer if not enough are available yet).
    Returns the average SER over all detected symbols.
    """
    x_feedback = x.clone()
    errors_per_t = []

    for t in range(k_initial, T):
        # Always include the initial k_initial pilots
        y_p_est = y[:, :k_initial]
        x_p_est = x_feedback[:, :k_initial]

        # Add a fixed number of most recent feedback symbols to the estimation
        if L_feedback_window_size > 0 and t > k_initial:
            # Determine the range of feedback symbols to include.
            # These are symbols from index k_initial up to t-1.
            # We want to take the last `L_feedback_window_size` of these.
            feedback_start_idx_in_frame = max(k_initial, t - L_feedback_window_size)

            y_f_est = y[:, feedback_start_idx_in_frame:t]
            x_f_est = x_feedback[:, feedback_start_idx_in_frame:t]

            # Concatenate pilots and feedback symbols for channel estimation
            y_p_est = torch.cat((y_p_est, y_f_est), dim=1)
            x_p_est = torch.cat((x_p_est, x_f_est), dim=1)

        h_hat = mmse_channel_estimate(y_p_est, x_p_est, sigma2)
        x_hat = mmse_detect(y[:, t], h_hat)
        errors_per_t.append((x_hat != x[:, t]).float())
        x_feedback[:, t] = x_hat.view(-1)

    return torch.stack(errors_per_t).mean().item()

# MONTE CARLO EVALUATION (FIXED K MMSE, SLIDING FEEDBACK WINDOW DF)
def evaluate_fixed_pilot_sliding_feedback(snr_db, k_initial_pilot, L_feedback_window_sizes):
    """
    Evaluates fixed MMSE (k_initial_pilot) and MMSE-DF with sweeping fixed-size feedback windows.
    """
    # --- MMSE with fixed k_initial_pilot ---
    mmse_fixed_k_errors = []
    for _ in range(MC_TRIALS):
        y, x, _, sigma2 = generate_frame(BATCH_SIZE, T, snr_db)
        mmse_fixed_k_errors.append(mmse_eval(y, x, sigma2, k_initial_pilot))
    fixed_mmse_ser = sum(mmse_fixed_k_errors) / MC_TRIALS

    # --- MMSE-DF sweeping fixed-size feedback window L_feedback_window_size ---
    df_sliding_window_ser_curve = []
    print(f"\nFixed MMSE (initial k={k_initial_pilot}): SER = {fixed_mmse_ser:.4f}")
    print(f"MMSE-DF (initial k={k_initial_pilot} + sweeping fixed feedback window):")
    for L_window_size in L_feedback_window_sizes:
        df_acc = 0.0
        for _ in range(MC_TRIALS):
            y, x, _, sigma2 = generate_frame(BATCH_SIZE, T, snr_db)
            df_acc += mmse_df_eval_fixed_feedback_window(y, x, sigma2, k_initial_pilot, L_window_size)
        
        current_df_ser = df_acc / MC_TRIALS
        df_sliding_window_ser_curve.append(current_df_ser)
        print(f"  Feedback window size L={L_window_size:2d}: SER = {current_df_ser:.4f}")

    return fixed_mmse_ser, df_sliding_window_ser_curve, L_feedback_window_sizes

def evaluate_snr(snr_db, k_values):
    mmse_curve = []
    df_curve = []

    for k in k_values:

        mmse_acc = 0.0
        df_acc = 0.0

        for _ in range(MC_TRIALS):
            y, x, _, sigma2 = generate_frame(BATCH_SIZE, T, snr_db)

            mmse_acc += mmse_eval(y, x, sigma2, k)
            df_acc += mmse_df_eval(y, x, sigma2, k)

        mmse_curve.append(mmse_acc / MC_TRIALS)
        df_curve.append(df_acc / MC_TRIALS)

        print(f"k={k:2d} | MMSE={mmse_curve[-1]:.4f} | MMSE-DF={df_curve[-1]:.4f}")

    return mmse_curve, df_curve

# RUN EXPERIMENT (FIXED K MMSE, SLIDING FEEDBACK WINDOW DF)
fixed_pilot_k = 1  # Fixed initial pilot for both MMSE and MMSE-DF
snr_db = 5

# Define the range of feedback window sizes to sweep (0 means only pilots are used, up to T-k_initial_pilot-1)
L_feedback_window_sizes = list(range(0, T - fixed_pilot_k))

fixed_mmse_ser, df_sliding_window_ser_curve, feedback_window_lengths_for_plot = \
    evaluate_fixed_pilot_sliding_feedback(snr_db, fixed_pilot_k, L_feedback_window_sizes)

# PLOT (FIXED K MMSE, SLIDING FEEDBACK WINDOW DF)
plt.figure(figsize=(10, 6))

# Plot fixed MMSE as a constant line
plt.plot(feedback_window_lengths_for_plot, [fixed_mmse_ser] * len(feedback_window_lengths_for_plot),
         marker='x', linestyle='--', color='gray', label=f"MMSE (k={fixed_pilot_k} fixed)")

# Plot MMSE-DF sweeping fixed feedback window size
plt.plot(feedback_window_lengths_for_plot, df_sliding_window_ser_curve,
         marker='o', label=f"MMSE-DF (k={fixed_pilot_k} + sliding feedback window)")

plt.xlabel("Decision Feedback Window Size (L)")
plt.ylabel("Symbol Error Rate (SER)")
plt.title(f"MMSE vs MMSE-DF (SNR = {snr_db} dB, Initial Pilot k={fixed_pilot_k})")
plt.grid(True)
plt.legend()
plt.show()