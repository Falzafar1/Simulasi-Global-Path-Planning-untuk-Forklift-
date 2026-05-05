"""
plot_results.py — Script untuk membuat grafik paper dari hasil training DQN.

Cara pakai:
    python plot_results.py              # otomatis cari model.pth
    python plot_results.py model.pth    # pakai file tertentu

Output (disimpan ke folder 'figures/'):
    fig1_reward_curve.png   → Kurva reward per episode (Fig. 1 paper)
    fig2_loss_curve.png     → Kurva training loss (Fig. 2 paper)
    fig3_epsilon_decay.png  → Kurva epsilon decay (Fig. 3 paper)
    fig4_goal_rate.png      → Persentase goal tiap 50 episode (Fig. 4 paper)
    summary.txt             → Statistik ringkasan untuk di-copy ke paper
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # tidak perlu display (aman di Colab & headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


# ══════════════════════════════════════════════════════════════
#  KONFIGURASI TAMPILAN — ubah sesuai selera
# ══════════════════════════════════════════════════════════════
STYLE = {
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f9f9f9",
    "axes.grid":          True,
    "grid.color":         "#dddddd",
    "grid.linestyle":     "--",
    "grid.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "lines.linewidth":    1.8,
}

# Warna utama
COLOR_RAW      = "#aecbf0"   # biru muda — reward mentah
COLOR_SMOOTH   = "#1a6fc4"   # biru tua  — moving average
COLOR_LOSS     = "#e07b54"   # oranye    — loss
COLOR_EPSILON  = "#5b9e6b"   # hijau     — epsilon
COLOR_GOAL     = "#9b5de5"   # ungu      — goal rate

OUTPUT_DIR = "figures"
WINDOW     = 50   # window moving average (ubah ke 100 untuk 1000 episode)


# ══════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════
def moving_average(data: list, window: int) -> np.ndarray:
    """Moving average sederhana."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def load_agent_data(model_path: str) -> dict:
    """
    Muat data training dari file .pth.
    Mengembalikan dict berisi ep_rewards, losses, episode_count, dsb.
    """
    print(f"[INFO] Memuat model dari: {model_path}")
    data = torch.load(model_path, map_location="cpu")

    ep_rewards    = data.get("ep_rewards", [])
    losses        = data.get("losses", [])
    episode_count = data.get("episode_count", len(ep_rewards))
    eps_final     = data.get("eps", 0.05)

    print(f"   Episode tersimpan : {episode_count}")
    print(f"   Panjang ep_rewards: {len(ep_rewards)}")
    print(f"   Panjang losses    : {len(losses)}")
    print(f"   Epsilon akhir     : {eps_final:.4f}")

    return {
        "ep_rewards":    ep_rewards,
        "losses":        losses,
        "episode_count": episode_count,
        "eps_final":     eps_final,
    }


def reconstruct_epsilon(n_episodes: int, eps_start=1.0,
                         eps_end=0.05, eps_decay=0.995) -> list:
    """Rekonstruksi kurva epsilon dari hyperparameter."""
    eps = eps_start
    result = []
    for _ in range(n_episodes):
        result.append(eps)
        eps = max(eps_end, eps * eps_decay)
    return result


def goal_rate_per_window(ep_rewards: list, window: int = 50,
                          goal_threshold: float = 90.0) -> tuple:
    """
    Hitung persentase episode yang mencapai goal tiap `window` episode.
    Returns: (x_positions, goal_rates)
    """
    x, y = [], []
    for i in range(0, len(ep_rewards), window):
        chunk = ep_rewards[i: i + window]
        rate  = sum(1 for r in chunk if r >= goal_threshold) / len(chunk) * 100
        x.append(i + window // 2)
        y.append(rate)
    return x, y


# ══════════════════════════════════════════════════════════════
#  GAMBAR 1 — Kurva Reward
# ══════════════════════════════════════════════════════════════
def plot_reward_curve(ep_rewards: list, out_path: str):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 4))

    episodes = np.arange(1, len(ep_rewards) + 1)
    smoothed  = moving_average(ep_rewards, WINDOW)
    sm_ep     = np.arange(WINDOW, len(ep_rewards) + 1)

    ax.plot(episodes, ep_rewards,
            color=COLOR_RAW, alpha=0.4, linewidth=0.8, label="Reward per Episode")
    ax.plot(sm_ep, smoothed,
            color=COLOR_SMOOTH, linewidth=2.2,
            label=f"Moving Average ({WINDOW} ep)")

    # Garis avg 100 ep terakhir
    avg_last = float(np.mean(ep_rewards[-100:])) if len(ep_rewards) >= 100 else float(np.mean(ep_rewards))
    ax.axhline(avg_last, color=COLOR_SMOOTH, linestyle=":", linewidth=1.5, alpha=0.7,
               label=f"Avg 100 ep terakhir = {avg_last:.1f}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Kurva Reward Training DQN")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Disimpan: {out_path}")


# ══════════════════════════════════════════════════════════════
#  GAMBAR 2 — Kurva Loss
# ══════════════════════════════════════════════════════════════
def plot_loss_curve(losses: list, out_path: str):
    if not losses:
        print("   [SKIP] Data loss kosong, Gambar 2 dilewati.")
        return

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 4))

    steps    = np.arange(1, len(losses) + 1)
    win_loss = max(50, len(losses) // 40)   # window adaptif
    smoothed = moving_average(losses, win_loss)
    sm_steps = np.arange(win_loss, len(losses) + 1)

    ax.plot(steps, losses,
            color="#f0c9b8", alpha=0.35, linewidth=0.6, label="Loss per Step")
    ax.plot(sm_steps, smoothed,
            color=COLOR_LOSS, linewidth=2.2,
            label=f"Moving Average ({win_loss} steps)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Kurva Training Loss DQN")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Disimpan: {out_path}")


# ══════════════════════════════════════════════════════════════
#  GAMBAR 3 — Kurva Epsilon Decay
# ══════════════════════════════════════════════════════════════
def plot_epsilon_decay(n_episodes: int, out_path: str,
                        eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 3.5))

    eps_curve = reconstruct_epsilon(n_episodes, eps_start, eps_end, eps_decay)
    episodes  = np.arange(1, n_episodes + 1)

    ax.plot(episodes, eps_curve, color=COLOR_EPSILON, linewidth=2.2)
    ax.fill_between(episodes, eps_curve, eps_end,
                    color=COLOR_EPSILON, alpha=0.12,
                    label="Area eksplorasi")
    ax.axhline(eps_end, color="#888888", linestyle="--", linewidth=1.2,
               label=f"ε minimum = {eps_end}")

    # Tandai titik di mana ε < 0.1
    cross_ep = next((i for i, e in enumerate(eps_curve) if e < 0.1), None)
    if cross_ep is not None:
        ax.axvline(cross_ep + 1, color="#cc4444", linestyle=":", linewidth=1.2, alpha=0.8,
                   label=f"ε < 0.10 pada ep {cross_ep+1}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Nilai ε (Epsilon)")
    ax.set_title("Kurva Epsilon Decay (Eksplorasi → Eksploitasi)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Disimpan: {out_path}")


# ══════════════════════════════════════════════════════════════
#  GAMBAR 4 — Goal Success Rate per Window Episode
# ══════════════════════════════════════════════════════════════
def plot_goal_rate(ep_rewards: list, out_path: str, window: int = 50):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 4))

    x, y = goal_rate_per_window(ep_rewards, window=window)

    ax.bar(x, y, width=window * 0.8,
           color=COLOR_GOAL, alpha=0.75, edgecolor="white", linewidth=0.5,
           label=f"Goal rate (per {window} ep)")

    # Garis tren
    if len(x) >= 3:
        z    = np.polyfit(x, y, 1)
        p    = np.poly1d(z)
        x_line = np.linspace(x[0], x[-1], 200)
        ax.plot(x_line, p(x_line),
                color="#4a0072", linewidth=1.8, linestyle="--", alpha=0.7,
                label="Tren linear")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Goal Success Rate (%)")
    ax.set_title(f"Persentase Goal Tercapai per {window} Episode")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Disimpan: {out_path}")


# ══════════════════════════════════════════════════════════════
#  RINGKASAN STATISTIK — untuk disalin ke paper
# ══════════════════════════════════════════════════════════════
def save_summary(data: dict, out_path: str):
    ep_rewards = data["ep_rewards"]
    losses     = data["losses"]
    n          = len(ep_rewards)

    goal_thresh = 90.0
    total_goals = sum(1 for r in ep_rewards if r >= goal_thresh)
    goal_pct    = total_goals / n * 100 if n > 0 else 0.0

    avg_100  = float(np.mean(ep_rewards[-100:])) if n >= 100 else float(np.mean(ep_rewards))
    max_r    = float(np.max(ep_rewards)) if ep_rewards else 0.0
    min_r    = float(np.min(ep_rewards)) if ep_rewards else 0.0
    std_100  = float(np.std(ep_rewards[-100:])) if n >= 100 else 0.0

    avg_loss = float(np.mean(losses[-1000:])) if losses else 0.0
    fin_loss = float(np.mean(losses[-100:]))  if len(losses) >= 100 else avg_loss

    lines = [
        "=" * 55,
        "   RINGKASAN HASIL TRAINING DQN — untuk paper",
        "=" * 55,
        f"  Total episode dilatih      : {n}",
        f"  Total goals tercapai       : {total_goals} / {n} ({goal_pct:.1f}%)",
        "",
        f"  Average reward (100 ep terakhir)  : {avg_100:.2f}",
        f"  Std reward     (100 ep terakhir)  : {std_100:.2f}",
        f"  Max reward episode                : {max_r:.2f}",
        f"  Min reward episode                : {min_r:.2f}",
        "",
        f"  Avg loss (1000 steps terakhir)    : {avg_loss:.6f}",
        f"  Avg loss (100 steps terakhir)     : {fin_loss:.6f}",
        f"  Epsilon akhir                     : {data['eps_final']:.4f}",
        "=" * 55,
        "",
        "  Kalimat siap pakai untuk paper:",
        f"  \"Agen DQN berhasil mencapai goal pada {goal_pct:.1f}% dari",
        f"   {n} episode training, dengan rata-rata reward {avg_100:.2f}",
        f"   (± {std_100:.2f}) pada 100 episode terakhir.\"",
        "=" * 55,
    ]
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n   [OK] Ringkasan disimpan: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # Tentukan path model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Cari model.pth di direktori yang sama dengan script ini
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, "model.pth")
        if not os.path.isfile(model_path):
            print("[ERROR] Tidak menemukan model.pth. Jalankan: python plot_results.py <path_model.pth>")
            sys.exit(1)

    # Buat folder output
    here    = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # Muat data
    data = load_agent_data(model_path)
    ep_rewards    = data["ep_rewards"]
    losses        = data["losses"]
    episode_count = data["episode_count"]

    if not ep_rewards:
        print("[ERROR] Data ep_rewards kosong. Pastikan model sudah dilatih.")
        sys.exit(1)

    print(f"\n[INFO] Membuat grafik ke folder '{OUTPUT_DIR}/'...\n")

    # ── Plot semua gambar
    plot_reward_curve(
        ep_rewards,
        os.path.join(out_dir, "fig1_reward_curve.png")
    )
    plot_loss_curve(
        losses,
        os.path.join(out_dir, "fig2_loss_curve.png")
    )
    plot_epsilon_decay(
        episode_count,
        os.path.join(out_dir, "fig3_epsilon_decay.png")
    )
    plot_goal_rate(
        ep_rewards,
        os.path.join(out_dir, "fig4_goal_rate.png"),
        window=max(10, episode_count // 10)  # adaptif: ~10 bar
    )
    save_summary(
        data,
        os.path.join(out_dir, "summary.txt")
    )

    print(f"\n[DONE] Semua file tersimpan di folder: {out_dir}/")
    print("   Masukkan gambar-gambar ini ke paper kamu sebagai Fig. 1-4.")


if __name__ == "__main__":
    main()
