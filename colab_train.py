"""
colab_train.py — Training DQN di Google Colab (tanpa pygame).

Cara pakai di Colab:
--------------------
    # 1. Upload semua file proyek ke Colab atau mount Google Drive
    # 2. Install dependensi:
    !pip install torch numpy

    # 3. Jalankan sel ini:
    %run colab_train.py

    # ATAU import dan kustomisasi:
    from colab_train import train
    train(map_path="maps/my_map.json", total_episodes=1000, save_path="model.pth")

Tidak ada dependensi pygame sama sekali.
"""

import os
import sys
import time
import math

# ── Pastikan root proyek ada di sys.path ────────────────────────────
# (saat dijalankan dari subfolder atau Colab)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment import RobotEnvironment
from dqn_agent   import DQNAgent
from core        import load_map, build_map_dict, TrainingConfig, run_training


# ════════════════════════════════════════════════════════════════════
#  FUNGSI UTAMA — bisa dipanggil langsung atau diimpor
# ════════════════════════════════════════════════════════════════════
def train(
    map_path:        str   = "",
    total_episodes:  int   = 500,
    max_steps:       int   = 300,
    gamma:           float = 0.99,
    lr:              float = 1e-3,
    eps_start:       float = 1.0,
    eps_end:         float = 0.05,
    eps_decay:       float = 0.995,
    log_every:       int   = 10,
    save_path:       str   = "model.pth",
    resume_from:     str   = "",
    checkpoint_every:int   = 10,    # simpan model setiap N episode (timpa file lama)
) -> DQNAgent:
    """
    Latih agen DQN.

    Parameters
    ----------
    map_path    : path ke file .json peta (kosong = pakai arena default 10×10)
    resume_from : path ke file .pth untuk melanjutkan training sebelumnya
    Semua parameter lain adalah hyperparameter training standar.

    Returns
    -------
    DQNAgent yang sudah dilatih
    """
    # ── Muat atau buat environment ─────────────────────────────────
    if map_path and os.path.isfile(map_path):
        print(f"📂 Memuat map: {map_path}")
        env = load_map(map_path)
    else:
        if map_path:
            print(f"⚠️  File map '{map_path}' tidak ditemukan, pakai default.")
        print("🗺  Menggunakan arena default 10×10 tanpa obstacle.")
        env = RobotEnvironment()

    env.reset()
    print(f"   Arena  : {env.arena_w:.1f} × {env.arena_h:.1f} m")
    print(f"   Lidar  : {env.lidar_num_rays} sinar, jangkauan {env.lidar_max_range:.1f} m")
    print(f"   Robot  : radius={env.robot.radius:.2f}m  "
          f"step={env.robot.step_linear*100:.1f}cm  "
          f"rot={math.degrees(env.robot.step_angular):.1f}°")
    print(f"   Obstacle: {len(env.obstacles)} buah")
    print()

    # ── Muat agent lama (opsional) ─────────────────────────────────
    agent = None
    if resume_from and os.path.isfile(resume_from):
        print(f"🔁 Melanjutkan training dari: {resume_from}")
        agent = DQNAgent.load(resume_from)
        # Override epsilon jika resume
        agent.eps     = eps_start
        agent.eps_end = eps_end
        print(f"   Sudah {agent.episode_count} episode sebelumnya.")
        print()

    # ── Konfigurasi ─────────────────────────────────────────────────
    cfg = TrainingConfig(
        total_episodes  = total_episodes,
        max_steps       = max_steps,
        gamma           = gamma,
        lr              = lr,
        eps_start       = eps_start,
        eps_end         = eps_end,
        eps_decay       = eps_decay,
        log_every       = log_every,
        auto_save_path  = save_path,
        checkpoint_every= checkpoint_every,
    )

    print(f"🚀 Mulai training: {total_episodes} episode, max {max_steps} langkah/episode")
    print(f"   γ={gamma}  lr={lr}  ε {eps_start}→{eps_end}  decay={eps_decay}")
    print(f"   Model akan disimpan ke: {save_path}")
    print("─" * 64)

    t0    = time.time()
    agent = run_training(env, cfg, agent=agent)
    elapsed = time.time() - t0

    print("─" * 64)
    print(f"✅ Training selesai dalam {elapsed/60:.1f} menit")
    print(f"   Episode      : {total_episodes}")
    print(f"   Goals reached: {sum(1 for r in agent.ep_rewards if r >= 90)}")
    print(f"   Avg reward   : {agent.avg_reward_100:.2f} (100 ep terakhir)")
    print(f"   Model disimpan: {save_path}")

    return agent


# ════════════════════════════════════════════════════════════════════
#  KONFIGURASI — edit bagian ini sebelum menjalankan
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train(
        map_path        = "",          # "" = default; atau "maps/my_map.json"
        total_episodes  = 500,
        max_steps       = 300,
        gamma           = 0.99,
        lr              = 1e-3,
        eps_start       = 1.0,
        eps_end         = 0.05,
        eps_decay       = 0.995,
        log_every       = 10,
        save_path       = "model.pth",
        resume_from     = "",          # "" = mulai dari awal
        checkpoint_every= 10,          # simpan tiap 10 episode, timpa file lama
    )
