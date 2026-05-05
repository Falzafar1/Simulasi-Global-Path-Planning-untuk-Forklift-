"""
core/training_loop.py — Loop training DQN murni, tanpa dependensi pygame.

Dapat diimpor di VSCode (pygame) maupun Google Colab (tanpa pygame).

Penggunaan minimal
------------------
    from core.training_loop import TrainingConfig, run_training
    from environment import RobotEnvironment

    env = RobotEnvironment()
    cfg = TrainingConfig(total_episodes=500, max_steps=300)
    agent = run_training(env, cfg)
    agent.save("model.pth")

Penggunaan dengan callback (untuk GUI atau Colab progress bar)
--------------------------------------------------------------
    def on_episode(info: dict):
        print(info["episode"], info["ep_reward"])

    agent = run_training(env, cfg, on_episode_end=on_episode)

Penggunaan dari thread (VSCode GUI)
------------------------------------
    import threading
    stop = threading.Event()
    agent = run_training(env, cfg, stop_event=stop)
"""

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from environment import RobotEnvironment
from dqn_agent import (
    DQNAgent, ACTIONS,
    make_state, compute_reward,
    DEFAULT_GAMMA, DEFAULT_LR,
    DEFAULT_EPS_START, DEFAULT_EPS_END, DEFAULT_EPS_DECAY,
    DEFAULT_TARGET_UPDATE,
)
from core.lidar import LidarSensor

GOAL_RADIUS = 0.5   # meter


# ══════════════════════════════════════════════
#  KONFIGURASI TRAINING
# ══════════════════════════════════════════════
@dataclass
class TrainingConfig:
    total_episodes:   int   = 500
    max_steps:        int   = 300
    gamma:            float = DEFAULT_GAMMA
    lr:               float = DEFAULT_LR
    eps_start:        float = DEFAULT_EPS_START
    eps_end:          float = DEFAULT_EPS_END
    eps_decay:        float = DEFAULT_EPS_DECAY
    target_update:    int   = DEFAULT_TARGET_UPDATE
    log_every:        int   = 10     # log setiap N episode
    auto_save_path:   str   = ""     # "" = tidak auto-save
    checkpoint_every: int   = 10     # simpan model setiap N episode (timpa file lama)


# ══════════════════════════════════════════════
#  HELPER INTERNAL
# ══════════════════════════════════════════════
def _random_goal(env: RobotEnvironment) -> tuple[float, float]:
    """Cari posisi goal acak yang bebas obstacle."""
    margin = max(env.robot.radius + 0.5, 1.0)
    for _ in range(200):
        gx = random.uniform(margin, env.arena_w - margin)
        gy = random.uniform(margin, env.arena_h - margin)
        if all(not o.collides_with_circle(gx, gy, GOAL_RADIUS + 0.1)
               for o in env.obstacles):
            return gx, gy
    return env.arena_w / 2, env.arena_h / 2


# ══════════════════════════════════════════════
#  FUNGSI UTAMA
# ══════════════════════════════════════════════
def run_training(
    env:             RobotEnvironment,
    cfg:             TrainingConfig = None,
    agent:           DQNAgent       = None,
    stop_event:      threading.Event | None = None,
    on_episode_end:  Callable[[dict], None] | None = None,
    on_step:         Callable[[dict], None] | None = None,
) -> DQNAgent:
    """
    Jalankan loop training DQN.

    Parameters
    ----------
    env            : lingkungan yang digunakan (akan di-snapshot per episode)
    cfg            : hyperparameter training
    agent          : DQNAgent yang sudah ada (opsional; dibuat baru jika None)
    stop_event     : threading.Event untuk menghentikan dari luar (opsional)
    on_episode_end : callback(info_dict) dipanggil tiap akhir episode
    on_step        : callback(step_dict) dipanggil tiap langkah (hati-hati: lambat)

    Returns
    -------
    DQNAgent yang telah dilatih
    """
    if cfg is None:
        cfg = TrainingConfig()

    lidar = LidarSensor(env.lidar_num_rays, env.lidar_max_range)
    state_dim = lidar.num_rays + 3  # lidar + dist_norm + sin(ang) + cos(ang)

    if agent is None:
        agent = DQNAgent(
            state_dim    = state_dim,
            gamma        = cfg.gamma,
            lr           = cfg.lr,
            eps_start    = cfg.eps_start,
            eps_end      = cfg.eps_end,
            eps_decay    = cfg.eps_decay,
            target_update= cfg.target_update,
        )

    env_dict      = env.to_dict()   # snapshot supaya aman lintas thread
    goals_reached = 0
    total_steps   = 0

    for ep in range(1, cfg.total_episodes + 1):
        if stop_event and stop_event.is_set():
            break

        # ── Setup episode ──────────────────────────
        local_env = RobotEnvironment.from_dict(env_dict)
        local_env.reset()
        local_lidar = LidarSensor(local_env.lidar_num_rays, local_env.lidar_max_range)

        gx, gy    = _random_goal(local_env)
        prev_dist = math.hypot(gx - local_env.robot.x, gy - local_env.robot.y)

        ld = local_lidar.scan(
            local_env.robot.x, local_env.robot.y, local_env.robot.theta,
            local_env.obstacles, local_env.arena_w, local_env.arena_h)
        state = make_state(ld, local_lidar.max_range,
                           local_env.robot.x, local_env.robot.y,
                           local_env.robot.theta, gx, gy,
                           local_env.arena_w, local_env.arena_h)

        ep_reward    = 0.0
        reached      = False
        stuck_steps  = 0                         # hitung langkah tanpa gerak
        last_pos     = (local_env.robot.x, local_env.robot.y)

        # ── Loop langkah ──────────────────────────
        for step in range(cfg.max_steps):
            if stop_event and stop_event.is_set():
                break

            action = agent.select_action(state)
            move   = ACTIONS[action]

            _, _, done, _ = local_env.step_manual(move)

            ld2 = local_lidar.scan(
                local_env.robot.x, local_env.robot.y, local_env.robot.theta,
                local_env.obstacles, local_env.arena_w, local_env.arena_h)

            curr_dist = math.hypot(gx - local_env.robot.x, gy - local_env.robot.y)
            reached   = curr_dist < GOAL_RADIUS

            # Hitung sudut relatif robot ke goal (untuk alignment reward)
            angle_to_goal = math.atan2(gy - local_env.robot.y,
                                       gx - local_env.robot.x) - local_env.robot.theta
            reward    = compute_reward(prev_dist, curr_dist,
                                       hit=done, reached_goal=reached,
                                       action=action, angle_to_goal=angle_to_goal)

            # Anti-stuck: beri penalty tambahan jika robot tidak bergerak
            curr_pos = (local_env.robot.x, local_env.robot.y)
            move_dist = math.hypot(curr_pos[0] - last_pos[0],
                                   curr_pos[1] - last_pos[1])
            if move_dist < 1e-4:   # robot nyaris tidak bergerak
                stuck_steps += 1
                if stuck_steps > 10:
                    reward -= 0.5  # penalty ekstra agar agent tidak diam terus
            else:
                stuck_steps = 0
            last_pos = curr_pos

            ep_reward  += reward
            prev_dist   = curr_dist
            total_steps += 1

            next_state = make_state(ld2, local_lidar.max_range,
                                    local_env.robot.x, local_env.robot.y,
                                    local_env.robot.theta, gx, gy,
                                    local_env.arena_w, local_env.arena_h)
            terminal = done or reached
            agent.push(state, action, reward, next_state, terminal)
            agent.optimize()
            state = next_state
            ld    = ld2

            # Callback per-step (opsional, berguna untuk visualisasi di GUI)
            if on_step:
                on_step({
                    "episode":    ep,
                    "step":       step,
                    "robot_x":    local_env.robot.x,
                    "robot_y":    local_env.robot.y,
                    "robot_theta":local_env.robot.theta,
                    "lidar":      ld2,
                    "goal_x":     gx,
                    "goal_y":     gy,
                    "reward":     reward,
                    "done":       terminal,
                })

            if terminal:
                break

        # ── Akhir episode ──────────────────────────
        agent.end_episode(ep_reward)
        if reached:
            goals_reached += 1

        info = {
            "episode":       ep,
            "total_episodes":cfg.total_episodes,
            "steps_in_ep":   step + 1,
            "total_steps":   total_steps,
            "ep_reward":     ep_reward,
            "avg_r100":      agent.avg_reward_100,
            "avg_loss100":   agent.avg_loss_100,
            "epsilon":       agent.eps,
            "goals_reached": goals_reached,
            "reached":       reached,
        }

        if on_episode_end:
            on_episode_end(info)

        # Log default ke stdout jika tidak ada callback
        if on_episode_end is None and (ep % cfg.log_every == 0 or ep <= 5):
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Ep {ep:4d}/{cfg.total_episodes}  "
                  f"R={ep_reward:7.1f}  avgR={agent.avg_reward_100:7.1f}  "
                  f"ε={agent.eps:.3f}  loss={agent.avg_loss_100:.4f}  "
                  f"goals={goals_reached}")

        # ── Checkpoint periodik (timpa file lama) ──────────
        if (cfg.auto_save_path
                and cfg.checkpoint_every > 0
                and ep % cfg.checkpoint_every == 0):
            try:
                agent.save(cfg.auto_save_path)
                if on_episode_end is None:
                    print(f"  💾 Checkpoint disimpan → {cfg.auto_save_path}  (ep {ep})")
            except Exception as e:
                print(f"  ⚠️  Checkpoint gagal: {e}")

    # ── Final save di akhir training ───────────
    if cfg.auto_save_path:
        try:
            agent.save(cfg.auto_save_path)
            print(f"Model final disimpan → {cfg.auto_save_path}")
        except Exception as e:
            print(f"Auto-save gagal: {e}")

    return agent
