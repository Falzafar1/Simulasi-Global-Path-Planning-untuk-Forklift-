"""
colab_inference.py — Inference & evaluasi DQN di Google Colab (tanpa pygame).

Cara pakai di Colab:
--------------------
    %run colab_inference.py

    # ATAU import dan kustomisasi:
    from colab_inference import demo, benchmark
    demo("model.pth", map_path="maps/my_map.json", n_episodes=5)
    benchmark("model.pth", map_path="maps/my_map.json", n_episodes=100)

Tidak ada dependensi pygame sama sekali.
Visualisasi menggunakan matplotlib (text atau plot ASCII).
"""

import os
import sys
import math

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment import RobotEnvironment
from dqn_agent   import DQNAgent
from core        import load_map, run_episode, run_benchmark, random_goal


# ════════════════════════════════════════════════════════════════════
#  VISUALISASI ASCII (opsional, tanpa matplotlib)
# ════════════════════════════════════════════════════════════════════
def _ascii_map(env: RobotEnvironment,
               trajectory: list[tuple[float, float]],
               goal_x: float, goal_y: float,
               cols: int = 40, rows: int = 20) -> str:
    """
    Render arena + trajectory + goal sebagai ASCII art.
    Berguna di Colab tanpa display.
    """
    grid = [["·"] * cols for _ in range(rows)]

    def w2g(wx, wy):
        col = int(wx / env.arena_w * (cols - 1))
        row = rows - 1 - int(wy / env.arena_h * (rows - 1))
        return max(0, min(cols - 1, col)), max(0, min(rows - 1, row))

    # Obstacle
    for obs in env.obstacles:
        for r in range(rows):
            for c in range(cols):
                wx = c / (cols - 1) * env.arena_w
                wy = (rows - 1 - r) / (rows - 1) * env.arena_h
                if obs.x <= wx <= obs.x + obs.w and obs.y <= wy <= obs.y + obs.h:
                    grid[r][c] = "█"

    # Trajectory
    for (tx, ty) in trajectory[1:-1]:
        c, r = w2g(tx, ty)
        if grid[r][c] == "·":
            grid[r][c] = "·"

    # Start
    if trajectory:
        c, r = w2g(*trajectory[0])
        grid[r][c] = "S"

    # Goal
    gc, gr = w2g(goal_x, goal_y)
    grid[gr][gc] = "G"

    # Posisi akhir robot
    if trajectory:
        c, r = w2g(*trajectory[-1])
        grid[r][c] = "R"

    border  = "+" + "─" * cols + "+"
    lines   = [border]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append(border)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  DEMO — beberapa episode dengan output verbose
# ════════════════════════════════════════════════════════════════════
def demo(
    model_path: str,
    map_path:   str  = "",
    n_episodes: int  = 5,
    max_steps:  int  = 500,
    show_ascii: bool = True,
) -> None:
    """
    Jalankan beberapa episode inference dengan output detail per episode.
    """
    # Muat model
    if not os.path.isfile(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        return
    print(f"📦 Memuat model: {model_path}")
    agent = DQNAgent.load(model_path)
    print(f"   State dim    : {agent.state_dim}")
    print(f"   Episode count: {agent.episode_count}")
    print()

    # Muat env
    if map_path and os.path.isfile(map_path):
        print(f"📂 Memuat map: {map_path}")
        env = load_map(map_path)
    else:
        print("🗺  Pakai arena default 10×10")
        env = RobotEnvironment()
    env.reset()

    print(f"Arena {env.arena_w:.1f}×{env.arena_h:.1f}m  "
          f"Obstacles: {len(env.obstacles)}  "
          f"Lidar: {env.lidar_num_rays} sinar")
    print("═" * 60)

    for ep in range(1, n_episodes + 1):
        result = run_episode(env, agent, max_steps=max_steps)

        status = "✅ GOAL" if result["reached"] else \
                 "💥 COLLISION" if result["collision"] else "⏱ TIMEOUT"
        print(f"\nEpisode {ep}/{n_episodes}  {status}")
        print(f"  Langkah  : {result['steps']}")
        print(f"  Goal     : ({result['goal_x']:.2f}, {result['goal_y']:.2f})")
        print(f"  Start    : ({result['trajectory'][0][0]:.2f}, "
              f"{result['trajectory'][0][1]:.2f})")
        print(f"  End      : ({result['trajectory'][-1][0]:.2f}, "
              f"{result['trajectory'][-1][1]:.2f})")

        if show_ascii:
            print(_ascii_map(env, result["trajectory"],
                             result["goal_x"], result["goal_y"]))


# ════════════════════════════════════════════════════════════════════
#  BENCHMARK — statistik agregat banyak episode
# ════════════════════════════════════════════════════════════════════
def benchmark(
    model_path:  str  = "model.pth",
    map_path:    str  = "",
    n_episodes:  int  = 100,
    max_steps:   int  = 500,
    verbose:     bool = True,
) -> dict:
    """
    Evaluasi model di N episode dan tampilkan statistik.
    """
    if not os.path.isfile(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        return {}
    agent = DQNAgent.load(model_path)

    if map_path and os.path.isfile(map_path):
        env = load_map(map_path)
    else:
        env = RobotEnvironment()
    env.reset()

    print(f"📊 Benchmark: {n_episodes} episode  model={model_path}")
    stats = run_benchmark(env, agent, n_episodes=n_episodes,
                          max_steps=max_steps, verbose=verbose)
    print()
    print("═" * 40)
    print(f"  Success rate  : {stats['success_rate']:.1%}")
    print(f"  Collision rate: {stats['collision_rate']:.1%}")
    print(f"  Avg steps     : {stats['avg_steps']:.1f}")
    print(f"  Goals reached : {stats['successes']} / {stats['n_episodes']}")
    print("═" * 40)
    return stats


# ════════════════════════════════════════════════════════════════════
#  KONFIGURASI — edit bagian ini
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MODEL_PATH = "model.pth"
    MAP_PATH   = ""           # "" = default; atau "maps/my_map.json"

    # Demo 5 episode dengan ASCII map
    demo(MODEL_PATH, map_path=MAP_PATH, n_episodes=5, show_ascii=True)

    # Benchmark 100 episode
    benchmark(MODEL_PATH, map_path=MAP_PATH, n_episodes=100)
