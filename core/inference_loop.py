"""
core/inference_loop.py — Loop inference DQN murni, tanpa dependensi pygame.

Dapat diimpor di VSCode (pygame) maupun Google Colab (tanpa pygame).

Penggunaan minimal
------------------
    from core.inference_loop import run_episode
    from dqn_agent import DQNAgent

    agent = DQNAgent.load("model.pth")
    result = run_episode(env, agent, goal_x=7.0, goal_y=7.0)
    print(result)

Penggunaan dengan callback (untuk animasi, logging, dsb.)
----------------------------------------------------------
    def on_step(info):
        print(info["step"], info["robot_x"], info["robot_y"])

    result = run_episode(env, agent, goal_x=7, goal_y=7, on_step=on_step)
"""

import math
import random
from typing import Callable

from environment import RobotEnvironment
from dqn_agent import DQNAgent, ACTIONS, make_state, compute_reward
from core.lidar import LidarSensor

GOAL_RADIUS = 0.5   # meter


def random_goal(env: RobotEnvironment) -> tuple[float, float]:
    """Pilih posisi goal acak yang bebas obstacle."""
    margin = max(env.robot.radius + 0.5, 1.0)
    for _ in range(200):
        gx = random.uniform(margin, env.arena_w - margin)
        gy = random.uniform(margin, env.arena_h - margin)
        if all(not o.collides_with_circle(gx, gy, GOAL_RADIUS + 0.1)
               for o in env.obstacles):
            return gx, gy
    return env.arena_w / 2, env.arena_h / 2


def run_episode(
    env:       RobotEnvironment,
    agent:     DQNAgent,
    goal_x:    float | None = None,
    goal_y:    float | None = None,
    max_steps: int   = 500,
    on_step:   Callable[[dict], None] | None = None,
) -> dict:
    """
    Jalankan satu episode inference (greedy, tanpa eksplorasi).

    Parameters
    ----------
    env       : lingkungan (akan di-reset sebelum episode)
    agent     : DQNAgent dengan model yang sudah dimuat
    goal_x/y  : posisi goal; jika None dipilih acak
    max_steps : batas langkah per episode
    on_step   : callback(step_dict) tiap langkah (opsional)

    Returns
    -------
    dict berisi ringkasan episode:
        steps, total_reward, reached, collision, trajectory
    """
    env.reset()
    lidar = LidarSensor(env.lidar_num_rays, env.lidar_max_range)

    if goal_x is None or goal_y is None:
        goal_x, goal_y = random_goal(env)

    ld = lidar.scan(env.robot.x, env.robot.y, env.robot.theta,
                    env.obstacles, env.arena_w, env.arena_h)
    # Fix: pass arena dimensions for accurate diagonal normalization
    state = make_state(ld, lidar.max_range,
                       env.robot.x, env.robot.y, env.robot.theta,
                       goal_x, goal_y,
                       env.arena_w, env.arena_h)

    total_reward = 0.0
    trajectory   = [(env.robot.x, env.robot.y)]
    reached      = False
    collision    = False
    # Fix: track prev_dist so compute_reward can be called each step
    prev_dist    = math.hypot(goal_x - env.robot.x, goal_y - env.robot.y)

    for step in range(max_steps):
        action = agent.select_action_greedy(state)
        move   = ACTIONS[action]

        _, _, done, info = env.step_manual(move)

        ld2 = lidar.scan(env.robot.x, env.robot.y, env.robot.theta,
                         env.obstacles, env.arena_w, env.arena_h)

        curr_dist = math.hypot(goal_x - env.robot.x, goal_y - env.robot.y)
        reached   = curr_dist < GOAL_RADIUS
        terminal  = done or reached
        collision = done and not reached

        # Hitung sudut relatif robot ke goal (untuk alignment reward, konsisten dgn training)
        angle_to_goal = math.atan2(goal_y - env.robot.y,
                                   goal_x - env.robot.x) - env.robot.theta

        # Fix: akumulasi reward per langkah (sebelumnya total_reward selalu 0)
        step_reward   = compute_reward(prev_dist, curr_dist,
                                       hit=done, reached_goal=reached,
                                       action=action, angle_to_goal=angle_to_goal)
        total_reward += step_reward
        prev_dist     = curr_dist


        trajectory.append((env.robot.x, env.robot.y))

        step_info = {
            "step":        step,
            "robot_x":     env.robot.x,
            "robot_y":     env.robot.y,
            "robot_theta": env.robot.theta,
            "lidar":       ld2,
            "goal_x":      goal_x,
            "goal_y":      goal_y,
            "action":      move,
            "reward":      step_reward,
            "dist_to_goal":curr_dist,
            "reached":     reached,
            "collision":   collision,
            "done":        terminal,
        }
        if on_step:
            on_step(step_info)

        next_state = make_state(ld2, lidar.max_range,
                                env.robot.x, env.robot.y, env.robot.theta,
                                goal_x, goal_y,
                                env.arena_w, env.arena_h)
        state = next_state
        ld    = ld2

        if terminal:
            break

    return {
        "steps":        step + 1,
        "total_reward": total_reward,
        "reached":      reached,
        "collision":    collision,
        "trajectory":   trajectory,
        "goal_x":       goal_x,
        "goal_y":       goal_y,
    }


def run_benchmark(
    env:        RobotEnvironment,
    agent:      DQNAgent,
    n_episodes: int = 100,
    max_steps:  int = 500,
    verbose:    bool = True,
) -> dict:
    """
    Jalankan N episode inference dan kembalikan statistik agregat.

    Berguna untuk evaluasi model di Colab tanpa GUI.

    Returns
    -------
    dict: success_rate, avg_steps, collision_rate, avg_reward, n_episodes
    """
    successes    = 0
    collisions   = 0
    steps_list   = []
    rewards_list = []

    for i in range(n_episodes):
        result = run_episode(env, agent, max_steps=max_steps)
        if result["reached"]:
            successes += 1
        if result["collision"]:
            collisions += 1
        steps_list.append(result["steps"])
        rewards_list.append(result["total_reward"])

        if verbose and (i + 1) % 10 == 0:
            print(f"  Episode {i+1}/{n_episodes}  "
                  f"success_rate={successes/(i+1):.2%}")

    return {
        "n_episodes":    n_episodes,
        "success_rate":  successes  / n_episodes,
        "collision_rate":collisions / n_episodes,
        "avg_steps":     sum(steps_list)   / n_episodes,
        "avg_reward":    sum(rewards_list) / n_episodes,
        "successes":     successes,
        "collisions":    collisions,
    }
