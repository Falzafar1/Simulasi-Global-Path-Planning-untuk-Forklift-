"""
core/map_io.py — Utilitas load/save peta JSON, tanpa dependensi pygame.

Dapat diimpor di VSCode (pygame) maupun Google Colab (tanpa pygame).
"""

import json
import math
from pathlib import Path

from environment import RobotEnvironment


def load_map(path: str) -> RobotEnvironment:
    """
    Baca file JSON peta dan kembalikan RobotEnvironment yang sudah dikonfigurasi.

    Parameters
    ----------
    path : path ke file .json

    Returns
    -------
    RobotEnvironment

    Raises
    ------
    FileNotFoundError, json.JSONDecodeError, KeyError — diteruskan ke pemanggil.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RobotEnvironment.from_dict(data)


def save_map(env: RobotEnvironment, path: str) -> None:
    """
    Simpan state RobotEnvironment ke file JSON.

    Parameters
    ----------
    env  : lingkungan yang akan disimpan
    path : path tujuan (akan dibuat jika belum ada)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = env.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_map_dict(arena_w: float, arena_h: float,
                   robot_x: float, robot_y: float,
                   robot_theta_deg: float = 0.0,
                   robot_radius: float = 0.2,
                   step_linear_cm: float = 5.0,
                   step_angular_deg: float = 15.7,
                   lidar_rays: int = 8,
                   lidar_range: float = 3.0,
                   obstacles: list[dict] | None = None) -> dict:
    """
    Bangun dict format map secara programatik (berguna di Colab tanpa editor GUI).

    obstacles : list of dict {"x": ..., "y": ..., "w": ..., "h": ...}

    Contoh
    ------
    >>> d = build_map_dict(10, 10, 5, 5, obstacles=[{"x":3,"y":3,"w":1,"h":1}])
    >>> env = RobotEnvironment.from_dict(d)
    """
    return {
        "arena": {"w": arena_w, "h": arena_h},
        "robot": {
            "start_x":      robot_x,
            "start_y":      robot_y,
            "start_theta":  math.radians(robot_theta_deg),
            "radius":       robot_radius,
            "step_linear":  step_linear_cm / 100.0,
            "step_angular": math.radians(step_angular_deg),
        },
        "lidar": {
            "num_rays":  lidar_rays,
            "max_range": lidar_range,
        },
        "obstacles": obstacles or [],
    }
