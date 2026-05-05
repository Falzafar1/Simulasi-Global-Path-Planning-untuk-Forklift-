"""
Differential Drive Robot Environment
Simulasi lingkungan 2D untuk robot differential drive.
Basis: https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM

Versi ini mendukung:
- Ukuran arena variabel
- Obstacle rectangular
- Parameter robot yang dapat dikustomisasi
"""

import math
import numpy as np


# ─────────────────────────────────────────────
#  KONSTANTA DEFAULT LINGKUNGAN
# ─────────────────────────────────────────────
DEFAULT_ARENA_WIDTH  = 10.0   # meter
DEFAULT_ARENA_HEIGHT = 10.0   # meter

ROBOT_RADIUS = 0.2            # meter

MAX_LINEAR_VEL  = 1.0         # m/s
MAX_ANGULAR_VEL = math.pi     # rad/s

DT = 0.05                     # detik

DISCRETE_DIRECTIONS = {
    "E":  0.0,
    "NE": math.pi / 4,
    "N":  math.pi / 2,
    "NW": 3 * math.pi / 4,
    "W":  math.pi,
    "SW": -3 * math.pi / 4,
    "SE": -math.pi / 4,
    "S":  -math.pi / 2,
}


# ─────────────────────────────────────────────
#  KELAS OBSTACLE
# ─────────────────────────────────────────────
class Obstacle:
    """
    Obstacle berbentuk rectangle di dunia (world coordinates, meter).
    x, y  : sudut kiri-bawah (world frame)
    w, h  : lebar dan tinggi (meter)
    """
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        """Cek apakah lingkaran (cx,cy,radius) bersinggungan dengan obstacle."""
        # Titik terdekat di rectangle terhadap pusat lingkaran
        nearest_x = max(self.x, min(cx, self.x + self.w))
        nearest_y = max(self.y, min(cy, self.y + self.h))
        dist_sq = (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2
        return dist_sq < radius ** 2

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, d: dict) -> "Obstacle":
        return cls(d["x"], d["y"], d["w"], d["h"])


# ─────────────────────────────────────────────
#  KELAS ROBOT
# ─────────────────────────────────────────────
class DifferentialDriveRobot:
    """
    Model kinematik robot differential drive.
    State  : (x, y, theta)
    Action : (v, omega)
    """

    def __init__(self, x: float = 5.0, y: float = 5.0, theta: float = 0.0,
                 radius: float = ROBOT_RADIUS,
                 step_linear: float = MAX_LINEAR_VEL * DT,
                 step_angular: float = MAX_ANGULAR_VEL * DT):
        self.start_x     = x
        self.start_y     = y
        self.start_theta = theta
        self.radius      = radius
        self.step_linear  = step_linear
        self.step_angular = step_angular

        self.x     = x
        self.y     = y
        self.theta = theta

        self.v     = 0.0
        self.omega = 0.0

        self.pose_history: list[tuple[float, float]] = [(x, y)]

    def step(self, v: float, omega: float, dt: float = DT) -> tuple[float, float, float]:
        v     = np.clip(v,     -MAX_LINEAR_VEL,  MAX_LINEAR_VEL)
        omega = np.clip(omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

        self.v     = v
        self.omega = omega

        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta  = self._normalize_angle(self.theta + omega * dt)

        self.pose_history.append((self.x, self.y))
        return self.x, self.y, self.theta

    def step_by_distance(self, direction: int) -> tuple[float, float, float]:
        """
        Gerak maju (1) atau mundur (-1) sejauh step_linear meter.
        """
        dist = direction * self.step_linear
        self.x += dist * math.cos(self.theta)
        self.y += dist * math.sin(self.theta)
        self.v  = dist / DT
        self.omega = 0.0
        self.pose_history.append((self.x, self.y))
        return self.x, self.y, self.theta

    def rotate_by_step(self, direction: int):
        """
        Rotasi searah (1) atau berlawanan (-1) jarum jam sebesar step_angular radian.
        """
        self.theta = self._normalize_angle(self.theta + direction * self.step_angular)
        self.v = 0.0
        self.omega = direction * self.step_angular / DT
        self.pose_history.append((self.x, self.y))

    def reset(self, x: float = None, y: float = None, theta: float = None):
        self.x     = x     if x     is not None else self.start_x
        self.y     = y     if y     is not None else self.start_y
        self.theta = theta if theta is not None else self.start_theta
        self.v, self.omega = 0.0, 0.0
        self.pose_history  = [(self.x, self.y)]

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega], dtype=np.float32)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


# ─────────────────────────────────────────────
#  KELAS LINGKUNGAN
# ─────────────────────────────────────────────
class RobotEnvironment:
    """
    Lingkungan simulasi robot 2D dengan dukungan obstacle.
    Gym-like interface: obs, reward, done, info = env.step(action)
    """

    def __init__(self,
                 arena_w: float = DEFAULT_ARENA_WIDTH,
                 arena_h: float = DEFAULT_ARENA_HEIGHT,
                 robot_x: float = 5.0,
                 robot_y: float = 5.0,
                 robot_theta: float = 0.0,
                 robot_radius: float = ROBOT_RADIUS,
                 step_linear: float = MAX_LINEAR_VEL * DT,
                 step_angular: float = MAX_ANGULAR_VEL * DT,
                 obstacles: list = None):

        self.arena_w   = arena_w
        self.arena_h   = arena_h
        self.obstacles: list[Obstacle] = obstacles if obstacles else []
        self.timestep  = 0
        # Lidar defaults (dapat di-override via from_dict atau editor)
        self.lidar_num_rays  = 8
        self.lidar_max_range = 3.0

        self.robot = DifferentialDriveRobot(
            x=robot_x, y=robot_y, theta=robot_theta,
            radius=robot_radius,
            step_linear=step_linear,
            step_angular=step_angular
        )

    def reset(self) -> np.ndarray:
        self.robot.reset()
        self.timestep = 0
        return self.robot.state

    def step(self, action: tuple[float, float]) -> tuple[np.ndarray, float, bool, dict]:
        v, omega = action
        self.robot.step(v, omega)
        self.timestep += 1

        hit_wall     = self._is_out_of_bounds()
        hit_obstacle = self._is_colliding_obstacle()
        done = hit_wall or hit_obstacle

        obs    = self.robot.state
        reward = 0.0
        info   = {
            "timestep":      self.timestep,
            "pose":          self.robot.pose,
            "out_of_bounds": hit_wall,
            "hit_obstacle":  hit_obstacle,
        }
        return obs, reward, done, info

    def step_manual(self, move: str) -> tuple[np.ndarray, float, bool, dict]:
        """
        Kontrol manual:
        move: 'forward' | 'backward' | 'rotate_left' | 'rotate_right'
        """
        if move == "forward":
            self.robot.step_by_distance(1)
        elif move == "backward":
            self.robot.step_by_distance(-1)
        elif move == "rotate_left":
            self.robot.rotate_by_step(1)
        elif move == "rotate_right":
            self.robot.rotate_by_step(-1)
        self.timestep += 1

        hit_wall     = self._is_out_of_bounds()
        hit_obstacle = self._is_colliding_obstacle()
        done = hit_wall or hit_obstacle

        obs  = self.robot.state
        info = {
            "timestep":      self.timestep,
            "pose":          self.robot.pose,
            "out_of_bounds": hit_wall,
            "hit_obstacle":  hit_obstacle,
        }
        return obs, 0.0, done, info

    def _is_out_of_bounds(self) -> bool:
        r = self.robot.radius
        return not (r <= self.robot.x <= self.arena_w - r and
                    r <= self.robot.y <= self.arena_h - r)

    def _is_colliding_obstacle(self) -> bool:
        for obs in self.obstacles:
            if obs.collides_with_circle(self.robot.x, self.robot.y, self.robot.radius):
                return True
        return False

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)

    def remove_obstacle_at(self, wx: float, wy: float):
        """Hapus obstacle yang mengandung titik (wx, wy)."""
        self.obstacles = [
            o for o in self.obstacles
            if not (o.x <= wx <= o.x + o.w and o.y <= wy <= o.y + o.h)
        ]

    def to_dict(self) -> dict:
        """Serialisasi lingkungan ke dict (untuk JSON)."""
        return {
            "arena": {"w": self.arena_w, "h": self.arena_h},
            "robot": {
                "start_x":      self.robot.start_x,
                "start_y":      self.robot.start_y,
                "start_theta":  self.robot.start_theta,
                "radius":       self.robot.radius,
                "step_linear":  self.robot.step_linear,
                "step_angular": self.robot.step_angular,
            },
            "lidar": {
                "num_rays":  getattr(self, "lidar_num_rays",  8),
                "max_range": getattr(self, "lidar_max_range", 3.0),
            },
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobotEnvironment":
        """Deserialisasi dari dict (dibaca dari JSON)."""
        arena    = d.get("arena", {})
        robot    = d.get("robot", {})
        lidar    = d.get("lidar", {})
        obs_list = [Obstacle.from_dict(o) for o in d.get("obstacles", [])]
        env = cls(
            arena_w      = arena.get("w", DEFAULT_ARENA_WIDTH),
            arena_h      = arena.get("h", DEFAULT_ARENA_HEIGHT),
            robot_x      = robot.get("start_x", 5.0),
            robot_y      = robot.get("start_y", 5.0),
            robot_theta  = robot.get("start_theta", 0.0),
            robot_radius = robot.get("radius", ROBOT_RADIUS),
            step_linear  = robot.get("step_linear", MAX_LINEAR_VEL * DT),
            step_angular = robot.get("step_angular", MAX_ANGULAR_VEL * DT),
            obstacles    = obs_list,
        )
        env.lidar_num_rays  = lidar.get("num_rays",  8)
        env.lidar_max_range = lidar.get("max_range", 3.0)
        return env