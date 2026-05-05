"""
core/lidar.py — LidarSensor tanpa dependensi pygame.

Dapat diimpor di VSCode (pygame) maupun Google Colab (tanpa pygame).
"""

import math


class LidarSensor:
    """
    Sensor lidar ray-casting 2D.

    Parameters
    ----------
    num_rays  : jumlah sinar (1–24)
    max_range : jangkauan maksimum (meter)
    """

    def __init__(self, num_rays: int = 8, max_range: float = 3.0):
        self.num_rays  = max(1, min(24, num_rays))
        self.max_range = max(0.1, max_range)

    def scan(self,
             robot_x: float, robot_y: float, robot_theta: float,
             obstacles, arena_w: float, arena_h: float,
             num_steps: int = 200) -> list[dict]:
        """
        Lakukan ray-casting dari posisi robot.

        Returns
        -------
        list of dict, satu per sinar:
            angle_abs  : sudut absolut (rad)
            angle_rel  : sudut relatif terhadap heading robot (rad)
            distance   : jarak ke halangan terdekat (meter)
            hit        : True jika mengenai sesuatu sebelum max_range
            end_x, end_y : koordinat ujung sinar di world frame
        """
        results = []
        step_m  = self.max_range / num_steps

        for i in range(self.num_rays):
            angle_rel = 2 * math.pi * i / self.num_rays
            angle_abs = robot_theta + angle_rel
            dist, hit = self.max_range, False

            for s in range(1, num_steps + 1):
                d  = s * step_m
                rx = robot_x + d * math.cos(angle_abs)
                ry = robot_y + d * math.sin(angle_abs)

                # Dinding arena
                if rx < 0 or rx > arena_w or ry < 0 or ry > arena_h:
                    dist, hit = d, True
                    break

                # Obstacle
                for obs in obstacles:
                    if obs.collides_with_circle(rx, ry, 0.01):
                        dist, hit = d, True
                        break

                if hit:
                    break

            results.append({
                "angle_abs": angle_abs,
                "angle_rel": angle_rel,
                "distance":  dist,
                "hit":       hit,
                "end_x":     robot_x + dist * math.cos(angle_abs),
                "end_y":     robot_y + dist * math.sin(angle_abs),
            })

        return results
