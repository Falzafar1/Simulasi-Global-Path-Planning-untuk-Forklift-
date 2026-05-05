"""
main.py — Entry point simulasi robot differential drive.

Menu:
    1. Environment Editor  — buat & simpan peta + parameter robot
    2. Manual Control      — kendalikan robot di peta yang dipilih
    3. Lidar Sensor View   — visualisasi lidar + kontrol keyboard ↑↓←→
    4. DQN Training        — latih agen DQN, simpan model .pth
    5. DQN Inference       — jalankan model terlatih, amati perilaku greedy

Jalankan:
    python main.py

Dependensi:
    pip install pygame numpy torch
"""

import sys
import pygame

from menu_main       import MainMenu
from menu_editor     import EditorMenu
from menu_control    import ControlMenu
from menu_lidar      import LidarMenu
from menu_training   import TrainingMenu
from menu_inference  import InferenceMenu
from menu_appearance import AppearanceMenu


def main():
    pygame.init()
    pygame.display.set_caption("Differential Drive Robot — RL Simulation")

    state           = "main"
    loaded_map_path = None

    while True:
        # ── Menu utama ──────────────────────────────────────
        if state == "main":
            result = MainMenu().run()
            if result == "editor":
                state = "editor"
            elif result == "control":
                state = "control"
                loaded_map_path = None
            elif result == "lidar":
                state = "lidar"
                loaded_map_path = None
            elif result == "training":
                state = "training"
                loaded_map_path = None
            elif result == "inference":
                state = "inference"
                loaded_map_path = None
            elif result == "appearance":
                state = "appearance"
            else:
                break   # quit

        # ── Editor ──────────────────────────────────────────
        elif state == "editor":
            result = EditorMenu().run()
            state  = "main" if result == "main" else "exit"

        # ── Manual Control ──────────────────────────────────
        elif state == "control":
            result = ControlMenu(map_path=loaded_map_path).run()
            state  = "main" if result == "main" else "exit"

        # ── Lidar View ──────────────────────────────────────
        elif state == "lidar":
            result = LidarMenu(map_path=loaded_map_path).run()
            state  = "main" if result == "main" else "exit"

        # ── DQN Training ────────────────────────────────────
        elif state == "training":
            result = TrainingMenu(map_path=loaded_map_path).run()
            state  = "main" if result == "main" else "exit"

        # ── DQN Inference ───────────────────────────────────
        elif state == "inference":
            result = InferenceMenu(map_path=loaded_map_path).run()
            state  = "main" if result == "main" else "exit"

        # ── Pengaturan Tampilan ──────────────────────────────
        elif state == "appearance":
            result = AppearanceMenu().run()
            state  = "main" if result == "main" else "exit"

        if state == "exit":
            break

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()