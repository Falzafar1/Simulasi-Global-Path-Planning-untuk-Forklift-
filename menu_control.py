"""
menu_control.py — Manual Control
Kiri : arena simulasi
Kanan: tombol kontrol + load map
"""

import sys
import math
import json
import os
import pygame
import tkinter as tk
from tkinter import filedialog

from environment import (
    RobotEnvironment, Obstacle,
    DEFAULT_ARENA_WIDTH, DEFAULT_ARENA_HEIGHT,
    ROBOT_RADIUS, MAX_LINEAR_VEL, MAX_ANGULAR_VEL, DT,
)
from ui_common import (
    BG_COLOR, PANEL_COLOR, ARENA_BG, ARENA_BORDER, GRID_COLOR,
    ROBOT_COLOR, ROBOT_DIR_COLOR, TRAIL_COLOR, AXIS_COLOR, TEXT_COLOR,
    HEADING_COLOR, HIGHLIGHT_COLOR, WARNING_COLOR, SUCCESS_COLOR,
    OBSTACLE_COLOR, OBSTACLE_BORDER, DIVIDER_COLOR,
    BTN_NORMAL, BTN_HOVER, BTN_PRESS,
    BTN_EXIT_NORMAL, BTN_EXIT_HOVER, BTN_EXIT_PRESS,
    BTN_SAVE_NORMAL, BTN_SAVE_HOVER, BTN_SAVE_PRESS,
    BTN_RESET_NORMAL, BTN_RESET_HOVER, BTN_RESET_PRESS,
    Button, Toast, draw_robot_sprite,
    draw_arena_background, draw_obstacle_rect,
)

# ── Layout ────────────────────────────────────────────────
ARENA_PANEL_W = 680
ARENA_MARGIN  = 40
RIGHT_W       = 300
WIN_W         = ARENA_PANEL_W + RIGHT_W
WIN_H         = 760
FPS           = 60
ARENA_PX      = ARENA_PANEL_W - 2 * ARENA_MARGIN

BTN_W = 240
BTN_H = 52
BTN_RADIUS = 8

# Warna khusus defeat overlay
DEFEAT_BG     = (120, 20, 20, 200)
DEFEAT_TEXT   = (255, 100, 80)
DEFEAT_SUB    = (255, 200, 180)


def _tk_root():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


class ControlMenu:
    def __init__(self, map_path: str = None):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Manual Control")

        self.font_xl   = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_lg   = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_md   = pygame.font.SysFont("monospace", 13)
        self.font_sm   = pygame.font.SysFont("monospace", 11)
        self.font_icon = pygame.font.SysFont("monospace", 20, bold=True)

        self.clock  = pygame.time.Clock()
        self.result = None
        self.toast  = Toast(self.font_lg)

        # ── State ─────────────────────────────────────────
        self.defeat       = False
        self.defeat_timer = 0
        self.map_name     = "Default"

        # ── Env default ───────────────────────────────────
        self.env = RobotEnvironment()
        self.env.reset()
        self._update_scale()

        self.trail_surf = self._new_trail_surf()

        self._build_buttons()

        if map_path:
            self._load_map(map_path)

    # ─────────────────────────────────────────
    #  Scale
    # ─────────────────────────────────────────
    def _update_scale(self):
        self.scale = min(
            ARENA_PX / self.env.arena_w,
            ARENA_PX / self.env.arena_h,
        )

    def _new_trail_surf(self):
        s = pygame.Surface((ARENA_PANEL_W, WIN_H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        return s

    # ─────────────────────────────────────────
    #  Koordinat
    # ─────────────────────────────────────────
    def w2s(self, wx, wy):
        sx = ARENA_MARGIN + wx * self.scale
        sy = ARENA_MARGIN + (self.env.arena_h - wy) * self.scale
        return int(sx), int(sy)

    # ─────────────────────────────────────────
    #  Build tombol
    # ─────────────────────────────────────────
    def _build_buttons(self):
        rp = ARENA_PANEL_W
        cx = rp + RIGHT_W // 2
        bx = cx - BTN_W // 2
        y0 = 80
        g  = BTN_H + 12

        self.btn_forward = Button(bx, y0,     BTN_W, BTN_H, "MAJU",       "▲",
                                  action=self._action_forward)
        self.btn_back    = Button(bx, y0+g,   BTN_W, BTN_H, "MUNDUR",     "▼",
                                  action=self._action_backward)
        self.btn_left    = Button(bx, y0+g*2, BTN_W, BTN_H, "ROT. KIRI",  "↺",
                                  action=self._action_rotate_left)
        self.btn_right   = Button(bx, y0+g*3, BTN_W, BTN_H, "ROT. KANAN", "↻",
                                  action=self._action_rotate_right)

        sep_y = y0 + g * 4 + 10
        sw = BTN_W
        self.btn_load  = Button(bx, sep_y,       sw, 40, "Load Map",
                                color_normal=BTN_SAVE_NORMAL,
                                color_hover =BTN_SAVE_HOVER,
                                color_press =BTN_SAVE_PRESS,
                                action=self._action_load)
        self.btn_reset = Button(bx, sep_y + 54,  sw, 40, "Reset",
                                color_normal=BTN_RESET_NORMAL,
                                color_hover =BTN_RESET_HOVER,
                                color_press =BTN_RESET_PRESS,
                                action=self._action_reset)
        self.btn_exit  = Button(bx, sep_y + 108, sw, 40, "Kembali ke Menu",
                                color_normal=BTN_EXIT_NORMAL,
                                color_hover =BTN_EXIT_HOVER,
                                color_press =BTN_EXIT_PRESS,
                                action=self._action_exit)

        self.move_buttons = [
            self.btn_forward, self.btn_back,
            self.btn_left, self.btn_right,
        ]
        self.all_buttons = self.move_buttons + [
            self.btn_load, self.btn_reset, self.btn_exit,
        ]

    # ─────────────────────────────────────────
    #  Aksi robot
    # ─────────────────────────────────────────
    def _action_forward(self):
        if not self.defeat:
            self._step("forward")

    def _action_backward(self):
        if not self.defeat:
            self._step("backward")

    def _action_rotate_left(self):
        if not self.defeat:
            self._step("rotate_left")

    def _action_rotate_right(self):
        if not self.defeat:
            self._step("rotate_right")

    def _step(self, move: str):
        obs, reward, done, info = self.env.step_manual(move)
        if done:
            self._trigger_defeat(info)

    def _trigger_defeat(self, info: dict):
        self.defeat       = True
        self.defeat_timer = 180   # ~3 detik pada 60 FPS

    def _action_reset(self):
        self.env.reset()
        self.trail_surf = self._new_trail_surf()
        self.defeat      = False
        self.defeat_timer = 0

    def _action_load(self):
        root = _tk_root()
        path = filedialog.askopenfilename(
            title="Pilih file map", filetypes=[("JSON Map", "*.json"), ("All", "*.*")])
        root.destroy()
        if not path:
            return
        self._load_map(path)

    def _load_map(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.env = RobotEnvironment.from_dict(data)
            self.env.reset()
            self._update_scale()
            self.trail_surf  = self._new_trail_surf()
            self.defeat      = False
            self.defeat_timer = 0
            self.map_name    = os.path.basename(path)
            self.toast.show(f"Map dimuat: {self.map_name}", SUCCESS_COLOR)
        except Exception as e:
            self.toast.show(f"Gagal load: {e}", WARNING_COLOR)

    def _action_exit(self):
        self.result = "main"

    # ─────────────────────────────────────────
    #  Run loop
    # ─────────────────────────────────────────
    def run(self) -> str:
        while self.result is None:
            self._handle_events()
            for b in self.all_buttons:
                b.update()
            self.toast.update()
            if self.defeat and self.defeat_timer > 0:
                self.defeat_timer -= 1
                if self.defeat_timer <= 0:
                    self._action_reset()
            self._draw()
            self.clock.tick(FPS)
        return self.result

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._action_exit()
                elif event.key == pygame.K_r:
                    self._action_reset()
                elif event.key == pygame.K_UP:
                    self._action_forward()
                elif event.key == pygame.K_DOWN:
                    self._action_backward()
                elif event.key == pygame.K_LEFT:
                    self._action_rotate_left()
                elif event.key == pygame.K_RIGHT:
                    self._action_rotate_right()
            for btn in self.all_buttons:
                btn.handle_event(event)

    # ─────────────────────────────────────────
    #  Draw
    # ─────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_arena()
        self._draw_grid()
        self._draw_obstacles()
        self._draw_robot_trail()
        self._draw_robot()
        self._draw_info_panel()
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (ARENA_PANEL_W, 0), (ARENA_PANEL_W, WIN_H), 2)
        self._draw_right_panel()

        if self.defeat:
            self._draw_defeat_overlay()

        self.toast.draw(self.screen, WIN_W // 2, WIN_H - 20)
        pygame.display.flip()

    def _draw_arena(self):
        arena_px_w = int(self.env.arena_w * self.scale)
        arena_px_h = int(self.env.arena_h * self.scale)
        r = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, arena_px_w, arena_px_h)
        draw_arena_background(self.screen, r)
        pygame.draw.rect(self.screen, ARENA_BORDER, r, 3)

    def _draw_grid(self):
        step = max(0.5, self.env.robot.step_linear)
        cols = int(self.env.arena_w / step) + 1
        rows = int(self.env.arena_h / step) + 1
        cell = int(step * self.scale)
        ax_w = int(self.env.arena_w * self.scale)
        ax_h = int(self.env.arena_h * self.scale)

        for i in range(cols + 1):
            x = ARENA_MARGIN + i * cell
            if x > ARENA_MARGIN + ax_w:
                break
            pygame.draw.line(self.screen, GRID_COLOR,
                             (x, ARENA_MARGIN), (x, ARENA_MARGIN + ax_h), 1)
        for j in range(rows + 1):
            y = ARENA_MARGIN + j * cell
            if y > ARENA_MARGIN + ax_h:
                break
            pygame.draw.line(self.screen, GRID_COLOR,
                             (ARENA_MARGIN, y), (ARENA_MARGIN + ax_w, y), 1)

    def _draw_obstacles(self):
        for obs in self.env.obstacles:
            sx, sy = self.w2s(obs.x, obs.y + obs.h)
            pw = max(2, int(obs.w * self.scale))
            ph = max(2, int(obs.h * self.scale))
            draw_obstacle_rect(self.screen, pygame.Rect(sx, sy, pw, ph))

    def _draw_robot_trail(self):
        h = self.env.robot.pose_history
        if len(h) >= 2:
            p1 = self.w2s(*h[-2])
            p2 = self.w2s(*h[-1])
            pygame.draw.line(self.trail_surf, (*TRAIL_COLOR, 180), p1, p2, 2)
        self.screen.blit(self.trail_surf, (0, 0))

    def _draw_robot(self):
        robot  = self.env.robot
        cx, cy = self.w2s(robot.x, robot.y)
        r_px   = max(4, int(robot.radius * self.scale))
        draw_robot_sprite(self.screen, cx, cy, r_px, robot.theta,
                          defeat=self.defeat)

    def _draw_info_panel(self):
        robot = self.env.robot
        py    = ARENA_MARGIN + int(self.env.arena_h * self.scale) + 8

        title = self.font_lg.render(
            f"MANUAL CONTROL  ·  Map: {self.map_name}", True, HEADING_COLOR)
        self.screen.blit(title, (ARENA_MARGIN, py))

        deg = math.degrees(robot.theta)
        self.screen.blit(
            self.font_md.render(
                f"Pose:  x={robot.x:6.3f}m  y={robot.y:6.3f}m  θ={deg:7.2f}°",
                True, TEXT_COLOR),
            (ARENA_MARGIN, py + 22))

        self.screen.blit(
            self.font_md.render(
                f"Step: {robot.step_linear*100:.1f}cm  "
                f"Rotasi: {math.degrees(robot.step_angular):.1f}°  "
                f"t={self.env.timestep}",
                True, TEXT_COLOR),
            (ARENA_MARGIN, py + 42))

        self.screen.blit(
            self.font_sm.render(
                "Keyboard: [↑↓] Maju/Mundur  [←→] Rotasi  [R] Reset  [ESC] Menu",
                True, (100, 100, 140)),
            (ARENA_MARGIN, py + 64))

    def _draw_right_panel(self):
        rp = ARENA_PANEL_W
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         pygame.Rect(rp, 0, RIGHT_W, WIN_H))

        title = self.font_xl.render("KONTROL", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(
            centerx=rp + RIGHT_W // 2, y=22))

        for btn in self.all_buttons:
            btn.draw(self.screen, self.font_lg, self.font_icon)

        # Info step
        robot = self.env.robot
        bx = ARENA_PANEL_W + (RIGHT_W - BTN_W) // 2
        iy = self.btn_right.rect.bottom + 10
        self.screen.blit(
            self.font_sm.render(
                f"Maju/Mundur : {robot.step_linear*100:.1f} cm/klik",
                True, (100, 110, 150)),
            (bx, iy))
        self.screen.blit(
            self.font_sm.render(
                f"Rotasi      : {math.degrees(robot.step_angular):.1f}° /klik",
                True, (100, 110, 150)),
            (bx, iy + 18))

        # Garis pemisah
        sep_y = self.btn_load.rect.top - 8
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (rp + 20, sep_y), (rp + RIGHT_W - 20, sep_y), 1)

    def _draw_defeat_overlay(self):
        """Overlay merah besar 'DEFEAT' saat robot menabrak."""
        # Background semi-transparan
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        alpha   = min(180, int((180 - self.defeat_timer) * 1.0))
        overlay.fill((120, 20, 20, max(0, alpha)))
        self.screen.blit(overlay, (0, 0))

        # Teks DEFEAT
        cx = WIN_W // 2
        cy = WIN_H // 2 - 40

        defeat_font = pygame.font.SysFont("monospace", 72, bold=True)
        txt = defeat_font.render("DEFEAT!", True, DEFEAT_TEXT)
        self.screen.blit(txt, txt.get_rect(centerx=cx, centery=cy))

        sub_font = pygame.font.SysFont("monospace", 20)
        sub = sub_font.render("Robot menabrak obstacle!", True, DEFEAT_SUB)
        self.screen.blit(sub, sub.get_rect(centerx=cx, centery=cy + 70))

        countdown = max(0, self.defeat_timer // 60) + 1
        cnt = sub_font.render(f"Reset dalam {countdown}...", True, DEFEAT_SUB)
        self.screen.blit(cnt, cnt.get_rect(centerx=cx, centery=cy + 100))

        # Tombol reset manual
        reset_r = pygame.Rect(cx - 100, cy + 140, 200, 44)
        pygame.draw.rect(self.screen, BTN_RESET_NORMAL, reset_r, border_radius=8)
        pygame.draw.rect(self.screen, (80, 200, 120), reset_r, 2, border_radius=8)
        rl = self.font_lg.render("Reset Sekarang [R]", True, (220, 255, 220))
        self.screen.blit(rl, rl.get_rect(center=reset_r.center))
