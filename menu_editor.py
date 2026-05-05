"""
menu_editor.py — Environment Editor
Kiri : kanvas arena + obstacle painting
Kanan: panel parameter robot & map
"""

import sys
import math
import json
import os
import pygame
import tkinter as tk
from tkinter import filedialog, messagebox

from environment import (
    RobotEnvironment, Obstacle,
    DEFAULT_ARENA_WIDTH, DEFAULT_ARENA_HEIGHT,
    ROBOT_RADIUS, MAX_LINEAR_VEL, MAX_ANGULAR_VEL, DT,
)
from ui_common import (
    BG_COLOR, PANEL_COLOR, ARENA_BG, ARENA_BORDER, GRID_COLOR,
    ROBOT_COLOR, ROBOT_DIR_COLOR, AXIS_COLOR, TEXT_COLOR,
    HEADING_COLOR, HIGHLIGHT_COLOR, WARNING_COLOR, SUCCESS_COLOR,
    OBSTACLE_COLOR, OBSTACLE_BORDER, DIVIDER_COLOR,
    BTN_NORMAL, BTN_HOVER, BTN_PRESS,
    BTN_EXIT_NORMAL, BTN_EXIT_HOVER, BTN_EXIT_PRESS,
    BTN_SAVE_NORMAL, BTN_SAVE_HOVER, BTN_SAVE_PRESS,
    BTN_WARN_NORMAL, BTN_WARN_HOVER, BTN_WARN_PRESS,
    BTN_RESET_NORMAL, BTN_RESET_HOVER, BTN_RESET_PRESS,
    Button, Checkbox, TextInput, Toast,
)

# ── Layout ────────────────────────────────────────────────
ARENA_PANEL_W = 680
ARENA_MARGIN  = 40
RIGHT_W       = 320
WIN_W         = ARENA_PANEL_W + RIGHT_W
WIN_H         = 760
FPS           = 60

ARENA_PX      = ARENA_PANEL_W - 2 * ARENA_MARGIN   # piksel lebar arena default

# Tombol
BTN_W = 260
BTN_H = 42
BTN_RADIUS = 8


def _tk_root():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


class EditorMenu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Environment Editor")

        self.font_xl  = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_lg  = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 13)
        self.font_sm  = pygame.font.SysFont("monospace", 11)

        self.clock  = pygame.time.Clock()
        self.result = None
        self.toast  = Toast(self.font_lg)

        # ── Parameter robot (akan dibaca dari input box) ──
        self.default_arena_w  = DEFAULT_ARENA_WIDTH
        self.default_arena_h  = DEFAULT_ARENA_HEIGHT
        self._init_params()

        # ── State editor ──────────────────────────────────
        self.obstacles: list[Obstacle] = []
        self.painted_cells: set        = set()   # set of (grid_col, grid_row)
        self.obstacle_mode = False
        self.painting      = False
        self.erase_mode    = False  # True = klik kanan hapus

        self._build_ui()
        self._update_scale()

    # ─────────────────────────────────────────
    #  Inisialisasi parameter default
    # ─────────────────────────────────────────
    def _init_params(self):
        self.arena_w    = self.default_arena_w
        self.arena_h    = self.default_arena_h
        self.robot_x    = self.arena_w / 2
        self.robot_y    = self.arena_h / 2
        self.robot_theta = 0.0
        self.robot_r    = ROBOT_RADIUS
        self.step_lin   = MAX_LINEAR_VEL * DT   # meter
        self.step_ang   = MAX_ANGULAR_VEL * DT  # radian
        self.lidar_rays  = 8
        self.lidar_range = 3.0

    # ─────────────────────────────────────────
    #  Hitung skala dari parameter
    # ─────────────────────────────────────────
    def _update_scale(self):
        self.scale_x = ARENA_PX / self.arena_w
        self.scale_y = ARENA_PX / self.arena_h
        # Gunakan skala seragam (ambil terkecil agar proporsional)
        self.scale   = min(self.scale_x, self.scale_y)

    # ─────────────────────────────────────────
    #  Step size dalam piksel
    # ─────────────────────────────────────────
    @property
    def cell_px(self) -> int:
        """Ukuran satu 'cell' obstacle = step_linear dalam piksel."""
        return max(4, int(self.step_lin * self.scale))

    # ─────────────────────────────────────────
    #  Konversi koordinat
    # ─────────────────────────────────────────
    def w2s(self, wx, wy):
        """World → Screen."""
        sx = ARENA_MARGIN + wx * self.scale
        sy = ARENA_MARGIN + (self.arena_h - wy) * self.scale
        return int(sx), int(sy)

    def s2w(self, sx, sy):
        """Screen → World (float)."""
        wx = (sx - ARENA_MARGIN) / self.scale
        wy = self.arena_h - (sy - ARENA_MARGIN) / self.scale
        return wx, wy

    def s2grid(self, sx, sy):
        """Screen koordinat → grid cell (col, row)."""
        wx, wy = self.s2w(sx, sy)
        col = int(wx / self.step_lin)
        row = int(wy / self.step_lin)
        return col, row

    def grid2obs(self, col, row) -> Obstacle:
        """Grid cell → Obstacle di world."""
        x = col * self.step_lin
        y = row * self.step_lin
        return Obstacle(x, y, self.step_lin, self.step_lin)

    def _in_arena(self, sx, sy) -> bool:
        return (ARENA_MARGIN <= sx <= ARENA_MARGIN + self.arena_w * self.scale and
                ARENA_MARGIN <= sy <= ARENA_MARGIN + self.arena_h * self.scale)

    # ─────────────────────────────────────────
    #  Build UI
    # ─────────────────────────────────────────
    def _build_ui(self):
        rp = ARENA_PANEL_W           # x start panel kanan
        cx = rp + RIGHT_W // 2
        bx = cx - BTN_W // 2
        iw = BTN_W - 10              # lebar input box
        ix = rp + 30
        iy = 28

        # ── Input: Ukuran Arena ─────────────
        self.inp_arena_w = TextInput(ix, iy + 20, 90, 28, self.font_md,
                                     label="Arena W (m)",
                                     default=str(self.arena_w), numeric=True,
                                     min_val=2.0, max_val=50.0)
        self.inp_arena_h = TextInput(ix + 110, iy + 20, 90, 28, self.font_md,
                                     label="Arena H (m)",
                                     default=str(self.arena_h), numeric=True,
                                     min_val=2.0, max_val=50.0)

        # ── Input: Langkah robot ────────────
        iy2 = iy + 76
        self.inp_step_lin = TextInput(ix, iy2 + 20, 90, 28, self.font_md,
                                      label="Step Linear (cm)",
                                      default=str(int(self.step_lin * 100)),
                                      numeric=True, min_val=1, max_val=500)
        self.inp_step_ang = TextInput(ix + 110, iy2 + 20, 90, 28, self.font_md,
                                      label="Step Rotasi (°)",
                                      default=str(round(math.degrees(self.step_ang), 1)),
                                      numeric=True, min_val=1, max_val=180)

        # ── Input: Ukuran robot ─────────────
        iy3 = iy2 + 76
        self.inp_robot_r = TextInput(ix, iy3 + 20, 90, 28, self.font_md,
                                     label="Radius Robot (cm)",
                                     default=str(int(self.robot_r * 100)),
                                     numeric=True, min_val=1, max_val=200)

        # ── Input: Posisi awal robot ────────
        iy4 = iy3 + 76
        self.inp_rx = TextInput(ix,       iy4 + 20, 80, 28, self.font_md,
                                label="Start X (m)",
                                default=str(self.robot_x), numeric=True)
        self.inp_ry = TextInput(ix + 100, iy4 + 20, 80, 28, self.font_md,
                                label="Start Y (m)",
                                default=str(self.robot_y), numeric=True)
        self.inp_rt = TextInput(ix + 200, iy4 + 20, 60, 28, self.font_md,
                                label="θ (°)",
                                default="0", numeric=True)

        # ── Checkbox obstacle mode ──────────
        iy5 = iy4 + 76
        self.chk_obstacle = Checkbox(
            ix, iy5 + 4, 22, "Mode Obstacle (drag kiri = tambah, kanan = hapus)",
            self.font_sm, checked=False,
            action=self._toggle_obstacle_mode)

        # ── Input: Lidar ─────────────────────────
        iy5b = iy5 + 44
        self.inp_lidar_rays = TextInput(ix, iy5b + 20, 90, 28, self.font_md,
                                        label="Jumlah Lidar (1-24)",
                                        default="8",
                                        numeric=True, min_val=1, max_val=24)
        self.inp_lidar_range = TextInput(ix + 110, iy5b + 20, 90, 28, self.font_md,
                                         label="Jangkauan Lidar (m)",
                                         default="3.0",
                                         numeric=True, min_val=0.1, max_val=20.0)

        # Tombol Terapkan Parameter
        iy6 = iy5b + 76
        self.btn_apply = Button(
            bx, iy6, BTN_W, BTN_H, "Terapkan Parameter",
            color_normal=BTN_WARN_NORMAL, color_hover=BTN_WARN_HOVER,
            color_press=BTN_WARN_PRESS,
            action=self._apply_params)

        # Tombol Clear
        iy7 = iy6 + BTN_H + 10
        self.btn_clear = Button(
            bx, iy7, BTN_W, BTN_H, "Clear Environment",
            color_normal=BTN_RESET_NORMAL, color_hover=BTN_RESET_HOVER,
            color_press=BTN_RESET_PRESS,
            action=self._action_clear)

        # Tombol Load Map
        iy8 = iy7 + BTN_H + 10
        self.btn_load = Button(
            bx, iy8, BTN_W, BTN_H, "Load Map",
            color_normal=BTN_NORMAL, color_hover=BTN_HOVER,
            color_press=BTN_PRESS,
            action=self._action_load)

        # Tombol Save Map
        iy9 = iy8 + BTN_H + 10
        self.btn_save = Button(
            bx, iy9, BTN_W, BTN_H, "Save Map",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS,
            action=self._action_save)

        # Tombol Exit
        iy10 = iy9 + BTN_H + 20
        self.btn_exit = Button(
            bx, iy10, BTN_W, BTN_H, "Kembali ke Menu",
            color_normal=BTN_EXIT_NORMAL, color_hover=BTN_EXIT_HOVER,
            color_press=BTN_EXIT_PRESS,
            action=self._action_exit)

        self.all_inputs = [
            self.inp_arena_w, self.inp_arena_h,
            self.inp_step_lin, self.inp_step_ang,
            self.inp_robot_r,
            self.inp_rx, self.inp_ry, self.inp_rt,
            self.inp_lidar_rays, self.inp_lidar_range,
        ]
        self.all_buttons = [
            self.btn_apply, self.btn_clear,
            self.btn_load, self.btn_save, self.btn_exit,
        ]

    # ─────────────────────────────────────────
    #  Aksi
    # ─────────────────────────────────────────
    def _toggle_obstacle_mode(self, state: bool):
        self.obstacle_mode = state

    def _apply_params(self):
        self.arena_w  = self.inp_arena_w.get_float(self.arena_w)
        self.arena_h  = self.inp_arena_h.get_float(self.arena_h)
        self.step_lin = self.inp_step_lin.get_float(self.step_lin * 100) / 100.0
        self.step_ang = math.radians(self.inp_step_ang.get_float(
                                         math.degrees(self.step_ang)))
        self.robot_r  = self.inp_robot_r.get_float(self.robot_r * 100) / 100.0
        self.robot_x  = self.inp_rx.get_float(self.robot_x)
        self.robot_y  = self.inp_ry.get_float(self.robot_y)
        self.robot_theta = math.radians(self.inp_rt.get_float(0.0))
        self.lidar_rays  = max(1, min(24, int(self.inp_lidar_rays.get_float(8))))
        self.lidar_range = max(0.1, self.inp_lidar_range.get_float(3.0))
        self._update_scale()
        # Hapus obstacles yang sudah tidak valid
        self._filter_obstacles()
        self.toast.show("Parameter diterapkan!", SUCCESS_COLOR)

    def _filter_obstacles(self):
        """Hapus obstacle di luar arena baru."""
        valid = []
        for o in self.obstacles:
            if o.x < self.arena_w and o.y < self.arena_h:
                valid.append(o)
        self.obstacles = valid

    def _action_clear(self):
        self.obstacles    = []
        self.painted_cells = set()
        self.toast.show("Environment di-clear.", WARNING_COLOR)

    def _action_load(self):
        root = _tk_root()
        path = filedialog.askopenfilename(
            title="Pilih file map", filetypes=[("JSON Map", "*.json"), ("All", "*.*")])
        root.destroy()
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            env = RobotEnvironment.from_dict(data)
            # Update state editor dari file
            self.arena_w    = env.arena_w
            self.arena_h    = env.arena_h
            self.step_lin   = env.robot.step_linear
            self.step_ang   = env.robot.step_angular
            self.robot_r    = env.robot.radius
            self.robot_x    = env.robot.start_x
            self.robot_y    = env.robot.start_y
            self.robot_theta = env.robot.start_theta
            self.obstacles  = env.obstacles
            self._update_scale()
            # Update input box
            self.inp_arena_w.text  = str(self.arena_w)
            self.inp_arena_h.text  = str(self.arena_h)
            self.inp_step_lin.text = str(int(self.step_lin * 100))
            self.inp_step_ang.text = str(round(math.degrees(self.step_ang), 1))
            self.inp_robot_r.text  = str(int(self.robot_r * 100))
            self.inp_rx.text       = str(self.robot_x)
            self.inp_ry.text       = str(self.robot_y)
            self.inp_rt.text       = str(round(math.degrees(self.robot_theta), 1))
            # Lidar
            self.lidar_rays  = env.lidar_num_rays
            self.lidar_range = env.lidar_max_range
            self.inp_lidar_rays.text  = str(self.lidar_rays)
            self.inp_lidar_range.text = str(self.lidar_range)
            self.toast.show(f"Map dimuat: {os.path.basename(path)}", SUCCESS_COLOR)
        except Exception as e:
            self.toast.show(f"Gagal load: {e}", WARNING_COLOR)

    def _action_save(self):
        root = _tk_root()
        path = filedialog.asksaveasfilename(
            title="Simpan map", defaultextension=".json",
            filetypes=[("JSON Map", "*.json"), ("All", "*.*")])
        root.destroy()
        if not path:
            return
        try:
            self._apply_params()   # pastikan nilai terbaru
            data = {
                "arena": {"w": self.arena_w, "h": self.arena_h},
                "robot": {
                    "start_x":      self.robot_x,
                    "start_y":      self.robot_y,
                    "start_theta":  self.robot_theta,
                    "radius":       self.robot_r,
                    "step_linear":  self.step_lin,
                    "step_angular": self.step_ang,
                },
                "lidar": {
                    "num_rays":  self.lidar_rays,
                    "max_range": self.lidar_range,
                },
                "obstacles": [o.to_dict() for o in self.obstacles],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.toast.show(f"Map disimpan: {os.path.basename(path)}", SUCCESS_COLOR)
        except Exception as e:
            self.toast.show(f"Gagal simpan: {e}", WARNING_COLOR)

    def _action_exit(self):
        self.result = "main"

    # ─────────────────────────────────────────
    #  Paint / erase obstacle
    # ─────────────────────────────────────────
    def _paint_at(self, sx, sy, erase=False):
        if not self._in_arena(sx, sy):
            return
        col, row = self.s2grid(sx, sy)
        obs = self.grid2obs(col, row)
        # Cek tidak menimpa area dalam arena
        if obs.x + obs.w > self.arena_w or obs.y + obs.h > self.arena_h:
            return
        if erase:
            key = (col, row)
            if key in self.painted_cells:
                self.painted_cells.discard(key)
                self.obstacles = [
                    o for o in self.obstacles
                    if not (abs(o.x - obs.x) < 0.001 and abs(o.y - obs.y) < 0.001)
                ]
        else:
            key = (col, row)
            if key not in self.painted_cells:
                self.painted_cells.add(key)
                self.obstacles.append(obs)

    # ─────────────────────────────────────────
    #  Run loop
    # ─────────────────────────────────────────
    def run(self) -> str:
        while self.result is None:
            self._handle_events()
            for b in self.all_buttons:
                b.update()
            self.toast.update()
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

            # Input & checkbox
            self.chk_obstacle.handle_event(event)
            for inp in self.all_inputs:
                inp.handle_event(event)
            for btn in self.all_buttons:
                btn.handle_event(event)

            # Obstacle painting
            if self.obstacle_mode:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self._in_arena(*event.pos):
                        self.painting   = True
                        self.erase_mode = False
                        self._paint_at(*event.pos, erase=False)
                    elif event.button == 3 and self._in_arena(*event.pos):
                        self.painting   = True
                        self.erase_mode = True
                        self._paint_at(*event.pos, erase=True)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.painting = False
                elif event.type == pygame.MOUSEMOTION and self.painting:
                    self._paint_at(*event.pos, erase=self.erase_mode)

    # ─────────────────────────────────────────
    #  Draw
    # ─────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_arena()
        self._draw_grid()
        self._draw_obstacles()
        self._draw_robot_start()
        self._draw_axis_labels()
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (ARENA_PANEL_W, 0), (ARENA_PANEL_W, WIN_H), 2)
        self._draw_right_panel()
        # Toast di tengah layar
        self.toast.draw(self.screen, WIN_W // 2, WIN_H - 30)
        pygame.display.flip()

    def _draw_arena(self):
        arena_px_w = int(self.arena_w * self.scale)
        arena_px_h = int(self.arena_h * self.scale)
        r = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, arena_px_w, arena_px_h)
        pygame.draw.rect(self.screen, ARENA_BG, r)
        pygame.draw.rect(self.screen, ARENA_BORDER, r, 3)

        # Highlight mode obstacle
        if self.obstacle_mode:
            s = pygame.Surface((arena_px_w, arena_px_h), pygame.SRCALPHA)
            s.fill((180, 60, 60, 18))
            self.screen.blit(s, (ARENA_MARGIN, ARENA_MARGIN))

    def _draw_grid(self):
        # Grid per step_lin
        cols = int(self.arena_w / self.step_lin) + 1
        rows = int(self.arena_h / self.step_lin) + 1
        cell = int(self.step_lin * self.scale)

        for i in range(cols + 1):
            x = ARENA_MARGIN + i * cell
            if x > ARENA_MARGIN + int(self.arena_w * self.scale):
                break
            pygame.draw.line(self.screen, GRID_COLOR,
                             (x, ARENA_MARGIN),
                             (x, ARENA_MARGIN + int(self.arena_h * self.scale)), 1)
        for j in range(rows + 1):
            y = ARENA_MARGIN + j * cell
            if y > ARENA_MARGIN + int(self.arena_h * self.scale):
                break
            pygame.draw.line(self.screen, GRID_COLOR,
                             (ARENA_MARGIN, y),
                             (ARENA_MARGIN + int(self.arena_w * self.scale), y), 1)

    def _draw_obstacles(self):
        for obs in self.obstacles:
            sx, sy = self.w2s(obs.x, obs.y + obs.h)   # top-left screen
            pw = max(2, int(obs.w * self.scale))
            ph = max(2, int(obs.h * self.scale))
            r  = pygame.Rect(sx, sy, pw, ph)
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, r)
            pygame.draw.rect(self.screen, OBSTACLE_BORDER, r, 1)

    def _draw_robot_start(self):
        sx, sy = self.w2s(self.robot_x, self.robot_y)
        r_px   = max(4, int(self.robot_r * self.scale))
        pygame.draw.circle(self.screen, ROBOT_COLOR, (sx, sy), r_px)
        pygame.draw.circle(self.screen, (200, 255, 200), (sx, sy), r_px, 2)
        # Arah
        dlen = r_px * 1.8
        dx = sx + int(dlen * math.cos(self.robot_theta))
        dy = sy - int(dlen * math.sin(self.robot_theta))
        pygame.draw.line(self.screen, ROBOT_DIR_COLOR, (sx, sy), (dx, dy), 2)
        # Label
        lbl = self.font_sm.render("START", True, HIGHLIGHT_COLOR)
        self.screen.blit(lbl, (sx + r_px + 3, sy - 7))

    def _draw_axis_labels(self):
        arena_px_w = int(self.arena_w * self.scale)
        arena_px_h = int(self.arena_h * self.scale)
        step = max(1, int(self.arena_w / 5))
        for i in range(0, int(self.arena_w) + 1, step):
            sx = ARENA_MARGIN + int(i * self.scale)
            lbl = self.font_sm.render(str(i), True, AXIS_COLOR)
            self.screen.blit(lbl, (sx - 5, ARENA_MARGIN + arena_px_h + 5))
        step_h = max(1, int(self.arena_h / 5))
        for j in range(0, int(self.arena_h) + 1, step_h):
            sy = ARENA_MARGIN + int((self.arena_h - j) * self.scale)
            lbl = self.font_sm.render(str(j), True, AXIS_COLOR)
            self.screen.blit(lbl, (ARENA_MARGIN - 24, sy - 7))

    def _draw_right_panel(self):
        rp = ARENA_PANEL_W
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         pygame.Rect(rp, 0, RIGHT_W, WIN_H))

        title = self.font_xl.render("EDITOR", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(
            centerx=rp + RIGHT_W // 2, y=6))

        # Section headers
        sections = [
            (8,   "── Ukuran Arena ──"),
            (84,  "── Langkah Robot ──"),
            (160, "── Ukuran Robot ──"),
            (236, "── Posisi Awal Robot ──"),
            (316, "── Mode Obstacle ──"),
            (368, "── Parameter Lidar ──"),
        ]
        for (dy, txt) in sections:
            lbl = self.font_sm.render(txt, True, (80, 100, 150))
            self.screen.blit(lbl, (rp + 20, dy))

        # Gambar semua input
        for inp in self.all_inputs:
            inp.draw(self.screen)
        self.chk_obstacle.draw(self.screen)

        # Info cell size
        lbl = self.font_sm.render(
            f"Cell obstacle = {self.step_lin*100:.0f}cm × {self.step_lin*100:.0f}cm",
            True, (100, 120, 160))
        self.screen.blit(lbl, (rp + 20, 348))

        # Tombol
        for btn in self.all_buttons:
            btn.draw(self.screen, self.font_md)

        # Info obstacle count
        info = self.font_sm.render(
            f"Obstacles: {len(self.obstacles)}  |  "
            f"Arena: {self.arena_w:.1f}×{self.arena_h:.1f} m",
            True, (100, 120, 160))
        self.screen.blit(info, info.get_rect(
            centerx=rp + RIGHT_W // 2, y=WIN_H - 18))