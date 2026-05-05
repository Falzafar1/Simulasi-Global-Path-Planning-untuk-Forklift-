"""
menu_inference.py — Menu 5: DQN Inference / Demo.

Menjalankan model DQN yang sudah dilatih (.pth) di atas peta pilihan.
Fitur:
    - Load model (.pth) + Load map (.json)
    - Pilih goal dengan klik kiri di arena
    - Jalankan / pause / reset episode
    - Visualisasi lidar, trail, dan statistik real-time
    - Kontrol kecepatan playback (step delay)
"""

import sys, math, json, os, time, random
import pygame
import tkinter as tk
from tkinter import filedialog

import numpy as np

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
    BTN_WARN_NORMAL, BTN_WARN_HOVER, BTN_WARN_PRESS,
    Button, Toast, draw_robot_sprite,
    draw_arena_background, draw_obstacle_rect,
)
from core.lidar          import LidarSensor          # subsistem bersama (pygame-free)
from core.inference_loop import run_episode, random_goal  # subsistem bersama
from dqn_agent  import DQNAgent, ACTIONS, N_ACTIONS, make_state

# ══════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════
ARENA_PANEL_W = 620
ARENA_MARGIN  = 30
RIGHT_W       = 380
WIN_W         = ARENA_PANEL_W + RIGHT_W
WIN_H         = 760
FPS           = 60
ARENA_PX      = ARENA_PANEL_W - 2 * ARENA_MARGIN

BTN_W = 320
BTN_H = 38

GOAL_COLOR    = (0, 128,   0)
LIDAR_FREE    = ( 60, 200, 130)
LIDAR_HIT     = (255, 100,  50)
GOAL_RADIUS   = 0.5          # meter
TRAIL_ALPHA   = 160

# Delay antar step (ms) — lebih besar = lebih lambat
SPEED_OPTIONS = [
    ("0.5×", 200),
    ("1×",    50),
    ("2×",    20),
    ("4×",     5),
    ("Max",    0),
]

MAX_TRAIL = 800              # titik trail disimpan
LOG_MAX   = 20


def _tk_root():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True); return root


# ══════════════════════════════════════════════
#  MENU INFERENCE
# ══════════════════════════════════════════════
class InferenceMenu:
    def __init__(self, map_path: str = None, model_path: str = None):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("DQN Inference — Menu 5")

        self.font_xl   = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_lg   = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_md   = pygame.font.SysFont("monospace", 12)
        self.font_sm   = pygame.font.SysFont("monospace", 10)
        self.font_mono = pygame.font.SysFont("monospace", 11)

        self.clock  = pygame.time.Clock()
        self.result = None
        self.toast  = Toast(self.font_lg)

        # ── State inference ────────────────────────
        self.running      = False    # sedang play?
        self.model_loaded = False
        self.map_name     = "Default"
        self.model_name   = "—"
        self.agent        = None

        # ── Env & lidar ────────────────────────────
        self.env  = RobotEnvironment()
        self.env.reset()
        self._update_scale()
        self.lidar      = LidarSensor(self.env.lidar_num_rays, self.env.lidar_max_range)
        self.lidar_data : list[dict] = []

        # ── Goal ───────────────────────────────────
        self.goal_x, self.goal_y = self._random_goal()
        self.goal_pending        = False   # menunggu klik user?

        # ── Trail ──────────────────────────────────
        self.trail : list[tuple[float, float]] = []

        # ── Stats ──────────────────────────────────
        self.ep_steps   = 0
        self.ep_reward  = 0.0
        self.ep_count   = 0
        self.goals_hit  = 0
        self.collisions = 0
        self.log_lines  : list[str] = []

        # ── Timing ─────────────────────────────────
        self.speed_idx       = 1          # index ke SPEED_OPTIONS
        self._last_step_ms   = 0

        # ── Current state vector ───────────────────
        self._do_scan()
        self._state = self._build_state()

        self._build_widgets()

        if map_path:   self._load_map(map_path)
        if model_path: self._load_model(model_path)

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════
    def _update_scale(self):
        self.scale = min(ARENA_PX / self.env.arena_w, ARENA_PX / self.env.arena_h)

    def w2s(self, wx, wy):
        """World coord → screen pixel."""
        return (int(ARENA_MARGIN + wx * self.scale),
                int(ARENA_MARGIN + (self.env.arena_h - wy) * self.scale))

    def s2w(self, sx, sy):
        """Screen pixel → world coord (untuk klik goal)."""
        wx = (sx - ARENA_MARGIN) / self.scale
        wy = self.env.arena_h - (sy - ARENA_MARGIN) / self.scale
        return wx, wy

    def _arena_rect(self) -> pygame.Rect:
        return pygame.Rect(ARENA_MARGIN, ARENA_MARGIN,
                           int(self.env.arena_w * self.scale),
                           int(self.env.arena_h * self.scale))

    def _do_scan(self):
        r = self.env.robot
        self.lidar_data = self.lidar.scan(
            r.x, r.y, r.theta,
            self.env.obstacles,
            self.env.arena_w, self.env.arena_h)

    def _build_state(self) -> np.ndarray:
        r = self.env.robot
        return make_state(self.lidar_data, self.lidar.max_range,
                          r.x, r.y, r.theta,
                          self.goal_x, self.goal_y)

    def _dist_to_goal(self) -> float:
        r = self.env.robot
        return math.hypot(self.goal_x - r.x, self.goal_y - r.y)

    def _reached_goal(self) -> bool:
        return self._dist_to_goal() < GOAL_RADIUS

    def _random_goal(self):
        """Delegasi ke core.inference_loop.random_goal."""
        return random_goal(self.env)

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-200:]

    # ══════════════════════════════════════════
    #  BUILD WIDGETS
    # ══════════════════════════════════════════
    def _build_widgets(self):
        rp  = ARENA_PANEL_W
        cx  = rp + RIGHT_W // 2
        bx  = cx - BTN_W // 2
        ix  = rp + 18

        y = 48   # di bawah judul

        # ── Baris 1: load map + load model ──
        bw2 = (BTN_W - 8) // 2
        self.btn_load_map = Button(
            bx, y, bw2, BTN_H, "Load Map",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS, action=self._action_load_map)
        self.btn_load_model = Button(
            bx + bw2 + 8, y, bw2, BTN_H, "Load Model",
            color_normal=(40, 80, 60), color_hover=(60, 120, 90),
            color_press=(25, 55, 40), action=self._action_load_model)
        y += BTN_H + 8

        # ── Baris 2: set goal klik / acak ──
        self.btn_set_goal = Button(
            bx, y, bw2, BTN_H, "Set Goal (klik)",
            color_normal=BTN_WARN_NORMAL, color_hover=BTN_WARN_HOVER,
            color_press=BTN_WARN_PRESS, action=self._action_set_goal_click)
        self.btn_rand_goal = Button(
            bx + bw2 + 8, y, bw2, BTN_H, "Goal Acak",
            color_normal=BTN_NORMAL, color_hover=BTN_HOVER,
            color_press=BTN_PRESS, action=self._action_rand_goal)
        y += BTN_H + 8

        # ── Baris 3: play / pause ──
        self.btn_play = Button(
            bx, y, bw2, BTN_H, "▶  PLAY",
            color_normal=(35, 80, 35), color_hover=(55, 120, 55),
            color_press=(20, 55, 20), action=self._action_play)
        self.btn_pause = Button(
            bx + bw2 + 8, y, bw2, BTN_H, "⏸  PAUSE",
            color_normal=(80, 70, 20), color_hover=(120, 105, 30),
            color_press=(55, 48, 12), action=self._action_pause)
        y += BTN_H + 8

        # ── Baris 4: reset episode ──
        self.btn_reset = Button(
            bx, y, BTN_W, BTN_H, "↺  Reset Episode",
            color_normal=BTN_RESET_NORMAL, color_hover=BTN_RESET_HOVER,
            color_press=BTN_RESET_PRESS, action=self._action_reset)
        y += BTN_H + 8

        # ── Baris 5: kecepatan ──
        speed_w = BTN_W // len(SPEED_OPTIONS)
        self.speed_btns = []
        for i, (label, _) in enumerate(SPEED_OPTIONS):
            c_n = (55, 45, 80) if i != self.speed_idx else (80, 60, 130)
            c_h = (80, 70, 120)
            c_p = (35, 28, 55)
            b = Button(bx + i * speed_w, y, speed_w - 2, BTN_H - 8, label,
                       color_normal=c_n, color_hover=c_h,
                       color_press=c_p, action=lambda idx=i: self._set_speed(idx))
            self.speed_btns.append(b)
        y += BTN_H + 2

        # ── Separator label ──
        self._sep_y = y
        y += 16

        # ── Baris 6: exit ──
        self.btn_exit = Button(
            bx, WIN_H - BTN_H - 12, BTN_W, BTN_H, "Kembali ke Menu [ESC]",
            color_normal=BTN_EXIT_NORMAL, color_hover=BTN_EXIT_HOVER,
            color_press=BTN_EXIT_PRESS, action=self._action_exit)

        # ── Log area ──
        self._log_y_start = y

        self.all_buttons = [
            self.btn_load_map, self.btn_load_model,
            self.btn_set_goal, self.btn_rand_goal,
            self.btn_play, self.btn_pause,
            self.btn_reset, self.btn_exit,
        ] + self.speed_btns

    # ══════════════════════════════════════════
    #  AKSI
    # ══════════════════════════════════════════
    def _action_load_map(self):
        root = _tk_root()
        path = filedialog.askopenfilename(
            title="Pilih map", filetypes=[("JSON","*.json"),("All","*.*")])
        root.destroy()
        if path: self._load_map(path)

    def _load_map(self, path: str):
        try:
            with open(path) as f: data = json.load(f)
            self.env = RobotEnvironment.from_dict(data)
            self.env.reset()
            self._update_scale()
            self.lidar = LidarSensor(self.env.lidar_num_rays, self.env.lidar_max_range)
            self.goal_x, self.goal_y = self._random_goal()
            self.trail.clear()
            self._do_scan()
            self._state = self._build_state()
            self.map_name = os.path.basename(path)
            # Jika model sudah ada dan state_dim tidak cocok, hapus model
            if self.agent and self.agent.state_dim != self.lidar.num_rays + 3:
                self.agent = None
                self.model_loaded = False
                self.model_name   = "—"
                self.toast.show("Map berubah — muat ulang model!", WARNING_COLOR)
            else:
                self.toast.show(f"Map: {self.map_name}", SUCCESS_COLOR)
            self._log(f"Map dimuat: {self.map_name}")
            self.running = False
            self._reset_stats()
        except Exception as e:
            self.toast.show(f"Gagal load map: {e}", WARNING_COLOR)

    def _action_load_model(self):
        root = _tk_root()
        path = filedialog.askopenfilename(
            title="Pilih model (.pth)", filetypes=[("PyTorch","*.pth"),("All","*.*")])
        root.destroy()
        if path: self._load_model(path)

    def _load_model(self, path: str):
        try:
            agent = DQNAgent.load(path)
            expected_dim = self.lidar.num_rays + 3
            if agent.state_dim != expected_dim:
                self.toast.show(
                    f"State dim tidak cocok! Model={agent.state_dim}, "
                    f"Env={expected_dim}", WARNING_COLOR, duration=200)
                self._log(f"GAGAL: state_dim model ({agent.state_dim}) "
                          f"≠ env ({expected_dim})")
                return
            self.agent        = agent
            self.model_loaded = True
            self.model_name   = os.path.basename(path)
            self._log(f"Model dimuat: {self.model_name}  "
                      f"(dim={agent.state_dim}, ep={agent.episode_count})")
            self.toast.show(f"Model: {self.model_name}", SUCCESS_COLOR)
            self._action_reset()
        except Exception as e:
            self.toast.show(f"Gagal load model: {e}", WARNING_COLOR)
            self._log(f"GAGAL load model: {e}")

    def _action_set_goal_click(self):
        """Aktifkan mode klik-untuk-set-goal."""
        self.goal_pending = True
        self.running      = False
        self.toast.show("Klik di arena untuk set posisi goal", HIGHLIGHT_COLOR, 150)

    def _action_rand_goal(self):
        self.goal_x, self.goal_y = self._random_goal()
        self._state = self._build_state()
        self._log(f"Goal acak: ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def _action_play(self):
        if not self.model_loaded:
            self.toast.show("Muat model (.pth) terlebih dahulu!", WARNING_COLOR)
            return
        self.running = True

    def _action_pause(self):
        self.running = False

    def _action_reset(self):
        self.running = False
        self.env.reset()
        self.trail.clear()
        self._do_scan()
        self._state = self._build_state()
        self._reset_stats()
        self._log("Episode di-reset")

    def _action_exit(self):
        self.running = False
        self.result  = "main"

    def _set_speed(self, idx: int):
        self.speed_idx = idx
        self._log(f"Kecepatan: {SPEED_OPTIONS[idx][0]}")

    def _reset_stats(self):
        self.ep_steps  = 0
        self.ep_reward = 0.0

    # ══════════════════════════════════════════
    #  STEP INFERENSI
    # ══════════════════════════════════════════
    def _inference_step(self):
        """Satu langkah greedy (tanpa eksplorasi)."""
        if not self.model_loaded or not self.running:
            return

        now = pygame.time.get_ticks()
        delay = SPEED_OPTIONS[self.speed_idx][1]
        if delay > 0 and (now - self._last_step_ms) < delay:
            return
        self._last_step_ms = now

        # Pilih aksi greedy
        action = self.agent.select_action_greedy(self._state)
        move   = ACTIONS[action]

        # Eksekusi di environment
        _, _, done, info = self.env.step_manual(move)
        self.ep_steps += 1

        # Scan lidar baru
        self._do_scan()

        # Trail
        r = self.env.robot
        self.trail.append((r.x, r.y))
        if len(self.trail) > MAX_TRAIL:
            self.trail = self.trail[-MAX_TRAIL:]

        # Reward (hanya untuk display)
        reached = self._reached_goal()
        dist    = self._dist_to_goal()
        if reached:
            self.ep_reward += 100.0
        elif done:
            self.ep_reward -= 10.0
        else:
            self.ep_reward += -0.1 + 0.5 * (getattr(self, "_prev_dist", dist) - dist)
        self._prev_dist = dist

        # Build state berikutnya
        self._state = self._build_state()

        # Terminal?
        if reached:
            self.goals_hit += 1
            self.ep_count  += 1
            self._log(f"[Ep {self.ep_count}] GOAL ✓  steps={self.ep_steps}  R={self.ep_reward:.1f}")
            self._episode_end()
        elif done:
            self.collisions += 1
            self.ep_count   += 1
            self._log(f"[Ep {self.ep_count}] COLLISION ✗  steps={self.ep_steps}  R={self.ep_reward:.1f}")
            self._episode_end()

    def _episode_end(self):
        """Reset env untuk episode berikutnya (tetap di posisi start)."""
        self.env.reset()
        self.trail.clear()
        self.goal_x, self.goal_y = self._random_goal()
        self._do_scan()
        self._state = self._build_state()
        self._reset_stats()

    # ══════════════════════════════════════════
    #  RUN LOOP
    # ══════════════════════════════════════════
    def run(self) -> str:
        while self.result is None:
            self._handle_events()
            self._inference_step()
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
                elif event.key == pygame.K_SPACE:
                    if self.running: self._action_pause()
                    else:            self._action_play()
                elif event.key == pygame.K_r:
                    self._action_reset()

            # Klik kiri di arena → set goal
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.goal_pending:
                    mx, my = event.pos
                    arena  = self._arena_rect()
                    if arena.collidepoint(mx, my):
                        gx, gy = self.s2w(mx, my)
                        gx = max(0.1, min(self.env.arena_w  - 0.1, gx))
                        gy = max(0.1, min(self.env.arena_h  - 0.1, gy))
                        self.goal_x, self.goal_y = gx, gy
                        self.goal_pending = False
                        self._state = self._build_state()
                        self._log(f"Goal di-set: ({gx:.2f}, {gy:.2f})")
                        self.toast.show(f"Goal: ({gx:.2f}, {gy:.2f})", SUCCESS_COLOR)

            for btn in self.all_buttons:
                btn.handle_event(event)

        for btn in self.all_buttons:
            btn.update()
        self.toast.update()

    # ══════════════════════════════════════════
    #  DRAW
    # ══════════════════════════════════════════
    def _draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_arena()
        self._draw_obstacles()
        self._draw_trail()
        self._draw_goal()
        self._draw_lidar()
        self._draw_robot()
        self._draw_info_strip()

        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (ARENA_PANEL_W, 0), (ARENA_PANEL_W, WIN_H), 2)
        self._draw_right_panel()

        self.toast.draw(self.screen, WIN_W // 2, WIN_H - 10)
        pygame.display.flip()

    # ── arena ──────────────────────────────────
    def _draw_arena(self):
        aw = int(self.env.arena_w * self.scale)
        ah = int(self.env.arena_h * self.scale)
        rect = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, aw, ah)
        draw_arena_background(self.screen, rect)
        pygame.draw.rect(self.screen, ARENA_BORDER, rect, 2)
        for i in range(int(self.env.arena_w) + 1):
            x = ARENA_MARGIN + int(i * self.scale)
            pygame.draw.line(self.screen, GRID_COLOR,
                             (x, ARENA_MARGIN), (x, ARENA_MARGIN + ah), 1)
        for j in range(int(self.env.arena_h) + 1):
            y = ARENA_MARGIN + int(j * self.scale)
            pygame.draw.line(self.screen, GRID_COLOR,
                             (ARENA_MARGIN, y), (ARENA_MARGIN + aw, y), 1)

        # Hint "klik untuk goal"
        if self.goal_pending:
            surf = pygame.Surface((aw, ah), pygame.SRCALPHA)
            surf.fill((255, 215, 0, 30))
            self.screen.blit(surf, (ARENA_MARGIN, ARENA_MARGIN))
            hint = self.font_lg.render("Klik untuk set goal", True, GOAL_COLOR)
            self.screen.blit(hint, hint.get_rect(
                centerx=ARENA_MARGIN + aw // 2,
                centery=ARENA_MARGIN + ah // 2))

    def _draw_obstacles(self):
        for obs in self.env.obstacles:
            sx, sy = self.w2s(obs.x, obs.y + obs.h)
            pw = max(2, int(obs.w * self.scale))
            ph = max(2, int(obs.h * self.scale))
            draw_obstacle_rect(self.screen, pygame.Rect(sx, sy, pw, ph))

    def _draw_trail(self):
        if len(self.trail) < 2:
            return
        pts = [self.w2s(x, y) for x, y in self.trail]
        for i in range(1, len(pts)):
            alpha = int(80 + 175 * i / len(pts))
            pygame.draw.line(self.screen, (*TRAIL_COLOR, alpha), pts[i-1], pts[i], 2)

    def _draw_goal(self):
        sx, sy = self.w2s(self.goal_x, self.goal_y)
        gr = max(4, int(GOAL_RADIUS * self.scale))
        pygame.draw.circle(self.screen, GOAL_COLOR, (sx, sy), gr)          # filled
        pygame.draw.circle(self.screen, (180, 140, 0), (sx, sy), gr, 2)    # border lebih gelap
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 3)      # titik tengah putih
        # Label "G"
        g_lbl = self.font_sm.render("G", True, (20, 20, 20))
        self.screen.blit(g_lbl, (sx + gr + 2, sy - g_lbl.get_height() // 2))


    def _draw_lidar(self):
        r = self.env.robot
        cx, cy = self.w2s(r.x, r.y)
        for ray in self.lidar_data:
            ex, ey = self.w2s(ray["end_x"], ray["end_y"])
            color  = LIDAR_HIT if ray["hit"] else LIDAR_FREE
            pygame.draw.line(self.screen, color, (cx, cy), (ex, ey), 1)

    def _draw_robot(self):
        r      = self.env.robot
        cx, cy = self.w2s(r.x, r.y)
        r_px   = max(4, int(r.radius * self.scale))
        draw_robot_sprite(self.screen, cx, cy, r_px, r.theta)

    def _draw_info_strip(self):
        py = ARENA_MARGIN + int(self.env.arena_h * self.scale) + 6
        status_col = (80, 220, 120) if self.running else (150, 150, 170)
        status_txt = "● RUNNING" if self.running else "○ Paused"
        self.screen.blit(self.font_lg.render(
            f"INFERENCE  ·  {self.map_name}", True, HEADING_COLOR),
            (ARENA_MARGIN, py))
        self.screen.blit(self.font_md.render(status_txt, True, status_col),
                         (ARENA_MARGIN, py + 20))
        self.screen.blit(self.font_md.render(
            f"Model: {self.model_name}   Ep={self.ep_count}  "
            f"Steps={self.ep_steps}  R={self.ep_reward:.1f}",
            True, TEXT_COLOR), (ARENA_MARGIN + 90, py + 20))
        self.screen.blit(self.font_sm.render(
            f"Goals={self.goals_hit}  Collisions={self.collisions}  "
            f"Goal:({self.goal_x:.2f},{self.goal_y:.2f})",
            True, AXIS_COLOR), (ARENA_MARGIN, py + 38))

    # ── panel kanan ──────────────────────────
    def _draw_right_panel(self):
        rp = ARENA_PANEL_W
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         pygame.Rect(rp, 0, RIGHT_W, WIN_H))

        # Judul
        title = self.font_xl.render("DQN INFERENCE", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(centerx=rp + RIGHT_W // 2, y=8))

        # Status model
        mdl_col  = SUCCESS_COLOR if self.model_loaded else WARNING_COLOR
        mdl_txt  = f"Model: {self.model_name}"
        ep_txt   = (f"  (ep={self.agent.episode_count}  "
                    f"avgR={self.agent.avg_reward_100:.1f})"
                    if self.agent else "")
        self.screen.blit(self.font_sm.render(mdl_txt + ep_txt, True, mdl_col),
                         (rp + 10, 32))

        # Tombol
        for btn in self.all_buttons:
            btn.draw(self.screen, self.font_md)

        # Highlight tombol kecepatan aktif
        cx_speed = ARENA_PANEL_W + (RIGHT_W - BTN_W) // 2
        bx = cx_speed
        speed_w = BTN_W // len(SPEED_OPTIONS)
        py_speed = None
        for i, b in enumerate(self.speed_btns):
            if i == self.speed_idx:
                pygame.draw.rect(self.screen, (100, 80, 170),
                                 b.rect, 2, border_radius=6)

        # Separator label kecepatan
        sep_lbl = self.font_sm.render(
            f"── Kecepatan: {SPEED_OPTIONS[self.speed_idx][0]} ──", True, AXIS_COLOR)
        self.screen.blit(sep_lbl,
                         sep_lbl.get_rect(centerx=rp + RIGHT_W // 2,
                                          y=self._sep_y - 14))

        # Shortcut hint
        hints = [
            "SPACE: play/pause",
            "R    : reset episode",
            "ESC  : kembali menu",
        ]
        hy = self._sep_y + 4
        for h in hints:
            self.screen.blit(self.font_sm.render(h, True, AXIS_COLOR),
                             (rp + 16, hy))
            hy += 13

        # Log
        log_rect = pygame.Rect(rp + 8,
                               self._log_y_start + 40,
                               RIGHT_W - 16,
                               self.btn_exit.rect.top - self._log_y_start - 50)
        if log_rect.height > 20:
            self._draw_log(log_rect)

    def _draw_log(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, (14, 14, 26), rect, border_radius=4)
        pygame.draw.rect(self.screen, (40, 50, 80), rect, 1, border_radius=4)
        visible  = self.log_lines[-LOG_MAX:]
        line_h   = self.font_mono.get_height() + 1
        for i, line in enumerate(visible):
            yp = rect.y + 4 + i * line_h
            if yp + line_h > rect.bottom: break
            col = (80, 220, 120) if "GOAL" in line \
                else (255, 120, 80)  if "COLLISION" in line \
                else (180, 190, 210)
            self.screen.blit(self.font_mono.render(line[:52], True, col),
                             (rect.x + 4, yp))