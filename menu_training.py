"""
menu_training.py — Menu 4: DQN Training.
Toggle visualisasi on/off, log ke panel kanan, kontrol start/stop.
"""

import sys, math, json, os, threading, time, random
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
    Button, TextInput, Checkbox, Toast, draw_robot_sprite,
    draw_arena_background, draw_obstacle_rect,
)
from core.lidar          import LidarSensor   # subsistem bersama (pygame-free)
from core.training_loop  import TrainingConfig, run_training  # subsistem bersama
from dqn_agent  import (
    DQNAgent, ACTIONS, N_ACTIONS,
    make_state, compute_reward,
    DEFAULT_GAMMA, DEFAULT_LR, DEFAULT_EPS_START,
    DEFAULT_EPS_END, DEFAULT_EPS_DECAY, DEFAULT_TARGET_UPDATE,
    REWARD_GOAL,
)

# ══════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════
ARENA_PANEL_W = 620
ARENA_MARGIN  = 30
RIGHT_W       = 420
WIN_W         = ARENA_PANEL_W + RIGHT_W
WIN_H         = 760
FPS           = 60
ARENA_PX      = ARENA_PANEL_W - 2 * ARENA_MARGIN

BTN_W   = 370
BTN_H   = 38
INP_W   = 170
INP_H   = 26

GOAL_COLOR    = (255, 215,  0)
LIDAR_FREE    = (60,  200, 130)
LIDAR_HIT     = (255, 100,  50)
LOG_MAX_LINES = 18
TRAIN_STEPS_PER_FRAME = 8   # langkah RL per frame (mode visualisasi)
GOAL_RADIUS   = 0.5         # meter


def _tk_root():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True); return root


class TrainingMenu:
    def __init__(self, map_path=None):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("DQN Training — Menu 4")

        self.font_xl   = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_lg   = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_md   = pygame.font.SysFont("monospace", 12)
        self.font_sm   = pygame.font.SysFont("monospace", 10)
        self.font_mono = pygame.font.SysFont("monospace", 11)

        self.clock  = pygame.time.Clock()
        self.result = None
        self.toast  = Toast(self.font_lg)

        # ── State ──────────────────────────────────
        self.training     = False      # sedang training?
        self.viz_on       = True       # toggle visualisasi
        self.map_name     = "Default"
        self.log_lines    : list[str] = []
        self.log_lock     = threading.Lock()

        # ── Env & lidar ────────────────────────────
        self.env   = RobotEnvironment()
        self.env.reset()
        self._update_scale()
        self.trail_surf  = self._new_trail()
        self.lidar       = LidarSensor(self.env.lidar_num_rays, self.env.lidar_max_range)
        self.lidar_data  : list[dict] = []

        # ── Goal ───────────────────────────────────
        self.goal_x, self.goal_y = self._random_goal()
        self.prev_dist           = self._dist_to_goal()

        # ── Agent ──────────────────────────────────
        state_dim   = self.lidar.num_rays + 3
        self.agent  = DQNAgent(state_dim=state_dim)
        self._do_scan()

        # ── Training thread ────────────────────────
        self._train_thread  : threading.Thread | None = None
        self._stop_event    = threading.Event()
        self._shared         = {   # data dibagi thread ↔ UI
            "episode":      0,
            "step":         0,
            "ep_reward":    0.0,
            "last_reward":  0.0,
            "epsilon":      self.agent.eps,
            "avg_r100":     0.0,
            "avg_loss100":  0.0,
            "total_steps":  0,
            "goals_reached":0,
        }
        self._vis_lock       = threading.Lock()
        self._vis_state      = None   # snapshot untuk visualisasi

        self._build_widgets()
        if map_path: self._load_map(map_path)

    # ── helpers ──────────────────────────────────
    def _update_scale(self):
        self.scale = min(ARENA_PX / self.env.arena_w, ARENA_PX / self.env.arena_h)

    def _new_trail(self):
        s = pygame.Surface((ARENA_PANEL_W, WIN_H), pygame.SRCALPHA)
        s.fill((0,0,0,0)); return s

    def w2s(self, wx, wy):
        return (int(ARENA_MARGIN + wx * self.scale),
                int(ARENA_MARGIN + (self.env.arena_h - wy) * self.scale))

    def _do_scan(self):
        r = self.env.robot
        self.lidar_data = self.lidar.scan(
            r.x, r.y, r.theta, self.env.obstacles,
            self.env.arena_w, self.env.arena_h)

    def _random_goal(self):
        margin = max(self.env.robot.radius + 0.5, 1.0)
        for _ in range(200):
            gx = random.uniform(margin, self.env.arena_w  - margin)
            gy = random.uniform(margin, self.env.arena_h - margin)
            # Pastikan tidak menimpa obstacle
            clear = True
            for obs in self.env.obstacles:
                if obs.collides_with_circle(gx, gy, GOAL_RADIUS + 0.1):
                    clear = False; break
            if clear: return gx, gy
        return self.env.arena_w / 2, self.env.arena_h / 2

    def _dist_to_goal(self):
        r = self.env.robot
        return math.hypot(self.goal_x - r.x, self.goal_y - r.y)

    def _reached_goal(self):
        return self._dist_to_goal() < GOAL_RADIUS

    # ── build widgets ────────────────────────────
    def _build_widgets(self):
        rp  = ARENA_PANEL_W
        cx  = rp + RIGHT_W // 2
        bx  = cx - BTN_W // 2
        ix  = rp + 18
        iw  = RIGHT_W - 36
        half= (iw - 8) // 2

        y = 10

        # ── Toggle visualisasi ──
        self.chk_viz = Checkbox(
            ix, y + 4, 20, "Tampilkan Visualisasi (matikan untuk training cepat)",
            self.font_sm, checked=True,
            action=lambda v: setattr(self, "viz_on", v))

        y += 34
        # ── Hyperparameter ──
        self.inp_episodes = TextInput(ix,        y+16, half, INP_H, self.font_md,
                                       label="Jumlah Episode", default="500",
                                       numeric=True, min_val=1, max_val=100000)
        self.inp_max_steps = TextInput(ix+half+8, y+16, half, INP_H, self.font_md,
                                        label="Max Steps/Episode", default="300",
                                        numeric=True, min_val=10, max_val=5000)
        y += 56
        self.inp_gamma = TextInput(ix,        y+16, half, INP_H, self.font_md,
                                    label="Gamma (γ)", default=str(DEFAULT_GAMMA),
                                    numeric=True, min_val=0.5, max_val=0.9999)
        self.inp_lr    = TextInput(ix+half+8, y+16, half, INP_H, self.font_md,
                                    label="Learning Rate", default=str(DEFAULT_LR),
                                    numeric=True, min_val=1e-6, max_val=0.1)
        y += 56
        self.inp_eps_start = TextInput(ix,        y+16, half, INP_H, self.font_md,
                                        label="Epsilon Start", default=str(DEFAULT_EPS_START),
                                        numeric=True, min_val=0.0, max_val=1.0)
        self.inp_eps_end   = TextInput(ix+half+8, y+16, half, INP_H, self.font_md,
                                        label="Epsilon End", default=str(DEFAULT_EPS_END),
                                        numeric=True, min_val=0.0, max_val=1.0)
        y += 56
        self.inp_eps_decay = TextInput(ix,        y+16, half, INP_H, self.font_md,
                                        label="Epsilon Decay", default=str(DEFAULT_EPS_DECAY),
                                        numeric=True, min_val=0.8, max_val=0.9999)
        self.inp_save_path = TextInput(ix+half+8, y+16, half, INP_H, self.font_md,
                                        label="Simpan ke (.pth)", default="model.pth",
                                        numeric=False)
        y += 56

        # ── Tombol load map, start, stop, save, exit ──
        bw_half = (BTN_W - 8) // 2
        self.btn_load_map = Button(bx, y,        BTN_W, BTN_H, "Load Map",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS, action=self._action_load_map)
        y += BTN_H + 6
        self.btn_start = Button(bx,          y, bw_half, BTN_H, "▶  START",
            color_normal=(35,80,35), color_hover=(55,120,55),
            color_press=(20,55,20), action=self._action_start)
        self.btn_stop  = Button(bx+bw_half+8,y, bw_half, BTN_H, "■  STOP",
            color_normal=(80,35,35), color_hover=(120,55,55),
            color_press=(55,20,20), action=self._action_stop)
        y += BTN_H + 6
        self.btn_save_model = Button(bx, y, BTN_W, BTN_H, "Simpan Model",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS, action=self._action_save_model)
        y += BTN_H + 6
        self.btn_exit = Button(bx, y, BTN_W, BTN_H, "Kembali ke Menu [ESC]",
            color_normal=BTN_EXIT_NORMAL, color_hover=BTN_EXIT_HOVER,
            color_press=BTN_EXIT_PRESS, action=self._action_exit)

        # ── Log area dimulai setelah tombol ──
        self._log_y_start = y + BTN_H + 12

        self.all_inputs  = [self.inp_episodes, self.inp_max_steps,
                            self.inp_gamma, self.inp_lr,
                            self.inp_eps_start, self.inp_eps_end,
                            self.inp_eps_decay, self.inp_save_path]
        self.all_buttons = [self.btn_load_map, self.btn_start, self.btn_stop,
                            self.btn_save_model, self.btn_exit]

    # ── aksi ─────────────────────────────────────
    def _action_load_map(self):
        root = _tk_root()
        path = filedialog.askopenfilename(
            title="Pilih map", filetypes=[("JSON","*.json"),("All","*.*")])
        root.destroy()
        if path: self._load_map(path)

    def _load_map(self, path):
        try:
            with open(path) as f: data = json.load(f)
            self.env = RobotEnvironment.from_dict(data)
            self.env.reset(); self._update_scale()
            self.trail_surf = self._new_trail()
            self.lidar = LidarSensor(self.env.lidar_num_rays, self.env.lidar_max_range)
            self.goal_x, self.goal_y = self._random_goal()
            self.prev_dist = self._dist_to_goal()
            self._do_scan()
            # Re-init agent dengan state_dim baru
            new_state_dim = self.lidar.num_rays + 3
            if new_state_dim != self.agent.state_dim:
                self.agent = DQNAgent(state_dim=new_state_dim)
            self.map_name = os.path.basename(path)
            self.toast.show(f"Map: {self.map_name}", SUCCESS_COLOR)
        except Exception as e:
            self.toast.show(f"Gagal load: {e}", WARNING_COLOR)

    def _action_start(self):
        if self.training: return
        # Baca hyperparameter
        episodes  = int(self.inp_episodes.get_float(500))
        max_steps = int(self.inp_max_steps.get_float(300))
        gamma     = self.inp_gamma.get_float(DEFAULT_GAMMA)
        lr        = self.inp_lr.get_float(DEFAULT_LR)
        eps_start = self.inp_eps_start.get_float(DEFAULT_EPS_START)
        eps_end   = self.inp_eps_end.get_float(DEFAULT_EPS_END)
        eps_decay = self.inp_eps_decay.get_float(DEFAULT_EPS_DECAY)

        state_dim  = self.lidar.num_rays + 3
        self.agent = DQNAgent(state_dim=state_dim, gamma=gamma, lr=lr,
                              eps_start=eps_start, eps_end=eps_end,
                              eps_decay=eps_decay)
        self._stop_event.clear()
        self.training = True
        self.log_lines.clear()
        self._log(f"Training dimulai: {episodes} ep, max {max_steps} steps/ep")
        self._log(f"γ={gamma}  lr={lr}  ε {eps_start}→{eps_end}  decay={eps_decay}")
        self._log(f"State dim={state_dim}  Actions={N_ACTIONS}")

        self._train_thread = threading.Thread(
            target=self._train_loop,
            args=(episodes, max_steps),
            daemon=True)
        self._train_thread.start()

    def _action_stop(self):
        if not self.training: return
        self._stop_event.set()
        self._log("── Training dihentikan oleh user ──")

    def _action_save_model(self):
        path = self.inp_save_path.get_str() or "model.pth"
        if not path.endswith(".pth"): path += ".pth"
        try:
            self.agent.save(path)
            self.toast.show(f"Model disimpan: {path}", SUCCESS_COLOR)
            self._log(f"Model disimpan: {path}")
        except Exception as e:
            self.toast.show(f"Gagal simpan: {e}", WARNING_COLOR)

    def _action_exit(self):
        self._stop_event.set()
        self.result = "main"

    # ── training loop (berjalan di thread terpisah) ──
    def _train_loop(self, total_episodes: int, max_steps: int):
        """
        Delegasikan seluruh logika ke core.training_loop.run_training.
        GUI menerima update via callback on_episode_end dan on_step.
        """
        cfg = TrainingConfig(
            total_episodes = total_episodes,
            max_steps      = max_steps,
            gamma          = self.agent.gamma,
            lr             = self.inp_lr.get_float(DEFAULT_LR),
            eps_start      = self.agent.eps,
            eps_end        = self.agent.eps_end,
            eps_decay      = self.agent.eps_decay,
            target_update  = self.agent.target_update,
            log_every      = 10,
            auto_save_path = "",   # GUI save manual via tombol
        )

        def _on_episode(info: dict):
            goals = info["goals_reached"]
            self._shared.update({
                "episode":       info["episode"],
                "step":          info["steps_in_ep"],
                "ep_reward":     info["ep_reward"],
                "epsilon":       info["epsilon"],
                "avg_r100":      info["avg_r100"],
                "avg_loss100":   info["avg_loss100"],
                "total_steps":   info["total_steps"],
                "goals_reached": goals,
            })
            ep = info["episode"]
            if ep % 10 == 0 or ep <= 5:
                self._log(
                    f"[Ep {ep:4d}/{total_episodes}] "
                    f"R={info['ep_reward']:7.1f}  "
                    f"avgR={info['avg_r100']:7.1f}  "
                    f"ε={info['epsilon']:.3f}  "
                    f"loss={info['avg_loss100']:.4f}")

        def _on_step(sd: dict):
            if self.viz_on:
                with self._vis_lock:
                    self._vis_state = {
                        "robot_x":     sd["robot_x"],
                        "robot_y":     sd["robot_y"],
                        "robot_theta": sd["robot_theta"],
                        "lidar":       sd["lidar"],
                        "goal_x":      sd["goal_x"],
                        "goal_y":      sd["goal_y"],
                    }

        run_training(
            self.env,
            cfg,
            agent          = self.agent,
            stop_event     = self._stop_event,
            on_episode_end = _on_episode,
            on_step        = _on_step,
        )

        self.training = False
        self._log("═══ Training selesai ═══")
        save_path = self.inp_save_path.get_str() or "model.pth"
        if not save_path.endswith(".pth"):
            save_path += ".pth"
        try:
            self.agent.save(save_path)
            self._log(f"Model auto-disimpan → {save_path}")
        except Exception as e:
            self._log(f"Auto-save gagal: {e}")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        with self.log_lock:
            self.log_lines.append(f"[{ts}] {msg}")
            if len(self.log_lines) > 200:
                self.log_lines = self.log_lines[-200:]

    # ── run loop ─────────────────────────────────
    def run(self):
        while self.result is None:
            self._handle_events()
            self._update()
            if self.viz_on:
                self._draw_full()
            else:
                self._draw_log_only()
            self.clock.tick(FPS)
        return self.result

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._stop_event.set(); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self._action_exit()
            for btn in self.all_buttons: btn.handle_event(event)
            for inp in self.all_inputs:  inp.handle_event(event)
            self.chk_viz.handle_event(event)

    def _update(self):
        for btn in self.all_buttons: btn.update()
        self.toast.update()

    # ── draw mode visualisasi ──────────────────────
    def _draw_full(self):
        self.screen.fill(BG_COLOR)

        # Snapshot dari thread
        snap = None
        if self.training:
            with self._vis_lock:
                snap = self._vis_state

        self._draw_arena()
        self._draw_obstacles()
        if snap:
            self._draw_goal_on_arena(snap["goal_x"], snap["goal_y"])
            self._draw_lidar_on_arena(snap["robot_x"], snap["robot_y"],
                                      snap["robot_theta"], snap["lidar"])
            self._draw_robot_on_arena(snap["robot_x"], snap["robot_y"],
                                      snap["robot_theta"])
        else:
            self._draw_goal_on_arena(self.goal_x, self.goal_y)
            r = self.env.robot
            self._draw_robot_on_arena(r.x, r.y, r.theta)

        self._draw_info_strip()
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (ARENA_PANEL_W,0),(ARENA_PANEL_W,WIN_H),2)
        self._draw_right_panel()
        self.toast.draw(self.screen, WIN_W//2, WIN_H-10)
        pygame.display.flip()

    def _draw_log_only(self):
        """Mode log-only: seluruh layar dipakai untuk log + panel parameter."""
        self.screen.fill(BG_COLOR)
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         pygame.Rect(0, 0, WIN_W, WIN_H))

        # Status bar atas
        sh = self._shared
        status_col = (80,220,120) if self.training else (150,150,170)
        status_txt = "● TRAINING..." if self.training else "○ Idle"
        self.screen.blit(self.font_xl.render(status_txt, True, status_col),
                         (20, 10))
        self.screen.blit(self.font_lg.render(
            f"Ep {sh['episode']}  |  "
            f"AvgR={sh['avg_r100']:.1f}  |  "
            f"ε={sh['epsilon']:.3f}  |  "
            f"Loss={sh['avg_loss100']:.4f}  |  "
            f"Goals={sh['goals_reached']}",
            True, HEADING_COLOR), (20, 38))

        # Log panel
        self._draw_log_panel(pygame.Rect(20, 68, WIN_W-440, WIN_H-90))

        # Panel kanan (tombol + params)
        rp = WIN_W - 420
        pygame.draw.rect(self.screen, (18,18,30), pygame.Rect(rp, 0, 420, WIN_H))
        pygame.draw.line(self.screen, DIVIDER_COLOR, (rp,0),(rp,WIN_H),2)
        self._draw_right_panel_inner(rp)

        self.toast.draw(self.screen, WIN_W//2, WIN_H-10)
        pygame.display.flip()

    # ── sub-draw arena ────────────────────────────
    def _draw_arena(self):
        aw = int(self.env.arena_w*self.scale); ah = int(self.env.arena_h*self.scale)
        r  = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, aw, ah)
        draw_arena_background(self.screen, r)
        pygame.draw.rect(self.screen, ARENA_BORDER, r, 2)
        for i in range(int(self.env.arena_w)+1):
            x = ARENA_MARGIN + int(i*self.scale)
            pygame.draw.line(self.screen, GRID_COLOR, (x,ARENA_MARGIN),(x,ARENA_MARGIN+ah),1)
        for j in range(int(self.env.arena_h)+1):
            y = ARENA_MARGIN + int(j*self.scale)
            pygame.draw.line(self.screen, GRID_COLOR, (ARENA_MARGIN,y),(ARENA_MARGIN+aw,y),1)

    def _draw_obstacles(self):
        for obs in self.env.obstacles:
            sx,sy = self.w2s(obs.x, obs.y+obs.h)
            pw = max(2,int(obs.w*self.scale)); ph = max(2,int(obs.h*self.scale))
            draw_obstacle_rect(self.screen, pygame.Rect(sx, sy, pw, ph))

    def _draw_goal_on_arena(self, gx, gy):
        sx, sy = self.w2s(gx, gy)
        gr     = max(4, int(GOAL_RADIUS * self.scale))
        pygame.draw.circle(self.screen, GOAL_COLOR, (sx, sy), gr)          # filled
        pygame.draw.circle(self.screen, (180, 140, 0), (sx, sy), gr, 2)    # border lebih gelap
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 3)      # titik tengah putih

    def _draw_lidar_on_arena(self, rx, ry, theta, ld):
        cx,cy = self.w2s(rx,ry)
        for ray in ld:
            ex,ey = self.w2s(ray["end_x"],ray["end_y"])
            c = LIDAR_HIT if ray["hit"] else LIDAR_FREE
            pygame.draw.line(self.screen, c, (cx,cy),(ex,ey),1)

    def _draw_robot_on_arena(self, rx, ry, theta):
        cx, cy = self.w2s(rx, ry)
        r_px   = max(4, int(self.env.robot.radius * self.scale))
        draw_robot_sprite(self.screen, cx, cy, r_px, theta)

    def _draw_info_strip(self):
        py = ARENA_MARGIN + int(self.env.arena_h*self.scale) + 6
        sh = self._shared
        self.screen.blit(self.font_lg.render(
            f"DQN TRAINING  ·  {self.map_name}", True, HEADING_COLOR),(ARENA_MARGIN,py))
        col = (80,220,120) if self.training else (150,150,170)
        txt = "● TRAINING" if self.training else "○ Idle"
        self.screen.blit(self.font_md.render(txt, True, col),(ARENA_MARGIN, py+20))
        self.screen.blit(self.font_md.render(
            f"Ep={sh['episode']}  R={sh['ep_reward']:.1f}  "
            f"avgR={sh['avg_r100']:.1f}  ε={sh['epsilon']:.3f}",
            True, TEXT_COLOR),(ARENA_MARGIN+90, py+20))
        self.screen.blit(self.font_sm.render(
            f"Loss={sh['avg_loss100']:.4f}  Steps={sh['total_steps']}  "
            f"Goals={sh['goals_reached']}",
            True, AXIS_COLOR),(ARENA_MARGIN, py+38))

    # ── panel kanan ───────────────────────────────
    def _draw_right_panel(self):
        rp = ARENA_PANEL_W
        pygame.draw.rect(self.screen, PANEL_COLOR, pygame.Rect(rp,0,RIGHT_W,WIN_H))
        self._draw_right_panel_inner(rp)

    def _draw_right_panel_inner(self, rp):
        title = self.font_xl.render("DQN TRAINING", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(centerx=rp+RIGHT_W//2, y=6))

        self.chk_viz.draw(self.screen)

        # Section headers
        for (txt, y) in [("── Parameter Training ──", 44),
                          ("── Simpan / Load ──",      self._log_y_start-18)]:
            lbl = self.font_sm.render(txt, True, (80,100,150))
            self.screen.blit(lbl, (rp+18, y))

        for inp in self.all_inputs: inp.draw(self.screen)
        for btn in self.all_buttons: btn.draw(self.screen, self.font_md)

        # Log di bawah tombol
        log_rect = pygame.Rect(rp+8, self._log_y_start, RIGHT_W-16, WIN_H-self._log_y_start-8)
        self._draw_log_panel(log_rect)

    def _draw_log_panel(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, (14,14,26), rect, border_radius=4)
        pygame.draw.rect(self.screen, (40,50,80), rect, 1, border_radius=4)
        with self.log_lock:
            visible = self.log_lines[-LOG_MAX_LINES:]
        line_h = self.font_mono.get_height() + 1
        for i, line in enumerate(visible):
            y_pos = rect.y + 4 + i*line_h
            if y_pos + line_h > rect.bottom: break
            color = (80,220,120) if "selesai" in line or "disimpan" in line \
                    else (255,160,60) if "Gagal" in line or "dihentikan" in line \
                    else (180,190,210)
            # Clip text ke lebar box
            txt_surf = self.font_mono.render(line[:68], True, color)
            self.screen.blit(txt_surf, (rect.x+4, y_pos))