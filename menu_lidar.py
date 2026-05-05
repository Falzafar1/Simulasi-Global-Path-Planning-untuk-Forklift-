"""
menu_lidar.py — Menu 3: Lidar Sensor Visualization
Kiri : arena + robot + ray casting lidar
Kanan: diagram kompas lidar (garis + jarak), parameter, kontrol keyboard

Kontrol:
    ↑ / ↓     : maju / mundur
    ← / →     : rotasi kiri / kanan
    R          : reset
    ESC        : kembali ke menu
"""

import sys, math, json, os
import pygame
import tkinter as tk
from tkinter import filedialog

from core.lidar import LidarSensor  # subsistem bersama (pygame-free)

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

ARENA_PANEL_W = 680
ARENA_MARGIN  = 40
RIGHT_W       = 380
WIN_W         = ARENA_PANEL_W + RIGHT_W
WIN_H         = 760
FPS           = 60
ARENA_PX      = ARENA_PANEL_W - 2 * ARENA_MARGIN

BTN_W = 310
BTN_H = 40

LIDAR_FREE_COLOR  = (60,  220, 160)
LIDAR_HIT_COLOR   = (255, 120,  60)
LIDAR_ENDPOINT    = (255, 200,  50)
COMPASS_BG        = (20,  22,  40)
COMPASS_RING      = (50,  65, 110)
COMPASS_LINE_FREE = (60,  200, 140)
COMPASS_LINE_HIT  = (255, 110,  50)
COMPASS_TEXT_FREE = (100, 230, 170)
COMPASS_TEXT_HIT  = (255, 160,  80)
COMPASS_CENTER    = (120, 180, 255)
DEFEAT_TEXT       = (255, 100,  80)
DEFEAT_SUB        = (255, 200, 180)


def _tk_root():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True); return root


# ══════════════════════════════════════════════════════════
#  MENU LIDAR
# ══════════════════════════════════════════════════════════
class LidarMenu:
    def __init__(self, map_path=None):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Lidar Sensor View — Menu 3")

        self.font_xl   = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_lg   = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_md   = pygame.font.SysFont("monospace", 13)
        self.font_sm   = pygame.font.SysFont("monospace", 11)
        self.font_icon = pygame.font.SysFont("monospace", 18, bold=True)

        self.clock  = pygame.time.Clock()
        self.result = None
        self.toast  = Toast(self.font_lg)

        self.defeat        = False
        self.defeat_alpha  = 0
        self.defeat_show_btns = False
        self.map_name      = "Default"

        self.env = RobotEnvironment()
        self.env.reset()
        self._update_scale()
        self.trail_surf = self._new_trail_surf()

        # Lidar dibaca dari env (default 8 ray, 3m)
        self.lidar      = LidarSensor(self.env.lidar_num_rays, self.env.lidar_max_range)
        self.lidar_data = []

        self._build_widgets()
        self._do_scan()

        if map_path:
            self._load_map(map_path)

    # ── helpers ──────────────────────────────
    def _update_scale(self):
        self.scale = min(ARENA_PX / self.env.arena_w, ARENA_PX / self.env.arena_h)

    def _new_trail_surf(self):
        s = pygame.Surface((ARENA_PANEL_W, WIN_H), pygame.SRCALPHA)
        s.fill((0,0,0,0)); return s

    def w2s(self, wx, wy):
        return (int(ARENA_MARGIN + wx * self.scale),
                int(ARENA_MARGIN + (self.env.arena_h - wy) * self.scale))

    def _do_scan(self):
        r = self.env.robot
        self.lidar_data = self.lidar.scan(
            r.x, r.y, r.theta, self.env.obstacles, self.env.arena_w, self.env.arena_h)

    # ── build widgets ─────────────────────────
    def _build_widgets(self):
        rp = ARENA_PANEL_W
        cx = rp + RIGHT_W // 2
        bx = cx - BTN_W // 2

        self.btn_load = Button(bx, 26, BTN_W, BTN_H, "Load Map",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS, action=self._action_load)

        self.btn_reset = Button(bx, WIN_H - 102, BTN_W, BTN_H, "Reset [R]",
            color_normal=BTN_RESET_NORMAL, color_hover=BTN_RESET_HOVER,
            color_press=BTN_RESET_PRESS, action=self._action_reset)

        self.btn_exit = Button(bx, WIN_H - 54, BTN_W, BTN_H, "Kembali ke Menu [ESC]",
            color_normal=BTN_EXIT_NORMAL, color_hover=BTN_EXIT_HOVER,
            color_press=BTN_EXIT_PRESS, action=self._action_exit)

        self.btn_retry = Button(WIN_W//2 - 110, WIN_H//2 + 60, 220, 50, "Retry [R]",
            color_normal=BTN_RESET_NORMAL, color_hover=BTN_RESET_HOVER,
            color_press=BTN_RESET_PRESS, action=self._action_reset)

        self.btn_def_menu = Button(WIN_W//2 - 110, WIN_H//2 + 124, 220, 44, "Main Menu [ESC]",
            color_normal=BTN_EXIT_NORMAL, color_hover=BTN_EXIT_HOVER,
            color_press=BTN_EXIT_PRESS, action=self._action_exit)

        self.all_buttons = [self.btn_load, self.btn_reset, self.btn_exit]

    # ── aksi ─────────────────────────────────
    def _step(self, move):
        if self.defeat: return
        _, _, done, _ = self.env.step_manual(move)
        self._do_scan()
        if done:
            self.defeat = True
            self.defeat_alpha = 0
            self.defeat_show_btns = False

    def _action_reset(self):
        self.env.reset(); self._update_scale()
        self.trail_surf = self._new_trail_surf()
        self.defeat = False; self.defeat_alpha = 0
        self.defeat_show_btns = False
        self._do_scan()

    def _action_load(self):
        root = _tk_root()
        path = filedialog.askopenfilename(
            title="Pilih file map", filetypes=[("JSON Map","*.json"),("All","*.*")])
        root.destroy()
        if path: self._load_map(path)

    def _load_map(self, path):
        try:
            with open(path) as f: data = json.load(f)
            self.env = RobotEnvironment.from_dict(data)
            self.env.reset(); self._update_scale()
            self.trail_surf = self._new_trail_surf()
            self.defeat = False; self.defeat_alpha = 0
            self.map_name = os.path.basename(path)
            # Baca lidar params dari map
            self.lidar = LidarSensor(self.env.lidar_num_rays, self.env.lidar_max_range)
            self._do_scan()
            self.toast.show(f"Map: {self.map_name}  |  "
                            f"{self.lidar.num_rays} sinar, {self.lidar.max_range:.1f}m",
                            SUCCESS_COLOR)
        except FileNotFoundError:
            self.toast.show(f"File tidak ditemukan!", WARNING_COLOR)
        except Exception as e:
            self.toast.show(f"Gagal load: {e}", WARNING_COLOR)

    def _action_exit(self): self.result = "main"

    # ── run loop ─────────────────────────────
    def run(self):
        while self.result is None:
            self._handle_events(); self._update(); self._draw()
            self.clock.tick(FPS)
        return self.result

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE: self._action_exit()
                elif k == pygame.K_r:    self._action_reset()
                elif not self.defeat:
                    if k == pygame.K_UP:    self._step("forward")
                    elif k == pygame.K_DOWN: self._step("backward")
                    elif k == pygame.K_LEFT: self._step("rotate_left")
                    elif k == pygame.K_RIGHT: self._step("rotate_right")
            if self.defeat and self.defeat_show_btns:
                self.btn_retry.handle_event(event)
                self.btn_def_menu.handle_event(event)
            else:
                for btn in self.all_buttons: btn.handle_event(event)

    def _update(self):
        for btn in self.all_buttons + [self.btn_retry, self.btn_def_menu]: btn.update()
        self.toast.update()
        if self.defeat:
            if self.defeat_alpha < 210: self.defeat_alpha = min(210, self.defeat_alpha + 7)
            if self.defeat_alpha >= 140: self.defeat_show_btns = True
        h = self.env.robot.pose_history
        if len(h) >= 2:
            pygame.draw.line(self.trail_surf, (*TRAIL_COLOR, 160),
                             self.w2s(*h[-2]), self.w2s(*h[-1]), 2)

    # ── draw ─────────────────────────────────
    def _draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_arena(); self._draw_grid(); self._draw_obstacles()
        self.screen.blit(self.trail_surf, (0, 0))
        self._draw_lidar_rays(); self._draw_robot(); self._draw_info_bar()
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (ARENA_PANEL_W, 0), (ARENA_PANEL_W, WIN_H), 2)
        self._draw_right_panel()
        if self.defeat: self._draw_defeat_overlay()
        self.toast.draw(self.screen, WIN_W // 2, WIN_H - 12)
        pygame.display.flip()

    def _draw_arena(self):
        aw, ah = int(self.env.arena_w*self.scale), int(self.env.arena_h*self.scale)
        r = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, aw, ah)
        draw_arena_background(self.screen, r)
        pygame.draw.rect(self.screen, ARENA_BORDER, r, 3)

    def _draw_grid(self):
        step = max(0.5, self.env.robot.step_linear)
        cpx  = max(4, int(step * self.scale))
        aw, ah = int(self.env.arena_w*self.scale), int(self.env.arena_h*self.scale)
        for i in range(int(self.env.arena_w/step)+2):
            x = ARENA_MARGIN + i*cpx
            if x > ARENA_MARGIN+aw: break
            pygame.draw.line(self.screen, GRID_COLOR, (x,ARENA_MARGIN),(x,ARENA_MARGIN+ah),1)
        for j in range(int(self.env.arena_h/step)+2):
            y = ARENA_MARGIN + j*cpx
            if y > ARENA_MARGIN+ah: break
            pygame.draw.line(self.screen, GRID_COLOR, (ARENA_MARGIN,y),(ARENA_MARGIN+aw,y),1)

    def _draw_obstacles(self):
        for obs in self.env.obstacles:
            sx, sy = self.w2s(obs.x, obs.y+obs.h)
            pw = max(2, int(obs.w*self.scale)); ph = max(2, int(obs.h*self.scale))
            draw_obstacle_rect(self.screen, pygame.Rect(sx, sy, pw, ph))

    def _draw_lidar_rays(self):
        robot = self.env.robot
        cx, cy = self.w2s(robot.x, robot.y)
        for ray in self.lidar_data:
            ex, ey = self.w2s(ray["end_x"], ray["end_y"])
            col = LIDAR_HIT_COLOR if ray["hit"] else LIDAR_FREE_COLOR
            pygame.draw.line(self.screen, col, (cx,cy), (ex,ey), 1)
            r = 4 if ray["hit"] else 2
            pygame.draw.circle(self.screen, col, (ex,ey), r)

    def _draw_robot(self):
        robot  = self.env.robot
        cx, cy = self.w2s(robot.x, robot.y)
        r_px   = max(4, int(robot.radius * self.scale))
        draw_robot_sprite(self.screen, cx, cy, r_px, robot.theta,
                          defeat=self.defeat)

    def _draw_info_bar(self):
        robot = self.env.robot
        py    = ARENA_MARGIN + int(self.env.arena_h*self.scale) + 8
        self.screen.blit(self.font_lg.render(
            f"LIDAR VIEW  ·  {self.lidar.num_rays} sinar  ·  "
            f"range {self.lidar.max_range:.1f}m  ·  {self.map_name}",
            True, HEADING_COLOR), (ARENA_MARGIN, py))
        deg = math.degrees(self.env.robot.theta)
        self.screen.blit(self.font_md.render(
            f"Pose: x={robot.x:.3f}m  y={robot.y:.3f}m  "
            f"θ={deg:.1f}°  t={self.env.timestep}",
            True, TEXT_COLOR), (ARENA_MARGIN, py+22))
        self.screen.blit(self.font_sm.render(
            "Keyboard: [↑↓] Maju/Mundur  [←→] Rotasi  [R] Reset  [ESC] Menu",
            True, (100,100,140)), (ARENA_MARGIN, py+44))

    # ── panel kanan ───────────────────────────
    def _draw_right_panel(self):
        rp = ARENA_PANEL_W
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         pygame.Rect(rp, 0, RIGHT_W, WIN_H))

        title = self.font_xl.render("LIDAR SENSOR", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(centerx=rp+RIGHT_W//2, y=6))

        self.btn_load.draw(self.screen, self.font_md)

        # Info lidar
        rp_x  = rp + 22
        info_y = 76
        self.screen.blit(self.font_sm.render(
            f"Jumlah sinar : {self.lidar.num_rays}",
            True, AXIS_COLOR), (rp_x, info_y))
        self.screen.blit(self.font_sm.render(
            f"Jangkauan    : {self.lidar.max_range:.1f} m",
            True, AXIS_COLOR), (rp_x, info_y+16))
        self.screen.blit(self.font_sm.render(
            "(Parameter lidar diatur di menu Editor)",
            True, (60,80,120)), (rp_x, info_y+32))

        # ── Kompas ──
        compass_top = 120
        compass_h   = WIN_H - compass_top - 116
        self._draw_lidar_compass(rp, compass_top, RIGHT_W, compass_h)

        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (rp+16, WIN_H-114),(rp+RIGHT_W-16, WIN_H-114), 1)
        self.btn_reset.draw(self.screen, self.font_md)
        self.btn_exit.draw(self.screen, self.font_md)

    def _draw_lidar_compass(self, panel_x, top, panel_w, panel_h):
        """
        Kompas lidar yang berputar sesuai orientasi robot.
        Arah heading robot selalu menunjuk ke atas kompas (FRONT).
        Setiap sinar digambar pada sudut relatif terhadap robot.
        """
        if not self.lidar_data: return

        cx = panel_x + panel_w // 2
        cy = top + panel_h // 2
        R  = min(panel_w, panel_h) // 2 - 36
        if R < 20: return

        robot_theta = self.env.robot.theta

        # ── Latar & ring ──
        pygame.draw.circle(self.screen, COMPASS_BG, (cx, cy), R + 8)
        for frac in (0.33, 0.66, 1.0):
            pygame.draw.circle(self.screen, (40, 55, 90), (cx, cy), int(R*frac), 1)
        pygame.draw.circle(self.screen, COMPASS_RING, (cx, cy), R + 8, 2)

        # Ring distance labels
        for frac, label in ((0.33, f"{self.lidar.max_range*0.33:.1f}m"),
                            (0.66, f"{self.lidar.max_range*0.66:.1f}m"),
                            (1.0,  f"{self.lidar.max_range:.1f}m")):
            rl = int(R * frac)
            lbl = self.font_sm.render(label, True, (60, 80, 120))
            self.screen.blit(lbl, (cx + 4, cy - rl - 1))

        # ── Cardinal labels (ikut berputar dengan robot) ──
        # N/S/E/W di world space → posisi di kompas ikut theta robot
        for world_angle, cardinal in ((0, "E"), (math.pi/2, "N"),
                                       (math.pi, "W"), (-math.pi/2, "S")):
            # Sudut relatif cardinal terhadap heading robot
            rel = world_angle - robot_theta
            # Di kompas: heading robot = atas = -π/2 layar
            sa  = -math.pi/2 + rel
            lx  = cx + int((R+18)*math.cos(sa))
            ly  = cy + int((R+18)*math.sin(sa))
            lbl = self.font_sm.render(cardinal, True, (70, 90, 130))
            self.screen.blit(lbl, lbl.get_rect(center=(lx, ly)))

        # ── Tiap sinar lidar ──
        for ray in self.lidar_data:
            # angle_rel = sudut sinar relatif ke heading robot
            angle_rel  = ray["angle_rel"]
            # Kompas: heading=atas → screen_angle
            screen_angle = -math.pi/2 + angle_rel

            frac    = max(0.05, min(1.0, ray["distance"] / self.lidar.max_range))
            line_r  = int(R * frac)

            ex = cx + int(line_r * math.cos(screen_angle))
            ey = cy + int(line_r * math.sin(screen_angle))

            c_line = COMPASS_LINE_HIT  if ray["hit"] else COMPASS_LINE_FREE
            c_txt  = COMPASS_TEXT_HIT  if ray["hit"] else COMPASS_TEXT_FREE

            pygame.draw.line(self.screen, c_line, (cx, cy), (ex, ey), 2)
            pygame.draw.circle(self.screen, c_line, (ex, ey), 4 if ray["hit"] else 3)

            # Label jarak
            label_r  = line_r + 14
            lbl_surf = self.font_sm.render(f"{ray['distance']:.2f}", True, c_txt)
            lx = cx + int(label_r*math.cos(screen_angle)) - lbl_surf.get_width()//2
            ly = cy + int(label_r*math.sin(screen_angle)) - lbl_surf.get_height()//2
            lx = max(panel_x+2, min(panel_x+panel_w-lbl_surf.get_width()-2, lx))
            ly = max(top+2,     min(top+panel_h-lbl_surf.get_height()-2, ly))
            self.screen.blit(lbl_surf, (lx, ly))

        # ── Titik tengah ──
        pygame.draw.circle(self.screen, COMPASS_CENTER, (cx, cy), 5)

        # ── Panah FRONT (selalu ke atas) ──
        arrow_len = R + 16
        ax = cx + int(arrow_len * math.cos(-math.pi/2))
        ay = cy + int(arrow_len * math.sin(-math.pi/2))
        pygame.draw.line(self.screen, ROBOT_DIR_COLOR, (cx, cy), (ax, ay), 3)
        pygame.draw.polygon(self.screen, ROBOT_DIR_COLOR,
                            [(ax, ay-8), (ax-5, ay+2), (ax+5, ay+2)])
        fl = self.font_sm.render("FRONT", True, ROBOT_DIR_COLOR)
        self.screen.blit(fl, fl.get_rect(centerx=ax, y=ay-22))

        # Judul
        t = self.font_md.render(
            f"Kompas Lidar  ({self.lidar.num_rays} sinar)", True, HEADING_COLOR)
        self.screen.blit(t, t.get_rect(centerx=panel_x+panel_w//2, y=top-18))

    def _draw_defeat_overlay(self):
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((80, 10, 10, min(180, self.defeat_alpha)))
        self.screen.blit(overlay, (0, 0))
        if self.defeat_alpha < 120: return
        cx, cy = WIN_W//2, WIN_H//2-60
        df = pygame.font.SysFont("monospace", 72, bold=True)
        t  = df.render("DEFEAT!", True, DEFEAT_TEXT)
        self.screen.blit(t, t.get_rect(centerx=cx, centery=cy))
        self.screen.blit(
            self.font_lg.render("Robot menabrak obstacle!", True, DEFEAT_SUB),
            self.font_lg.render("Robot menabrak obstacle!", True, DEFEAT_SUB
                               ).get_rect(centerx=cx, centery=cy+65))
        self.screen.blit(
            self.font_md.render(f"Total langkah: {self.env.timestep}", True, DEFEAT_SUB),
            self.font_md.render(f"Total langkah: {self.env.timestep}", True, DEFEAT_SUB
                               ).get_rect(centerx=cx, centery=cy+92))
        if self.defeat_show_btns:
            self.btn_retry.draw(self.screen, self.font_lg)