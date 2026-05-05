"""
menu_appearance.py  —  Menu Pengaturan Tampilan

Fitur:
    - Ubah warna background arena (palette 16 warna + upload gambar)
    - Ubah warna obstacle        (palette 16 warna + upload gambar)
    - Preview real-time mini arena
    - Reset ke default
"""

import os
import sys
import math
import pygame
import tkinter as tk
from tkinter import filedialog

from ui_common import (
    BG_COLOR, PANEL_COLOR, ARENA_BORDER, GRID_COLOR,
    HEADING_COLOR, AXIS_COLOR, TEXT_COLOR, DIVIDER_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, ROBOT_DIR_COLOR,
    BTN_NORMAL, BTN_HOVER, BTN_PRESS,
    BTN_EXIT_NORMAL, BTN_EXIT_HOVER, BTN_EXIT_PRESS,
    BTN_SAVE_NORMAL, BTN_SAVE_HOVER, BTN_SAVE_PRESS,
    ARENA_BG, OBSTACLE_COLOR, OBSTACLE_BORDER,
    Button, Toast,
    set_arena_bg_color, load_arena_bg_image, clear_arena_bg_image,
    set_obstacle_color, load_obstacle_image, clear_obstacle_image,
    get_arena_bg_color, get_obstacle_color,
    has_arena_bg_image, has_obstacle_image,
    draw_arena_background, draw_obstacle_rect,
    draw_robot_sprite,
)

# ══════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════
WIN_W = 860
WIN_H = 720
FPS   = 60

COL_L  = 20
COL_W  = (WIN_W - 60) // 2
COL_R  = COL_L + COL_W + 20

SWATCH_SIZE = 36
SWATCH_GAP  = 6
SW_COLS     = 8
SW_ROWS     = 2

PREVIEW_H = 130
PREVIEW_Y = WIN_H - PREVIEW_H - 55

# ══════════════════════════════════════════════════════════
#  PALET WARNA
# ══════════════════════════════════════════════════════════
ARENA_PALETTE = [
    (30,  30,  50),   (20,  40,  20),   (40,  20,  20),   (20,  20,  40),
    (50,  40,  20),   (20,  40,  40),   (40,  30,  40),   (35,  35,  35),
    (200, 210, 220),  (210, 230, 200),  (230, 210, 200),  (200, 200, 230),
    (220, 230, 210),  (230, 220, 200),  (100, 120,  80),  (180, 160, 130),
]

OBSTACLE_PALETTE = [
    (180,  60,  60),  ( 60, 120, 180),  ( 60, 180,  80),  (200, 150,  40),
    (120,  60, 180),  (180,  60, 160),  ( 80,  80,  80),  (200, 200,  60),
    ( 60, 180, 180),  (180, 100,  60),  (100,  60,  40),  ( 50,  50,  50),
    (220, 220, 220),  (140,  40,  40),  ( 40,  80, 140),  ( 40, 120,  80),
]


# ══════════════════════════════════════════════════════════
#  SWATCH GRID
# ══════════════════════════════════════════════════════════
class SwatchGrid:
    def __init__(self, x, y, palette,
                 cols=SW_COLS, rows=SW_ROWS,
                 size=SWATCH_SIZE, gap=SWATCH_GAP):
        self.x        = x
        self.y        = y
        self.palette  = palette
        self.cols     = cols
        self.rows     = rows
        self.size     = size
        self.gap      = gap
        self.selected = 0

    def _rect_at(self, idx):
        row = idx // self.cols
        col = idx % self.cols
        return pygame.Rect(
            self.x + col * (self.size + self.gap),
            self.y + row * (self.size + self.gap),
            self.size, self.size)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i in range(len(self.palette)):
                if self._rect_at(i).collidepoint(event.pos):
                    self.selected = i
                    return True
        return False

    def draw(self, surface):
        for i, color in enumerate(self.palette):
            r = self._rect_at(i)
            pygame.draw.rect(surface, color, r, border_radius=4)
            border = (255, 255, 255) if i == self.selected else (70, 70, 90)
            thick  = 3 if i == self.selected else 1
            pygame.draw.rect(surface, border, r, thick, border_radius=4)

    def get_color(self):
        return self.palette[self.selected]

    @property
    def bottom(self):
        return self.y + self.rows * (self.size + self.gap)


# ══════════════════════════════════════════════════════════
#  MENU UTAMA
# ══════════════════════════════════════════════════════════
class AppearanceMenu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Pengaturan Tampilan")

        self.font_xl = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 13)
        self.font_sm = pygame.font.SysFont("monospace", 11)

        self.clock  = pygame.time.Clock()
        self.result = None
        self.toast  = Toast(self.font_lg)

        self._arena_img_name    = ""
        self._obstacle_img_name = ""

        # Swatch grids
        swatch_y = 112
        self.arena_sw = SwatchGrid(COL_L + 4, swatch_y, ARENA_PALETTE)
        self.obs_sw   = SwatchGrid(COL_R + 4, swatch_y, OBSTACLE_PALETTE)

        # Sinkronisasi ke warna aktif
        cur = get_arena_bg_color()
        if cur in ARENA_PALETTE:
            self.arena_sw.selected = ARENA_PALETTE.index(cur)
        cur = get_obstacle_color()
        if cur in OBSTACLE_PALETTE:
            self.obs_sw.selected = OBSTACLE_PALETTE.index(cur)

        self._build_buttons()

    # ── Build tombol ──────────────────────────────────────
    def _build_buttons(self):
        bw  = COL_W - 8
        bh  = 36
        gap = 8
        y_base = self.arena_sw.bottom + 16

        # Kolom kiri — Arena BG
        lx = COL_L + 4
        self.btn_arena_apply  = Button(lx, y_base,          bw, bh, "Terapkan Warna BG",
            color_normal=(35,75,35), color_hover=(55,115,55), color_press=(20,50,20),
            action=self._apply_arena_color)
        self.btn_arena_upload = Button(lx, y_base+bh+gap,   bw, bh, "Upload Gambar BG",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS, action=self._upload_arena)
        self.btn_arena_reset  = Button(lx, y_base+2*(bh+gap),bw, bh, "Reset Default BG",
            color_normal=(65,35,35), color_hover=(100,55,55), color_press=(42,22,22),
            action=self._reset_arena)

        # Kolom kanan — Obstacle
        rx = COL_R + 4
        self.btn_obs_apply    = Button(rx, y_base,          bw, bh, "Terapkan Warna Obstacle",
            color_normal=(35,75,35), color_hover=(55,115,55), color_press=(20,50,20),
            action=self._apply_obs_color)
        self.btn_obs_upload   = Button(rx, y_base+bh+gap,   bw, bh, "Upload Gambar Obstacle",
            color_normal=BTN_SAVE_NORMAL, color_hover=BTN_SAVE_HOVER,
            color_press=BTN_SAVE_PRESS, action=self._upload_obs)
        self.btn_obs_reset    = Button(rx, y_base+2*(bh+gap),bw, bh, "Reset Default Obstacle",
            color_normal=(65,35,35), color_hover=(100,55,55), color_press=(42,22,22),
            action=self._reset_obs)

        # Kembali
        bw2 = 300
        self.btn_back = Button(WIN_W//2 - bw2//2, WIN_H - 44,
                               bw2, 36, "Kembali ke Menu Utama  [ESC]",
                               color_normal=BTN_EXIT_NORMAL,
                               color_hover=BTN_EXIT_HOVER,
                               color_press=BTN_EXIT_PRESS,
                               action=lambda: setattr(self, "result", "main"))

        self.all_buttons = [
            self.btn_arena_apply, self.btn_arena_upload, self.btn_arena_reset,
            self.btn_obs_apply,   self.btn_obs_upload,   self.btn_obs_reset,
            self.btn_back,
        ]

    # ── Aksi ─────────────────────────────────────────────
    def _apply_arena_color(self):
        set_arena_bg_color(self.arena_sw.get_color())
        self._arena_img_name = ""
        self.toast.show("Warna background arena diterapkan!", SUCCESS_COLOR)

    def _upload_arena(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Pilih gambar background arena",
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        root.destroy()
        if path:
            ok = load_arena_bg_image(path)
            self._arena_img_name = os.path.basename(path) if ok else ""
            self.toast.show(
                f"Background: {self._arena_img_name}" if ok else "Gagal memuat gambar!",
                SUCCESS_COLOR if ok else WARNING_COLOR)

    def _reset_arena(self):
        set_arena_bg_color(ARENA_BG); clear_arena_bg_image()
        self._arena_img_name = ""; self.arena_sw.selected = 0
        self.toast.show("Background arena direset.", SUCCESS_COLOR)

    def _apply_obs_color(self):
        set_obstacle_color(self.obs_sw.get_color())
        self._obstacle_img_name = ""
        self.toast.show("Warna obstacle diterapkan!", SUCCESS_COLOR)

    def _upload_obs(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Pilih gambar tekstur obstacle",
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        root.destroy()
        if path:
            ok = load_obstacle_image(path)
            self._obstacle_img_name = os.path.basename(path) if ok else ""
            self.toast.show(
                f"Obstacle: {self._obstacle_img_name}" if ok else "Gagal memuat gambar!",
                SUCCESS_COLOR if ok else WARNING_COLOR)

    def _reset_obs(self):
        set_obstacle_color(OBSTACLE_COLOR); clear_obstacle_image()
        self._obstacle_img_name = ""; self.obs_sw.selected = 0
        self.toast.show("Obstacle direset ke default.", SUCCESS_COLOR)

    # ── Run loop ─────────────────────────────────────────
    def run(self) -> str:
        while self.result is None:
            self._handle_events()
            for b in self.all_buttons: b.update()
            self.toast.update()
            self._draw()
            self.clock.tick(FPS)
        return self.result

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.result = "main"
            self.arena_sw.handle_event(event)
            self.obs_sw.handle_event(event)
            for b in self.all_buttons: b.handle_event(event)

    # ── Draw ─────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG_COLOR)

        # Judul
        title = self.font_xl.render("PENGATURAN TAMPILAN", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(centerx=WIN_W // 2, y=14))
        sub = self.font_sm.render(
            "Ubah tampilan visual arena & obstacle  |  tidak memengaruhi logika simulasi",
            True, AXIS_COLOR)
        self.screen.blit(sub, sub.get_rect(centerx=WIN_W // 2, y=44))

        # Garis vertikal pemisah kolom
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (WIN_W // 2, 62), (WIN_W // 2, PREVIEW_Y - 18), 1)

        # ── Kolom kiri: Arena Background ──
        self._draw_col_header(COL_L, 65, COL_W, "BACKGROUND ARENA", (100, 160, 255))
        self.arena_sw.draw(self.screen)
        self._draw_status(COL_L + 4, self.arena_sw.bottom + 2,
                          has_arena_bg_image(), self._arena_img_name,
                          self.arena_sw.get_color())

        # ── Kolom kanan: Obstacle ──
        self._draw_col_header(COL_R, 65, COL_W, "OBSTACLE", (255, 140, 100))
        self.obs_sw.draw(self.screen)
        self._draw_status(COL_R + 4, self.obs_sw.bottom + 2,
                          has_obstacle_image(), self._obstacle_img_name,
                          self.obs_sw.get_color())

        # Tombol
        for b in self.all_buttons: b.draw(self.screen, self.font_md)

        # Divider atas preview
        pygame.draw.line(self.screen, DIVIDER_COLOR,
                         (20, PREVIEW_Y - 16), (WIN_W - 20, PREVIEW_Y - 16), 1)
        plbl = self.font_sm.render("── PREVIEW REAL-TIME ──", True, AXIS_COLOR)
        self.screen.blit(plbl, plbl.get_rect(centerx=WIN_W // 2, y=PREVIEW_Y - 14))

        # Preview
        self._draw_preview()

        self.toast.draw(self.screen, WIN_W // 2, WIN_H - 56)
        pygame.display.flip()

    def _draw_col_header(self, x, y, w, text, color):
        lbl = self.font_lg.render(text, True, color)
        self.screen.blit(lbl, (x + 4, y + 4))
        pygame.draw.line(self.screen, color,
                         (x + 4, y + 26), (x + w - 8, y + 26), 1)

    def _draw_status(self, x, y, has_img, img_name, swatch_color):
        # Kotak warna terpilih
        sr = pygame.Rect(x, y + 2, 18, 18)
        pygame.draw.rect(self.screen, swatch_color, sr, border_radius=3)
        pygame.draw.rect(self.screen, (120, 120, 140), sr, 1, border_radius=3)

        if has_img and img_name:
            txt, col = f"Gambar: {img_name}", SUCCESS_COLOR
        elif has_img:
            txt, col = "Gambar terpasang", SUCCESS_COLOR
        else:
            r, g, b = swatch_color
            txt, col = f"Warna: #{r:02X}{g:02X}{b:02X}  (klik swatch lalu Terapkan)", AXIS_COLOR
        lbl = self.font_sm.render(txt, True, col)
        self.screen.blit(lbl, (x + 24, y + 4))

    def _draw_preview(self):
        px, py = 20, PREVIEW_Y
        pw, ph = WIN_W - 40, PREVIEW_H

        # Panel luar
        pygame.draw.rect(self.screen, (14, 14, 26),
                         pygame.Rect(px - 2, py - 2, pw + 4, ph + 4),
                         border_radius=6)

        # Arena
        m = 10
        arena = pygame.Rect(px + m, py + m, pw - 2 * m, ph - 2 * m)
        draw_arena_background(self.screen, arena)
        pygame.draw.rect(self.screen, ARENA_BORDER, arena, 2)

        # Grid tipis
        step = max(10, arena.w // 12)
        for i in range(0, arena.w, step):
            pygame.draw.line(self.screen, (70, 70, 90),
                             (arena.x + i, arena.y), (arena.x + i, arena.bottom), 1)
        for j in range(0, arena.h, step):
            pygame.draw.line(self.screen, (70, 70, 90),
                             (arena.x, arena.y + j), (arena.right, arena.y + j), 1)

        # Obstacle contoh (3 buah)
        ow = arena.w // 8
        oh = arena.h // 2
        for ox, oy in [(arena.x + 15, arena.y + 10),
                       (arena.centerx - ow // 2, arena.y + 8),
                       (arena.right - ow - 15, arena.bottom - oh - 10)]:
            r = pygame.Rect(ox, oy, ow, oh).clip(arena)
            if r.w > 2 and r.h > 2:
                draw_obstacle_rect(self.screen, r)

        # Robot
        rcx = arena.x + arena.w * 3 // 4
        rcy = arena.centery
        rr  = max(6, arena.h // 5)
        draw_robot_sprite(self.screen, rcx, rcy, rr, math.pi * 0.15)

        # Goal
        gx, gy = arena.x + arena.w // 5, arena.centery
        pygame.draw.circle(self.screen, (255, 215, 0), (gx, gy), 8, 2)
        pygame.draw.line(self.screen, (255, 215, 0), (gx - 8, gy), (gx + 8, gy), 1)
        pygame.draw.line(self.screen, (255, 215, 0), (gx, gy - 8), (gx, gy + 8), 1)
