"""
menu_main.py — Layar menu utama dengan 5 pilihan menu.

Menu:
    1. Environment Editor
    2. Manual Control
    3. Lidar Sensor View
    4. DQN Training
    5. DQN Inference
"""

import sys
import math
import os
import pygame
import tkinter as tk
from tkinter import filedialog

from ui_common import (
    BG_COLOR, HEADING_COLOR, TEXT_COLOR, AXIS_COLOR,
    BTN_NORMAL, BTN_HOVER, BTN_PRESS,
    BTN_EXIT_NORMAL, BTN_EXIT_HOVER, BTN_EXIT_PRESS,
    BTN_SAVE_NORMAL, BTN_SAVE_HOVER, BTN_SAVE_PRESS,
    Button, load_robot_sprite, clear_robot_sprite,
    SUCCESS_COLOR, WARNING_COLOR,
)

WIN_W = 700
WIN_H = 780   # lebih tinggi untuk tombol sprite + appearance


class MainMenu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Differential Drive Robot — Menu Utama")

        self.font_xl  = pygame.font.SysFont("monospace", 26, bold=True)
        self.font_lg  = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 13)
        self.font_sm  = pygame.font.SysFont("monospace", 11)

        self.clock   = pygame.time.Clock()
        self.result  = None
        self._anim_t = 0.0

        cx = WIN_W // 2
        bw, bh = 340, 52
        bx = cx - bw // 2

        # ── 5 menu utama ──
        self.buttons = [
            Button(bx, 188, bw, bh, "  Environment Editor",
                   color_normal=BTN_SAVE_NORMAL,
                   color_hover =BTN_SAVE_HOVER,
                   color_press =BTN_SAVE_PRESS,
                   action=lambda: self._select("editor")),

            Button(bx, 252, bw, bh, "  Manual Control",
                   color_normal=BTN_NORMAL,
                   color_hover =BTN_HOVER,
                   color_press =BTN_PRESS,
                   action=lambda: self._select("control")),

            Button(bx, 316, bw, bh, "  Lidar Sensor View",
                   color_normal=(35, 70, 80),
                   color_hover =(50, 105, 120),
                   color_press =(22, 50, 58),
                   action=lambda: self._select("lidar")),

            Button(bx, 380, bw, bh, "  DQN Training",
                   color_normal=(55, 35, 80),
                   color_hover =(85, 55, 130),
                   color_press =(35, 22, 55),
                   action=lambda: self._select("training")),

            Button(bx, 444, bw, bh, "  DQN Inference / Demo",
                   color_normal=(35, 70, 50),
                   color_hover =(55, 110, 75),
                   color_press =(22, 48, 34),
                   action=lambda: self._select("inference")),

            Button(bx, 508, bw, bh, "  Pengaturan Tampilan",
                   color_normal=(55, 45, 80),
                   color_hover =(85, 70, 130),
                   color_press =(35, 28, 55),
                   action=lambda: self._select("appearance")),

            Button(bx, 572, bw, 40, "Keluar",
                   color_normal=BTN_EXIT_NORMAL,
                   color_hover =BTN_EXIT_HOVER,
                   color_press =BTN_EXIT_PRESS,
                   action=lambda: self._select("quit")),
        ]

        # ── Tombol sprite ──
        sprite_bw = (bw - 8) // 2
        self.btn_load_sprite = Button(
            bx, 626, sprite_bw, 36, "Ganti Sprite Robot",
            color_normal=(50, 60, 90), color_hover=(75, 90, 140),
            color_press=(30, 38, 65),
            action=self._action_load_sprite)
        self.btn_clear_sprite = Button(
            bx + sprite_bw + 8, 626, sprite_bw, 36, "Reset ke Lingkaran",
            color_normal=(70, 40, 40), color_hover=(110, 60, 60),
            color_press=(45, 25, 25),
            action=self._action_clear_sprite)
        self.sprite_buttons = [self.btn_load_sprite, self.btn_clear_sprite]

        # Nama file sprite aktif
        self._sprite_name = "(default: lingkaran)"

        # (y_garis_tengah_tombol, teks deskripsi)
        self.subtitles = [
            (214, "Buat / edit peta dan simpan parameter robot"),
            (278, "Kendalikan robot secara manual di peta yang dipilih"),
            (342, "Visualisasi sensor lidar  ·  kontrol keyboard ↑↓←→"),
            (406, "Latih agen DQN  ·  atur hyperparameter & simpan model"),
            (470, "Jalankan model terlatih & amati performa greedy"),
            (534, "Ubah background arena, obstacle & sprite robot"),
        ]

    def _select(self, val):
        self.result = val

    def _action_load_sprite(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Pilih gambar sprite robot",
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp *.gif"),
                       ("All", "*.*")])
        root.destroy()
        if path:
            ok = load_robot_sprite(path)
            if ok:
                self._sprite_name = os.path.basename(path)
            else:
                self._sprite_name = "(gagal dimuat)"

    def _action_clear_sprite(self):
        clear_robot_sprite()
        self._sprite_name = "(default: lingkaran)"

    def run(self) -> str:
        while self.result is None:
            dt = self.clock.tick(60)
            self._anim_t += dt * 0.001
            self._handle_events()
            for b in self.buttons + self.sprite_buttons:
                b.update()
            self._draw()
        return self.result

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            for b in self.buttons + self.sprite_buttons:
                b.handle_event(event)

    def _draw(self):
        self.screen.fill(BG_COLOR)

        # Lingkaran animasi latar
        for i in range(5):
            r    = 60 + i * 40
            a    = self._anim_t + i * 1.2
            cx_  = WIN_W // 2 + int(math.sin(a) * 30)
            cy_  = 90 + int(math.cos(a * 0.7) * 15)
            surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (30 + i*5, 40 + i*5, 70 + i*5, 18), (r, r), r)
            self.screen.blit(surf, (cx_ - r, cy_ - r))

        # Judul
        title = self.font_xl.render("DIFFERENTIAL DRIVE ROBOT", True, HEADING_COLOR)
        self.screen.blit(title, title.get_rect(centerx=WIN_W//2, y=36))
        sub = self.font_md.render("RL Simulation Framework", True, AXIS_COLOR)
        self.screen.blit(sub, sub.get_rect(centerx=WIN_W//2, y=72))

        # Robot ikon animasi
        _draw_robot_icon(self.screen, WIN_W // 2, 130, self._anim_t)

        # Tombol
        for b in self.buttons:
            b.draw(self.screen, self.font_lg)

        # Deskripsi di bawah tiap tombol
        for (y, txt) in self.subtitles:
            lbl = self.font_sm.render(txt, True, AXIS_COLOR)
            self.screen.blit(lbl, lbl.get_rect(centerx=WIN_W//2, y=y))

        # ── Area sprite robot ──
        sprite_lbl = self.font_sm.render(
            "Sprite Robot:", True, AXIS_COLOR)
        self.screen.blit(sprite_lbl, (WIN_W//2 - 170, 612))
        col = SUCCESS_COLOR if self._sprite_name != "(default: lingkaran)" \
              and self._sprite_name != "(gagal dimuat)" else AXIS_COLOR
        name_lbl = self.font_sm.render(self._sprite_name, True, col)
        self.screen.blit(name_lbl, (WIN_W//2 - 170 + sprite_lbl.get_width() + 6, 612))
        for b in self.sprite_buttons:
            b.draw(self.screen, self.font_sm)

        # Versi
        ver = self.font_sm.render(
            "v1.2  ·  DQN Navigation  ·  basis: DRL-robot-navigation-IR-SIM",
            True, (60, 60, 90))
        self.screen.blit(ver, ver.get_rect(centerx=WIN_W//2, y=WIN_H - 18))

        pygame.display.flip()


def _draw_robot_icon(surface, cx, cy, t):
    theta = t * 1.5
    r = 18
    pygame.draw.circle(surface, (50, 70, 110), (cx, cy), r + 4)
    pygame.draw.circle(surface, (80, 200, 120), (cx, cy), r)
    pygame.draw.circle(surface, (200, 255, 200), (cx, cy), r, 2)
    dx = cx + int(r * 1.5 * math.cos(theta))
    dy = cy - int(r * 1.5 * math.sin(theta))
    pygame.draw.line(surface, (240, 240, 80), (cx, cy), (dx, dy), 3)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)