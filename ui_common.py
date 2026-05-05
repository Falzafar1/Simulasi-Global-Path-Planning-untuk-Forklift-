"""
ui_common.py — Komponen UI bersama (warna, tombol, text input).
"""

import math
import pygame

# ══════════════════════════════════════════════════════════
#  PALET WARNA
# ══════════════════════════════════════════════════════════
BG_COLOR        = (18,  18,  30)
PANEL_COLOR     = (22,  22,  38)
ARENA_BG        = (30,  30,  50)
ARENA_BORDER    = (80, 120, 200)
GRID_COLOR      = (45,  45,  70)
ROBOT_COLOR     = (80, 200, 120)
ROBOT_DIR_COLOR = (240, 240,  80)
TRAIL_COLOR     = (60, 160, 100)
AXIS_COLOR      = (100, 100, 140)
TEXT_COLOR      = (210, 210, 230)
HEADING_COLOR   = (130, 180, 255)
HIGHLIGHT_COLOR = (255, 200,  60)
WARNING_COLOR   = (255,  80,  80)
SUCCESS_COLOR   = (80,  220, 120)
DIVIDER_COLOR   = (50,  60,  90)
OBSTACLE_COLOR  = (180,  60,  60)
OBSTACLE_BORDER = (220, 100, 100)

BTN_NORMAL      = (45,  55,  90)
BTN_HOVER       = (65,  85, 140)
BTN_PRESS       = (30,  40,  70)
BTN_BORDER      = (80, 120, 200)
BTN_EXIT_NORMAL = (90,  30,  30)
BTN_EXIT_HOVER  = (140, 50,  50)
BTN_EXIT_PRESS  = (60,  20,  20)
BTN_RESET_NORMAL= (45,  75,  45)
BTN_RESET_HOVER = (65, 110,  60)
BTN_RESET_PRESS = (30,  55,  30)
BTN_SAVE_NORMAL = (40,  70, 100)
BTN_SAVE_HOVER  = (60, 100, 150)
BTN_SAVE_PRESS  = (25,  50,  75)
BTN_WARN_NORMAL = (100, 70,  20)
BTN_WARN_HOVER  = (150, 100, 30)
BTN_WARN_PRESS  = (70,  50,  10)
BTN_TEXT        = (220, 230, 255)
ICON_COLOR      = (255, 220,  80)

INPUT_BG        = (28,  28,  48)
INPUT_BORDER    = (70,  90, 160)
INPUT_ACTIVE    = (100, 140, 220)
INPUT_TEXT      = (220, 230, 255)

BTN_RADIUS      = 8


# ══════════════════════════════════════════════════════════
#  TOMBOL
# ══════════════════════════════════════════════════════════
class Button:
    def __init__(self, x, y, w, h, label, icon="",
                 color_normal=BTN_NORMAL,
                 color_hover=BTN_HOVER,
                 color_press=BTN_PRESS,
                 action=None, enabled=True):
        self.rect         = pygame.Rect(x, y, w, h)
        self.label        = label
        self.icon         = icon
        self.c_normal     = color_normal
        self.c_hover      = color_hover
        self.c_press      = color_press
        self.action       = action
        self.enabled      = enabled
        self._press_timer = 0

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._press_timer = 6
                if self.action:
                    self.action()
                return True
        return False

    def update(self):
        if self._press_timer > 0:
            self._press_timer -= 1

    def draw(self, surface, font_lbl, font_icon=None):
        if not font_icon:
            font_icon = font_lbl

        if not self.enabled:
            color = (35, 35, 55)
            border = (50, 50, 70)
        else:
            hovered = self.rect.collidepoint(pygame.mouse.get_pos())
            if self._press_timer > 0:
                color = self.c_press
            elif hovered:
                color = self.c_hover
            else:
                color = self.c_normal
            border = BTN_BORDER

        pygame.draw.rect(surface, color,  self.rect, border_radius=BTN_RADIUS)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=BTN_RADIUS)

        txt_color = (120, 120, 140) if not self.enabled else BTN_TEXT

        if self.icon:
            ico = font_icon.render(self.icon, True, ICON_COLOR if self.enabled else (80, 80, 100))
            surface.blit(ico, ico.get_rect(centerx=self.rect.centerx,
                                           centery=self.rect.centery - 9))
            lbl = font_lbl.render(self.label, True, txt_color)
            surface.blit(lbl, lbl.get_rect(centerx=self.rect.centerx,
                                            centery=self.rect.centery + 13))
        else:
            lbl = font_lbl.render(self.label, True, txt_color)
            surface.blit(lbl, lbl.get_rect(center=self.rect.center))


# ══════════════════════════════════════════════════════════
#  CHECKBOX TOGGLE
# ══════════════════════════════════════════════════════════
class Checkbox:
    def __init__(self, x, y, size, label, font, checked=False, action=None):
        self.rect    = pygame.Rect(x, y, size, size)
        self.label   = label
        self.font    = font
        self.checked = checked
        self.action  = action

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.action:
                    self.action(self.checked)
                return True
        return False

    def draw(self, surface):
        color = INPUT_ACTIVE if self.checked else INPUT_BORDER
        pygame.draw.rect(surface, INPUT_BG,  self.rect, border_radius=4)
        pygame.draw.rect(surface, color,     self.rect, 2, border_radius=4)
        if self.checked:
            cx, cy = self.rect.center
            s = self.rect.width // 3
            pygame.draw.line(surface, SUCCESS_COLOR,
                             (cx - s, cy), (cx - s//3, cy + s), 2)
            pygame.draw.line(surface, SUCCESS_COLOR,
                             (cx - s//3, cy + s), (cx + s, cy - s), 2)

        lbl = self.font.render(self.label, True, TEXT_COLOR)
        surface.blit(lbl, (self.rect.right + 8,
                            self.rect.centery - lbl.get_height() // 2))


# ══════════════════════════════════════════════════════════
#  TEXT INPUT BOX
# ══════════════════════════════════════════════════════════
class TextInput:
    def __init__(self, x, y, w, h, font, label="", default="",
                 numeric=False, min_val=None, max_val=None):
        self.rect    = pygame.Rect(x, y, w, h)
        self.label   = label
        self.font    = font
        self.text    = str(default)
        self.active  = False
        self.numeric = numeric
        self.min_val = min_val
        self.max_val = max_val

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.active = False
            else:
                ch = event.unicode
                if self.numeric:
                    if ch in "0123456789." or (ch == "-" and len(self.text) == 0):
                        self.text += ch
                else:
                    self.text += ch

    def get_float(self, fallback=0.0) -> float:
        try:
            v = float(self.text)
            if self.min_val is not None:
                v = max(self.min_val, v)
            if self.max_val is not None:
                v = min(self.max_val, v)
            return v
        except ValueError:
            return fallback

    def get_str(self) -> str:
        return self.text.strip()

    def draw(self, surface, label_x=None, label_y=None):
        border = INPUT_ACTIVE if self.active else INPUT_BORDER
        pygame.draw.rect(surface, INPUT_BG,  self.rect, border_radius=4)
        pygame.draw.rect(surface, border,    self.rect, 2, border_radius=4)

        txt_surf = self.font.render(self.text, True, INPUT_TEXT)
        # Clip text ke dalam box
        clip = pygame.Rect(self.rect.x + 4, self.rect.y,
                           self.rect.w - 8, self.rect.h)
        surface.set_clip(clip)
        surface.blit(txt_surf, (self.rect.x + 4,
                                self.rect.centery - txt_surf.get_height() // 2))
        surface.set_clip(None)

        if self.label:
            lx = label_x if label_x is not None else self.rect.x
            ly = label_y if label_y is not None else self.rect.y - 18
            lbl = self.font.render(self.label, True, AXIS_COLOR)
            surface.blit(lbl, (lx, ly))


# ══════════════════════════════════════════════════════════
#  HELPER — TOAST NOTIFIKASI SEMENTARA
# ══════════════════════════════════════════════════════════
class Toast:
    def __init__(self, font):
        self.font    = font
        self.msg     = ""
        self.color   = SUCCESS_COLOR
        self.timer   = 0

    def show(self, msg, color=None, duration=120):
        self.msg   = msg
        self.color = color if color else SUCCESS_COLOR
        self.timer = duration

    def update(self):
        if self.timer > 0:
            self.timer -= 1

    def draw(self, surface, cx, cy):
        if self.timer <= 0:
            return
        alpha = min(255, self.timer * 4)
        surf  = self.font.render(self.msg, True, self.color)
        r     = surf.get_rect(center=(cx, cy))
        bg    = pygame.Surface((r.w + 20, r.h + 10), pygame.SRCALPHA)
        bg.fill((10, 10, 20, min(200, alpha)))
        surface.blit(bg, (r.x - 10, r.y - 5))
        surf.set_alpha(alpha)
        surface.blit(surf, r)


# ══════════════════════════════════════════════════════════
#  ROBOT SPRITE — gambar kustom pengganti lingkaran
# ══════════════════════════════════════════════════════════
# Cache gambar: diisi sekali lewat load_robot_sprite(), dipakai semua menu.
_robot_sprite_original: pygame.Surface | None = None   # surface asli (belum di-scale)


def load_robot_sprite(image_path: str) -> bool:
    """
    Muat gambar dari disk sebagai sprite robot.

    Parameters
    ----------
    image_path : path ke file gambar (PNG, JPG, BMP, dsb.)

    Returns
    -------
    True  jika berhasil dimuat
    False jika gagal (sprite tidak berubah, fallback ke lingkaran)
    """
    global _robot_sprite_original
    try:
        img = pygame.image.load(image_path).convert_alpha()
        _robot_sprite_original = img
        return True
    except Exception as e:
        print(f"[robot sprite] Gagal memuat '{image_path}': {e}")
        return False


def clear_robot_sprite():
    """Hapus sprite — kembali ke rendering lingkaran default."""
    global _robot_sprite_original
    _robot_sprite_original = None


def draw_robot_sprite(surface: pygame.Surface,
                      cx: int, cy: int,
                      r_px: int,
                      theta: float,
                      fallback_color=None,
                      defeat: bool = False):
    """
    Gambar robot di posisi layar (cx, cy).

    Jika sprite sudah dimuat via load_robot_sprite(), gambar itu
    di-scale ke ukuran 2*r_px × 2*r_px lalu dirotasi sesuai theta.

    Jika belum ada sprite (atau defeat=True dengan warna override),
    jatuh ke gambar lingkaran hijau seperti semula.

    Parameters
    ----------
    surface       : pygame.Surface tujuan gambar
    cx, cy        : posisi tengah robot dalam piksel layar
    r_px          : radius robot dalam piksel
    theta         : orientasi robot dalam radian (world frame)
    fallback_color: warna lingkaran fallback (default: ROBOT_COLOR)
    defeat        : jika True, paksa gunakan lingkaran merah (defeat overlay)
    """
    color = fallback_color if fallback_color else ROBOT_COLOR

    if _robot_sprite_original is not None and not defeat:
        # ── Gambar sprite ──────────────────────────────────────
        diameter = max(8, r_px * 2)
        # Scale ke ukuran yang sesuai
        scaled = pygame.transform.smoothscale(
            _robot_sprite_original, (diameter, diameter))
        # Rotasi: pygame rotasi berlawanan jarum jam, theta dunia berlawanan jarum jam
        # konversi: layar y ke bawah, jadi negate theta
        angle_deg = math.degrees(theta)
        rotated   = pygame.transform.rotate(scaled, angle_deg)
        # Blit di tengah
        rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rect)

        # Tetap gambar garis arah tipis di atas sprite supaya orientasi jelas
        dlen = r_px * 1.6
        ex   = cx + int(dlen * math.cos(theta))
        ey   = cy - int(dlen * math.sin(theta))
        pygame.draw.line(surface, ROBOT_DIR_COLOR, (cx, cy), (ex, ey), 2)

    else:
        # ── Fallback: lingkaran asli ───────────────────────────
        draw_col = WARNING_COLOR if defeat else color
        pygame.draw.circle(surface, draw_col, (cx, cy), r_px)
        pygame.draw.circle(surface, (200, 255, 200), (cx, cy), r_px, 2)

        dlen = r_px * 1.8
        ex   = cx + int(dlen * math.cos(theta))
        ey   = cy - int(dlen * math.sin(theta))
        pygame.draw.line(surface, ROBOT_DIR_COLOR, (cx, cy), (ex, ey), 3)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)


# ══════════════════════════════════════════════════════════
#  APPEARANCE — warna & gambar arena background + obstacle
# ══════════════════════════════════════════════════════════

# ── Arena background ──
_arena_bg_color: tuple = ARENA_BG
_arena_bg_image: pygame.Surface | None = None

# ── Obstacle appearance ──
_obstacle_color: tuple = OBSTACLE_COLOR
_obstacle_image: pygame.Surface | None = None


# ── Getters ───────────────────────────────────────────────
def get_arena_bg_color() -> tuple:
    return _arena_bg_color

def get_obstacle_color() -> tuple:
    return _obstacle_color

def has_arena_bg_image() -> bool:
    return _arena_bg_image is not None

def has_obstacle_image() -> bool:
    return _obstacle_image is not None


# ── Setters / loaders ─────────────────────────────────────
def set_arena_bg_color(color: tuple):
    """Ubah warna background arena. Hapus gambar jika ada."""
    global _arena_bg_color, _arena_bg_image
    _arena_bg_color = color
    _arena_bg_image = None


def load_arena_bg_image(path: str) -> bool:
    """Muat gambar sebagai background arena. Return True jika berhasil."""
    global _arena_bg_image
    try:
        img = pygame.image.load(path).convert()
        _arena_bg_image = img
        return True
    except Exception as e:
        print(f"[arena bg] Gagal memuat '{path}': {e}")
        return False


def clear_arena_bg_image():
    """Hapus gambar background — kembali ke warna solid."""
    global _arena_bg_image
    _arena_bg_image = None


def set_obstacle_color(color: tuple):
    """Ubah warna obstacle. Hapus gambar jika ada."""
    global _obstacle_color, _obstacle_image
    _obstacle_color = color
    _obstacle_image = None


def load_obstacle_image(path: str) -> bool:
    """Muat gambar sebagai tekstur obstacle. Return True jika berhasil."""
    global _obstacle_image
    try:
        img = pygame.image.load(path).convert_alpha()
        _obstacle_image = img
        return True
    except Exception as e:
        print(f"[obstacle] Gagal memuat '{path}': {e}")
        return False


def clear_obstacle_image():
    """Hapus gambar obstacle — kembali ke warna solid."""
    global _obstacle_image
    _obstacle_image = None


# ── Drawing helpers (dipakai semua menu) ─────────────────
def draw_arena_background(surface: pygame.Surface, rect: pygame.Rect):
    """
    Gambar background arena.
    Jika gambar dimuat, scale ke rect.
    Jika tidak, pakai warna solid.
    """
    if _arena_bg_image is not None:
        scaled = pygame.transform.smoothscale(
            _arena_bg_image, (rect.width, rect.height))
        surface.blit(scaled, rect.topleft)
    else:
        pygame.draw.rect(surface, _arena_bg_color, rect)


def draw_obstacle_rect(surface: pygame.Surface, rect: pygame.Rect):
    """
    Gambar satu obstacle.
    Jika gambar dimuat, scale ke rect lalu gambar border.
    Jika tidak, pakai warna solid + border.
    """
    if _obstacle_image is not None:
        # Pastikan obstacle cukup besar untuk terlihat
        if rect.width >= 2 and rect.height >= 2:
            scaled = pygame.transform.smoothscale(
                _obstacle_image, (rect.width, rect.height))
            surface.blit(scaled, rect.topleft)
    else:
        pygame.draw.rect(surface, _obstacle_color, rect)
    # Border selalu digambar agar obstacle selalu terlihat
    border_col = tuple(min(255, c + 40) for c in
                       (_obstacle_color if _obstacle_image is None else OBSTACLE_BORDER))
    pygame.draw.rect(surface, border_col, rect, 1)


