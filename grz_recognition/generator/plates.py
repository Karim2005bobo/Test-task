import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generator import strokefont
from grz.plate_format import DIGITS, RU_LETTERS

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
TTF_FONTS = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
             "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"]
_font_cache = {}

OTHER_KINDS = ["trailer", "moto", "transit", "police", "military", "diplomatic",
               "foreign_eu", "foreign_by", "foreign_kz", "foreign_ua"]


@dataclass
class Plate:
    image: np.ndarray            # BGR, uint8
    mask: np.ndarray             # uint8 0/255, форма знака (скругления)
    text: str                    # символы знака без пробелов (разметка)
    plate_type: str              # type1 / type1a / type1b / other
    kind: str                    # детальный подтип
    two_line: bool
    size_mm: tuple = field(default=(520, 112))

REGIONS_3 = ["102", "113", "116", "121", "122", "123", "124", "125", "126", "134", "136", "138", "142",
             "147", "150", "152", "154", "156", "159", "161", "163", "164", "173", "174", "177", "178",
             "186", "190", "193", "196", "197", "198", "199", "277", "299", "323", "550", "702", "716",
             "725", "750", "761", "763", "774", "777", "790", "797", "799"]


def _rand_region(rng):
    if rng.random() < 0.35:
        if rng.random() < 0.8:
            return str(rng.choice(REGIONS_3))
        return str(rng.choice([1, 2, 3, 5, 7])) + "".join(rng.choice(list(DIGITS), 2))
    r = int(rng.integers(1, 100))
    return f"{r:02d}"


def rand_ru_number(rng, plate_type="type1"):
    """Случайный номер по маске типа: type1/1a – A123BC77(7), type1b – AB12377(7)."""
    L = list(RU_LETTERS)
    if plate_type == "type1b":
        return "".join(rng.choice(L, 2)) + "".join(rng.choice(list(DIGITS), 3)) + _rand_region(rng)
    return (rng.choice(L) + "".join(rng.choice(list(DIGITS), 3))
            + "".join(rng.choice(L, 2)) + _rand_region(rng))


def _ttf(name, size):
    key = (name, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _font_cache[key]


def _draw_ttf_char(img, ch, x, y, w, h, color, font_name):
    size = max(int(h * 1.6), 8)
    font = _ttf(font_name, size)
    canvas = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(canvas).text((size // 2, size // 4), ch, font=font, fill=255)
    arr = np.array(canvas)
    ys, xs = np.nonzero(arr > 10)
    if len(xs) == 0:
        return
    glyph = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    tw, th = max(int(round(w)), 1), max(int(round(h)), 1)
    glyph = cv2.resize(glyph, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x0, y0 = int(round(x)), int(round(y))
    H, W = img.shape[:2]
    x1, y1 = min(x0 + tw, W), min(y0 + th, H)
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
        return
    g = glyph[: y1 - y0, : x1 - x0, None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - g) + np.array(color, np.float32) * g).astype(np.uint8)


class Painter:

    def __init__(self, img, ppm, fg, style, rng):
        self.img, self.ppm, self.fg, self.style, self.rng = img, ppm, fg, style, rng
        self.stroke_mm = rng.uniform(8.5, 11.0)   # по ГОСТ ≈10 мм

    def char(self, ch, x, y, w, h):
        p = self.ppm
        if self.style == "stroke" and strokefont.has_glyph(ch):
            stroke = self.stroke_mm * (h / 76.0) ** 0.6 * p
            strokefont.draw_char(self.img, ch, x * p, y * p, w * p, h * p, stroke, self.fg,
                                 jitter=self.rng)
        else:
            font = self.style if self.style != "stroke" else TTF_FONTS[0]
            _draw_ttf_char(self.img, ch, x * p, y * p, w * p, h * p, self.fg, font)

    def text_small(self, s, x, y, w, h):
        font = TTF_FONTS[0] if self.style == "stroke" else self.style
        n = len(s)
        cw = w / max(n, 1)
        for i, ch in enumerate(s):
            _draw_ttf_char(self.img, ch, (x + i * cw + cw * 0.06) * self.ppm, y * self.ppm,
                           cw * 0.88 * self.ppm, h * self.ppm, self.fg, font)

    def row(self, items, x0, x1, bottom):
        total = sum(w + g for _, w, _, g in items)
        x = x0 + (x1 - x0 - total) / 2
        for ch, w, h, g in items:
            x += g
            self.char(ch, x, bottom - h, w, h)
            x += w


def _base(size_mm, ppm, bg, fg, rng, border=True, radius_mm=6):
    W, H = int(round(size_mm[0] * ppm)), int(round(size_mm[1] * ppm))
    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    mask = np.zeros((H, W), np.uint8)
    r = int(radius_mm * ppm)
    cv2.rectangle(mask, (r, 0), (W - 1 - r, H - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (W - 1, H - 1 - r), 255, -1)
    for cx, cy in [(r, r), (W - 1 - r, r), (r, H - 1 - r), (W - 1 - r, H - 1 - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    if border:
        t = max(1, int(round(rng.uniform(1.5, 2.5) * ppm)))
        ins = int(round(rng.uniform(2.5, 4.0) * ppm))
        cv2.rectangle(img, (ins, ins), (W - 1 - ins, H - 1 - ins), fg, t, cv2.LINE_AA)
    return img, mask


def _flag(img, x, y, w, h, ppm, colors=((255, 255, 255), (165, 57, 0), (32, 20, 213))):
    X, Y, Wp, Hp = [int(round(v * ppm)) for v in (x, y, w, h)]
    n = len(colors)
    for i, c in enumerate(colors):
        cv2.rectangle(img, (X, Y + i * Hp // n), (X + Wp, Y + (i + 1) * Hp // n), c, -1)
    cv2.rectangle(img, (X, Y), (X + Wp, Y + Hp), (60, 60, 60), max(1, int(ppm * 0.4)))


def _region_field(pt, region, x0, x1, top, h, rng, ppm, rus=True):
    n = len(region)
    w, g = (40, 5) if n == 2 else (34, 4)
    items = [(c, w, h, g if i else 0) for i, c in enumerate(region)]
    pt.row(items, x0, x1, top + h)
    if rus:
        pt.text_small("RUS", x0 + 10, top + h + 9, 50, 17)
        _flag(pt.img, x0 + 66, top + h + 9, 36, 17, ppm)


def _screws(img, ppm, rng, pts_mm):
    if rng.random() < 0.5:
        return
    for x, y in pts_mm:
        c = (int(rng.integers(90, 200)),) * 3
        cv2.circle(img, (int(x * ppm), int(y * ppm)), int(rng.uniform(3, 5) * ppm), c, -1, cv2.LINE_AA)

def _white(rng):
    v = rng.uniform(205, 252)
    tint = rng.normal(0, 5, 3)
    return tuple(int(np.clip(v + t, 0, 255)) for t in tint)


def _yellow(rng):
    b = rng.uniform(0, 70)
    g = rng.uniform(165, 225)
    r = rng.uniform(215, 255)
    return (int(b), int(g), int(r))


def _black(rng):
    v = int(rng.uniform(5, 45))
    return (v, v, v)

def render_type1(text, rng, ppm=2.0, yellow=False, style="stroke"):
    fg = _black(rng)
    bg = _yellow(rng) if yellow else _white(rng)
    img, mask = _base((520, 112), ppm, bg, fg, rng)
    pt = Painter(img, ppm, fg, style, rng)
    lw, lh, dw, dh = 50, 58, 48, 76
    bottom = 96
    if yellow:
        L12, D, region = text[:2], text[2:5], text[5:]
        items = [(c, lw, lh, 0 if i == 0 else 8) for i, c in enumerate(L12)] \
            + [(c, dw, dh, 16 if i == 0 else 8) for i, c in enumerate(D)]
    else:
        # тип 1: Л ЦЦЦ ЛЛ | регион
        L1, D, L23, region = text[0], text[1:4], text[4:6], text[6:]
        items = [(L1, lw, lh, 0)] + [(c, dw, dh, 12 if i == 0 else 7) for i, c in enumerate(D)] \
            + [(c, lw, lh, 12 if i == 0 else 7) for i, c in enumerate(L23)]
    pt.row(items, 12, 382, bottom)
    sep = int(386 * ppm)
    cv2.line(img, (sep, int(4 * ppm)), (sep, int(108 * ppm)), fg, max(1, int(2 * ppm)))
    _region_field(pt, region, 390, 512, 12, 58, rng, ppm)
    _screws(img, ppm, rng, [(150, 56), (330, 56)])
    return Plate(img, mask, text, "type1b" if yellow else "type1",
                 "type1b" if yellow else "type1", False, (520, 112))


def render_type1a(text, rng, ppm=2.0, style="stroke"):
    fg = _black(rng)
    bg = _white(rng)
    img, mask = _base((290, 170), ppm, bg, fg, rng)
    pt = Painter(img, ppm, fg, style, rng)
    L1, D, L23, region = text[0], text[1:4], text[4:6], text[6:]
    top = [(L1, 48, 56, 0)] + [(c, 46, 74, 14 if i == 0 else 7) for i, c in enumerate(D)]
    pt.row(top, 8, 282, 84)
    L = [(c, 46, 58, 0 if i == 0 else 7) for i, c in enumerate(L23)]
    pt.row(L, 12, 118, 158)
    n = len(region)
    w, g = (44, 6) if n == 2 else (36, 4)
    pt.row([(c, w, 58, g if i else 0) for i, c in enumerate(region)], 118, 246, 158)
    _flag(img, 250, 106, 28, 14, ppm)
    pt.text_small("RUS", 248, 126, 32, 13)
    _screws(img, ppm, rng, [(145, 90)])
    return Plate(img, mask, text, "type1a", "type1a", True, (290, 170))

def _rand(rng, alphabet, n):
    return "".join(rng.choice(list(alphabet), n))


def render_other(kind, rng, ppm=2.0, style="stroke"):
    L, D = RU_LETTERS, DIGITS
    if kind == "moto":
        fg, bg = _black(rng), _white(rng)
        img, mask = _base((190, 145), ppm, bg, fg, rng)
        pt = Painter(img, ppm, fg, style, rng)
        num, ser, reg = _rand(rng, D, 4), _rand(rng, L, 2), _rand_region(rng)
        pt.row([(c, 36, 62, 0 if i == 0 else 5) for i, c in enumerate(num)], 8, 182, 72)
        pt.row([(c, 34, 48, 0 if i == 0 else 5) for i, c in enumerate(ser)], 6, 90, 136)
        pt.row([(c, 28, 48, 0 if i == 0 else 4) for i, c in enumerate(reg)], 90, 184, 126)
        return Plate(img, mask, num + ser + reg, "other", kind, True, (190, 145))

    if kind.startswith("foreign"):
        return _render_foreign(kind, rng, ppm)

    # Однострочные российские «прочие» знаки 520x112
    if kind == "trailer":
        fg, bg, main = _black(rng), _white(rng), _rand(rng, L, 2) + _rand(rng, D, 4)
    elif kind == "transit":
        fg, bg, main = _black(rng), _white(rng), _rand(rng, L, 2) + _rand(rng, D, 3) + _rand(rng, L, 1)
    elif kind == "police":
        fg = (int(rng.uniform(230, 255)),) * 3
        bg = (int(rng.uniform(120, 170)), int(rng.uniform(50, 80)), int(rng.uniform(10, 40)))
        main = _rand(rng, L, 1) + _rand(rng, D, 4)
    elif kind == "military":
        fg, bg = (int(rng.uniform(220, 255)),) * 3, _black(rng)
        main = _rand(rng, D, 4) + _rand(rng, L, 2)
    else:  # diplomatic
        fg = (int(rng.uniform(225, 255)),) * 3
        bg = (int(rng.uniform(20, 50)), int(rng.uniform(20, 50)), int(rng.uniform(150, 200)))
        mid = rng.choice(["D", "CD", "T"])
        main = _rand(rng, D, 3) + mid + _rand(rng, D, 3 if mid != "CD" else 1)
    reg = _rand_region(rng)
    img, mask = _base((520, 112), ppm, bg, fg, rng)
    pt = Painter(img, ppm, fg, style, rng)
    items = []
    for i, c in enumerate(main):
        is_d = c.isdigit()
        items.append((c, 44 if is_d else 46, 76 if is_d else 58, 0 if i == 0 else 7))
    pt.row(items, 12, 382, 96)
    sep = int(386 * ppm)
    cv2.line(img, (sep, int(4 * ppm)), (sep, int(108 * ppm)), fg, max(1, int(2 * ppm)))
    _region_field(pt, reg, 390, 512, 12, 58, rng, ppm, rus=kind != "military" or rng.random() < 0.5)
    return Plate(img, mask, main + reg, "other", kind, False, (520, 112))


def _render_foreign(kind, rng, ppm):
    AZ = "ABCDEFGHIJKLMNOPRSTUVWXYZ"
    font = TTF_FONTS[int(rng.integers(0, 2))]
    fg, bg = _black(rng), _white(rng)
    if kind == "foreign_kz" and rng.random() < 0.3:
        bg = _yellow(rng)
    if kind == "foreign_eu":
        text = _rand(rng, AZ, int(rng.integers(1, 3))) + _rand(rng, AZ, 2) + _rand(rng, DIGITS, int(rng.integers(2, 5)))
        size = (520, 110)
    elif kind == "foreign_by":
        text = _rand(rng, DIGITS, 4) + _rand(rng, "ABEIKMHOPCTX", 2) + _rand(rng, "1234567", 1)
        size = (520, 112)
    elif kind == "foreign_kz":
        text = _rand(rng, DIGITS, 3) + _rand(rng, AZ, 3) + _rand(rng, DIGITS, 2)
        size = (520, 112)
    else:
        text = _rand(rng, "ABCEHIKMOPTX", 2) + _rand(rng, DIGITS, 4) + _rand(rng, "ABCEHIKMOPTX", 2)
        size = (520, 112)
    img, mask = _base(size, ppm, bg, fg, rng, radius_mm=5)
    strip_w = 44
    X = int(round(strip_w * ppm))
    if kind in ("foreign_eu", "foreign_ua"):
        cv2.rectangle(img, (int(3 * ppm), int(3 * ppm)), (X, img.shape[0] - int(3 * ppm)), (153, 51, 0), -1)
        code = rng.choice(["D", "F", "PL", "LT", "LV", "FIN", "EST", "CZ", "UA", "GE", "AM"]) \
            if kind == "foreign_eu" else "UA"
        pt = Painter(img, ppm, (255, 255, 255), font, rng)
        pt.text_small(code, 6, 70, strip_w - 8, 26)
        if kind == "foreign_ua":
            _flag(img, 10, 18, 26, 18, ppm, ((183, 91, 0), (0, 213, 255)))
    elif kind == "foreign_by":
        _flag(img, 8, 30, 30, 20, ppm, ((46, 39, 200), (46, 39, 200), (40, 160, 0)))
    else:
        cv2.rectangle(img, (int(3 * ppm), int(3 * ppm)), (X, img.shape[0] - int(3 * ppm)), (200, 170, 0), -1)
    pt = Painter(img, ppm, fg, font, rng)
    n = len(text)
    cw = min(52, (500 - strip_w - 10) / n - 6)
    items = [(c, cw, 70, 0 if i == 0 else rng.uniform(5, 14)) for i, c in enumerate(text)]
    pt.row(items, strip_w + 6, 512, 92)
    return Plate(img, mask, text, "other", kind, False, size)


def render_random(rng, ppm=2.0, type_probs=None, text=None):
    probs = type_probs or {"type1": 0.25, "type1a": 0.3, "type1b": 0.3, "other": 0.15}
    t = rng.choice(list(probs), p=np.array(list(probs.values())) / sum(probs.values()))
    style = "stroke" if rng.random() < 0.75 else TTF_FONTS[int(rng.integers(0, len(TTF_FONTS)))]
    if t == "other":
        return render_other(rng.choice(OTHER_KINDS), rng, ppm, style)
    text = text or rand_ru_number(rng, t)
    if t == "type1a":
        return render_type1a(text, rng, ppm, style)
    return render_type1(text, rng, ppm, yellow=(t == "type1b"), style=style)
