"""Animated diagrams for the Caught On Code explainers: the picture behind
Dean's narration, drawn from code so it explains instead of decorating.

Each narrated scene carries 1-3 "visuals", each a diagram spec Claude picks
from a fixed set of templates (see TEMPLATES / explainer.py's prompt):

  flow     boxes in a row joined by arrows, a packet travelling along them
           (how a stream gets from a PC to a viewer)
  network  players around a server, packets going back and forth, the
           server ticking (netcode, matchmaking, 100-player lobbies)
  race     two or three lanes on a timeline in milliseconds, events and
           the delays between them (peeker's advantage, input lag)
  bars     values side by side (tick rates, latencies)
  bignum   one number, big (validated against the research)
  grid     N dots, some highlighted (100 players, 1 in 20 matches)
  layers   a stack (game / Windows / kernel / hardware), one highlighted
  neural   a small neural network lighting up
  compare  two columns, A vs B
  quiz     "pause and guess": a question, options, a countdown, the answer
  words    one short line in big type
  arena    a little top-down game, slowed down: players, a wall, where the
           enemy really is vs where your screen shows him. Modes: peek
           (peeker's advantage), ticks (what the server knows at a tick
           rate), rewind (lag compensation: shot after he reached cover)

Every element appears when its phrase is spoken (`at`, matched against the
Whisper word timings of Dean's take, like the documentary's keyword
visuals). Frames are drawn at 2x and scaled down for smooth edges, then
piped into ffmpeg with the take as sound, with the same word-by-word
captions and watermark as the documentaries.
"""
from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .longform_beats import (FPS, H, W, Captions, F, back, clamp, ease_in_out, ease_out, find_phrase, prog,
                             script_timings, text_w, watermark, wrap)
from .visual_sources import emoji_file

S = 2  # drawn at 2x, scaled down: smooth lines and circles
BG = (15, 17, 23)
PANEL = (27, 33, 48)
EDGE = (64, 74, 96)
YEL = (250, 204, 21)
BLUE = (96, 165, 250)
RED = (248, 113, 113)
GREEN = (74, 222, 128)
WHITE = (243, 244, 246)
MUTED = (148, 155, 170)

TEMPLATES = ("flow", "network", "race", "bars", "bignum", "grid", "layers", "neural", "compare", "quiz", "words",
             "arena")
FADE = 0.35  # crossfade between two visuals in one scene


def mix(a, b, k: float):
    k = clamp(k)
    return tuple(int(round(x + (y - x) * k)) for x, y in zip(a, b))


class Canvas:
    """Drawing in 1920x1080 units on a 2x image."""

    def __init__(self, img):
        from PIL import ImageDraw

        self.img = img
        self.d = ImageDraw.Draw(img)

    def font(self, name: str, size: float):
        return F(name, int(size * S))

    def tw(self, text: str, name: str, size: float) -> float:
        return text_w(self.font(name, size), text) / S

    def line(self, pts, color, width: float = 3):
        self.d.line([(x * S, y * S) for x, y in pts], fill=color, width=max(1, int(width * S)), joint="curve")

    def circle(self, x, y, r, fill=None, outline=None, width: float = 3):
        self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=fill, outline=outline,
                       width=max(1, int(width * S)))

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, width: float = 3):
        self.d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), int(r * S), fill=fill, outline=outline,
                                 width=max(1, int(width * S)) if outline else 0)

    def text(self, x, y, s: str, name: str, size: float, color, anchor: str = "la", stroke: float = 0):
        self.d.text((x * S, y * S), s, font=self.font(name, size), fill=color, anchor=anchor,
                    stroke_width=int(stroke * S), stroke_fill=BG)

    def block(self, x, y, s: str, name: str, size: float, color, max_w: float, align: str = "center",
              line_h: float = 1.15, max_lines: int = 3) -> float:
        """Wrapped text; (x, y) is the top centre (align=center) or top left.
        Returns the height used."""
        fnt = self.font(name, size)
        rows = wrap(fnt, s, max_w * S)[:max_lines]
        for i, row in enumerate(rows):
            yy = y + i * size * line_h
            if align == "center":
                self.text(x, yy, row, name, size, color, "ma")
            else:
                self.text(x, yy, row, name, size, color, "la")
        return len(rows) * size * line_h

    def arrow(self, x0, y0, x1, y1, color, width: float = 4, head: float = 16, k: float = 1.0):
        if k <= 0:
            return
        x1, y1 = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
        self.line([(x0, y0), (x1, y1)], color, width)
        ang = math.atan2(y1 - y0, x1 - x0)
        if k > 0.2:
            pts = [(x1, y1), (x1 - head * math.cos(ang - 0.45), y1 - head * math.sin(ang - 0.45)),
                   (x1 - head * math.cos(ang + 0.45), y1 - head * math.sin(ang + 0.45))]
            self.d.polygon([(px * S, py * S) for px, py in pts], fill=color)

    def dashed(self, x0, y0, x1, y1, color, width: float = 2, dash: float = 10):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / dash))
        for i in range(0, n, 2):
            a, b = i / n, min(1.0, (i + 1) / n)
            self.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], color, width)

    def emoji(self, ch: str, x, y, size: float, k: float = 1.0):
        img = _emoji_img(ch, int(size * S))
        if img is None or k <= 0:
            return
        if k < 1:
            from PIL import Image

            sz = max(2, int(size * S * (0.6 + 0.4 * k)))
            img = img.resize((sz, sz), Image.LANCZOS)
            a = img.getchannel("A").point(lambda v: int(v * k))
            img.putalpha(a)
        self.img.paste(img, (int(x * S - img.width / 2), int(y * S - img.height / 2)), img)


_emoji_cache: Dict[Tuple[str, int], object] = {}


def _emoji_img(ch: str, px: int):
    from PIL import Image

    key = (ch, px)
    if key not in _emoji_cache:
        p = emoji_file(ch) if ch else None
        _emoji_cache[key] = Image.open(p).convert("RGBA").resize((px, px), Image.LANCZOS) if p else None
    return _emoji_cache[key]


_bg_cache: dict = {}


def _background():
    from PIL import Image

    if "bg" not in _bg_cache:
        img = Image.new("RGB", (W * S, H * S), BG)
        c = Canvas(img)
        for gx in range(0, W + 1, 60):
            for gy in range(0, H + 1, 60):
                c.circle(gx, gy, 1.6, fill=(30, 35, 48))
        _bg_cache["bg"] = img
    return _bg_cache["bg"].copy()


def _kicker(c: Canvas, spec: dict, k: float):
    title = (spec.get("title") or "").strip().upper()
    if title and k > 0:
        c.rrect(110, 64, 110 + 10, 64 + 44, 3, fill=YEL)
        c.text(136, 64, title[:48], "Inter-Black", 34, mix(BG, WHITE, k))


# ----------------------------------------------------------------- timing ---

def element_times(items: List[dict], timed, t0: float, t1: float, first_at_start: bool = True) -> List[float]:
    """When each element appears: its `at` phrase if said, else spread out
    over the first 60% of the visual."""
    n = len(items)
    out = []
    for i, it in enumerate(items):
        t = find_phrase(str(it.get("at") or ""), timed) if isinstance(it, dict) and it.get("at") else None
        if t is None or t < t0 - 0.05 or t >= t1:
            t = t0 + 0.15 + (t1 - t0) * 0.6 * (i / max(1, n))
        out.append(max(t0 + (0.1 if first_at_start else 0), t - 0.1))
    return out


def plan(scene: dict, timed, duration: float) -> List[dict]:
    """The scene's visuals with their time ranges."""
    vis = [v for v in scene.get("visuals") or [] if isinstance(v, dict) and v.get("type") in TEMPLATES]
    if not vis:
        vis = [{"type": "words", "text": " ".join((scene.get("narration") or "").split()[:8])}]
    starts = []
    for i, v in enumerate(vis):
        t = 0.0 if i == 0 else find_phrase(str(v.get("at") or ""), timed)
        if t is None:
            t = duration * i / len(vis)
        starts.append(max(0.0, t - 0.1))
    order = sorted(range(len(vis)), key=lambda i: starts[i])
    vis, starts = [vis[i] for i in order], [starts[i] for i in order]
    starts[0] = 0.0
    out = []
    for i, v in enumerate(vis):
        t1 = starts[i + 1] if i + 1 < len(vis) else duration
        if t1 - starts[i] < 1.2 and i + 1 < len(vis):
            continue
        out.append({"spec": v, "t0": starts[i], "t1": t1})
    for i, p in enumerate(out):
        p["t1"] = out[i + 1]["t0"] if i + 1 < len(out) else duration
        p["times"] = _times(p["spec"], timed, p["t0"], p["t1"])
    return out


def _times(spec: dict, timed, t0: float, t1: float) -> dict:
    kind = spec["type"]
    key = {"flow": "nodes", "network": "clients", "race": "events", "bars": "items", "layers": "items"}.get(kind)
    out = {"t0": t0, "t1": t1}
    if key:
        out["items"] = element_times(spec.get(key) or [], timed, t0, t1)
    if kind == "compare":
        rows = [{"at": it.get("at")} if isinstance(it, dict) else {} for side in ("left", "right")
                for it in (spec.get(side) or {}).get("items") or []]
        out["items"] = element_times(rows, timed, t0 + 0.6, t1)
    if kind == "arena":
        go = find_phrase(str(spec.get("go_at") or ""), timed) if spec.get("go_at") else None
        out["go"] = go if go is not None and t0 <= go < t1 - 1.0 else t0 + 0.8
    for name in ("highlight_at", "reveal_at", "result_at"):
        if spec.get(name):
            tt = find_phrase(str(spec[name]), timed)
            out[name] = tt if tt is not None and t0 <= tt < t1 else None
    return out


# -------------------------------------------------------------- templates ---

def _a(t: float, te: float, length: float = 0.45) -> float:
    return ease_out(prog(t, te, length))


def _node(c: Canvas, cx, cy, w, h, label: str, emoji: str, k: float, active: bool, sub: str = ""):
    if k <= 0:
        return
    s = 0.88 + 0.12 * back(k)
    ww, hh = w * s, h * s
    border = YEL if active else EDGE
    c.rrect(cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, 22, fill=mix(BG, PANEL, k), outline=mix(BG, border, k), width=4)
    ty = cy - hh / 2 + 22
    emoji = emoji if emoji and _emoji_img(emoji, 84 * S) is not None else ""
    if emoji:
        c.emoji(emoji, cx, cy - hh / 2 + 22 + 44, 84 * s, k)
        ty = cy - hh / 2 + 22 + 98
    lh = c.block(cx, ty if emoji else cy - 22 - (18 if len(label) > 14 else 0), label, "Inter-Bold", 32, mix(BG, WHITE, k), ww - 30, max_lines=2)
    if sub:
        c.block(cx, (ty if emoji else cy) + lh + 2, sub, "Inter-Regular", 24, mix(BG, MUTED, k), ww - 30, max_lines=1)


def draw_flow(c: Canvas, spec: dict, t: float, tm: dict):
    nodes = (spec.get("nodes") or [])[:6]
    n = max(1, len(nodes))
    gap = 90
    bw = min(330, (1680 - (n - 1) * gap) / n)
    total = n * bw + (n - 1) * gap
    x0 = (W - total) / 2 + bw / 2
    cy = 480
    xs = [x0 + i * (bw + gap) for i in range(n)]
    times = tm["items"]
    shown = [i for i in range(n) if t >= times[i]]
    last = shown[-1] if shown else -1
    for i in range(n - 1):
        if t >= times[i + 1] - 0.1:
            c.arrow(xs[i] + bw / 2 + 8, cy, xs[i + 1] - bw / 2 - 8, cy, mix(BG, MUTED, 1), 5, 20, _a(t, times[i + 1] - 0.15, 0.4))
    for i, nd in enumerate(nodes):
        _node(c, xs[i], cy, bw, 270, str(nd.get("label") or ""), nd.get("emoji") or "", _a(t, times[i]), i == last,
              str(nd.get("sub") or ""))
    if spec.get("packet", True) and len(shown) >= 2:
        span = xs[shown[-1]] - xs[0]
        u = (t * 0.45) % 1.0
        px = xs[0] + span * ease_in_out(u)
        c.line([(xs[0], cy + 150), (xs[shown[-1]], cy + 150)], (40, 46, 62), 3)
        c.circle(px, cy + 150, 26, outline=mix(BG, YEL, 0.45), width=3)
        c.circle(px, cy + 150, 16, fill=YEL)


def draw_network(c: Canvas, spec: dict, t: float, tm: dict):
    center = spec.get("center") or {"label": "Server", "emoji": "🖥️"}
    clients = (spec.get("clients") or [])[:8]
    cx, cy = W / 2, 470
    n = max(1, len(clients))
    pos = []
    for i in range(n):
        # on the diagonals, so nothing sits right above or below the server
        ang = math.pi + 2 * math.pi * (i + 0.5) / n if n > 2 else (math.pi if i == 0 else 0.0)
        pos.append((cx + 640 * math.cos(ang), cy + 250 * math.sin(ang)))
    times = tm["items"]
    for i, cl in enumerate(clients):
        k = _a(t, times[i])
        if k > 0:
            px, py = pos[i]
            c.line([(cx + (px - cx) * 0.18, cy + (py - cy) * 0.18), (cx + (px - cx) * (0.18 + 0.64 * k), cy + (py - cy) * (0.18 + 0.64 * k))],
                   (52, 60, 80), 4)
    # packets: up to the server (blue) and back out (yellow)
    for i in range(n):
        if t < times[i] + 0.5:
            continue
        px, py = pos[i]
        for phase, col, out in ((0.0, BLUE, False), (0.5, YEL, True)):
            u = ((t - times[i]) * 0.6 + phase + i * 0.13) % 1.0
            f = 0.18 + 0.64 * (u if out else 1 - u)
            c.circle(cx + (px - cx) * f, cy + (py - cy) * f, 9, fill=col)
    # the server, ticking
    rate = spec.get("tick_rate")
    pulse = (t * 2.0) % 1.0
    c.circle(cx, cy, 130 + 60 * pulse, outline=mix(BG, YEL, (1 - pulse) * 0.5), width=4)
    _node(c, cx, cy, 250, 220, str(center.get("label") or "Server"), center.get("emoji") or "🖥️", 1.0, True)
    if rate:
        c.text(cx, cy + 150, f"{rate} ticks per second", "Inter-Bold", 30, YEL, "ma")
    for i, cl in enumerate(clients):
        px, py = pos[i]
        _node(c, px, py, 220, 170, str(cl.get("label") or ""), cl.get("emoji") or "", _a(t, times[i]), False, str(cl.get("sub") or ""))


def draw_race(c: Canvas, spec: dict, t: float, tm: dict):
    lanes = (spec.get("lanes") or ["You", "Server", "Enemy"])[:3]
    events = (spec.get("events") or [])[:8]
    unit = spec.get("unit") or "ms"
    max_v = float(spec.get("max") or 0) or max([float(e.get("ms") or 0) for e in events] + [1]) * 1.15
    xl, xr = 420, 1760
    ys = [300 + i * (420 / max(1, len(lanes) - 1)) for i in range(len(lanes))] if len(lanes) > 1 else [480]
    xv = lambda v: xl + (xr - xl) * clamp(float(v) / max_v)  # noqa: E731
    for i, lane in enumerate(lanes):
        c.text(390, ys[i], str(lane), "Inter-Black", 36, WHITE, "rm")
        c.line([(xl, ys[i]), (xr, ys[i])], (44, 52, 70), 4)
    axis_y = ys[-1] + 90
    c.line([(xl, axis_y), (xr, axis_y)], MUTED, 2)
    for k in range(5):
        v = max_v * k / 4
        c.line([(xv(v), axis_y), (xv(v), axis_y + 10)], MUTED, 2)
        c.text(xv(v), axis_y + 18, f"{v:.0f} {unit}", "Inter-Regular", 24, MUTED, "ma")
    times = tm["items"]
    prev = None
    for j, ev in enumerate(events):
        lane = int(ev.get("lane") or 0) % len(lanes)
        x, y = xv(ev.get("ms") or 0), ys[lane]
        k = _a(t, times[j], 0.5)
        if k <= 0:
            prev = (x, y, lane)
            continue
        if prev and prev[2] != lane:
            c.arrow(prev[0], prev[1], x, y, mix(BG, BLUE, 1), 4, 18, ease_in_out(prog(t, times[j], 0.6)))
        c.dashed(x, y, x, axis_y, mix(BG, MUTED, 0.6 * k))
        col = YEL if ev.get("highlight") else BLUE
        c.circle(x, y, 13 + 6 * back(k), fill=mix(BG, col, k))
        label = str(ev.get("label") or "")
        if label:
            c.block(x, y - 78, label, "Inter-Bold", 27, mix(BG, WHITE, k), 280, max_lines=2)
        prev = (x, y, lane)
    if spec.get("example"):
        c.rrect(1560, 120, 1810, 164, 10, fill=PANEL)
        c.text(1685, 128, "EXAMPLE TIMINGS", "Inter-Black", 22, MUTED, "ma")


# ----------------------------------------------------------------- arena ---
# A small top-down game, slowed down so milliseconds become visible. All of
# it is drawn from code; the only numbers on screen are the spec's own
# (delay_ms / tick_rates, checked against the research in explainer.py) and
# plain arithmetic on them. Always tagged SIMPLIFIED.

ARENA = (160, 150, 1760, 690)
_NICE_SLOW = (2, 4, 5, 8, 10, 20, 25, 40, 50, 100)


def _nice_slow(want: float) -> int:
    return min(_NICE_SLOW, key=lambda v: abs(math.log(v / max(1.0, want))))


def _ms(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 0.05 else f"{round(v)}"


def _blocked(a, b, walls) -> bool:
    for i in range(1, 100):
        s = i / 100
        x, y = a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in walls):
            return True
    return False


def _arena_tags(c: Canvas, spec: dict, slow: int, k: float):
    title = (spec.get("title") or "").strip().upper()[:48]
    x = (136 + c.tw(title, "Inter-Black", 34) + 28) if title else 110
    for tag in [f"SIMPLIFIED · SLOWED DOWN {slow}×"] + (["EXAMPLE NUMBERS"] if spec.get("example") else []):
        tw = c.tw(tag, "Inter-Black", 20)
        c.rrect(x, 70, x + tw + 36, 104, 17, outline=mix(BG, YEL, k), width=2)
        c.text(x + 18 + tw / 2, 77, tag, "Inter-Black", 20, mix(BG, YEL, k), "ma")
        x += tw + 50


def _arena_floor(c: Canvas, walls, k: float, wall_label: str = "WALL"):
    x0, y0, x1, y1 = ARENA
    c.rrect(x0, y0, x1, y1, 22, fill=mix(BG, PANEL, k), outline=mix(BG, EDGE, k), width=2)
    for gx in range(x0 + 80, x1, 80):
        c.line([(gx, y0 + 4), (gx, y1 - 4)], mix(PANEL, (38, 45, 62), k), 1)
    for gy in range(y0 + 80, y1, 80):
        c.line([(x0 + 4, gy), (x1 - 4, gy)], mix(PANEL, (38, 45, 62), k), 1)
    for i, r in enumerate(walls):
        c.rrect(*r, 10, fill=mix(PANEL, (58, 66, 88), k), outline=mix(PANEL, EDGE, k), width=2)
        if i == 0 and wall_label and r[2] - r[0] > 140:
            c.text((r[0] + r[2]) / 2, (r[1] + r[3]) / 2 - 12, wall_label, "Inter-Black", 22, mix(PANEL, MUTED, k), "ma")


def _arena_note(c: Canvas, spec: dict, k: float):
    if spec.get("note") and not spec.get("example"):
        c.text(ARENA[2] - 60, ARENA[3] - 38, str(spec["note"])[:70], "Inter-Bold", 18, mix(PANEL, MUTED, k), "ra")


def _player(c: Canvas, p, col, k: float = 1.0, ghost: bool = False, label: Optional[str] = None,
            label_above: bool = False, label_bg: bool = False):
    if k <= 0:
        return
    x, y = p
    if ghost:
        for i in range(0, 18, 2):
            c.d.arc(((x - 26) * S, (y - 26) * S, (x + 26) * S, (y + 26) * S), i * 20, i * 20 + 20,
                    fill=mix(PANEL, col, k), width=4 * S)
    else:
        c.circle(x, y, 30, fill=mix(PANEL, col, 0.35 * k))
        c.circle(x, y, 22, fill=mix(PANEL, col, k))
    if label:
        ly = y - 72 if label_above else y + 40
        if label_bg:
            w = c.tw(label, "Inter-Black", 24)
            c.rrect(x - w / 2 - 10, ly - 4, x + w / 2 + 10, ly + 32, 8, fill=mix(PANEL, BG, k))
        c.text(x, ly, label, "Inter-Black", 24, mix(PANEL, col if ghost else WHITE, k), "ma")


def _pill(c: Canvas, x, y, text: str, col, k: float = 1.0):
    w = c.tw(text, "Inter-Black", 24)
    c.rrect(x, y, x + w + 40, y + 40, 10, fill=mix(PANEL, BG, k))
    c.text(x + 20 + w / 2, y + 6, text, "Inter-Black", 24, mix(BG, col, k), "ma")


def _pill_c(c: Canvas, cx, y, text: str, col, k: float = 1.0):
    _pill(c, cx - c.tw(text, "Inter-Black", 24) / 2 - 20, y, text, col, k)


def _clock(c: Canvas, ms: float):
    c.text(ARENA[2] - 60, ARENA[1] + 22, f"+{_ms(ms)} ms", "Inter-Black", 44, WHITE, "ra")


def _move(a, b, k):
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)


def _arena_peek(c: Canvas, spec: dict, t: float, tm: dict):
    """Peeker's advantage: the enemy steps past a corner; his screen shows
    you at once, yours shows him delay_ms later."""
    walls = [(160, 400, 900, 470), (1420, 210, 1510, 290), (330, 540, 420, 620), (1560, 520, 1640, 600)]
    you, e0, e1 = (1180, 590), (560, 320), (1040, 320)
    delay = float(spec.get("delay_ms") or 100)
    slow = _nice_slow(1.1 / (delay / 1000))
    lag = delay / 1000 * slow
    go, move = tm["go"], 1.4
    fade = _a(t, tm["t0"], 0.6)
    enemy_at = lambda tt: _move(e0, e1, ease_in_out(clamp((tt - go) / move)))  # noqa: E731
    t_vis = next((go + i / 50 for i in range(int(move * 50) + 1)
                  if not _blocked(you, enemy_at(go + i / 50), walls)), go + move)
    t_you = t_vis + lag
    game = max(0.0, (t - t_vis) / slow * 1000)
    span = max(50.0, math.ceil(delay * 1.45 / 25) * 25)
    _arena_tags(c, spec, slow, fade)
    _arena_floor(c, walls, fade)
    c.circle(walls[0][2], walls[0][1], 6, fill=mix(PANEL, YEL, 0.6 * fade))
    e, g = enemy_at(t), enemy_at(t - lag)
    if t >= t_vis:
        k = ease_out(clamp((t - t_vis) / 0.35))
        c.line([e, _move(e, you, k)], RED, 4)
        if k >= 1:
            m = _move(e, you, 0.3)
            _pill(c, m[0] + 20, m[1] - 46, "He can see you", RED)
    if t >= t_you:
        k = ease_out(clamp((t - t_you) / 0.35))
        b = _move(you, g, k)
        c.dashed(you[0], you[1], b[0], b[1], BLUE, 4, 14)
        if k >= 1:
            m = _move(g, you, 0.72)
            _pill(c, m[0] + 24, m[1] - 4, "You can see him", BLUE)
    _player(c, you, BLUE, fade, label="YOU")
    if t >= go and math.hypot(g[0] - e[0], g[1] - e[1]) > 3:
        near = math.hypot(g[0] - e[0], g[1] - e[1]) < 260
        _player(c, g, RED, 0.9 * clamp((t - go) / 0.3), ghost=True, label=None if near else "ON YOUR SCREEN", label_bg=True)
    _player(c, e, RED, fade)
    c.text(e[0], e[1] - 72, "ENEMY (REAL)" if t >= go + 0.2 else "ENEMY", "Inter-Black", 24, mix(PANEL, WHITE, fade), "ma")
    # timeline
    tx0, tx1, ty = 260, 1660, 790
    c.line([(tx0, ty), (tx1, ty)], mix(BG, EDGE, fade), 3)
    step = 25 if span <= 200 else 50 if span <= 400 else 100
    for v in range(0, int(span) + 1, step):
        x = tx0 + (tx1 - tx0) * v / span
        c.line([(x, ty - 8), (x, ty + 8)], mix(BG, EDGE, fade), 2)
        c.text(x, ty + 16, f"{v} ms", "Inter-Bold", 20, mix(BG, MUTED, fade), "ma")
    if t >= t_vis:
        gm = min(span, game)
        px = tx0 + (tx1 - tx0) * gm / span
        c.line([(tx0, ty), (px, ty)], WHITE, 5)
        c.circle(px, ty, 9, fill=WHITE)
        c.circle(tx0, ty, 12, fill=RED)
        c.text(tx0, ty - 52, "He sees you", "Inter-Black", 24, RED, "ma")
        _clock(c, gm)
    xg = tx0 + (tx1 - tx0) * delay / span
    if t >= t_you:
        c.circle(xg, ty, 12, fill=BLUE)
        c.text(xg, ty - 52, "You see him", "Inter-Black", 24, BLUE, "ma")
        res = tm.get("result_at") if tm.get("result_at") and tm["result_at"] > t_you else t_you + 0.8
        k = ease_out(clamp((t - res) / 0.6))
        if k > 0:
            c.rrect(tx0, ty - 22, tx0 + (xg - tx0) * k, ty - 16, 3, fill=YEL)
            if k >= 1:
                c.text((tx0 + xg) / 2, ty - 88, f"~{_ms(delay)} ms head start", "Inter-Black", 30, YEL, "ma")
    _arena_note(c, spec, fade)


def _arena_ticks(c: Canvas, spec: dict, t: float, tm: dict):
    """What the server knows: the enemy runs smoothly, but the server only
    takes a snapshot every 1000 / tick_rate ms -- one corridor per rate."""
    rates = [float(r) for r in (spec.get("tick_rates") or [64])[:2] if float(r) > 0] or [64.0]
    gap = 1000 / min(rates)
    slow = _nice_slow(0.45 / (gap / 1000))
    go, run = tm["go"], 3.6
    fade = _a(t, tm["t0"], 0.6)
    lanes = [300, 540] if len(rates) == 2 else [420]
    _arena_tags(c, spec, slow, fade)
    _arena_floor(c, [], fade)
    xa, xb = 470, 1650
    game = max(0.0, (t - go) / slow * 1000)
    total = run / slow * 1000
    for rate, y in zip(rates, lanes):
        every = 1000 / rate
        c.rrect(200, y - 70, 1720, y + 70, 16, outline=mix(PANEL, EDGE, fade), width=2)
        c.text(240, y - 30, f"{_ms(rate)} TICK", "Inter-Black", 30, mix(PANEL, WHITE, fade), "la")
        c.text(240, y + 8, f"every {_ms(every)} ms", "Inter-Bold", 22, mix(PANEL, MUTED, fade), "la")
        pos = lambda ms: xa + (xb - xa) * clamp(ms / total)  # noqa: E731
        if t < go:
            _player(c, (xa, y), RED, fade)
            continue
        snap = math.floor(min(game, total) / every) * every
        n = int(snap // every)
        for i in range(n + 1):  # every snapshot the server took, left behind as a dot
            c.circle(pos(i * every), y, 6, fill=mix(PANEL, YEL, 0.75))
        _player(c, (pos(min(game, total)), y), RED, 0.9, ghost=True)
        _player(c, (pos(snap), y), RED)
        if game >= total:
            c.text(xb - 30, y - 62, f"{n + 1} snapshots", "Inter-Black", 24, YEL, "ra")
    lx = 980
    c.circle(lx, ARENA[3] - 28, 8, fill=YEL)
    c.text(lx + 16, ARENA[3] - 40, "a snapshot", "Inter-Bold", 20, mix(PANEL, MUTED, fade), "la")
    c.circle(lx + 210, ARENA[3] - 28, 12, fill=mix(PANEL, RED, fade))
    c.text(lx + 230, ARENA[3] - 40, "what the server knows", "Inter-Bold", 20, mix(PANEL, MUTED, fade), "la")
    c.d.arc(((lx + 480 - 12) * S, (ARENA[3] - 40) * S, (lx + 480 + 12) * S, (ARENA[3] - 16) * S), 0, 360, fill=RED, width=3 * S)
    c.text(lx + 502, ARENA[3] - 40, "where he really is", "Inter-Bold", 20, mix(PANEL, MUTED, fade), "la")
    if t >= go:
        _clock(c, min(game, total))
    _arena_note(c, spec, fade)


def _arena_rewind(c: Canvas, spec: dict, t: float, tm: dict):
    """Lag compensation: you shoot where your screen shows him; he has
    already reached cover on his. The server rewinds by delay_ms, finds
    the hit, and he dies behind the wall."""
    walls = [(1180, 380, 1600, 450), (300, 220, 390, 300), (1580, 560, 1660, 640)]
    you, e0, e1 = (820, 610), (760, 300), (1460, 300)
    delay = float(spec.get("delay_ms") or 50)
    slow = _nice_slow(1.0 / (delay / 1000))
    lag = delay / 1000 * slow
    go, move = tm["go"], 3.0
    fade = _a(t, tm["t0"], 0.6)
    enemy_at = lambda tt: _move(e0, e1, ease_in_out(clamp((tt - go) / move)))  # noqa: E731
    t_hide = next((go + i / 50 for i in range(int(move * 50) + 1)
                   if _blocked(you, enemy_at(go + i / 50), walls)), go + move)
    t_shot = t_hide + min(0.3, lag * 0.4)
    t_rew = t_shot + 0.9
    t_hit = t_rew + 0.9
    _arena_tags(c, spec, slow, fade)
    _arena_floor(c, walls, fade)
    e = enemy_at(t)
    g = enemy_at(min(t, t_shot) - lag)  # your screen; frozen at the moment you fire
    shot_at = enemy_at(t_shot - lag)
    if t >= t_shot:
        k = ease_out(clamp((t - t_shot) / 0.15))
        b = _move(you, shot_at, k)
        c.line([you, b], YEL, 5)
        if t < t_rew:
            _pill_c(c, shot_at[0], shot_at[1] + 48, "You shoot here", YEL)
    _player(c, you, BLUE, fade, label="YOU")
    if t >= go and (t < t_rew or t >= t_hit):
        _player(c, g, RED, 0.9 * clamp((t - go) / 0.3), ghost=True,
                label="ON YOUR SCREEN" if t < t_shot else None, label_bg=True)
    _player(c, e, RED, fade)
    c.text(e[0], e[1] - 72, "ENEMY (REAL)" if t >= go + 0.2 else "ENEMY", "Inter-Black", 24, mix(PANEL, WHITE, fade), "ma")
    if t_shot <= t and _blocked(you, e, walls):
        _pill_c(c, min(e[0], 1720 - c.tw("Safe behind the wall on his screen", "Inter-Black", 24) / 2 - 20), e[1] + 168,
                "Safe behind the wall on his screen", MUTED)
    if t >= t_rew:
        k = ease_in_out(clamp((t - t_rew) / 0.8))
        r = _move(e, shot_at, k)  # the server's rewound copy of him
        if t < t_hit:
            _player(c, r, YEL, 0.9, ghost=True)
        _pill_c(c, shot_at[0], shot_at[1] + 48, f"Server rewinds {_ms(delay)} ms", YEL)
    if t >= t_hit:
        k = back(clamp((t - t_hit) / 0.4))
        x, y = shot_at
        for dx, dy in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
            c.line([(x + dx * 10, y + dy * 10), (x + dx * 34 * k, y + dy * 34 * k)], YEL, 6)
        c.text(x, y - 96, "HIT", "Inter-Black", 44, YEL, "ma")
    if t >= go:
        c.text(ARENA[2] - 60, ARENA[1] + 22, f"your ping ~{_ms(delay)} ms", "Inter-Black", 30, WHITE, "ra")
    _arena_note(c, spec, fade)


def draw_arena(c: Canvas, spec: dict, t: float, tm: dict):
    {"ticks": _arena_ticks, "rewind": _arena_rewind}.get(spec.get("mode"), _arena_peek)(c, spec, t, tm)


def _fmt_value(v: float) -> str:
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"


def draw_bars(c: Canvas, spec: dict, t: float, tm: dict):
    items = (spec.get("items") or [])[:6]
    unit = spec.get("unit") or ""
    vmax = max([float(it.get("value") or 0) for it in items] + [1])
    n = max(1, len(items))
    bh = min(90, 560 / n - 26)
    y0 = 470 - (n * (bh + 26) - 26) / 2
    times = tm["items"]
    for i, it in enumerate(items):
        y = y0 + i * (bh + 26)
        k = ease_out(prog(t, times[i], 0.9))
        label = str(it.get("label") or "")
        c.text(520, y + bh / 2, label[:28], "Inter-Bold", 34, mix(BG, WHITE, min(1, k * 3)), "rm")
        if k <= 0:
            continue
        v = float(it.get("value") or 0)
        bw = max(8, 1040 * (v / vmax) * k)
        col = YEL if it.get("highlight") else BLUE
        c.rrect(550, y, 550 + bw, y + bh, 10, fill=col)
        c.text(550 + bw + 18, y + bh / 2, f"{_fmt_value(v * k)} {unit}".strip(), "Inter-Black", 36, WHITE, "lm")


def draw_bignum(c: Canvas, spec: dict, t: float, tm: dict):
    k = ease_out(prog(t, tm["t0"] + 0.1, 0.9))
    value = str(spec.get("value") or "")
    shown = value
    if k < 0.999:  # counting up; the finished number shows exactly as written
        try:
            num = float(value.replace(",", ""))
            shown = _fmt_value(num * k) if "," not in value else f"{int(num * k):,}"
        except ValueError:
            pass
    size = 230 if len(value) <= 7 else 170
    c.text(W / 2, 430, shown, "Anton", size * (0.9 + 0.1 * back(min(1, k * 1.3))), YEL, "mm")
    c.block(W / 2, 600, str(spec.get("label") or ""), "Inter-Bold", 50, mix(BG, WHITE, k), 1400, max_lines=2)
    if spec.get("sub"):
        c.block(W / 2, 730, str(spec["sub"]), "Inter-Regular", 32, mix(BG, MUTED, k), 1300, max_lines=2)


def draw_grid(c: Canvas, spec: dict, t: float, tm: dict):
    n = max(1, min(400, int(spec.get("count") or 100)))
    hl = max(0, min(n, int(spec.get("highlight") or 0)))
    cols = max(1, round(math.sqrt(n * 2.6)))
    rows = math.ceil(n / cols)
    cell = min(1500 / cols, 520 / rows)
    r = cell * 0.36
    x0 = (W - cols * cell) / 2 + cell / 2
    y0 = 200 + (520 - rows * cell) / 2 + cell / 2
    t0 = tm["t0"] + 0.1
    th = tm.get("highlight_at") or (t0 + 1.6)
    picked = set(random.Random(n * 7 + hl).sample(range(n), hl)) if hl else set()
    for i in range(n):
        row, col = divmod(i, cols)
        k = ease_out(prog(t, t0 + 1.2 * i / n, 0.3))
        if k <= 0:
            continue
        on = i in picked and t >= th + 0.4 * (i / n)
        c.circle(x0 + col * cell, y0 + row * cell, r * (0.6 + 0.4 * k), fill=YEL if on else mix(BG, (70, 82, 110), k))
    label = str(spec.get("label") or "")
    if label:
        c.block(W / 2, y0 + rows * cell + 20, label, "Inter-Bold", 44, WHITE, 1500, max_lines=2)


def draw_layers(c: Canvas, spec: dict, t: float, tm: dict):
    items = (spec.get("items") or [])[:6]
    n = max(1, len(items))
    hl = spec.get("highlight")
    th = tm.get("highlight_at") if tm.get("highlight_at") is not None else (tm["items"][-1] + 0.8 if items else 0)
    lh = min(110, 600 / n - 16)
    y0 = 470 - (n * (lh + 16) - 16) / 2
    times = tm["items"]
    for i, it in enumerate(items):
        label = it.get("label") if isinstance(it, dict) else str(it)
        sub = it.get("sub") if isinstance(it, dict) else ""
        k = _a(t, times[i], 0.5)
        if k <= 0:
            continue
        y = y0 + i * (lh + 16) + (1 - k) * -30
        on = hl is not None and int(hl) == i and t >= th
        c.rrect(460, y, 1460, y + lh, 16, fill=mix(BG, (60, 52, 10) if on else PANEL, k), outline=mix(BG, YEL if on else EDGE, k), width=4)
        c.text(500, y + lh / 2 - (12 if sub else 0), str(label or ""), "Inter-Black", 38, mix(BG, WHITE, k), "lm")
        if sub:
            c.text(500, y + lh / 2 + 24, str(sub), "Inter-Regular", 26, mix(BG, MUTED, k), "lm")
        if on:
            pin = str(spec.get("pin") or "")
            kk = _a(t, th, 0.4)
            c.arrow(1700, y + lh / 2, 1480, y + lh / 2, YEL, 6, 24, kk)
            if pin:
                c.text(1710, y + lh / 2, pin, "Inter-Black", 32, YEL, "lm")


def draw_neural(c: Canvas, spec: dict, t: float, tm: dict):
    sizes = [max(1, min(8, int(s))) for s in (spec.get("layers") or [4, 6, 6, 3])][:5]
    xs = [460 + i * (1000 / max(1, len(sizes) - 1)) for i in range(len(sizes))]
    pos = [[(xs[i], 470 + (j - (n - 1) / 2) * 78) for j in range(n)] for i, n in enumerate(sizes)]
    rnd = random.Random(sum(sizes))
    k = _a(t, tm["t0"] + 0.1, 0.8)
    for a, b in zip(pos, pos[1:]):
        for p in a:
            for q in b:
                w = rnd.uniform(-1, 1)
                col = BLUE if w > 0 else RED
                c.line([p, q], mix(BG, mix(BG, col, 0.25 + 0.4 * abs(w)), k), 1.5 + 2 * abs(w))
    wave = ((t - tm["t0"]) * 0.7) % 1.4
    for i, layer in enumerate(pos):
        lit = clamp(1 - abs(wave - i / max(1, len(pos) - 1)) * 3)
        for j, (x, y) in enumerate(layer):
            c.circle(x, y, 22, fill=mix(BG, mix(PANEL, YEL, lit * (0.4 + 0.6 * ((j * 7 + i) % 3) / 2)), k),
                     outline=mix(BG, WHITE, k), width=3)
    outs = spec.get("outputs") or []
    for j, (x, y) in enumerate(pos[-1][:len(outs)]):
        c.text(x + 40, y, str(outs[j])[:16], "Inter-Bold", 28, mix(BG, WHITE, k), "lm")
    if spec.get("label"):
        c.block(W / 2, 830, str(spec["label"]), "Inter-Bold", 36, mix(BG, MUTED, k), 1500, max_lines=1)


def draw_compare(c: Canvas, spec: dict, t: float, tm: dict):
    times = tm.get("items") or []
    k0 = _a(t, tm["t0"] + 0.1, 0.5)
    rows = max(1, max(len((spec.get(side) or {}).get("items") or []) for side in ("left", "right")))
    rows = min(rows, 5)
    hgt = 190 + rows * 108 + 30
    top = 470 - hgt / 2
    idx = 0
    for side, x0, col in (("left", 180, BLUE), ("right", 1000, YEL)):
        part = spec.get(side) or {}
        c.rrect(x0, top, x0 + 740, top + hgt, 24, fill=mix(BG, PANEL, k0), outline=mix(BG, col, k0), width=4)
        c.block(x0 + 370, top + 35, str(part.get("title") or ""), "Anton", 72, mix(BG, col, k0), 680, max_lines=1)
        for j, it in enumerate((part.get("items") or [])[:5]):
            text = it.get("text") if isinstance(it, dict) else str(it)
            te = times[idx] if idx < len(times) else tm["t0"] + 0.8
            idx += 1
            k = _a(t, te, 0.4)
            if k <= 0:
                continue
            y = top + 175 + j * 108
            c.circle(x0 + 62, y + 24, 11, fill=mix(BG, col, k))
            c.block(x0 + 96, y, str(text or ""), "Inter-Bold", 42, mix(BG, WHITE, k), 610, align="left", max_lines=2)


def draw_quiz(c: Canvas, spec: dict, t: float, tm: dict):
    t0, t1 = tm["t0"], tm["t1"]
    reveal = tm.get("reveal_at")
    if reveal is None:
        reveal = max(t0 + 2.5, t1 - 1.4)
    k = _a(t, t0 + 0.05, 0.5)
    c.text(W / 2, 120, "PAUSE AND GUESS", "Inter-Black", 40, mix(BG, YEL, k), "ma")
    qh = c.block(W / 2, 185, str(spec.get("question") or ""), "Anton", 74, mix(BG, WHITE, k), 1500, max_lines=2)
    opts = (spec.get("options") or [])[:4]
    ans = int(spec.get("answer") or 0)
    y = 200 + qh + 40
    for i, o in enumerate(opts):
        ko = _a(t, t0 + 0.6 + 0.35 * i, 0.4)
        if ko <= 0:
            continue
        right = i == ans and t >= reveal
        col = GREEN if right else EDGE
        c.rrect(420, y, 1500, y + 92, 18, fill=mix(BG, (20, 56, 36) if right else PANEL, ko), outline=mix(BG, col, ko), width=5 if right else 3)
        c.text(470, y + 46, "ABCD"[i], "Inter-Black", 40, mix(BG, YEL, ko), "lm")
        c.text(540, y + 46, str(o)[:60], "Inter-Bold", 38, mix(BG, WHITE, ko), "lm")
        if right:
            c.line([(1418, y + 48), (1436, y + 66), (1468, y + 28)], GREEN, 7)
        y += 112
    count_from = t0 + 0.6 + 0.35 * len(opts) + 0.3
    if count_from < t < reveal:
        left = reveal - t
        n = max(1, math.ceil(left))
        frac = left - math.floor(left)
        c.circle(1700, 780, 64, outline=MUTED, width=5)
        c.d.arc(((1700 - 64) * S, (780 - 64) * S, (1700 + 64) * S, (780 + 64) * S), -90, -90 + 360 * frac, fill=YEL, width=6 * S)
        c.text(1700, 780, str(n), "Anton", 70, WHITE, "mm")


def draw_words(c: Canvas, spec: dict, t: float, tm: dict):
    text = str(spec.get("text") or "").strip()
    hl = {w.lower().strip(".,!?") for w in str(spec.get("highlight") or "").split()}
    fnt_size = 120 if len(text) <= 26 else 92
    fnt = c.font("Anton", fnt_size)
    rows = wrap(fnt, text.upper(), 1600 * S)[:3]
    y = 470 - len(rows) * fnt_size * 1.1 / 2
    i = 0
    for row in rows:
        x = (W - text_w(fnt, row) / S) / 2
        for w in row.split():
            k = _a(t, tm["t0"] + 0.1 + 0.12 * i, 0.35)
            col = YEL if w.lower().strip(".,!?") in hl else WHITE
            c.text(x, y + (1 - k) * 20, w, "Anton", fnt_size, mix(BG, col, k))
            x += text_w(fnt, w + " ") / S
            i += 1
        y += fnt_size * 1.1


DRAW = {"flow": draw_flow, "network": draw_network, "race": draw_race, "bars": draw_bars, "bignum": draw_bignum,
        "grid": draw_grid, "layers": draw_layers, "neural": draw_neural, "compare": draw_compare, "quiz": draw_quiz,
        "words": draw_words, "arena": draw_arena}


def frame(visual: dict, t: float):
    """One frame (1920x1080 RGB) of a planned visual at scene time t."""
    img = _background()
    c = Canvas(img)
    spec, tm = visual["spec"], visual["times"]
    _kicker(c, spec, _a(t, tm["t0"], 0.4))
    DRAW[spec["type"]](c, spec, t, tm)
    return img.reduce(S) if S > 1 else img


# --------------------------------------------------------------- rendering ---

def render_explain(ffmpeg: str, scene: dict, duration: float, take: Path, take_words: List[dict], out: Path,
                   brand: str) -> None:
    """One narrated explainer scene: its diagrams, the take as sound, and
    word-by-word captions."""
    from PIL import Image

    timed = script_timings(scene.get("narration") or "", take_words, duration)
    visuals = plan(scene, timed, duration)
    caps = Captions(timed)
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(take), "-filter_complex", "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[a]",
         "-map", "0:v", "-map", "[a]", "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    total = max(1, round(duration * FPS))
    try:
        for i in range(total):
            t = (i + 0.5) / FPS
            k = next((j for j, v in enumerate(visuals) if v["t0"] <= t < v["t1"]), len(visuals) - 1)
            img = frame(visuals[k], t)
            if k > 0 and t - visuals[k]["t0"] < FADE:
                img = Image.blend(frame(visuals[k - 1], t), img, (t - visuals[k]["t0"]) / FADE)
            watermark(img, brand)
            caps.draw(img, t)
            if i < 8:
                img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, (i + 1) / 9)
            enc.stdin.write(img.tobytes())
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
    except BaseException:
        enc.kill()
        raise


def still(scene: dict, which: int = -1, duration: Optional[float] = None):
    """A finished-looking frame of one of the scene's visuals (all its
    steps shown), without captions -- for the story editor's preview and
    for thumbnails."""
    words = (scene.get("narration") or "").split()
    dur = duration or max(4.0, len(words) / 2.5)
    timed = script_timings(scene.get("narration") or "", [], dur)
    visuals = plan(scene, timed, dur)
    v = visuals[which if -len(visuals) <= which < len(visuals) else -1]
    tm = dict(v["times"])
    # everything already appeared; a quiz shows its answer
    tm["items"] = [v["t0"]] * len(tm.get("items") or [])
    tm["highlight_at"] = v["t0"]
    if v["spec"]["type"] == "quiz":
        tm["reveal_at"] = v["t0"]
    return frame({"spec": v["spec"], "times": tm}, min(v["t1"] - 0.01, v["t0"] + 6.0) if v["t1"] > v["t0"] else v["t0"])
