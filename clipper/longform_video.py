"""Visuals and render for long-form videos (see longform.py for the script
and recording side).

Every scene gets one main visual, timed to exactly where the narration
says it (the scene start times saved when the takes were joined):

  map      a dark OpenStreetMap map with the route drawing itself and the
           airports / key places labelled
  cockpit  a cockpit-voice-recorder style card, lines typed out
  chart    altitude / speed / time data drawing itself as a line
  report   a photo or diagram from the NTSB report itself, slow zoom
  stock    free Pexels footage (needs PEXELS_API_KEY; without it the
           scene falls back to a report image or a plain title card)

plus a short caption lower-third and a small channel watermark. Claude
plans the specifics (coordinates, which cockpit lines, chart numbers,
stock search words) from the report in plan_visuals; everything visual is
drawn here with Pillow and encoded with ffmpeg. Nothing is AI-generated
imagery.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

W, H, FPS = 1920, 1080, 30
BRAND = os.environ.get("CLIPPER_LONGFORM_BRAND", "Seconds to Decide")
END_CARD_SECONDS = 12.0

BG = (8, 12, 22)
TEXT = (243, 244, 246)
MUTED = (156, 163, 175)
AMBER = (245, 158, 11)
RED = (239, 68, 68)
PINK = (244, 114, 182)
CYAN = (34, 211, 238)

VISUAL_TYPES = ["map", "cockpit", "chart", "stock", "report"]


# ----------------------------------------------------------------- fonts ---

_FONT_FILES = {
    "sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    "monobold": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"],
}
_FC_NAMES = {"sans": "DejaVu Sans", "bold": "DejaVu Sans:bold", "mono": "DejaVu Sans Mono", "monobold": "DejaVu Sans Mono:bold"}
_font_cache: dict = {}


def font(size: int, kind: str = "sans"):
    from PIL import ImageFont

    key = (kind, size)
    if key in _font_cache:
        return _font_cache[key]
    path = next((p for p in _FONT_FILES[kind] if Path(p).exists()), None)
    if path is None:
        try:
            out = subprocess.run(["fc-match", "-f", "%{file}", _FC_NAMES[kind]], capture_output=True, text=True, timeout=10)
            path = out.stdout.strip() or None
        except Exception:
            path = None
    f = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
    _font_cache[key] = f
    return f


def _wrap(draw, text: str, fnt, max_w: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------- planning ---

def _plan_prompt(report_text: str, incident_title: str, scenes: List[dict]) -> str:
    listing = "\n".join(
        f'Scene {i + 1} (currently "{s.get("visual")}", note: {s.get("visual_note") or "-"}):\n{s["narration"]}'
        for i, s in enumerate(scenes)
    )
    return f"""You are planning the visuals for a narrated YouTube documentary about a
real aviation incident, built only from the official investigation report.

Incident: {incident_title}

For EACH scene below, pick one main visual and give the details needed to
draw it. Visual types:
- "map": a map showing where the aircraft is. Give "points" (airports,
  landing or crash site, key places named in the scene) with real-world
  latitude/longitude, and optionally a "path" (the route flown so far, as
  [lat, lon] pairs in order). Use coordinates you are confident of: airport
  reference points, or positions stated in the report. Keep a map to one
  local area (don't mix far-apart places unless the scene is about the
  whole route).
- "cockpit": lines from the cockpit voice recorder or radio transcript in
  the report, copied WORD FOR WORD, with the speaker as the report labels
  it (e.g. "CAPTAIN", "FIRST OFFICER", "ATC"). Only if the report actually
  contains those exact lines. At most 5 lines.
- "chart": a value changing over time taken from numbers in the report
  (e.g. altitude after the bird strike). "points" are [x, y] pairs with x
  in seconds from the first point; give "y_label" with the unit, and an
  optional "marker" ({{"x": seconds, "label": "..."}}) for a key moment.
  Only numbers the report states; at least 3 points.
- "report": a photo or diagram from the report (the app picks from the
  report's own images). Give "report_hint" saying what it should show.
- "stock": general aviation footage. Give "stock_query": 2 to 4 plain
  search words for a free stock video site (e.g. "airliner takeoff",
  "cockpit night", "river rescue boats"). No airline names.

Use a mix; don't repeat the same visual more than twice in a row. Prefer
"map", "cockpit" and "chart" where the scene's content supports them.

Every scene also gets a short "caption" for a lower-third (under 45
characters): a place, time or fact from the report, e.g. "LaGuardia
Airport, New York" or "15:27:10 · 2,818 feet". Use "" if nothing fits.

Scenes:
{listing}

Respond with ONLY a JSON array with exactly one object per scene, in
order, like:
[
  {{"scene": 1, "visual": "map", "caption": "...", "points": [{{"label": "LGA", "lat": 40.7769, "lon": -73.8740}}], "path": [[40.77, -73.87], [40.80, -73.90]]}},
  {{"scene": 2, "visual": "cockpit", "caption": "...", "time": "15:27:10", "lines": [{{"speaker": "CAPTAIN", "text": "birds."}}]}},
  {{"scene": 3, "visual": "chart", "caption": "...", "title": "Altitude", "y_label": "feet", "points": [[0, 2800], [30, 2000]], "marker": {{"x": 0, "label": "both engines lost"}}}},
  {{"scene": 4, "visual": "stock", "caption": "...", "stock_query": "airliner takeoff"}},
  {{"scene": 5, "visual": "report", "caption": "...", "report_hint": "..."}}
]

Report:
{report_text}
"""


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def normalize_spec(item: dict, fallback_visual: str = "stock") -> dict:
    """Validate one planned visual; anything unusable falls back to a plain
    type rather than rendering something wrong."""
    item = item if isinstance(item, dict) else {}
    visual = str(item.get("visual") or fallback_visual).lower()
    caption = " ".join(str(item.get("caption") or "").split())[:60]
    spec: dict = {"visual": visual, "caption": caption}

    if visual == "map":
        pts = []
        for p in item.get("points") or []:
            if not isinstance(p, dict):
                continue
            lat, lon = _num(p.get("lat")), _num(p.get("lon"))
            if lat is not None and lon is not None and -85 <= lat <= 85 and -180 <= lon <= 180:
                pts.append({"label": " ".join(str(p.get("label") or "").split())[:28], "lat": lat, "lon": lon})
        path = []
        for p in item.get("path") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                lat, lon = _num(p[0]), _num(p[1])
                if lat is not None and lon is not None and -85 <= lat <= 85 and -180 <= lon <= 180:
                    path.append([lat, lon])
        if not pts and len(path) < 2:
            return normalize_spec({**item, "visual": "stock"})
        spec.update(points=pts[:8], path=path[:200])
    elif visual == "cockpit":
        lines = []
        for ln in item.get("lines") or []:
            if isinstance(ln, dict) and str(ln.get("text") or "").strip():
                lines.append({"speaker": " ".join(str(ln.get("speaker") or "").split()).upper()[:24],
                              "text": " ".join(str(ln.get("text")).split())[:220]})
        if not lines:
            return normalize_spec({**item, "visual": "report"})
        spec.update(lines=lines[:5], time=" ".join(str(item.get("time") or "").split())[:20])
    elif visual == "chart":
        pts = []
        for p in item.get("points") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = _num(p[0]), _num(p[1])
                if x is not None and y is not None:
                    pts.append([x, y])
        pts.sort(key=lambda p: p[0])
        if len(pts) < 3 or pts[0][0] == pts[-1][0]:
            return normalize_spec({**item, "visual": "stock"})
        marker = item.get("marker") if isinstance(item.get("marker"), dict) else None
        mx = _num(marker.get("x")) if marker else None
        spec.update(points=pts[:120], title=str(item.get("title") or "")[:40], y_label=str(item.get("y_label") or "")[:20],
                    marker={"x": mx, "label": str(marker.get("label") or "")[:40]} if mx is not None else None)
    elif visual == "report":
        spec.update(report_hint=str(item.get("report_hint") or "")[:120], image_index=None)
    else:
        spec["visual"] = "stock"
        q = " ".join(str(item.get("stock_query") or "").split())[:60] or "airliner flying clouds"
        spec.update(stock_query=q, stock_index=0)
    return spec


def plan_visuals(report_text: str, incident_title: str, scenes: List[dict]) -> List[dict]:
    from .longform import trim_report
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    data = _ask_claude_for_json(_plan_prompt(trim_report(report_text), incident_title, scenes), None, DEFAULT_MODEL)
    items = data if isinstance(data, list) else []
    by_scene = {}
    for k, it in enumerate(items):
        if isinstance(it, dict):
            n = int(_num(it.get("scene")) or (k + 1))
            by_scene.setdefault(n - 1, it)
    return [normalize_spec(by_scene.get(i, {"visual": s.get("visual")}), s.get("visual") or "stock")
            for i, s in enumerate(scenes)]


# --------------------------------------------------------- report images ---

def extract_report_images(pdf_bytes: bytes, out_dir: Path, max_images: int = 40) -> List[dict]:
    """Photos and diagrams embedded in the report PDF (US government work,
    so free to use), big enough to fill a 1080p frame reasonably."""
    from PIL import ImageStat
    from pypdf import PdfReader

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if reader.is_encrypted:
        reader.decrypt("")
    found, seen = [], set()
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            imgs = list(page.images)
        except Exception:
            continue
        for k, img in enumerate(imgs):
            try:
                im = img.image.convert("RGB")
            except Exception:
                continue
            w, h = im.size
            if w < 500 or h < 300 or w / h > 4 or h / w > 3:
                continue
            small = im.resize((16, 16))
            if max(ImageStat.Stat(small.convert("L")).stddev) < 8:
                continue  # blank / near-solid
            digest = hashlib.md5(small.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            name = f"p{page_no:03d}_{k}.jpg"
            im.save(out_dir / name, quality=88)
            found.append({"file": name, "page": page_no, "w": w, "h": h})
            if len(found) >= max_images:
                return found
    return found


# ----------------------------------------------------------------- stock ---

def pexels_search(query: str, per_page: int = 12) -> List[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    import requests

    resp = requests.get(
        "https://api.pexels.com/videos/search",
        params={"query": query, "orientation": "landscape", "size": "medium", "per_page": per_page},
        headers={"Authorization": key}, timeout=20,
    )
    resp.raise_for_status()
    out = []
    for v in resp.json().get("videos") or []:
        files = [f for f in v.get("video_files") or [] if (f.get("file_type") == "video/mp4" and (f.get("width") or 0) >= 1280)]
        if not files:
            continue
        best = min(files, key=lambda f: abs((f.get("width") or 0) - 1920))
        out.append({"id": v.get("id"), "url": best["link"], "image": v.get("image"),
                    "user": (v.get("user") or {}).get("name"), "page": v.get("url"), "duration": v.get("duration")})
    return out


def fetch_stock(query: str, index: int, cache_dir: Path) -> Optional[dict]:
    """The index-th Pexels result for the query, downloaded (cached)."""
    try:
        results = pexels_search(query)
    except Exception as e:
        print(f"[longform_video] Pexels search failed: {e}", flush=True)
        return None
    if not results:
        return None
    item = results[index % len(results)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pexels_{item['id']}.mp4"
    if not path.exists():
        import requests

        with requests.get(item["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
    return {**item, "path": str(path), "count": len(results)}


# ------------------------------------------------------------------- map ---

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_UA = "clipper-longform/1.0 (personal YouTube documentary tool)"


def _world_px(lat: float, lon: float, z: int) -> Tuple[float, float]:
    n = 256 * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def fit_view(coords: List[Tuple[float, float]], pad: int = 220) -> Tuple[int, float, float]:
    """Zoom and centre (world pixels) that fit every coordinate on screen."""
    if len(coords) == 1:
        z = 12
        cx, cy = _world_px(coords[0][0], coords[0][1], z)
        return z, cx, cy
    z = 3
    for zz in range(15, 2, -1):
        xs, ys = zip(*(_world_px(la, lo, zz) for la, lo in coords))
        if max(xs) - min(xs) <= W - 2 * pad and max(ys) - min(ys) <= H - 2 * pad:
            z = zz
            break
    xs, ys = zip(*(_world_px(la, lo, z) for la, lo in coords))
    return z, (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2


def _fetch_tile(z: int, x: int, y: int, cache_dir: Path):
    from PIL import Image

    n = 2 ** z
    if not 0 <= y < n:
        return None
    x %= n
    path = cache_dir / f"{z}_{x}_{y}.png"
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            path.unlink(missing_ok=True)
    import requests

    resp = requests.get(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": TILE_UA}, timeout=15)
    resp.raise_for_status()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _darken_map(img):
    """OSM's light style -> a dark 'night' map: inverted greys (labels come
    out light) tinted navy, with water picked out in blue."""
    import numpy as np
    from PIL import Image

    a = np.asarray(img).astype(np.float32)
    water = np.abs(a - np.array([170, 211, 223], dtype=np.float32)).sum(axis=2) < 70
    gray = a.mean(axis=2)
    inv = 255.0 - gray
    out = np.empty_like(a)
    for c, base in enumerate((10, 16, 30)):
        out[..., c] = base + inv * 0.62
    out[water] = (22, 52, 92)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _grid_map(z: int, cx: float, cy: float):
    """Fallback when map tiles can't be fetched: a plain dark grid."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (12, 18, 32))
    d = ImageDraw.Draw(img)
    step = 256
    ox, oy = (cx - W / 2) % step, (cy - H / 2) % step
    for x in range(int(-ox), W, step):
        d.line([(x, 0), (x, H)], fill=(24, 34, 56), width=1)
    for y in range(int(-oy), H, step):
        d.line([(0, y), (W, y)], fill=(24, 34, 56), width=1)
    return img


def base_map(z: int, cx: float, cy: float, cache_dir: Path, fetch_tiles: bool = True) -> Tuple[object, bool]:
    from PIL import Image

    if not fetch_tiles:
        return _grid_map(z, cx, cy), False
    left, top = cx - W / 2, cy - H / 2
    img = Image.new("RGB", (W, H))
    try:
        for tx in range(int(math.floor(left / 256)), int(math.floor((left + W) / 256)) + 1):
            for ty in range(int(math.floor(top / 256)), int(math.floor((top + H) / 256)) + 1):
                tile = _fetch_tile(z, tx, ty, cache_dir)
                if tile is not None:
                    img.paste(tile, (int(tx * 256 - left), int(ty * 256 - top)))
    except Exception as e:
        print(f"[longform_video] map tiles unavailable ({e}); using a plain grid", flush=True)
        return _grid_map(z, cx, cy), False
    return _darken_map(img), True


# -------------------------------------------------------------- overlays ---

def overlay_layer(caption: str, attribution: str = ""):
    """Caption lower-third + channel watermark (+ map attribution), as one
    transparent layer composited over every frame of a scene."""
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    wm = BRAND.upper()
    wf = font(24, "bold")
    tw = d.textlength(wm, font=wf)
    d.text((W - 60 - tw, 48), wm, font=wf, fill=(255, 255, 255, 150))
    if caption:
        cf = font(38, "bold")
        cw = d.textlength(caption, font=cf)
        x, y = 90, H - 190
        d.rounded_rectangle([x, y, x + cw + 64, y + 76], radius=10, fill=(8, 12, 22, 205))
        d.rectangle([x, y, x + 8, y + 76], fill=RED + (255,))
        d.text((x + 34, y + 16), caption, font=cf, fill=TEXT + (255,))
    if attribution:
        af = font(18)
        aw = d.textlength(attribution, font=af)
        d.text((W - 24 - aw, H - 34), attribution, font=af, fill=(255, 255, 255, 140))
    return layer


# -------------------------------------------------------- frame drawers ---

def _map_frames(spec: dict, duration: float, cache_dir: Path, fetch_tiles: bool) -> Tuple[Iterator, str]:
    from PIL import ImageDraw

    pts = spec.get("points") or []
    path = spec.get("path") or []
    coords = [(p["lat"], p["lon"]) for p in pts] + [tuple(p) for p in path]
    z, cx, cy = fit_view(coords)
    base, real = base_map(z, cx, cy, cache_dir, fetch_tiles)

    def scr(lat, lon):
        x, y = _world_px(lat, lon, z)
        return x - cx + W / 2, y - cy + H / 2

    route = [scr(la, lo) for la, lo in path]
    seglen = [math.dist(route[i], route[i + 1]) for i in range(len(route) - 1)]
    total = sum(seglen) or 1.0
    draw_t = min(8.0, max(2.0, duration * 0.6))
    marks = [(scr(p["lat"], p["lon"]), p["label"]) for p in pts]
    lf = font(30, "bold")

    def partial(progress):
        target = total * progress
        acc, out = 0.0, [route[0]] if route else []
        for i, L in enumerate(seglen):
            if acc + L >= target:
                f = (target - acc) / L if L else 0
                out.append((route[i][0] + (route[i + 1][0] - route[i][0]) * f, route[i][1] + (route[i + 1][1] - route[i][1]) * f))
                return out, math.atan2(route[i + 1][1] - route[i][1], route[i + 1][0] - route[i][0])
            acc += L
            out.append(route[i + 1])
        ang = math.atan2(route[-1][1] - route[-2][1], route[-1][0] - route[-2][0]) if len(route) > 1 else 0
        return out, ang

    def frame(t):
        img = base.copy()
        d = ImageDraw.Draw(img)
        if len(route) > 1:
            d.line(route, fill=(70, 80, 110), width=3)
            pr = _ease(t / draw_t)
            done, ang = partial(pr)
            if len(done) > 1:
                d.line(done, fill=PINK, width=7, joint="curve")
            hx, hy = done[-1]
            s = 20
            tri = [(hx + s * math.cos(ang), hy + s * math.sin(ang)),
                   (hx + s * 0.7 * math.cos(ang + 2.5), hy + s * 0.7 * math.sin(ang + 2.5)),
                   (hx + s * 0.7 * math.cos(ang - 2.5), hy + s * 0.7 * math.sin(ang - 2.5))]
            d.polygon(tri, fill=(255, 255, 255))
        a = _ease(t / 0.8)
        for (x, y), label in marks:
            r = 11
            d.ellipse([x - r - 5, y - r - 5, x + r + 5, y + r + 5], outline=AMBER, width=3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=AMBER)
            if label and a > 0.05:
                tw = d.textlength(label, font=lf)
                bx, by = x + 26, y - 26
                if bx + tw + 24 > W - 20:
                    bx = x - 26 - tw - 24
                d.rounded_rectangle([bx, by, bx + tw + 24, by + 50], radius=8, fill=(8, 12, 22))
                d.text((bx + 12, by + 7), label, font=lf, fill=TEXT)
        return img

    frame.static_after = max(draw_t if len(route) > 1 else 0.0, 0.8) + 0.05
    return frame, ("© OpenStreetMap contributors" if real else "")


def _cockpit_frames(spec: dict, duration: float) -> Callable:
    from PIL import Image, ImageDraw

    lines = spec.get("lines") or []
    bg = Image.new("RGB", (W, H), (6, 7, 10))
    d0 = ImageDraw.Draw(bg)
    for y in range(0, H, 6):
        d0.line([(0, y), (W, y)], fill=(10, 11, 16))
    hf, sf, tf = font(30, "mono"), font(52, "monobold"), font(52, "mono")
    header = "COCKPIT VOICE RECORDER" + (f"  ·  {spec['time']}" if spec.get("time") else "")
    d0.text((180, 170), header, font=hf, fill=MUTED)
    d0.rectangle([180, 222, 330, 228], fill=RED)
    cps = 24.0
    start = 0.5
    wrapped = []
    for ln in lines:
        sp = (ln.get("speaker") or "").strip()
        label = f"{sp}:" if sp else ""
        lw = d0.textlength(label + " ", font=sf) if label else 0
        wrapped.append((label, lw, _wrap(d0, f"“{ln['text']}”", tf, W - 360 - int(lw))))

    def frame(t):
        img = bg.copy()
        d = ImageDraw.Draw(img)
        budget = max(0.0, (t - start) * cps)
        y = 290
        for label, lw, rows in wrapped:
            if budget <= 0:
                break
            if label:
                d.text((180, y), label, font=sf, fill=AMBER)
            for row in rows:
                shown = row[: int(budget)]
                budget -= len(row)
                d.text((180 + lw, y), shown, font=tf, fill=TEXT)
                y += 72
                if budget <= 0:
                    break
            budget -= 8  # a beat between lines
            y += 26
        return img

    total_chars = sum(len(r) for _l, _w, rows in wrapped for r in rows) + 8 * len(wrapped)
    frame.static_after = start + total_chars / cps + 0.1
    return frame


def _nice_ticks(lo: float, hi: float, n: int = 4) -> List[float]:
    span = hi - lo or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _chart_frames(spec: dict, duration: float) -> Callable:
    from PIL import Image, ImageDraw

    pts = spec["points"]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    lo, hi = min(ys), max(ys)
    padv = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - padv, hi + padv
    if min(ys) >= 0 and lo < 0:
        lo = 0.0
    L, R, T, B = 240, W - 160, 230, H - 250
    bg = Image.new("RGB", (W, H), (9, 13, 24))
    d0 = ImageDraw.Draw(bg)
    title = (spec.get("title") or "").upper()
    d0.text((L, 120), title, font=font(46, "bold"), fill=TEXT)
    if spec.get("y_label"):
        d0.text((L, 178), spec["y_label"], font=font(28), fill=MUTED)
    tf = font(26)
    for tv in _nice_ticks(lo, hi):
        y = B - (tv - lo) / (hi - lo) * (B - T)
        d0.line([(L, y), (R, y)], fill=(28, 36, 56), width=2)
        label = f"{tv:,.0f}" if abs(tv) >= 10 else f"{tv:g}"
        d0.text((L - 20 - d0.textlength(label, font=tf), y - 15), label, font=tf, fill=MUTED)
    for tv in _nice_ticks(x0, x1, 5):
        x = L + (tv - x0) / ((x1 - x0) or 1) * (R - L)
        label = f"{tv:g}s"
        d0.text((x - d0.textlength(label, font=tf) / 2, B + 16), label, font=tf, fill=MUTED)
    scr = [(L + (p[0] - x0) / ((x1 - x0) or 1) * (R - L), B - (p[1] - lo) / (hi - lo) * (B - T)) for p in pts]
    draw_t = min(8.0, max(2.5, duration * 0.65))
    marker = spec.get("marker")
    mf = font(30, "bold")

    def frame(t):
        img = bg.copy()
        d = ImageDraw.Draw(img)
        pr = _ease(t / draw_t)
        cutoff_x = L + pr * (R - L)
        shown = [p for p in scr if p[0] <= cutoff_x]
        nxt = next((i for i, p in enumerate(scr) if p[0] > cutoff_x), None)
        if nxt is not None and nxt > 0:
            a, b = scr[nxt - 1], scr[nxt]
            f = (cutoff_x - a[0]) / ((b[0] - a[0]) or 1)
            shown.append((cutoff_x, a[1] + (b[1] - a[1]) * f))
        if len(shown) > 1:
            d.line(shown, fill=CYAN, width=7, joint="curve")
        if shown:
            hx, hy = shown[-1]
            d.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], fill=(255, 255, 255))
        if marker and marker.get("x") is not None:
            mx = L + (marker["x"] - x0) / ((x1 - x0) or 1) * (R - L)
            if cutoff_x >= mx:
                for yy in range(T, B, 22):
                    d.line([(mx, yy), (mx, min(yy + 12, B))], fill=AMBER, width=3)
                if marker.get("label"):
                    lw = d.textlength(marker["label"], font=mf)
                    lx = mx + 18 if mx + 18 + lw < R else mx - 18 - lw
                    d.rounded_rectangle([lx - 10, T + 6, lx + lw + 10, T + 52], radius=8, fill=(9, 13, 24))
                    d.text((lx, T + 12), marker["label"], font=mf, fill=AMBER)
        return img

    frame.static_after = draw_t + 0.05
    return frame


def _image_frames(image_path: Path, duration: float) -> Callable:
    from PIL import Image, ImageFilter

    src = Image.open(image_path).convert("RGB")
    sw, sh = src.size
    aspect = sw / sh
    if 1.3 <= aspect <= 2.2:
        # Photo: fill the frame and slowly push in (Ken Burns).
        scale = max(W / sw, H / sh) * 1.12
        big = src.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
        bw, bh = big.size
        rnd = random.Random(str(image_path))
        dx, dy = rnd.choice([-1, 1]), rnd.choice([-1, 1])

        def frame(t):
            p = _ease(t / max(duration, 0.1))
            cw = W * (1.12 - 0.10 * p)
            ch = cw * H / W
            cw, ch = min(cw, bw), min(ch, bh)
            ox = (bw - cw) / 2 + dx * (bw - cw) / 2 * 0.7 * p
            oy = (bh - ch) / 2 + dy * (bh - ch) / 2 * 0.7 * p
            return big.resize((W, H), Image.BILINEAR, box=(ox, oy, ox + cw, oy + ch))
        return frame

    # Diagram or odd shape: show it whole on a blurred, darkened backdrop.
    cover = max(W / sw, H / sh)
    backdrop = src.resize((int(sw * cover) + 2, int(sh * cover) + 2)).crop((0, 0, W, H)).filter(ImageFilter.GaussianBlur(40))
    backdrop = Image.blend(backdrop, Image.new("RGB", (W, H), BG), 0.7)
    fit = min((W - 240) / sw, (H - 240) / sh)
    fg = src.resize((int(sw * fit), int(sh * fit)), Image.LANCZOS)

    def frame(t):
        p = _ease(t / max(duration, 0.1))
        z = 1.0 + 0.04 * p
        f2 = fg.resize((int(fg.width * z), int(fg.height * z)), Image.BILINEAR) if p > 0 else fg
        img = backdrop.copy()
        img.paste(f2, ((W - f2.width) // 2, (H - f2.height) // 2))
        return img
    return frame


def _title_frames(text: str, duration: float) -> Callable:
    """Fallback visual: the caption (or incident title) on a slowly moving
    dark gradient."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for r in range(800, 0, -80):
        c = int(22 * (1 - r / 800))
        d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=(8 + c, 12 + c, 22 + c * 2))
    tf = font(64, "bold")
    rows = _wrap(d, text, tf, W - 400)[:3]
    y = H / 2 - len(rows) * 45
    for row in rows:
        d.text(((W - d.textlength(row, font=tf)) / 2, y), row, font=tf, fill=TEXT)
        y += 90

    def frame(t):
        return img
    frame.static_after = 0.0
    return frame


def end_card_frame():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for r in range(900, 0, -90):
        c = int(18 * (1 - r / 900))
        d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=(8 + c, 12 + c, 22 + c * 2))
    bf = font(84, "bold")
    t = BRAND.upper()
    d.text(((W - d.textlength(t, font=bf)) / 2, 120), t, font=bf, fill=TEXT)
    d.rectangle([W / 2 - 110, 236, W / 2 + 30, 244], fill=RED)
    d.rectangle([W / 2 + 44, 236, W / 2 + 110, 244], fill=AMBER)
    sf = font(34)
    s = "New episode every week"
    d.text(((W - d.textlength(s, font=sf)) / 2, 272), s, font=sf, fill=MUTED)
    return img


# ------------------------------------------------------------- encoding ---

def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


def _encode_frames(frame_fn: Callable, overlay, duration: float, out: Path) -> None:
    from PIL import Image

    n = max(1, round(duration * FPS))
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-vf", "fade=t=in:st=0:d=0.3", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-r", str(FPS), str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Once a visual has finished animating its frames stop changing, so the
    # last one is reused instead of redrawn -- most of a long scene is a hold.
    static_after = getattr(frame_fn, "static_after", None)
    held = None
    try:
        for i in range(n):
            t = i / FPS
            if held is None or static_after is None or t < static_after:
                img = frame_fn(t).convert("RGBA")
                if overlay is not None:
                    img = Image.alpha_composite(img, overlay)
                b = img.convert("RGB").tobytes()
                if static_after is not None and t >= static_after:
                    held = b
            else:
                b = held
            proc.stdin.write(b)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding a scene: {err[-400:]}")


def _encode_stock(stock_path: Path, overlay, duration: float, out: Path, work: Path) -> None:
    ov = work / f"{out.stem}_overlay.png"
    overlay.save(ov)
    try:
        subprocess.run(
            [_ffmpeg(), "-y", "-v", "error", "-stream_loop", "-1", "-i", str(stock_path), "-i", str(ov),
             "-filter_complex",
             f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},"
             f"eq=brightness=-0.05:saturation=0.85,setsar=1[v];[v][1:v]overlay=0:0,fade=t=in:st=0:d=0.3,format=yuv420p[o]",
             "-map", "[o]", "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-r", str(FPS), str(out)],
            check=True, capture_output=True, timeout=900,
        )
    finally:
        ov.unlink(missing_ok=True)


# --------------------------------------------------------------- scenes ---

class SceneContext:
    """What a scene render needs besides its own spec."""

    def __init__(self, project_dir: Path, title: str, report_images: List[dict], cache_dir: Path, fetch_tiles: bool = True):
        self.project_dir = project_dir
        self.title = title
        self.report_images = report_images
        self.cache_dir = cache_dir
        self.fetch_tiles = fetch_tiles

    def report_image(self, index: Optional[int], scene_index: int) -> Optional[Path]:
        if not self.report_images:
            return None
        i = index if index is not None else scene_index
        return self.project_dir / "report_images" / self.report_images[i % len(self.report_images)]["file"]


def scene_frame_source(spec: dict, duration: float, ctx: SceneContext, scene_index: int):
    """(frame_fn or None, stock_path or None, attribution, credit) for a
    scene -- stock scenes are encoded by ffmpeg directly, the rest drawn
    frame by frame."""
    visual = spec.get("visual")
    if visual == "map":
        fn, attribution = _map_frames(spec, duration, ctx.cache_dir / "tiles", ctx.fetch_tiles)
        return fn, None, attribution, "Map data © OpenStreetMap contributors" if attribution else None
    if visual == "cockpit":
        return _cockpit_frames(spec, duration), None, "", None
    if visual == "chart":
        return _chart_frames(spec, duration), None, "", None
    if visual == "stock":
        item = fetch_stock(spec.get("stock_query") or "airliner", int(spec.get("stock_index") or 0), ctx.cache_dir / "stock")
        if item:
            return None, Path(item["path"]), "", f"Stock footage: {item.get('user') or 'Pexels'} via Pexels ({item.get('page')})"
    img = ctx.report_image(spec.get("image_index"), scene_index) if visual in ("report", "stock") else None
    if img and img.exists():
        return _image_frames(img, duration), None, "", None
    return _title_frames(spec.get("caption") or ctx.title, duration), None, "", None


def spec_hash(spec: dict, duration: float) -> str:
    blob = json.dumps({"spec": spec, "d": round(duration, 3), "brand": BRAND, "v": 1}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def render_scene(spec: dict, duration: float, ctx: SceneContext, scene_index: int, out_dir: Path) -> Tuple[Path, Optional[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = spec_hash(spec, duration)
    out = out_dir / f"scene{scene_index:02d}_{key}.mp4"
    credit_file = out.with_suffix(".credit")
    if out.exists():
        return out, (credit_file.read_text(encoding="utf-8") if credit_file.exists() else None)
    fn, stock, attribution, credit = scene_frame_source(spec, duration, ctx, scene_index)
    overlay = overlay_layer(spec.get("caption") or "", attribution)
    tmp = out.with_name(f".{out.name}")
    if stock is not None:
        _encode_stock(stock, overlay, duration, tmp, out_dir)
    else:
        _encode_frames(fn, overlay, duration, tmp)
    tmp.replace(out)
    if credit:
        credit_file.write_text(credit, encoding="utf-8")
    return out, credit


def preview_still(spec: dict, duration: float, ctx: SceneContext, scene_index: int, out: Path) -> Path:
    """One representative frame of a scene (for the Visuals grid), without
    encoding the whole scene. Stock scenes use Pexels' own still."""
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    fn, stock, attribution, _credit = scene_frame_source(spec, duration, ctx, scene_index)
    if stock is not None:
        tmp = out.with_suffix(".src.jpg")
        subprocess.run([_ffmpeg(), "-y", "-v", "error", "-ss", "1", "-i", str(stock), "-frames:v", "1",
                        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}", str(tmp)],
                       check=True, capture_output=True, timeout=120)
        img = Image.open(tmp).convert("RGBA")
        tmp.unlink(missing_ok=True)
    else:
        img = fn(min(duration, 8.0) * 0.95).convert("RGBA")
    img = Image.alpha_composite(img, overlay_layer(spec.get("caption") or "", attribution)).convert("RGB")
    img.resize((640, 360), Image.LANCZOS).save(out, quality=85)
    return out


# ------------------------------------------------------------- assembly ---

def render_video(
    project_dir: Path, scenes: List[dict], scene_starts: List[float], narration_wav: Path, narration_duration: float,
    title: str, report_images: List[dict], music: Optional[Path], cache_dir: Path,
    on_progress: Callable[[float, str], None] = lambda p, m: None, fetch_tiles: bool = True,
) -> Tuple[Path, List[str]]:
    """Render every scene timed to the narration, add the end card, lay the
    narration (and optional music, ducked under the voice) over it."""
    rdir = project_dir / "render"
    rdir.mkdir(parents=True, exist_ok=True)
    ctx = SceneContext(project_dir, title, report_images, cache_dir, fetch_tiles)
    n = len(scenes)
    bounds = list(scene_starts) + [narration_duration + 0.8]
    parts, credits = [], []
    for i, sc in enumerate(scenes):
        dur = max(1.0, bounds[i + 1] - bounds[i])
        on_progress(0.05 + 0.8 * i / n, f"Rendering scene {i + 1} of {n}...")
        path, credit = render_scene(sc["spec"], dur, ctx, i, rdir)
        parts.append(path)
        if credit and credit not in credits:
            credits.append(credit)
    on_progress(0.87, "Rendering the end card...")
    end = rdir / "endcard.mp4"
    if not end.exists():
        card = end_card_frame()
        _encode_frames(lambda t: card, None, END_CARD_SECONDS, end)
    parts.append(end)

    on_progress(0.9, "Joining scenes...")
    listing = rdir / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    video_only = rdir / "video_only.mp4"
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(video_only)],
                   check=True, capture_output=True, timeout=600)
    total = bounds[-1] + END_CARD_SECONDS

    on_progress(0.95, "Mixing the audio...")
    final = project_dir / "final.mp4"
    tmp = project_dir / ".final.partial.mp4"
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", str(video_only), "-i", str(narration_wav)]
    if music is not None and music.exists():
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        fc = (f"[1:a]aresample=48000,apad,atrim=0:{total:.3f},asplit=2[voice][key];"
              f"[2:a]aresample=48000,atrim=0:{total:.3f},volume=0.22,afade=t=in:d=2,afade=t=out:st={max(0.0, total - 4):.3f}:d=4[bed];"
              f"[bed][key]sidechaincompress=threshold=0.02:ratio=10:attack=30:release=600[ducked];"
              f"[voice][ducked]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        fc = f"[1:a]aresample=48000,apad,atrim=0:{total:.3f}[a]"
    cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-t", f"{total:.3f}", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    tmp.replace(final)
    video_only.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)
    on_progress(1.0, "Done.")
    return final, credits


# ------------------------------------------------------- title & chapters ---

def _fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def write_publish_text(incident_title: str, scenes: List[dict], scene_starts: List[float], credits: List[str], report_ref: str) -> dict:
    """Three title options and a description with chapters (YouTube wants
    the first at 0:00, at least 3, each at least 10 seconds long)."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    listing = "\n".join(f"Scene {i + 1} (starts {_fmt_ts(scene_starts[i])}): {s['narration'][:400]}" for i, s in enumerate(scenes))
    prompt = f"""You write YouTube titles and descriptions for a documentary channel called
"{BRAND}" that tells true aviation incident stories from official reports.

Incident: {incident_title}

The video's scenes:
{listing}

Write:
- "titles": 3 title options, each under 70 characters. Specific and
  intriguing, built on the real stakes (seconds, altitude, the decision),
  factual, no clickbait words, no ALL CAPS.
- "description": 2 short paragraphs (under 90 words total) saying what
  the video covers, plain and factual.
- "chapters": 5 to 8 chapters as {{"scene": <scene number where it starts>, "title": "..."}},
  the first at scene 1, titles under 40 characters.

Respond with ONLY a JSON array holding one object:
[{{"titles": ["...", "...", "..."], "description": "...", "chapters": [{{"scene": 1, "title": "..."}}]}}]
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    obj = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else (data if isinstance(data, dict) else {})
    titles = [" ".join(str(t).split())[:100] for t in obj.get("titles") or [] if str(t).strip()][:3] or [incident_title]

    chapters, last = [], -999.0
    for ch in obj.get("chapters") or []:
        n = int(_num((ch or {}).get("scene")) or 0) - 1 if isinstance(ch, dict) else -1
        if not 0 <= n < len(scene_starts):
            continue
        t = 0.0 if not chapters else scene_starts[n]
        if chapters and t - last < 10:
            continue
        chapters.append((t, " ".join(str(ch.get("title") or "").split())[:60] or f"Part {len(chapters) + 1}"))
        last = t
    if chapters and chapters[0][0] != 0.0:
        chapters.insert(0, (0.0, "Introduction"))
    lines = [" ".join(str(obj.get("description") or "").split("\n\n")).strip()]
    if len(chapters) >= 3:
        lines += ["", "Chapters"] + [f"{_fmt_ts(t)} {name}" for t, name in chapters]
    src = ["", "Sources and credits", f"Official investigation report: {report_ref}" if report_ref else "Official investigation report (NTSB)"]
    src += credits
    lines += src
    return {"titles": titles, "description": "\n".join(lines).strip()}
