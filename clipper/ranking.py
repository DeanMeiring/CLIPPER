"""Ranking Shorts: "Ranking <streamer>'s Funniest Moments", the format Dean
liked from other channels -- a 1-5 list on screen from the first frame,
filled in out of order as each moment plays, #1 last.

What makes ours more than a re-upload: Dean (or a voice he picked on the
Voices page) says where each moment lands and why, right after it plays
-- a blind-rank call like "That's a four. He knew it too." YouTube's
Shorts feed cuts reach for clips "edited together with little or no
narrative"; a ranking with the narrator's own take is commentary.

Built from clips the normal pipeline already found (their downloaded
source and cached transcript), so nothing is downloaded again:

  plan()    Claude orders 3-5 moments (1 = best) and, for each, picks the
            seconds to show, a 2-4 word label, an emoji and the narrator's
            line. normalize_plan() keeps every pick inside its clip.
  render()  one 1080x1920 segment per moment (in the reveal order) and a
            concat. A wide stream sits in the middle under the list; a
            vertical one fills the screen with the list over it, like the
            channels Dean showed. The streamer's words are captioned the
            same way as the Shorts; the narrator's line is captioned too.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from .longform_beats import F, text_w

VW, VH = 1080, 1920
MIN_SLOTS, MAX_SLOTS = 3, 5
MOMENT_MIN, MOMENT_MAX = 3.0, 9.0
MAX_TOTAL = 58.0          # the app uploads Shorts up to 60 s
FREEZE_PAD = 0.35         # held after the narrator's line
LINE_MAX_CHARS = 110
LINE_MAX_SECONDS = 8.0    # a recorded line longer than this isn't a quick call
LABEL_MAX_CHARS = 26
YEL = (250, 204, 21)
WHITE = (255, 255, 255)
RANK_COLOURS = {1: (250, 204, 21), 2: (214, 219, 228), 3: (255, 146, 56)}
EMOJI_DIR = Path(__file__).parent / "assets" / "emoji"

# Names Claude picks from -> the bundled Fluent Emoji file.
EMOJI = {
    "skull": "1f480", "tears-of-joy": "1f602", "rolling-laughing": "1f923", "screaming": "1f631",
    "sobbing": "1f62d", "flushed": "1f633", "fire": "1f525", "mind-blown": "1f92f", "open-mouth": "1f62e",
    "angry": "1f621", "cursing": "1f92c", "eyes": "1f440", "cool": "1f60e", "grimacing": "1f62c",
    "hot": "1f975", "cold": "1f976", "clown": "1f921", "trophy": "1f3c6", "hundred": "1f4af",
    "brain": "1f9e0", "ghost": "1f47b", "tired": "1f62b", "eye-roll": "1f644", "smirk": "1f60f",
}
DEFAULT_EMOJI = "tears-of-joy"
THEMES = ["Funniest Moments", "Biggest Rage Moments", "Scariest Moments", "Best Clutches", "Most Awkward Moments"]


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


# ------------------------------------------------------------------ plan ---

def reveal_order(n: int, seed: str = "") -> List[int]:
    """The order the ranks fill in: shuffled, #1 always last -- viewers stay
    to see what takes the top spot."""
    others = list(range(2, n + 1))
    random.Random(seed or n).shuffle(others)
    return others + [1]


def transcript(words: List[dict], start: float = 0.0, end: Optional[float] = None) -> str:
    """What was said in [start, end] of a clip, a line per ~2 s with its
    time, and the gaps marked, so Claude can cut on the words."""
    ws = [w for w in words if w["s"] >= start - 0.05 and (end is None or w["e"] <= end + 0.05)]
    if not ws:
        return "(no speech)"
    lines, cur, t0, last = [], [], None, start
    for w in ws:
        if w["s"] - last >= 1.5:
            if cur:
                lines.append(f"[{t0:.1f}] {' '.join(cur)}")
                cur = []
            lines.append(f"(silence {w['s'] - last:.0f}s)")
        if not cur:
            t0 = w["s"]
        cur.append(w["w"])
        last = w["e"]
        if w["e"] - t0 >= 2.0:
            lines.append(f"[{t0:.1f}] {' '.join(cur)}")
            cur = []
    if cur:
        lines.append(f"[{t0:.1f}] {' '.join(cur)}")
    return "\n".join(lines)


def _prompt(clips: List[dict], streamer: str, theme: str) -> str:
    blocks = []
    for c in clips:
        blocks.append(
            f"## Moment {c['key']} ({c['duration']:.0f} s)\n"
            f"Picked because: {c.get('reason') or c.get('title') or ''}\n"
            f"Said (seconds from the start of the clip):\n{transcript(c['words'])}"
        )
    return f"""You're helping make a YouTube Short in the "ranking" format: "Ranking {streamer}'s {theme}".
A list numbered 1 to {len(clips)} is on screen the whole time. The moments below play one by one, and after each one
the narrator says where it goes in the list, in his own words. Number 1 is shown last.

{chr(10).join(blocks)}

For every moment, decide:
- rank: 1 is the best of these for "{theme}". Use each rank once.
- start, end: the seconds of the clip to show, {MOMENT_MIN:.0f}-{MOMENT_MAX:.0f} seconds. Start one or two seconds before the
  turn so a viewer gets it without the build-up, and end right after the reaction. No dead air at either end.
- label: 2-4 words naming the moment the way a viewer would remember it ("Blamed the controller", "Mom walked in").
  Plain words, no hype words, no emoji, not all caps.
- emoji: one of {", ".join(EMOJI)}.
- line: what the narrator says right after the moment as he puts it in the list, 4-12 words, said out loud.
  It gives his own opinion or a quick reason, and says the rank ("That's a four. He knew it too.",
  "Number two, and honestly it's close."). Never just describe what happened on screen.

Also write a YouTube title (under 70 characters, names the streamer) and a one or two sentence description.

Return only JSON:
{{"moments": [{{"key": "A", "rank": 1, "start": 0.0, "end": 6.5, "label": "...", "emoji": "skull", "line": "..."}}],
  "title": "...", "description": "..."}}
"""


def _clean(text, n: int) -> str:
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️‍\"]", "", str(text or ""))
    return " ".join(t.split())[:n].strip()


def normalize_plan(ans: dict, clips: List[dict], streamer: str, theme: str) -> dict:
    """Claude's answer made safe: every clip ranked exactly once (missing
    ones keep the order they were given in), cuts inside the clip and
    MOMENT_MIN-MOMENT_MAX long, labels and lines short and plain."""
    by_key = {c["key"]: c for c in clips}
    got = {}
    for m in (ans or {}).get("moments") or []:
        if isinstance(m, dict) and m.get("key") in by_key and m["key"] not in got:
            got[m["key"]] = m
    try_rank = lambda k: (int(got[k].get("rank")) if str((got.get(k) or {}).get("rank", "")).isdigit() else 99)
    order = sorted(by_key, key=lambda k: (try_rank(k), [c["key"] for c in clips].index(k)))
    slots = []
    for rank, key in enumerate(order, start=1):
        c, m = by_key[key], got.get(key) or {}
        dur = float(c["duration"])
        try:
            s, e = float(m.get("start")), float(m.get("end"))
        except (TypeError, ValueError):
            s, e = max(0.0, dur - MOMENT_MAX), dur
        s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
        if e - s > MOMENT_MAX:
            s = e - MOMENT_MAX
        if e - s < MOMENT_MIN:
            e = min(dur, s + MOMENT_MIN)
            s = max(0.0, e - MOMENT_MIN)
        emoji = m.get("emoji") if m.get("emoji") in EMOJI else DEFAULT_EMOJI
        label = _clean(m.get("label"), LABEL_MAX_CHARS) or _clean(c.get("title"), LABEL_MAX_CHARS) or f"Moment {key}"
        line = _clean(m.get("line"), LINE_MAX_CHARS) or f"Number {rank}."
        slots.append({"key": key, "rank": rank, "start": round(s, 2), "end": round(e, 2),
                      "label": label, "emoji": emoji, "line": line})
    title = _clean((ans or {}).get("title"), 100) or f"Ranking {streamer}'s {theme}"
    return {"slots": slots, "title": title, "description": _clean((ans or {}).get("description"), 400)}


def plan(clips: List[dict], streamer: str, theme: str) -> dict:
    """clips: [{"key", "duration", "title", "reason", "words": [{"w","s","e"}]
    relative to the clip start}]. Returns normalize_plan()'s dict."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    try:
        ans = _ask_claude_for_json(_prompt(clips, streamer, theme), None, DEFAULT_MODEL)
        if isinstance(ans, list):
            ans = ans[0] if ans and isinstance(ans[0], dict) else {}
    except Exception as e:  # noqa: BLE001 - a ranking in clip order still works
        print(f"[ranking] Claude failed, keeping the clips' order: {e}", flush=True)
        ans = {}
    return normalize_plan(ans if isinstance(ans, dict) else {}, clips, streamer, theme)


def fit_lengths(slots: List[dict], line_seconds: List[float]) -> List[dict]:
    """Shorten the moments (from their start, keeping the payoff) until the
    whole Short with the narrator's lines fits in MAX_TOTAL."""
    talk = sum(t + FREEZE_PAD for t in line_seconds if t)
    cap = max(MOMENT_MIN, min(MOMENT_MAX, (MAX_TOTAL - talk) / max(1, len(slots))))
    out = []
    for s in slots:
        s = dict(s)
        if s["end"] - s["start"] > cap:
            s["start"] = round(s["end"] - cap, 2)
        out.append(s)
    return out


# ---------------------------------------------------------------- render ---

def _italic(img, xy, s: str, fnt, fill, sw: int = 7, shear: float = 0.2) -> float:
    """Bold italic text with a black outline (Inter Black, sheared) -- the
    look of the ranking channels' lists."""
    from PIL import Image, ImageDraw

    h = int(fnt.size * 1.5 + sw * 2)
    pad = int(h * shear)
    layer = Image.new("RGBA", (int(text_w(fnt, s) + sw * 2 + pad + 20), h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((sw + pad, sw), s, font=fnt, fill=fill, stroke_width=sw, stroke_fill=(0, 0, 0))
    layer = layer.transform(layer.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), Image.BICUBIC)
    img.alpha_composite(layer, (int(xy[0]), int(xy[1])))
    return text_w(fnt, s)


def _emoji(img, key: str, xy, size: int) -> None:
    from PIL import Image

    path = EMOJI_DIR / f"{EMOJI.get(key, EMOJI[DEFAULT_EMOJI])}.webp"
    if path.is_file():
        e = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
        img.alpha_composite(e, (int(xy[0]), int(xy[1])))


def _fit_font(name: str, size: int, text: str, max_w: float, min_size: int = 30):
    while size > min_size and text_w(F(name, size), text) > max_w:
        size -= 2
    return F(name, size)


class Layout:
    """Where things go. wide: the stream frame across the middle under the
    list. tall: a vertical stream fills the screen, the list over its left
    side (as in the ranking Shorts Dean sent)."""

    def __init__(self, src_w: int, src_h: int, n: int):
        self.n = n
        self.wide = src_w / max(1, src_h) >= 1.2
        if self.wide:
            self.fg_h = int(round(VW * src_h / src_w / 2)) * 2
            self.video_y = 860
            self.list_y, self.row = 330, 96
            self.mark_y = self.video_y + self.fg_h + 14
        else:
            self.fg_h, self.video_y = VH, 0
            self.list_y, self.row = 560, 104
            self.mark_y = 1440

    def video_filter(self, src: str, out: str) -> str:
        if self.wide:
            return (f"[{src}]split[s1][s2];[s1]scale={VW}:{VH}:force_original_aspect_ratio=increase,crop={VW}:{VH},"
                    f"boxblur=28:3,eq=brightness=-0.35[bg];[s2]scale={VW}:{self.fg_h}:force_original_aspect_ratio=decrease,"
                    f"scale=trunc(iw/2)*2:trunc(ih/2)*2[fg];"
                    f"[bg][fg]overlay=(main_w-overlay_w)/2:{self.video_y}+({self.fg_h}-overlay_h)/2[{out}]")
        return f"[{src}]scale={VW}:{VH}:force_original_aspect_ratio=increase,crop={VW}:{VH}[{out}]"


def _board(lay: Layout, title: str, theme: str, brand: str, slots_by_rank: dict, revealed: List[int],
           only: Optional[int] = None):
    """The overlay: title + list with `revealed` ranks filled in. With
    `only`, just that one entry (the one popping in)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    if only is None:
        if not lay.wide:  # darken behind the title over a full-screen video
            shade = Image.new("RGBA", (VW, 420), (0, 0, 0, 0))
            d = ImageDraw.Draw(shade)
            for y in range(420):
                d.line([(0, y), (VW, y)], fill=(0, 0, 0, int(170 * (1 - y / 420))))
            img.alpha_composite(shade, (0, 0))
        head, name = title
        f1 = _fit_font("Inter-Black", 76, f"{head}{name}", VW - 120)
        x = (VW - text_w(f1, head + name)) / 2 - 18
        x += _italic(img, (x, 110), head, f1, WHITE)
        _italic(img, (x, 110), name, f1, YEL)
        f2 = _fit_font("Inter-Black", 76, theme, VW - 120)
        _italic(img, ((VW - text_w(f2, theme)) / 2 - 18, 200), theme, f2, WHITE)
        bf = F("Inter-Black", 40)
        _italic(img, ((VW - text_w(bf, brand)) / 2 - 10, lay.mark_y), brand, bf, YEL, sw=4)
    nf = F("Inter-Black", 72 if lay.n <= 4 else 68)
    for r in range(1, lay.n + 1):
        if only is not None and r != only:
            continue
        y = lay.list_y + (r - 1) * lay.row
        filled = r in revealed or r == only
        _italic(img, (36, y), f"{r}.", nf, RANK_COLOURS.get(r, WHITE) if filled else WHITE)
        if filled:
            s = slots_by_rank[r]
            lf = _fit_font("Inter-Black", 50, s["label"], VW - 150 - 110)
            w = _italic(img, (136, y + 16), s["label"], lf, WHITE, sw=6)
            _emoji(img, s["emoji"], (136 + w + 34, y + 10), 64)
    return img


def _segment_ass(words: List[dict], line: str, line_at: float, line_dur: float, path: Path) -> Path:
    """The streamer's words (the Shorts' caption style) and then the
    narrator's line, a few words at a time while it's said."""
    from .captions import _escape_ass_text, _fmt_ts, build_ass
    from .transcribe import Word

    build_ass([Word(text=w["w"], start=w["s"], end=w["e"]) for w in words], 0.0, path, punchy=True)
    if line and line_dur > 0:
        parts = line.split()
        groups = [parts[i:i + 3] for i in range(0, len(parts), 3)]
        step = line_dur / max(1, len(groups))
        extra = []
        for k, g in enumerate(groups):
            a, b = line_at + k * step, line_at + (k + 1) * step
            extra.append(f"Dialogue: 2,{_fmt_ts(a)},{_fmt_ts(b)},CaptionPop,,0,0,0,,{_escape_ass_text(' '.join(g).upper())}")
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(extra))
    return path


def _has_audio(path: Path) -> bool:
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in r.stderr


def _run(cmd: list, what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {r.stderr[-600:]}")


def video_size(path: Path) -> tuple:
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


def render(items: List[dict], streamer: str, theme: str, brand: str, out: Path, seed: str = "") -> dict:
    """items, in rank order (1 first): {"rank", "video" (Path), "start",
    "end" (in that file), "words" ([{"w","s","e"}] relative to start),
    "label", "emoji", "line", "take" (Path to the narrator's wav or None),
    "take_seconds"}. Writes `out`; returns {"duration", "reveal"}."""
    from .render import _escape_for_filter

    n = len(items)
    by_rank = {it["rank"]: it for it in items}
    reveal = reveal_order(n, seed)
    w0, h0 = video_size(items[0]["video"])
    lay = Layout(w0, h0, n)
    tmp = out.parent / f".{out.stem}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    head = ("Ranking ", f"{streamer}'s")
    segs, revealed, total = [], [], 0.0
    try:
        for i, rank in enumerate(reveal):
            it = by_rank[rank]
            length = round(float(it["end"]) - float(it["start"]), 3)
            take = it.get("take")
            talk = float(it.get("take_seconds") or 0) if take else 0.0
            hold = talk + FREEZE_PAD if take else 0.0
            seg_len = length + hold
            pop_at = length if take else max(0.6, length - 1.2)
            base = tmp / f"base{i}.png"
            _board(lay, head, theme, brand, by_rank, list(revealed)).save(base)
            pop = tmp / f"pop{i}.png"
            _board(lay, head, theme, brand, by_rank, [], only=rank).save(pop)
            ass = _segment_ass(it.get("words") or [], it.get("line", "") if take else "", length + 0.05, talk, tmp / f"cap{i}.ass")
            cmd = [_ffmpeg(), "-y", "-v", "error", "-ss", f"{float(it['start']):.3f}", "-t", f"{length:.3f}", "-i", str(it["video"]),
                   "-loop", "1", "-i", str(base), "-loop", "1", "-i", str(pop)]
            vf = (f"[0:v]fps=30,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={hold:.3f}[v0];"
                  + lay.video_filter("v0", "v1") +
                  f";[v1][1:v]overlay=0:0[v2];[2:v]format=rgba,fade=t=in:st={pop_at:.3f}:d=0.2:alpha=1[p];"
                  f"[v2][p]overlay=0:0:enable='gte(t,{pop_at:.3f})'[v3];"
                  f"[v3]ass='{_escape_for_filter(ass)}',format=yuv420p[v]")
            k = 3
            if _has_audio(it["video"]):
                clip_a = "[0:a]"
            else:
                cmd += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                clip_a, k = f"[{k}:a]", k + 1
            af = (f"{clip_a}aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,loudnorm=I=-16:TP=-1.5:LRA=11,"
                  f"aresample=48000,afade=t=out:st={max(0.0, length - 0.25):.3f}:d=0.25,apad=whole_dur={seg_len:.3f}[ca]")
            if take:
                cmd += ["-i", str(take)]
                af += (f";[{k}:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I=-15:TP=-1.5:LRA=11,aresample=48000,"
                       f"adelay={int(length * 1000)}|{int(length * 1000)},apad=whole_dur={seg_len:.3f}[va];"
                       f"[ca][va]amix=inputs=2:normalize=0:duration=first[a]")
            else:
                af += ";[ca]anull[a]"
            seg = tmp / f"seg{i}.mp4"
            _run(cmd + ["-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{seg_len:.3f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
                        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k", str(seg)], f"moment {rank}")
            segs.append(seg)
            revealed.append(rank)
            total += seg_len
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{s.name}'\n" for s in segs), encoding="utf-8")
        _run([_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
              "-movflags", "+faststart", str(out)], "joining the moments")
    finally:
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()
    return {"duration": round(total, 2), "reveal": reveal}


def publish_text(title: str, description: str, streamer: str) -> dict:
    """Upload title/description; the streamer's name as a hashtag."""
    desc = description.strip()
    tag = re.sub(r"[^A-Za-z0-9]", "", streamer)
    if tag:
        desc = (desc + "\n\n" if desc else "") + f"#{tag} #ranking"
    return {"title": title[:100], "description": desc}
