"""Picture packs for Ball Evolution's "merge into final form" videos.

A pack is 6-9 pictures in order, from the most normal to the most cursed
(the last one is the final form nobody sees coming). Dean asked for meme
pictures instead of emoji ("the emoji doesn't grab attention"), after a
music.bouncingball reel with cut-out meme cats did 308K likes.

Packs come from two places:

* **Upload** -- Dean picks the pictures for one video and puts them in
  order on /ball-evolution.
* **Automatic** -- Claude plans a chain of GIPHY sticker searches from
  normal to cursed, the app searches GIPHY (stickers are already cut out,
  like the reference), and Claude then looks at every candidate and keeps
  only ones that fit and are safe: no nudity, gore, real people, slurs.
  So the autopilot never posts a picture nothing has looked at. Needs
  ``GIPHY_API_KEY`` (free) and ``ANTHROPIC_API_KEY``.

Stored in ``BASE_DIR/_balls/packs/<id>/``: ``pack.json`` and ``01.png`` ..
``09.png`` (RGBA, trimmed, at most 512 px).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image, ImageSequence

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")
MIN_ITEMS, MAX_ITEMS, AUTO_ITEMS = 6, 9, 8
MAX_SIDE = 512
GIPHY_SEARCH = "https://api.giphy.com/v1/stickers/search"


# ---- images ----------------------------------------------------------------

def normalize(data: bytes) -> Image.Image:
    """Any picture (png/jpg/webp/gif, first frame) -> RGBA, see-through
    margins trimmed, at most MAX_SIDE px."""
    im = Image.open(io.BytesIO(data))
    im = next(iter(ImageSequence.Iterator(im))).convert("RGBA")
    box = im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
    if box:
        im = im.crop(box)
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im


def is_cutout(im: Image.Image) -> bool:
    """True when the picture has a see-through background (a sticker), so it
    is drawn as it is; a plain photo is drawn in a circle instead."""
    a = im.getchannel("A")
    w, h = im.size
    edge = [a.getpixel((x, y)) for x in (0, w - 1) for y in range(0, h, max(1, h // 12))]
    edge += [a.getpixel((x, y)) for y in (0, h - 1) for x in range(0, w, max(1, w // 12))]
    return sum(1 for v in edge if v < 128) > len(edge) * 0.3


# ---- the store -------------------------------------------------------------

class PackStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, pid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{10}", pid or ""):
            raise KeyError(pid)
        return self.root / pid

    def save(self, title: str, items: List[tuple], source: str, extra: Optional[dict] = None) -> dict:
        """items: [(name, RGBA image)] in order, last = final form."""
        if not MIN_ITEMS <= len(items) <= MAX_ITEMS:
            raise ValueError(f"a pack needs {MIN_ITEMS}-{MAX_ITEMS} pictures")
        pid = secrets.token_hex(5)
        d = self.root / pid
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for k, (name, im) in enumerate(items, 1):
            fn = f"{k:02d}.png"
            im.save(d / fn)
            out.append({"name": (name or f"Level {k}").strip()[:30], "file": fn, "cutout": is_cutout(im)})
        pack = {"id": pid, "title": (title or "").strip()[:60] or f"Pack {pid[:4]}", "source": source,
                "items": out, "created_at": time.time(), "used_by": None, **(extra or {})}
        (d / "pack.json").write_text(json.dumps(pack, indent=1))
        return pack

    def load(self, pid: str) -> dict:
        return json.loads((self.dir(pid) / "pack.json").read_text())

    def packs(self) -> List[dict]:
        out = []
        if self.root.is_dir():
            for d in self.root.iterdir():
                try:
                    out.append(self.load(d.name))
                except (KeyError, OSError, ValueError):
                    continue
        return sorted(out, key=lambda p: p.get("created_at") or 0)

    def mark_used(self, pid: str, vid: str) -> None:
        p = self.load(pid)
        p["used_by"] = vid
        (self.dir(pid) / "pack.json").write_text(json.dumps(p, indent=1))

    def delete(self, pid: str) -> None:
        shutil.rmtree(self.dir(pid), ignore_errors=True)

    def next_upload(self) -> Optional[dict]:
        """The oldest uploaded pack no video has used yet."""
        return next((p for p in self.packs() if p.get("source") == "upload" and not p.get("used_by")), None)

    def recipe_pack(self, pack: dict) -> dict:
        """What goes into a video's recipe: enough to draw it from disk."""
        return {"id": pack["id"], "dir": str(self.dir(pack["id"])), "title": pack["title"],
                "source": pack.get("source"), "credits": bool(pack.get("credits")),
                "items": [{"name": i["name"], "file": i["file"], "cutout": i["cutout"]} for i in pack["items"]]}


# ---- automatic packs: Claude plans, GIPHY finds, Claude checks ----------------

_PLAN_PROMPT = """You plan the pictures for a short "merge into final form" video: \
round pictures fall into a ring, two of the same merge into the next one, and the \
chain ends on a final form nobody saw coming. Viewers stay to see the last one.

Plan a chain of exactly {n} steps, from completely normal to the most cursed, \
absurd or unexpected. Each step is a search on GIPHY *stickers* (cut-out \
pictures). Good chains escalate one idea: e.g. kitten -> chubby cat -> cat in \
a costume -> buff cat -> cat with a human face -> ... -> something absurd.

Rules:
- Funny, weird, cursed is the point. No real people (no celebrities, no \
streamers, no faces of actual humans), no nudity, gore, drugs, weapons, \
slurs, politics or religion.
- Search words must be short and common sticker words (2-4 words), so GIPHY \
actually finds stickers. Give two different searches per step.
- Names are what the viewer sees ("NEW: Buff cat!"): 1-3 words.
{avoid}
Answer with JSON only:
{{"title": "<pack title, 2-4 words>", "hook": "<question for the top of the video, \
max 40 characters, e.g. 'What will the cat become?'>", \
"steps": [{{"name": "...", "queries": ["...", "..."]}}, ...]}}"""

_PICK_PROMPT = """These are candidate stickers for a "merge into final form" video. \
The chain goes from normal to cursed; the last step is the final form.

{steps}

The pictures follow, labelled "step S candidate C". For every step pick the ONE \
candidate that best shows that step and stands out as a clear, funny picture on \
a black background. Reject (do not pick) anything with nudity or sexual content, \
gore or injury, a real human's face or body, text that is offensive, logos, or \
anything a general audience would find disturbing rather than funny. Prefer \
pictures that read well small. Use -1 when no candidate of a step is fine. \
Never pick the same picture twice.

Answer with JSON only: {{"picks": [<candidate index or -1 for step 1>, ...]}}"""


def _claude_text(content, api_key: str, max_tokens: int = 1500) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(model=DEFAULT_MODEL, max_tokens=max_tokens,
                                  messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def _pick_list(raw: str, n: int) -> list:
    """The picks from Claude's answer, read leniently: the numbers inside
    "picks": [...] (comments or stray text around them don't matter)."""
    m = re.search(r'"?picks"?\s*:\s*\[([^\]]*)\]', raw)
    nums = re.findall(r"-?\d+", m.group(1) if m else "")
    if len(nums) < n:
        raise ValueError(f"couldn't read the picture choices: {raw[:200]!r}")
    return [int(x) for x in nums[:n]]


def _claude_json(content, api_key: str, max_tokens: int = 1500):
    raw = _claude_text(content, api_key, max_tokens).strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)


def plan_chain(api_key: str, avoid: List[str], n: int = AUTO_ITEMS) -> dict:
    av = ("- Don't repeat these recent packs: " + "; ".join(avoid[-20:])) if avoid else ""
    plan = _claude_json(_PLAN_PROMPT.format(n=n, avoid=av), api_key)
    steps = [s for s in plan.get("steps") or [] if s.get("name") and s.get("queries")]
    if len(steps) != n:
        raise ValueError(f"the plan had {len(steps)} steps, not {n}")
    return {"title": str(plan.get("title") or "")[:60], "hook": str(plan.get("hook") or "")[:60], "steps": steps}


def giphy_search(query: str, key: str, limit: int = 4, http=None) -> List[dict]:
    import requests
    http = http or requests
    r = http.get(GIPHY_SEARCH, params={"api_key": key, "q": query, "limit": limit, "rating": "pg",
                                       "lang": "en"}, timeout=20)
    r.raise_for_status()
    out = []
    for g in r.json().get("data") or []:
        imgs = g.get("images") or {}
        # still versions first (a plain picture is all that's needed), then the animated ones
        urls = [((imgs.get(k) or {}).get(f)) for k, f in (("original_still", "url"), ("fixed_height_still", "url"),
                                                            ("original", "webp"), ("fixed_height", "webp"),
                                                            ("original", "url"))]
        urls = [u for u in dict.fromkeys(urls) if u]
        if urls:
            out.append({"id": g.get("id"), "title": g.get("title") or "", "urls": urls,
                        "page": g.get("url") or ""})
    return out


def _thumb_b64(im: Image.Image, side: int = 192) -> str:
    t = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    s = im.copy()
    s.thumbnail((side, side))
    t.alpha_composite(s, ((side - s.width) // 2, (side - s.height) // 2))
    buf = io.BytesIO()
    t.convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def auto_pack(store: PackStore, giphy_key: str, api_key: str, avoid: List[str],
              log: Callable[[str], None] = print, http=None, n: int = AUTO_ITEMS) -> dict:
    """Plan a chain, search GIPHY stickers for every step, let Claude pick a
    safe, fitting one per step, save the pack. Raises when it can't fill
    every step."""
    import requests
    http = http or requests
    plan = plan_chain(api_key, avoid, n)
    log(f"[packs] plan: {plan['title']}: " + " > ".join(s["name"] for s in plan["steps"]))
    cands: List[List[tuple]] = []
    seen = set()
    for s in plan["steps"]:
        got, found, last_err = [], 0, None
        for q in s["queries"][:2]:
            try:
                results = giphy_search(q, giphy_key, http=http)
            except Exception as e:
                log(f"[packs] search '{q}' failed: {e}")
                continue
            found += len(results)
            for g in results:
                if g["id"] in seen or len(got) >= 6:
                    continue
                im = None
                for url in g["urls"]:
                    try:
                        r = http.get(url, timeout=20)
                        r.raise_for_status()
                        im = normalize(r.content)
                        break
                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"
                if im is None or min(im.size) < 48:
                    continue
                seen.add(g["id"])
                got.append((g, im))
        log(f"[packs] {s['name']}: {found} found, {len(got)} usable" + (f" (last error: {last_err})" if last_err else ""))
        if not got:
            raise ValueError(f"GIPHY had no usable sticker for '{s['name']}'")
        cands.append(got)
    content: list = [{"type": "text", "text": _PICK_PROMPT.format(
        steps="\n".join(f"Step {i + 1}: {s['name']}" + (" (FINAL FORM)" if i == n - 1 else "")
                        for i, s in enumerate(plan["steps"])))}]
    for i, got in enumerate(cands):
        for c, (g, im) in enumerate(got):
            content.append({"type": "text", "text": f"step {i + 1} candidate {c}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": _thumb_b64(im)}})
    picks = _pick_list(_claude_text(content, api_key, max_tokens=400), n)
    items, credits = [], []
    for i, s in enumerate(plan["steps"]):
        k = picks[i] if i < len(picks) else -1
        if not isinstance(k, int) or not 0 <= k < len(cands[i]):
            raise ValueError(f"no usable sticker for step {i + 1} ({s['name']})")
        g, im = cands[i][k]
        items.append((s["name"], im))
        credits.append(g["page"])
    return store.save(plan["title"], items, "giphy", {"hook": plan["hook"], "credits": credits})
