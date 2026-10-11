"""Cursed packs from real, free photos (Dean wanted the chad-cat look of
real cursed pictures without taking other people's memes).

1. Openverse, limited to CC0 and public-domain photos (free to use with no
   conditions), is searched for one animal's face.
2. Claude looks at the results and picks one where a single animal faces
   the camera (no people, no text or watermarks), and says where its face
   and eyes are.
3. The face is cut out as a circle and goes through ball_cursed's edits
   (sunglasses, buff arms, sigma or chad jaw, cursed lips and eyes, deep
   fried laser eyes, god), so every picture in the pack is our own edit
   of a free photo.
"""
from __future__ import annotations

import base64
import io
import random
import re
from typing import Callable, List, Optional

from PIL import Image, ImageDraw

from . import ball_cursed as bc
from .ball_packs import PackStore, _claude_text

OPENVERSE = "https://api.openverse.org/v1/images/"
UA = "Mozilla/5.0 (compatible; clipper-ball-evolution)"
MAX_BYTES = 15 * 1024 * 1024

# what to search for, the name the viewer sees
SUBJECTS = [("cat face", "Cat"), ("dog face", "Dog"), ("pug face", "Pug"), ("hamster face", "Hamster"),
            ("frog face", "Frog"), ("owl face", "Owl"), ("capybara", "Capybara"), ("raccoon face", "Raccoon"),
            ("rabbit face", "Rabbit"), ("goat face", "Goat"), ("cow face", "Cow"), ("sheep face", "Sheep"),
            ("llama face", "Llama"), ("seal face", "Seal"), ("duck face", "Duck"), ("parrot face", "Parrot"),
            ("pig face", "Pig"), ("fox face", "Fox"), ("hedgehog face", "Hedgehog"), ("lemur face", "Lemur")]

_PICK = """These are {n} free photos found for "{q}". I want to cut one {name}'s face \
out of one of them for a funny video.

Pick the best one: a single {name} looking at the camera (front view, both eyes \
visible), sharp, its face big in the picture. Never pick a photo with a person or part of \
a person, text, a logo or a watermark on it. If none is good, the pick is -1.

For the photo you pick, say where the face is and where the eyes are, as fractions of \
the photo's width and height (0 = left/top, 1 = right/bottom).
Answer with JSON only:
{{"pick": <photo number or -1>, "face": [left, top, right, bottom], "eyes": [[x, y], [x, y]]}}"""


def search(query: str, http, n: int = 10) -> List[dict]:
    r = http.get(OPENVERSE, params={"q": query, "license": "cc0,pdm", "page_size": 20, "mature": "false"},
                 headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    out = []
    for x in r.json().get("results") or []:
        if int(x.get("width") or 0) >= 500 and int(x.get("height") or 0) >= 400 and x.get("url"):
            out.append({"url": x["url"], "thumb": x.get("thumbnail") or x["url"], "creator": x.get("creator") or "",
                        "license": (x.get("license") or "cc0").upper(), "page": x.get("foreign_landing_url") or ""})
        if len(out) >= n:
            break
    return out


def _get_image(url: str, http, side: int) -> Image.Image:
    r = http.get(url, headers={"User-Agent": UA}, timeout=40)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError("too large")
    im = Image.open(io.BytesIO(r.content))
    im.draft("RGB", (side, side))
    im = im.convert("RGB")
    im.thumbnail((side, side))
    return im


def _b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _parse(raw: str) -> dict:
    """Claude's pick, read leniently: the numbers after "pick", "face" and
    "eyes" (brackets and stray text don't matter)."""
    raw = raw or ""
    pick = re.search(r'"pick"\s*:\s*(-?\d+)', raw)
    if not pick:
        raise ValueError(f"couldn't read the photo pick: {raw[:200]!r}")

    def nums(key: str, until: str) -> list:
        m = re.search(rf'"{key}"\s*:(.*?)(?:"{until}"|}}\s*$|$)', raw, re.S)
        return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))] if m else []
    return {"pick": int(pick.group(1)), "face": nums("face", "eyes")[:4], "eyes": nums("eyes", "face")[:4]}


def cut_face(photo: Image.Image, face: list, eyes: list) -> tuple:
    """(round RGBA face picture, eye height, half the eye distance), both as
    fractions of the picture: the square is centred on the eyes."""
    W, H = photo.size
    (lx, ly), (rx, ry) = sorted(((eyes[0], eyes[1]), (eyes[2], eyes[3])))
    lx, rx, ly, ry = lx * W, rx * W, ly * H, ry * H
    fx0, fy0, fx1, fy1 = face[0] * W, face[1] * H, face[2] * W, face[3] * H
    d = max(8.0, rx - lx)
    side = max(fx1 - fx0, fy1 - fy0, d * 2.6) * 1.12
    side = min(side, W, H)
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    x0 = min(max(0.0, mx - side / 2), W - side)
    y0 = min(max(0.0, my - side * 0.45), H - side)
    sq = photo.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((440, 440), Image.LANCZOS)
    big = 440 * 3
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
    out = sq.convert("RGBA")
    out.putalpha(mask.resize((440, 440), Image.LANCZOS))
    rim = Image.new("RGBA", (464, 464), (0, 0, 0, 0))
    ImageDraw.Draw(rim).ellipse((0, 0, 463, 463), fill=(255, 255, 255, 255))
    rim.alpha_composite(out, (12, 12))
    eye_y = (12 + (my - y0) / side * 440) / 464
    eye_dx = (d / 2) / side * 440 / 464
    return rim, eye_y, eye_dx


def photo_pack(store: PackStore, api_key: str, log: Callable[[str], None] = print, http=None,
               subject: Optional[tuple] = None, seed: Optional[int] = None) -> dict:
    """Find a free photo, cut the face out, make a cursed pack (source
    "photo"). Raises when nothing usable turns up."""
    import requests
    http = http or requests
    seed = seed if seed is not None else random.randrange(10 ** 6)
    rng = random.Random(seed)
    if subject is None:
        used = [p.get("photo_subject") for p in store.packs() if p.get("photo_subject")]
        recent = set(used[-12:])
        subject = rng.choice([s for s in SUBJECTS if s[1] not in recent] or SUBJECTS)
    query, name = subject
    found = search(query, http)
    rng.shuffle(found)
    found = found[:6]
    thumbs = []
    for f in found:
        try:
            thumbs.append((f, _get_image(f["thumb"], http, 512)))
        except Exception as e:
            log(f"[packs] photo thumbnail failed: {e}")
    if not thumbs:
        raise ValueError(f"no free photos found for {query!r}")
    content: list = []
    for k, (_, im) in enumerate(thumbs):
        content += [{"type": "text", "text": f"photo {k}"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(im)}}]
    content.append({"type": "text", "text": _PICK.format(n=len(thumbs), q=query, name=name.lower())})
    ans = _parse(_claude_text(content, api_key, max_tokens=4000, log=log))
    k = ans["pick"]
    if not 0 <= k < len(thumbs) or len(ans["face"]) < 4 or len(ans["eyes"]) < 4:
        raise ValueError(f"none of the {len(thumbs)} {name.lower()} photos was usable")
    info, _ = thumbs[k]
    photo = _get_image(info["url"], http, 1600)
    face_im, eye_y, eye_dx = cut_face(photo, ans["face"], ans["eyes"])
    face = bc.Face(None, name, eye_y, eye_dx, im=face_im)
    p = bc.make(f"photo-{name}", seed, face=face, steps=bc.PHOTO_STEPS)
    by = f" by {info['creator']}" if info["creator"] else ""
    log(f"[packs] photo pack: {p['title']} from {info['page'] or info['url']}")
    return store.save(p["title"], p["items"], "photo",
                      {"hook": p["hook"], "photo_subject": name, "photo_page": info["page"],
                       "credit": f"Photo{by}, {info['license']} via Openverse"})
