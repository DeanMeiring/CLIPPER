"""Free visuals for the long-form documentaries: where a keyword visual's
picture or footage comes from. Everything here is free with no paid tier
needed, and free for commercial use on a monetised channel:

  emoji     Microsoft Fluent Emoji 3D, bundled in clipper/assets/emoji (MIT)
  stock     Pexels videos/photos (free API key, Pexels licence) then
            Pixabay (free API key, Pixabay content licence) -- both
            royalty-free with no attribution required on the video itself;
            their API terms ask apps to credit them, which the YouTube
            description does
  photo     the above, then Openverse restricted to CC0 / public-domain
            images (no key needed; no attribution needed either)
  posts     an X post the creator pasted, read through X's public embed
            endpoint (the one embedded tweets use; no key)

Downloads land in the project's visuals/ folder with a .json sidecar that
records the source and credit, so a render is repeatable and the credits
can go in the description.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional

ASSETS = Path(__file__).parent / "assets"
EMOJI_DIR = ASSETS / "emoji"
FONT_DIR = ASSETS / "fonts"
_UA = "clipper-documentaries/1.0 (personal YouTube documentary tool)"
MAX_DOWNLOAD = 80 * 1024 * 1024


# ----------------------------------------------------------------- emoji ---

def _emoji_code(ch: str) -> str:
    return "-".join(f"{ord(c):x}" for c in ch if ord(c) != 0xFE0F)


def emoji_file(ch: str) -> Optional[Path]:
    code = _emoji_code(ch.strip())
    for name in (f"{code}.webp", f"{code}-fe0f.webp"):
        p = EMOJI_DIR / name
        if p.is_file():
            return p
    return None


def emoji_list() -> str:
    """Every bundled emoji as one string, for the planning prompt."""
    out = []
    for p in sorted(EMOJI_DIR.glob("*.webp")):
        try:
            out.append("".join(chr(int(h, 16)) for h in p.stem.split("-") if h != "fe0f"))
        except ValueError:
            continue
    return "".join(out)


# ------------------------------------------------------------- downloads ---

def _get(url: str, **kw):
    import requests

    headers = {"User-Agent": _UA, **kw.pop("headers", {})}
    r = requests.get(url, headers=headers, timeout=kw.pop("timeout", 25), **kw)
    r.raise_for_status()
    return r


def download(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    with requests.get(url, headers={"User-Agent": _UA}, timeout=60, stream=True) as r:
        r.raise_for_status()
        size = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise RuntimeError("file too large")
                f.write(chunk)
    tmp.replace(dest)
    return dest


def _cache_key(kind: str, query: str) -> str:
    return f"{kind}_" + hashlib.sha1(query.strip().lower().encode()).hexdigest()[:10]


def _cached(dest_dir: Path, key: str) -> Optional[dict]:
    meta = dest_dir / f"{key}.json"
    if meta.is_file():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if info.get("file") and (dest_dir / info["file"]).is_file():
                return info
            if info.get("none"):
                return info
        except (OSError, ValueError):
            pass
    return None


def _save(dest_dir: Path, key: str, info: dict) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"{key}.json").write_text(json.dumps(info), encoding="utf-8")
    return info


# ----------------------------------------------------------------- stock ---

def _pexels_video(query: str) -> Optional[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return None
    data = _get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                params={"query": query, "orientation": "landscape", "per_page": 8}).json()
    for v in data.get("videos") or []:
        if float(v.get("duration") or 0) < 4:
            continue
        files = [f for f in v.get("video_files") or [] if (f.get("file_type") or "").endswith("mp4") and f.get("width")]
        files = [f for f in files if 1280 <= int(f["width"]) <= 2560] or files
        if not files:
            continue
        best = min(files, key=lambda f: abs(int(f["width"]) - 1920))
        user = v.get("user") or {}
        return {"url": best["link"], "source": "Pexels", "author": user.get("name") or "", "page": v.get("url") or "", "ext": ".mp4"}
    return None


def _pixabay_video(query: str) -> Optional[dict]:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return None
    data = _get("https://pixabay.com/api/videos/", params={"key": key, "q": query[:100], "safesearch": "true", "per_page": 8}).json()
    for h in data.get("hits") or []:
        vids = h.get("videos") or {}
        pick = next((vids[k] for k in ("large", "medium") if (vids.get(k) or {}).get("url") and int(vids[k].get("width") or 0) >= 1280), None)
        if pick:
            return {"url": pick["url"], "source": "Pixabay", "author": h.get("user") or "", "page": h.get("pageURL") or "", "ext": ".mp4"}
    return None


def _pexels_photo(query: str) -> Optional[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return None
    data = _get("https://api.pexels.com/v1/search", headers={"Authorization": key},
                params={"query": query, "orientation": "landscape", "per_page": 8}).json()
    for p in data.get("photos") or []:
        src = p.get("src") or {}
        url = src.get("large2x") or src.get("original")
        if url:
            return {"url": url, "source": "Pexels", "author": p.get("photographer") or "", "page": p.get("url") or "", "ext": ".jpg"}
    return None


def _pixabay_photo(query: str) -> Optional[dict]:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return None
    data = _get("https://pixabay.com/api/", params={"key": key, "q": query[:100], "image_type": "photo",
                                                    "orientation": "horizontal", "safesearch": "true", "per_page": 8}).json()
    for h in data.get("hits") or []:
        if h.get("largeImageURL"):
            return {"url": h["largeImageURL"], "source": "Pixabay", "author": h.get("user") or "", "page": h.get("pageURL") or "", "ext": ".jpg"}
    return None


def _openverse_photo(query: str) -> Optional[dict]:
    """Openverse, limited to CC0 and public-domain images: free to use
    with no conditions at all."""
    data = _get("https://api.openverse.org/v1/images/", params={"q": query, "license": "cc0,pdm", "page_size": 12,
                                                                 "aspect_ratio": "wide", "mature": "false"}).json()
    for r in data.get("results") or []:
        if int(r.get("width") or 0) >= 1200 and r.get("url"):
            ext = Path(r["url"].split("?")[0]).suffix.lower()
            return {"url": r["url"], "source": "Openverse (" + (r.get("license") or "cc0").upper() + ")",
                    "author": r.get("creator") or "", "page": r.get("foreign_landing_url") or "",
                    "ext": ext if ext in (".jpg", ".jpeg", ".png", ".webp") else ".jpg"}
    return None


def find_stock(kind: str, query: str, dest_dir: Path) -> Optional[dict]:
    """Search the free libraries in order and download the first good
    match. kind is "video" or "photo". Returns {"file", "source", "author",
    "page", "kind"} or None. Results (and misses) are cached per query."""
    query = " ".join(query.split())[:80]
    if not query:
        return None
    key = _cache_key(kind, query)
    hit = _cached(dest_dir, key)
    if hit is not None:
        return None if hit.get("none") else hit
    finders = [_pexels_video, _pixabay_video] if kind == "video" else [_pexels_photo, _pixabay_photo, _openverse_photo]
    errored = False
    for finder in finders:
        try:
            found = finder(query)
        except Exception as e:
            print(f"[visuals] {finder.__name__} {query!r} failed: {e}", flush=True)
            errored = True
            continue
        if not found:
            continue
        name = f"{key}{found.pop('ext')}"
        try:
            download(found.pop("url"), dest_dir / name)
        except Exception as e:
            print(f"[visuals] download for {query!r} failed: {e}", flush=True)
            errored = True
            continue
        return _save(dest_dir, key, {**found, "file": name, "kind": kind, "query": query})
    # Remember a clean "nothing found" so re-renders don't search again --
    # but not an outage, and not a video search with no keys set (adding a
    # key later should find footage).
    if not errored and (kind == "photo" or any(stock_available().values())):
        _save(dest_dir, key, {"none": True, "query": query})
    return None


def stock_available() -> dict:
    return {"pexels": bool(os.environ.get("PEXELS_API_KEY")), "pixabay": bool(os.environ.get("PIXABAY_API_KEY"))}


# ----------------------------------------------------------------- posts ---

_X_STATUS = re.compile(r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/([A-Za-z0-9_]+)/status(?:es)?/(\d+)")


def is_x_post(url: str) -> bool:
    return bool(_X_STATUS.match(url.strip()))


def x_post(url: str) -> Optional[dict]:
    """Text, author, date and like count of a public X post, from the
    endpoint X's own embedded posts load from."""
    m = _X_STATUS.match(url.strip())
    if not m:
        return None
    data = _get("https://cdn.syndication.twimg.com/tweet-result", params={"id": m.group(2), "lang": "en", "token": "0"}).json()
    text = data.get("text") or ""
    for ent in (data.get("entities") or {}).get("urls") or []:
        text = text.replace(ent.get("url") or "\0", ent.get("display_url") or "")
    text = re.sub(r"\s*https://t\.co/\S+$", "", text).strip()
    user = data.get("user") or {}
    if not text:
        return None
    return {"url": url.strip(), "text": text, "name": user.get("name") or m.group(1), "handle": user.get("screen_name") or m.group(1),
            "date": (data.get("created_at") or "")[:10], "likes": int(data.get("favorite_count") or 0),
            "avatar": (user.get("profile_image_url_https") or "").replace("_normal", "_200x200")}


def fetch_image(url: str, dest: Path) -> Optional[Path]:
    """An avatar or other small picture, cached."""
    if not url:
        return None
    if dest.is_file():
        return dest
    try:
        return download(url, dest)
    except Exception as e:
        print(f"[visuals] image {url} failed: {e}", flush=True)
        return None


def credits(infos: List[dict]) -> List[str]:
    """Description lines crediting the stock libraries and creators used."""
    by_source: dict = {}
    for i in infos:
        if not i or not i.get("source"):
            continue
        by_source.setdefault(i["source"], set())
        if i.get("author"):
            by_source[i["source"]].add(i["author"])
    lines = []
    for src, authors in sorted(by_source.items()):
        who = ", ".join(sorted(authors)[:12])
        lines.append(f"Stock visuals from {src}" + (f" by {who}" if who else ""))
    return lines
