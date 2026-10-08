"""The Ball Evolution channel's videos on disk (the /ball-evolution page).

BASE_DIR/_balls/:
  settings.json            posting time, name on the videos, default description
  videos/<id>/video.json   recipe, status, title/description, YouTube info
  videos/<id>/video.mp4    the rendered Short (deleted a few days after it's public)

A video's life: queued -> picking (simulating seeds) -> rendering -> ready
-> scheduled / posted, or error. The recipes of every video ever made (kept
in video.json after the mp4 is gone) are the history ``ball_evolution
.pick_recipe`` uses to avoid repeats.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List

_ID_RE = re.compile(r"^[0-9a-f]{12}$")

DEFAULT_SETTINGS = {
    "post_time": "17:00",
    "watermark": "",
    "description": "",
    "bed": "auto",          # background sound: auto / rain / air / hush / off
    # autopilot: make videos ahead and post one at each slot (SA time)
    "autopilot": False,
    "slots": ["08:00", "15:00"],
    "auto_instagram": True,
    "ahead_days": 2,
}


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()

    def video_dir(self, vid: str) -> Path:
        if not _ID_RE.match(vid or ""):
            raise KeyError(vid)
        return self.root / "videos" / vid

    def new_video(self, options: dict) -> dict:
        vid = uuid.uuid4().hex[:12]
        (self.root / "videos" / vid).mkdir(parents=True, exist_ok=True)
        video = {"id": vid, "created_at": time.time(), "status": "queued", "options": options,
                 "recipe": None, "title": None, "description": None, "progress": 0.0}
        self._write(video)
        return video

    def load(self, vid: str) -> dict:
        p = self.video_dir(vid) / "video.json"
        if not p.is_file():
            raise KeyError(vid)
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, video: dict) -> None:
        d = self.root / "videos" / video["id"]
        tmp = d / "video.json.tmp"
        tmp.write_text(json.dumps(video), encoding="utf-8")
        tmp.replace(d / "video.json")

    def update(self, vid: str, fn) -> dict:
        with self.lock:
            video = self.load(vid)
            fn(video)
            self._write(video)
            return video

    def videos(self) -> List[dict]:
        base = self.root / "videos"
        out = []
        if base.is_dir():
            for d in base.iterdir():
                try:
                    out.append(self.load(d.name))
                except (KeyError, OSError, ValueError):
                    continue
        out.sort(key=lambda v: v.get("created_at") or 0)
        return out

    def delete(self, vid: str) -> None:
        import shutil
        with self.lock:
            shutil.rmtree(self.video_dir(vid), ignore_errors=True)

    def history(self) -> List[dict]:
        """Recipes of every video made, oldest first."""
        return [v["recipe"] for v in self.videos() if v.get("recipe")]

    def settings(self) -> dict:
        try:
            data = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return {**DEFAULT_SETTINGS, **data}

    def save_settings(self, data: dict) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        merged = {**self.settings(), **data}
        (self.root / "settings.json").write_text(json.dumps(merged), encoding="utf-8")
        return merged
