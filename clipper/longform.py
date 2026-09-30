"""Recording side of long-form videos (see documentary.py for the streamer
documentaries built on it): project storage, and scene-by-scene narration
recorded in the web app.

Each recorded take is transcribed with Whisper and compared against the
scene's script (check_take) so a skipped line, a misread phrase or a
restarted sentence gets caught right away and re-recorded, instead of being
found by ear in a 10-minute edit.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import List

# Reading pace used for time estimates (words per minute of narration).
NARRATION_WPM = 150

# Pause after each narrated scene, so scenes don't run into each other.
SCENE_GAP_SECONDS = 0.6

MAX_TAKE_BYTES = 40 * 1024 * 1024


# ------------------------------------------------------------ take check ---

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "fifteenth", "twentieth", "oh", "am", "pm",
}
_FILLERS = {"um", "uh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm"}

# A run of this many script words missing (skipped or misread) or extra
# words spoken (a restarted sentence, an ad-lib) flags the take. Shorter
# slips are left alone: Whisper itself mishears a single word now and then,
# especially names, and stopping for those would just be annoying.
_RUN_FLAG = 3
_MIN_COVERAGE = 0.85


def _norm(token: str) -> str:
    token = token.lower().replace("’", "'").replace("‘", "'")
    token = re.sub(r"[^a-z0-9']", "", token).strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    return token


def _numberish(token: str) -> bool:
    # Numbers are skipped entirely: "2,800" read aloud comes back from
    # Whisper as digits or as words depending on the take.
    return any(ch.isdigit() for ch in token) or token in _NUMBER_WORDS


def check_take(script: str, spoken_words: List[str]) -> dict:
    """Compare a scene's script with what was actually said. Returns
    {"ok", "coverage", "missed" (indices into script.split() to highlight),
    "skipped" (missing/misread phrases), "extra" (added/repeated phrases),
    "message"}."""
    display = script.split()
    comp_idx = [i for i, w in enumerate(display) if _norm(w) and not _numberish(_norm(w))]
    script_toks = [_norm(display[i]) for i in comp_idx]

    spoken: List[str] = []
    for w in spoken_words:
        for part in str(w).split():
            n = _norm(part)
            if n and not _numberish(n) and n not in _FILLERS:
                spoken.append(n)
    if not spoken:
        return {"ok": False, "coverage": 0.0, "missed": comp_idx, "skipped": [], "extra": [],
                "message": "Couldn't hear any speech in that take -- check your mic and read the scene again."}

    # Near-miss spellings of script words (names especially) count as the
    # script word, so "Sullenburger" doesn't read as a misread.
    vocab = sorted(set(script_toks))
    canon = []
    for s in spoken:
        if s in vocab:
            canon.append(s)
        else:
            close = get_close_matches(s, vocab, n=1, cutoff=0.8)
            canon.append(close[0] if close else s)

    matched = [False] * len(script_toks)
    extra_runs: List[List[str]] = []
    sm = SequenceMatcher(None, script_toks, canon, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                matched[k] = True
        elif tag in ("insert", "replace") and (j2 - j1) >= _RUN_FLAG and (tag == "insert" or (j2 - j1) > (i2 - i1)):
            extra_runs.append(canon[j1:j2])

    missed_runs: List[List[int]] = []
    run: List[int] = []
    for k, ok in enumerate(matched):
        if not ok:
            run.append(k)
        elif run:
            missed_runs.append(run)
            run = []
    if run:
        missed_runs.append(run)

    coverage = (sum(matched) / len(script_toks)) if script_toks else 1.0
    missed_total = sum(len(r) for r in missed_runs)
    long_missed = [r for r in missed_runs if len(r) >= _RUN_FLAG]

    def _phrase(r: List[int]) -> str:
        first, last = comp_idx[r[0]], comp_idx[r[-1]]
        return re.sub(r"^[^\w]+|[^\w]+$", "", " ".join(display[first:last + 1]))

    skipped = [_phrase(r) for r in long_missed]
    extra = [" ".join(r) for r in extra_runs]
    flagged = bool(long_missed or extra_runs or (coverage < _MIN_COVERAGE and missed_total >= _RUN_FLAG))

    highlight_runs = (long_missed or missed_runs) if flagged else []
    highlight = [comp_idx[k] for r in highlight_runs for k in r]

    if not flagged:
        message = "Sounds right."
    elif skipped:
        message = "Sounds like you skipped or misread: “" + "”, “".join(skipped) + "”. Read this scene again."
    elif extra:
        message = "Sounds like you repeated or added: “" + "”, “".join(extra) + "”. Read this scene again."
    else:
        message = "A lot of that didn't match the script. Read this scene again."
    return {"ok": not flagged, "coverage": round(coverage, 3), "missed": highlight,
            "skipped": skipped, "extra": extra, "message": message}


# ----------------------------------------------------------------- audio ---

_whisper_model = None
_whisper_lock = threading.Lock()


def _whisper():
    global _whisper_model
    from faster_whisper import WhisperModel

    if _whisper_model is None:
        _whisper_model = WhisperModel("small", compute_type="int8")
    return _whisper_model


def transcribe_take(wav_path: Path) -> List[str]:
    """Words Whisper hears in a take. One model is kept loaded between
    takes -- a recording session is dozens of short checks in a row."""
    with _whisper_lock:
        segments, _info = _whisper().transcribe(str(wav_path), language="en")
        return [seg.text for seg in segments]


def transcribe_words(media_path: Path) -> List[dict]:
    """Word-level transcript ({"w", "s", "e"}) of a clip, for subtitles and
    for picking where a moment starts and ends."""
    with _whisper_lock:
        segments, _info = _whisper().transcribe(str(media_path), language="en", word_timestamps=True)
        out = []
        for seg in segments:
            for w in seg.words or []:
                if w.word.strip():
                    out.append({"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)})
        return out


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True, timeout=120,
    )


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(result.stdout.strip())


# -------------------------------------------------------------- projects ---

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class ProjectStore:
    """Each long-form video is a folder under base_dir with a project.json,
    its research and clips, and the recorded takes."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.lock = threading.Lock()

    def path(self, pid: str) -> Path:
        if not _ID_RE.match(pid or ""):
            raise KeyError(pid)
        return self.base_dir / pid

    def create(self, fields: dict) -> dict:
        pid = uuid.uuid4().hex[:12]
        d = self.base_dir / pid
        (d / "takes").mkdir(parents=True, exist_ok=True)
        now = time.time()
        project = {
            "id": pid, "created_at": now, "updated_at": now,
            "status": "new", "error": None, "scenes": [],
            **fields,
        }
        self._write(project)
        return project

    def load(self, pid: str) -> dict:
        p = self.path(pid) / "project.json"
        if not p.exists():
            raise KeyError(pid)
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, project: dict) -> None:
        project["updated_at"] = time.time()
        d = self.base_dir / project["id"]
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "project.json.tmp"
        tmp.write_text(json.dumps(project), encoding="utf-8")
        tmp.replace(d / "project.json")

    def update(self, pid: str, fn) -> dict:
        """Load, apply fn(project) in place, save -- under the store lock so
        concurrent requests (a take check finishing while a script edit
        saves) can't overwrite each other."""
        with self.lock:
            project = self.load(pid)
            fn(project)
            self._write(project)
            return project

    def list(self) -> List[dict]:
        out = []
        if not self.base_dir.exists():
            return out
        for d in self.base_dir.iterdir():
            if not _ID_RE.match(d.name):
                continue
            try:
                out.append(self.load(d.name))
            except (KeyError, OSError, ValueError):
                continue
        out.sort(key=lambda p: p.get("created_at", 0), reverse=True)
        return out

    def delete(self, pid: str) -> None:
        d = self.path(pid)
        with self.lock:
            shutil.rmtree(d, ignore_errors=True)


def needs_take(scene: dict) -> bool:
    """Only narrated scenes are recorded; clip moments and chapter cards
    carry their own sound (or none)."""
    return scene.get("kind", "narrate") == "narrate"


def scene_ready(scene: dict) -> bool:
    if not needs_take(scene):
        return True
    take = scene.get("take") or {}
    return bool(take.get("file")) and bool(take.get("ok") or take.get("kept"))


def summary(project: dict) -> dict:
    scenes = project.get("scenes") or []
    narrated = [s for s in scenes if needs_take(s)]
    words = sum(len(s.get("narration", "").split()) for s in narrated)
    moments = sum(max(0.0, float(s.get("end") or 0) - float(s.get("start") or 0)) for s in scenes if s.get("kind") == "moment")
    return {
        "id": project["id"],
        "kind": project.get("kind") or "aviation",
        "title": project.get("title") or (project.get("incident") or {}).get("title") or "Untitled",
        "status": project.get("status"),
        "created_at": project.get("created_at"),
        "scenes": len(narrated),
        "recorded": sum(1 for s in narrated if scene_ready(s)),
        "minutes": round(words / NARRATION_WPM + moments / 60, 1) if scenes else 0,
        "rendered": (project.get("render") or {}).get("status") == "done",
        "youtube_video_id": project.get("youtube_video_id"),
    }
