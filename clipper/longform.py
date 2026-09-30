"""Long-form aviation-story videos for the second channel: an incident's
official NTSB report -> a narrated script (written by Claude, split into
scenes) -> the creator reads it in the web app one scene at a time.

Each recorded take is transcribed with Whisper and compared against the
scene's script (check_take) so a skipped line, a misread phrase or a
restarted sentence gets caught right away and re-recorded, instead of being
found by ear in a 10-minute edit. The accepted takes are joined into one
narration track whose scene boundaries are known exactly -- what the
visuals step times itself against.

Building the video from the narration (maps, cockpit cards, charts, stock
footage) is a later step; nothing here renders video.
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

# Curated starting points -- incidents with a detailed NTSB report (cockpit
# transcript, flight data, diagrams) and, first, ones where people survived:
# gripping without being grim, and advertiser-friendly. Any other NTSB
# report can be used by pasting its link or uploading the PDF.
INCIDENTS = [
    {
        "id": "us1549",
        "icon": "🌊",
        "title": "US Airways 1549: “Miracle on the Hudson”",
        "subtitle": "15 Jan 2009 · Airbus A320 · LaGuardia, New York · NTSB AAR-10/03",
        "report_url": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR1003.pdf",
        "tags": [["All 155 survived", "good"], ["Cockpit transcript", ""], ["Flight data", ""]],
    },
    {
        "id": "ac759",
        "icon": "🛬",
        "title": "Air Canada 759: the near-landing on a taxiway",
        "subtitle": "7 Jul 2017 · Airbus A320 · San Francisco · NTSB AIR-18/01",
        "report_url": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AIR1801.pdf",
        "tags": [["No injuries", "good"], ["Near miss", ""], ["Cockpit transcript", ""]],
    },
    {
        "id": "wn1380",
        "icon": "🔥",
        "title": "Southwest 1380: engine failure at 32,000 ft",
        "subtitle": "17 Apr 2018 · Boeing 737-700 · Philadelphia · NTSB AAR-19/03",
        "report_url": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR1903.pdf",
        "tags": [["1 fatality", "warn"], ["Cockpit transcript", ""], ["Flight data", ""]],
    },
    {
        "id": "aq243",
        "icon": "✈️",
        "title": "Aloha 243: the roof tore off at 24,000 ft",
        "subtitle": "28 Apr 1988 · Boeing 737-200 · Maui, Hawaii · NTSB AAR-89/03",
        "report_url": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR8903.pdf",
        "tags": [["1 fatality", "warn"], ["Cockpit transcript", ""], ["Diagrams", ""]],
    },
]

VISUAL_TYPES = ["map", "cockpit", "chart", "stock", "report"]

# Reading pace used for time estimates (words per minute of narration).
NARRATION_WPM = 150

# How much report text Claude gets. A full NTSB report can run 200+ pages;
# the front (summary, history of the flight, factual findings) and the back
# (conclusions, probable cause, recommendations, cockpit transcript
# appendix) carry the story.
_REPORT_HEAD_CHARS = 110_000
_REPORT_TAIL_CHARS = 40_000

# Silence between scenes when the takes are joined into one narration track.
SCENE_GAP_SECONDS = 0.6

MAX_REPORT_BYTES = 60 * 1024 * 1024
MAX_TAKE_BYTES = 40 * 1024 * 1024


# ---------------------------------------------------------------- report ---

def download_report(url: str) -> bytes:
    import requests

    resp = requests.get(
        url, timeout=90, headers={"User-Agent": "Mozilla/5.0 (clipper long-form; report download)"},
    )
    resp.raise_for_status()
    data = resp.content
    if not data.startswith(b"%PDF"):
        raise RuntimeError("That link didn't return a PDF -- try uploading the report file instead.")
    if len(data) > MAX_REPORT_BYTES:
        raise RuntimeError("That report is too large to read.")
    return data


def pdf_text(data: bytes) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_report(text: str) -> str:
    if len(text) <= _REPORT_HEAD_CHARS + _REPORT_TAIL_CHARS:
        return text
    return text[:_REPORT_HEAD_CHARS] + "\n\n[... middle of report omitted ...]\n\n" + text[-_REPORT_TAIL_CHARS:]


# ---------------------------------------------------------------- script ---

def _script_prompt(report_text: str, incident_title: str) -> str:
    return f"""You write narration scripts for a YouTube channel that tells the true
stories of aviation incidents, based strictly on the official investigation
report. The creator reads your script aloud; the video is built around their
voice with maps, cockpit-transcript cards, charts, photos and stock footage.

Incident: {incident_title}

Write a script of about 1,500 to 1,800 words (10 to 12 minutes read aloud),
split into 12 to 16 scenes.

Story:
- Scene 1 is a short cold open: the most gripping moment or line, then the
  question the video answers. No "welcome to the channel".
- Then tell it in order: the flight and crew, the moment things went wrong,
  each decision the crew faced and the seconds they had, how it ended, what
  the investigation found, and what changed in aviation because of it.
- Explain any technical idea in one plain sentence the first time it comes
  up. Spell out an abbreviation the first time it is used.

Accuracy and tone:
- Use ONLY facts in the report below. Never invent dialogue, times, numbers
  or thoughts. Quote cockpit or radio lines only when the report gives them,
  word for word.
- Be respectful: no graphic injury detail, and never name passengers or
  victims. Crew members named in the report may be named.
- Calm, clear, documentary voice. No exaggerated or clickbait words.

For reading aloud:
- Short, natural spoken sentences that are easy to say in one breath.
- Write numbers as digits (2,800 feet, 3:27 pm, 155 people).
- Each scene's narration is 60 to 160 words.

For each scene also choose the main visual, one of:
- "map": where the aircraft is (airports, route, turn, landing spot)
- "cockpit": a card showing a cockpit or radio line from the report
- "chart": altitude, speed or timeline data changing over time
- "stock": general aviation footage (takeoff, cabin, clouds, airport)
- "report": a photo, diagram or page from the report itself
and a short "visual_note" saying what it should show.

Respond with ONLY a JSON array, no other text, in this exact shape (escape
any double-quote characters inside a string value):
[
  {{"narration": "...", "visual": "map", "visual_note": "..."}}
]

Report:
{report_text}
"""


def write_script(report_text: str, incident_title: str) -> List[dict]:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    data = _ask_claude_for_json(_script_prompt(trim_report(report_text), incident_title), None, DEFAULT_MODEL)
    scenes = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        narration = " ".join(str(item.get("narration") or "").split())
        if not narration:
            continue
        visual = str(item.get("visual") or "").strip().lower()
        scenes.append({
            "narration": narration,
            "visual": visual if visual in VISUAL_TYPES else "stock",
            "visual_note": " ".join(str(item.get("visual_note") or "").split()),
            "take": None,
        })
    if not scenes:
        raise RuntimeError("Claude didn't return a usable script -- try Rewrite script.")
    return scenes


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


def transcribe_take(wav_path: Path) -> List[str]:
    """Words Whisper hears in a take. One model is kept loaded between
    takes -- a recording session is dozens of short checks in a row."""
    global _whisper_model
    from faster_whisper import WhisperModel

    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel("small", compute_type="int8")
        segments, _info = _whisper_model.transcribe(str(wav_path), language="en")
        return [seg.text for seg in segments]


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


def join_takes(wavs: List[Path], out_wav: Path, out_m4a: Path) -> None:
    """Join the accepted scene takes in order, with a short breath of
    silence between scenes, into one narration track (WAV for the video
    build, M4A to download)."""
    work = out_wav.parent
    gap = work / "_gap.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
         "-t", str(SCENE_GAP_SECONDS), "-c:a", "pcm_s16le", str(gap)],
        check=True, capture_output=True, timeout=60,
    )
    listing = work / "_concat.txt"
    lines = []
    for i, w in enumerate(wavs):
        if i:
            lines.append(f"file '{gap.resolve()}'")
        lines.append(f"file '{Path(w).resolve()}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out_wav)],
            check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(out_wav), "-c:a", "aac", "-b:a", "192k", str(out_m4a)],
            check=True, capture_output=True, timeout=300,
        )
    finally:
        listing.unlink(missing_ok=True)
        gap.unlink(missing_ok=True)



# -------------------------------------------------------------- projects ---

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class ProjectStore:
    """Each long-form video is a folder under base_dir with a project.json,
    the report text, and the recorded takes."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.lock = threading.Lock()

    def path(self, pid: str) -> Path:
        if not _ID_RE.match(pid or ""):
            raise KeyError(pid)
        return self.base_dir / pid

    def create(self, incident: dict) -> dict:
        pid = uuid.uuid4().hex[:12]
        d = self.base_dir / pid
        (d / "takes").mkdir(parents=True, exist_ok=True)
        now = time.time()
        project = {
            "id": pid, "created_at": now, "updated_at": now,
            "incident": incident, "status": "new", "error": None,
            "scenes": [], "narration": None,
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


def scene_ready(scene: dict) -> bool:
    take = scene.get("take") or {}
    return bool(take.get("file")) and bool(take.get("ok") or take.get("kept"))


def summary(project: dict) -> dict:
    scenes = project.get("scenes") or []
    words = sum(len(s.get("narration", "").split()) for s in scenes)
    return {
        "id": project["id"],
        "title": (project.get("incident") or {}).get("title") or "Untitled",
        "status": project.get("status"),
        "created_at": project.get("created_at"),
        "scenes": len(scenes),
        "recorded": sum(1 for s in scenes if scene_ready(s)),
        "minutes": round(words / NARRATION_WPM, 1) if words else 0,
        "has_narration": bool(project.get("narration")),
    }
