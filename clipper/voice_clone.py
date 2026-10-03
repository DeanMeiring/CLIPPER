"""Dean's own voice, cloned, for long-form narration he doesn't want to
re-read: a line he keeps fluffing, a one-sentence bridge added after
recording. His real voice stays the default -- this only fills gaps.

Uses Kyutai's Pocket TTS (CC-BY-4.0 weights, MIT code): a 100M-parameter
model that runs on CPU at a few times real-time in about 0.6 GB of memory,
so it runs on the same Railway box as everything else. Chatterbox (MIT)
was the other candidate but needs ~7.5 GB on CPU, which the 8 GB box can't
spare. The voice-cloning weights are gated on Hugging Face (free, auto
approved after accepting "only clone voices with consent"), so the server
needs HF_TOKEN set from an account that accepted them.

One sample of Dean reading SAMPLE_TEXT (30 seconds or so) is stored under
the long-form folder and reused for every project. Generated lines go
through the same misread check as a recorded take, so a garbled line is
caught the same way a fluffed one is.

Other people's voices work the same way, one folder each with its own
sample and a voice.json (name, who agreed and when). Dean's friend lives
far away, so a sample can also be an uploaded voice note, or recorded by
the friend on a private link (see webapp /voice-sample/<token>).
Every other voice needs the person's consent first -- the model's terms
require it -- and videos using one are marked as altered or synthetic
content on YouTube.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
import time
import wave
from pathlib import Path

# What Dean reads for his voice sample: about 30 seconds at a normal pace,
# with a question, a list and a quote so the model hears his range, not
# just one flat sentence.
SAMPLE_TEXT = (
    "This is the story of how one stream changed everything. "
    "It started in a small bedroom, with a cheap microphone and about twelve viewers. "
    "Nobody expected much. But then something happened that nobody saw coming. "
    "Was it luck? Was it timing? Or was it something else entirely? "
    "Over the next three years, the numbers went up, the streams got longer, "
    "and the moments got bigger. As he put it himself: \"I just kept turning the camera on.\" "
    "So let's go back to where it all began."
)

MIN_SAMPLE_SECONDS = 10.0
MAX_SAMPLE_SECONDS = 60.0
# An uploaded file (a voice note sent from far away) can be longer: the
# first KEEP_SECONDS of it, from where the talking starts, become the sample.
UPLOAD_MAX_SECONDS = 600.0
KEEP_SECONDS = 45.0
MIN_SAMPLE_WORDS = 15

SETUP_HINT = (
    "Your AI voice isn't switched on yet. One-time setup (free): make a Hugging Face account, "
    "open huggingface.co/kyutai/pocket-tts and accept the terms, create a Read token under "
    "Settings > Access Tokens, and add it on Railway as the variable HF_TOKEN."
)

_model = None
_model_error: str | None = None
_voice_state = None
_voice_key = None
_lock = threading.Lock()


def installed() -> bool:
    # find_spec, not import: importing it pulls in torch (slow, ~300 MB),
    # which should only happen once the voice is actually used.
    try:
        return importlib.util.find_spec("pocket_tts") is not None
    except (ImportError, ValueError):
        return False


def sample_path(voice_dir: Path) -> Path:
    return voice_dir / "sample.wav"


def status(voice_dir: Path) -> dict:
    """What the page shows in the "Your voice" box."""
    meta_path = voice_dir / "sample.json"
    sample = None
    if sample_path(voice_dir).is_file() and meta_path.is_file():
        try:
            sample = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            sample = None
    ok = installed()
    configured = ok and bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return {
        "installed": ok,
        "configured": configured,
        "sample": sample,
        "ready": bool(configured and sample and not _model_error),
        "error": _model_error,
        "setup_hint": None if configured else SETUP_HINT,
        "sample_text": SAMPLE_TEXT,
    }


def read_meta(voice_dir: Path) -> dict:
    """voice.json of another person's voice: {"name", "consent", "link"...}.
    Empty for Dean's own voice, which has none."""
    try:
        return json.loads((voice_dir / "voice.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_meta(voice_dir: Path, meta: dict) -> None:
    voice_dir.mkdir(parents=True, exist_ok=True)
    tmp = voice_dir / "voice.json.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(voice_dir / "voice.json")


def trim_sample(src: Path, dst: Path, seconds: float = KEEP_SECONDS) -> None:
    """Mono 48 kHz wav of `src` from where the talking starts, at most
    `seconds` long. A voice note often opens with a second of silence or
    fumbling, which the model would copy as part of the voice."""
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-af", "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2",
         "-t", f"{seconds:.1f}", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True, timeout=120,
    )


def save_sample_meta(voice_dir: Path, duration: float, check: dict) -> dict:
    meta = {"recorded_at": time.time(), "duration": round(duration, 1),
            "coverage": check.get("coverage"), "heard": check.get("heard", "")}
    (voice_dir / "sample.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _load():
    global _model, _model_error
    if _model is not None:
        return _model
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    if not getattr(model, "has_voice_cloning", True):
        # Loaded the public weights instead of the gated cloning ones:
        # HF_TOKEN missing, wrong, or the terms weren't accepted.
        _model_error = ("Hugging Face didn't let the server download the voice-cloning model. "
                        "Check HF_TOKEN on Railway and that the terms at huggingface.co/kyutai/pocket-tts "
                        "were accepted with the same account.")
        raise RuntimeError(_model_error)
    _model_error = None
    _model = model
    return model


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())


# ------------------------------------------------------ numbers as words ---
# The scripts write numbers as digits ("2019", "1,240,000", "4.3M"), which
# Pocket TTS reads badly. The AI voice gets them spelled out the way a
# person says them; the script itself (and so the captions) keep digits.

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}


def _under_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def number_words(n: int) -> str:
    if n < 0:
        return "minus " + number_words(-n)
    if n < 1000:
        return _under_1000(n)
    parts = []
    for size, name in _SCALES:
        if n >= size:
            parts.append(f"{number_words(n // size)} {name}")
            n %= size
    if n:
        parts.append(_under_1000(n))
    return " ".join(parts)


def _decimal_words(s: str) -> str:
    whole, _, frac = s.replace(",", "").partition(".")
    out = number_words(int(whole or 0))
    if frac:
        out += " point " + " ".join(_ONES[int(d)] for d in frac)
    return out


def year_words(y: int) -> str:
    if 2000 <= y <= 2009:
        return "two thousand" + (f" {_ONES[y - 2000]}" if y > 2000 else "")
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{_under_1000(hi)} hundred"
    return f"{_under_1000(hi)} {'oh ' + _ONES[lo] if lo < 10 else _under_1000(lo)}"


def _ordinal(words: str) -> str:
    head, sep, last = words.rpartition("-" if "-" in words.split(" ")[-1] else " ")
    if last in _ORD:
        last = _ORD[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def _plural(words: str) -> str:
    head, sep, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return head + sep + last


_YEAR = r"(?:1[1-9]\d\d|20\d\d)"
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# Not part of a bigger number: "2,019" or "1.2019" aren't years, but
# "since 2020, he..." is.
_B = r"(?<!\d)(?<!\d,)(?<!\d\.)"
_A = r"(?!\d|,\d|\.\d)"
_SUFFIX = {"k": "thousand", "m": "million", "b": "billion", "thousand": "thousand", "million": "million", "billion": "billion"}


def spoken_text(text: str) -> str:
    """The narration with every number written the way it's said aloud:
    "2019" -> "twenty nineteen", "1,240,000" -> "one million two hundred
    forty thousand", "4.3M views" -> "four point three million views",
    "50%" -> "fifty percent", "$1.5M" -> "one point five million dollars",
    "3rd" -> "third", "2019-2021" -> "twenty nineteen to twenty twenty-one",
    "the 2010s" -> "the twenty tens", "#1" -> "number one"."""
    t = text
    t = re.sub(r"#(\d+)\b", lambda m: "number " + number_words(int(m.group(1))), t)

    def money(m):
        amount = _decimal_words(m.group(1))
        scale = _SUFFIX.get((m.group(2) or "").lower())
        return f"{amount}{' ' + scale if scale else ''} dollars"
    t = re.sub(rf"\$({_NUM})(?:\s?(k|m|b|thousand|million|billion)\b)?", money, t, flags=re.I)
    t = re.sub(rf"({_NUM})\s?%", lambda m: _decimal_words(m.group(1)) + " percent", t)
    t = re.sub(rf"{_B}({_YEAR})\s?[-\u2013\u2014]\s?({_YEAR}){_A}",
               lambda m: f"{year_words(int(m.group(1)))} to {year_words(int(m.group(2)))}", t)
    t = re.sub(rf"{_B}({_YEAR})s\b", lambda m: _plural(year_words(int(m.group(1)))), t)
    t = re.sub(r"(?<![\w,.])'?([1-9]0)s\b", lambda m: _plural(number_words(int(m.group(1)))), t)
    t = re.sub(rf"(?<![\w,.])({_NUM})([kmb])\b", lambda m: f"{_decimal_words(m.group(1))} {_SUFFIX[m.group(2).lower()]}", t, flags=re.I)
    t = re.sub(r"(?<![\w,.])(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(number_words(int(m.group(1)))), t, flags=re.I)
    t = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b(?![\d:])",
               lambda m: f"{m.group(1)} {_ordinal(number_words(int(m.group(2))))}", t)
    t = re.sub(rf"{_B}({_YEAR}){_A}", lambda m: year_words(int(m.group(1))), t)
    t = re.sub(rf"(?<![\w,.])({_NUM})(?![\w])", lambda m: _decimal_words(m.group(1)), t)
    return t


def synthesize(voice_dir: Path, text: str, out_path: Path) -> None:
    """Say `text` in Dean's voice and write it as a mono 16-bit wav at the
    model's own rate (24 kHz); callers convert it like any other take."""
    global _voice_state, _voice_key
    sample = sample_path(voice_dir)
    if not sample.is_file():
        raise RuntimeError("Record your voice sample first.")
    with _lock:
        model = _load()
        key = (str(sample), sample.stat().st_mtime)
        if _voice_key != key:
            _voice_state = model.get_state_for_audio_prompt(sample, truncate=True)
            _voice_key = key
        audio = model.generate_audio(_voice_state, spoken_text(text).strip())
        _write_wav(out_path, audio.detach().cpu().numpy(), model.sample_rate)
