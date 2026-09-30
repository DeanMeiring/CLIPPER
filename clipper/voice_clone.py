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
"""
from __future__ import annotations

import importlib.util
import json
import os
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
        audio = model.generate_audio(_voice_state, text.strip())
        _write_wav(out_path, audio.detach().cpu().numpy(), model.sample_rate)
