"""Bounce sounds besides the piano tune (Dean, 11 Oct: "a cursed meow,
heavily deep, that gets deeper the more it bounces" and "a song people put
under mog videos").

Both are made here from nothing (numpy only), so nothing can be claimed:

* ``meow``: a synthesised cat meow ("mee-ow": a pitch glide through
  i -> a -> u vowel formants), played on every bounce and slowed down a
  little more each time, like a tape running slow, so it gets deeper and
  more cursed.
* ``phonk``: the sound mog edits use, as an original riff -- a pitched 808
  cowbell melody in a minor key, one note per bounce, with an 808 bass hit
  every other note. The actual songs are labels' tracks: Content ID would
  claim every video, so none of them is used.

And ``song`` (Dean, 11 Oct: "better songs" than the piano): a free song he
uploaded on the page (NCS, YouTube Audio Library...). Every bounce lets the
song play on for a moment, so it only moves forward while the ball bounces,
the way the bouncing-ball reels do it.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .ball_evolution import SR

KINDS = ("tune", "meow", "phonk", "song", "knock", "none")
MEOW_GAP = 0.22         # seconds between meows at most
PHONK_GAP = 0.12
KNOCK_GAP = 0.05
SONG_OPEN = 0.42        # how long the song plays on after each bounce
SONG_FADE_IN, SONG_FADE_OUT = 0.006, 0.05


def meow(slow: float = 1.0) -> np.ndarray:
    """One meow; slow > 1 plays it slower and deeper (pitch and formants
    both drop, like a slowed tape)."""
    dur = 0.6 * slow
    n = int(SR * dur)
    t = np.arange(n) / n                                   # 0..1 through the meow
    f0 = (430 + 330 * np.sin(np.pi * np.minimum(1, t * 1.4)) - 120 * t) / slow
    phase = 2 * np.pi * np.cumsum(f0) / SR
    # vowel formants: "i" (300/2300) -> "a" (800/1300) -> "u" (380/900)
    k1 = np.clip(t * 2.2, 0, 1)
    k2 = np.clip(t * 2.2 - 1.1, 0, 1)
    f1 = (300 + 500 * k1 - 420 * k2) / slow
    f2 = (2300 - 1000 * k1 - 400 * k2) / slow
    out = np.zeros(n)
    for h in range(1, 40):
        fh = f0 * h
        if np.min(fh) > SR / 2.2:
            break
        amp = (np.exp(-((fh - f1) / (140 / slow)) ** 2) + 0.7 * np.exp(-((fh - f2) / (220 / slow)) ** 2)
               + 0.12 / h)
        amp = np.where(fh < SR / 2.2, amp, 0)
        out += amp * np.sin(h * phase)
    env = np.minimum(1, t / 0.08) * np.minimum(1, (1 - t) / 0.25)
    out = np.tanh(out * env * 1.6)                          # a little grit
    # a short slap-back makes it sound like a room, not a synth
    d = int(SR * 0.045)
    out[d:] += 0.35 * out[:-d]
    return (out / max(1e-6, np.max(np.abs(out))) * 0.5).astype(np.float32)


def cowbell(midi: int, dur: float = 0.32) -> np.ndarray:
    """An 808-style cowbell (two square-ish tones a fifth-ish apart),
    pitched to midi."""
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(SR * dur)
    t = np.arange(n) / SR
    tone = np.sign(np.sin(2 * np.pi * f * t)) + np.sign(np.sin(2 * np.pi * f * 1.48 * t))
    # soften the squares (a crude low-pass) and let it ring out
    k = 6
    tone = np.convolve(tone, np.ones(k) / k, mode="same")
    env = np.exp(-t * 11) * np.minimum(1, t / 0.002)
    return (tone * env * 0.22).astype(np.float32)


def bass808(midi: int, dur: float = 0.7) -> np.ndarray:
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(SR * dur)
    t = np.arange(n) / SR
    glide = f * (1 + 1.5 * np.exp(-t * 30))                # the punchy pitch drop at the start
    out = np.sin(2 * np.pi * np.cumsum(glide) / SR)
    out = np.tanh(out * 2.2) * np.exp(-t * 3.5)
    return (out * 0.42).astype(np.float32)


# an original riff in E minor (cowbell, played one note per bounce)
PHONK_RIFF = [76, 79, 81, 79, 76, 74, 76, 71, 76, 79, 83, 81, 79, 76, 74, 71]
PHONK_BASS = [40, 40, 43, 38]


def knock(seed: int = 0) -> np.ndarray:
    """A plain ball tap: a short, dull noise burst with a low thump (no
    pitch, no tune)."""
    t = np.arange(int(SR * 0.09)) / SR
    noise = np.convolve(np.random.default_rng(seed).standard_normal(len(t)), np.ones(6) / 6, mode="same")
    thump = np.sin(2 * np.pi * np.cumsum(110 + 120 * np.exp(-t * 60)) / SR)
    return ((noise * 0.6 + thump * 0.5) * np.exp(-t * 55) * 0.5).astype(np.float32)


def track(kind: str, hits: list, n: int) -> np.ndarray:
    """The bounce sound for a whole video: one meow / riff note / tap per
    hit ("none": silence)."""
    out = np.zeros(n, dtype=np.float32)
    if kind == "none":
        return out
    last, k, cache = -1.0, 0, {}

    def put(t, w, vol):
        i = int(t * SR)
        seg = out[i:i + len(w)]
        seg += w[:len(seg)] * vol

    for t, vol in hits:
        if kind == "knock":
            if t - last < KNOCK_GAP:
                continue
            w = cache.get(k % 4)
            if w is None:
                w = cache[k % 4] = knock(k % 4)
            put(t, w, 0.4 + 0.6 * vol)
        elif kind == "meow":
            if t - last < MEOW_GAP:
                continue
            # a bit deeper every bounce: 1.0 -> about 3.5x slower by bounce 60
            slow = round(min(3.5, 1.0 + k * 0.045), 2)
            w = cache.get(slow)
            if w is None:
                w = cache[slow] = meow(slow)
            put(t, w, 0.5 + 0.5 * vol)
        else:
            if t - last < PHONK_GAP:
                continue
            m = PHONK_RIFF[k % len(PHONK_RIFF)]
            w = cache.get(m)
            if w is None:
                w = cache[m] = cowbell(m)
            put(t, w, 0.6 + 0.4 * vol)
            if k % 2 == 0:
                b = PHONK_BASS[(k // 4) % len(PHONK_BASS)]
                wb = cache.get(("b", b))
                if wb is None:
                    wb = cache[("b", b)] = bass808(b)
                put(t, wb, 0.8)
        last = t
        k += 1
    return out


def load_song(path: str, start: float, seconds: float) -> np.ndarray:
    """`seconds` of the song from `start`, mono at SR (ffmpeg); it wraps back
    to the beginning if the song ends first."""
    ff = os.environ.get("CLIPPER_FFMPEG", "ffmpeg")

    def read(at: float, dur: float) -> np.ndarray:
        r = subprocess.run([ff, "-v", "error", "-ss", f"{max(0.0, at):.2f}", "-t", f"{dur:.2f}", "-i", str(path),
                            "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"couldn't read the song: {r.stderr.decode(errors='replace')[-300:]}")
        return np.frombuffer(r.stdout, dtype=np.float32)

    out = read(start, seconds)
    need = int(SR * seconds)
    while len(out) < need:
        more = read(0.0, seconds)
        if not len(more):
            break
        out = np.concatenate([out, more])
    if not len(out):
        raise RuntimeError("the song has no sound")
    return out[:need]


def song_track(song: dict, hits: list, n: int) -> np.ndarray:
    """The song, heard only around the bounces: each one keeps it playing for
    SONG_OPEN s; between bounces it stops, and it picks up where it stopped."""
    gates = []                                   # (start, end) in samples
    for t, _vol in hits:
        a, b = int(t * SR), int((t + SONG_OPEN) * SR)
        if gates and a <= gates[-1][1]:
            gates[-1] = (gates[-1][0], max(gates[-1][1], b))
        else:
            gates.append((a, b))
    gates = [(a, min(b, n)) for a, b in gates if a < n]
    out = np.zeros(n, dtype=np.float32)
    total = sum(b - a for a, b in gates)
    if not total:
        return out
    audio = load_song(song["path"], float(song.get("start") or 0.0), total / SR + 0.5)
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * 0.9
    fi, fo = int(SR * SONG_FADE_IN), int(SR * SONG_FADE_OUT)
    cur = 0
    for a, b in gates:
        piece = audio[cur:cur + (b - a)].copy()
        cur += len(piece)
        if not len(piece):
            break
        k = min(fi, len(piece))
        piece[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        k = min(fo, len(piece))
        piece[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
        out[a:a + len(piece)] = piece
    return out
