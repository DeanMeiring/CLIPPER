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
"""
from __future__ import annotations

import numpy as np

from .ball_evolution import SR

KINDS = ("tune", "meow", "phonk")
MEOW_GAP = 0.22         # seconds between meows at most
PHONK_GAP = 0.12


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


def track(kind: str, hits: list, n: int) -> np.ndarray:
    """The bounce sound for a whole video: one meow / riff note per hit."""
    out = np.zeros(n, dtype=np.float32)
    last, k, cache = -1.0, 0, {}

    def put(t, w, vol):
        i = int(t * SR)
        seg = out[i:i + len(w)]
        seg += w[:len(seg)] * vol

    for t, vol in hits:
        if kind == "meow":
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
