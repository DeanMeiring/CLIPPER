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

KINDS = ("tune", "meow", "phonk", "song", "knock", "none",
         "demon_meow", "quack", "huh", "bruh", "goat", "bonk", "boing")
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


# ---- meme sounds (Dean, 9 Nov) ------------------------------------------------------
# "What is satisfying is when the balls fill up and the audio goes deep with
# that meow" -- so more of those: a deeper meow and other meme-style sounds,
# all made here from nothing (no recordings, nothing to claim). Each one gets
# slower and deeper as the VIDEO goes on (by time, not by bounce count, so
# the deepest one lands at the end, when it's full -- the old meow hit its
# lowest after ~60 bounces and then stayed there).
# kind -> (how much slower it is at the end, the gap between two at the start)
MEMES = {
    "demon_meow": (6.0, 0.22),
    "quack": (4.5, 0.14),
    "huh": (4.0, 0.26),
    "bruh": (4.0, 0.3),
    "goat": (4.0, 0.3),
    "bonk": (3.5, 0.1),
    "boing": (3.0, 0.2),
}
LABELS = {"meow": "🐱 cursed meow", "demon_meow": "😈 demon meow", "quack": "🦆 deep quack",
          "huh": "❓ huh?", "bruh": "😐 bruh", "goat": "🐐 goat", "bonk": "🔨 bonk", "boing": "🌀 boing",
          "phonk": "🥁 phonk", "knock": "🔊 ball sounds"}
TRACK_KINDS = ("meow", "phonk", "knock", "none") + tuple(MEMES)


def _harmonics(f0: np.ndarray, formants: list, n: int, buzz: float = 0.12, sub: float = 0.0) -> np.ndarray:
    """A voice made of harmonics of f0 (Hz per sample), each as loud as the
    formants (centre Hz per sample, width Hz, weight) let through -- the
    same way the meow is made. ``sub`` adds a growl an octave below."""
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    hs = ([0.5] if sub else []) + list(range(1, 60))
    for h in hs:
        fh = f0 * h
        if np.min(fh) > SR / 2.2:
            break
        amp = buzz / max(h, 1) ** 0.7
        for fc, bw, wgt in formants:
            amp = amp + wgt * np.exp(-((fh - fc) / bw) ** 2)
        if h == 0.5:
            amp = amp * 0 + sub
        amp = np.where(fh < SR / 2.2, amp, 0)
        out += amp * np.sin(h * phase)
    return out


def _shaped_noise(n: int, formants: list, seed: int) -> np.ndarray:
    """Breath: white noise through the formants (fixed centres here)."""
    x = np.random.default_rng(seed).standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = sum(w * np.exp(-((f - fc) / bw) ** 2) for fc, bw, w in formants) + 0.004
    return np.fft.irfft(np.fft.rfft(x) * g, n)


def _finish(out: np.ndarray, grit: float = 1.6, room: float = 0.35, peak: float = 0.5,
            slow: float = 1.0) -> np.ndarray:
    out = np.tanh(out * grit)
    d = int(SR * 0.045)
    if len(out) > d:
        out[d:] += room * out[:-d]
    # a slowed tape loses its highs too: without this the grit and the breath
    # stay bright while the voice drops, and it sounds hissy instead of deep
    f = np.fft.rfftfreq(len(out), 1 / SR)
    out = np.fft.irfft(np.fft.rfft(out) / (1 + (f / (9000 / slow ** 0.6)) ** 4), len(out))
    return (out / max(1e-6, np.max(np.abs(out))) * peak).astype(np.float32)


def _env(t: np.ndarray, attack: float, release: float) -> np.ndarray:
    return np.minimum(1, t / attack) * np.clip((1 - t) / release, 0, 1)


def demon_meow(slow: float = 1.0) -> np.ndarray:
    """The cursed meow taken further: past ~2x it growls (a rough flutter and
    an octave below) and gets a cave echo; up to 6x slower by the end."""
    dur = 0.6 * slow
    n = int(SR * dur)
    t = np.arange(n) / n
    g = float(np.clip((slow - 1.5) / 4.0, 0, 1))          # how demonic: 0 at the start
    f0 = (430 + 330 * np.sin(np.pi * np.minimum(1, t * 1.4)) - 120 * t) / slow
    k1 = np.clip(t * 2.2, 0, 1)
    k2 = np.clip(t * 2.2 - 1.1, 0, 1)
    fs = slow ** 0.75             # the formants drop less than the pitch: a demon voice that a phone still plays
    f1 = (300 + 500 * k1 - 420 * k2) / fs
    f2 = (2300 - 1000 * k1 - 400 * k2) / fs
    out = _harmonics(f0, [(f1, 140 / fs, 1.0), (f2, 220 / fs, 0.7)], n, buzz=0.12 + 0.3 * g, sub=0.35 * g)
    secs = np.arange(n) / SR
    out *= _env(t, 0.08, 0.25) * (1 - 0.4 * g + 0.4 * g * np.sin(2 * np.pi * 27 * secs))
    out = np.tanh(out * (1.6 + 1.4 * g))
    for dl, gain in ((0.07, 0.3), (0.13, 0.22), (0.21, 0.15)):      # the cave
        d = int(SR * dl * (1 + g))
        if len(out) > d:
            out[d:] += gain * g * out[:-d]
    return _finish(out, grit=1.0, room=0.3, slow=slow)


def quack(slow: float = 1.0) -> np.ndarray:
    """A duck's quack (a nasal, buzzy "aah" with a rasp); slowed down it's a
    giant goose honk."""
    n = int(SR * 0.2 * slow)
    t = np.arange(n) / n
    f0 = (260 - 70 * t + 30 * np.exp(-t * 12)) / slow
    form = [(950 / slow + 0 * t, 160 / slow, 1.0), (1650 / slow + 0 * t, 200 / slow, 0.8),
            (2700 / slow + 0 * t, 260 / slow, 0.5)]
    out = _harmonics(f0, form, n, buzz=0.35)
    secs = np.arange(n) / SR
    out *= _env(t, 0.06, 0.35) * (0.75 + 0.25 * np.sin(2 * np.pi * 44 / slow * secs))
    return _finish(out, grit=2.6, room=0.2, slow=slow)


def huh(slow: float = 1.0) -> np.ndarray:
    """A puzzled "huh?": a breathy h, then "uh" rising like a question."""
    n = int(SR * 0.42 * slow)
    t = np.arange(n) / n
    vowel = [(650 / slow, 120 / slow, 1.0), (1200 / slow, 160 / slow, 0.6), (2500 / slow, 240 / slow, 0.25)]
    h_end = 0.2
    v = np.clip((t - h_end) / (1 - h_end), 0, 1)
    f0 = (108 + 75 * v ** 1.6) / slow
    voiced = _harmonics(f0, [(fc + 0 * t, bw, w) for fc, bw, w in vowel], n, buzz=0.1 + 0.08 * slow)
    voiced *= np.clip((t - h_end + 0.04) / 0.08, 0, 1) * np.clip((1 - t) / 0.3, 0, 1)
    breath = _shaped_noise(n, vowel, 7)
    breath *= np.clip(t / 0.05, 0, 1) * np.clip((h_end + 0.05 - t) / 0.12, 0, 1)
    out = voiced / max(1e-6, np.max(np.abs(voiced))) + breath / max(1e-6, np.max(np.abs(breath))) * 0.45
    return _finish(out, grit=1.3, room=0.25, slow=slow)


def bruh(slow: float = 1.0) -> np.ndarray:
    """A flat, unimpressed "bruh": a soft b, an r gliding into "uh", the
    pitch falling, a breathy end."""
    n = int(SR * 0.5 * slow)
    t = np.arange(n) / n
    k = np.clip((t - 0.08) / 0.3, 0, 1)                    # r -> uh
    f1 = (420 + 230 * k) / slow
    f2 = (1050 + 150 * k) / slow
    f3 = (1550 + 950 * k) / slow                           # a low F3 is what makes the "r"
    f0 = (138 - 42 * t) / slow
    voiced = _harmonics(f0, [(f1, 110 / slow, 1.0), (f2, 150 / slow, 0.6), (f3, 200 / slow, 0.35)], n, buzz=0.1 + 0.08 * slow)
    voiced *= np.clip(t / 0.03, 0, 1) * np.clip((0.86 - t) / 0.25, 0, 1)
    secs = np.arange(n) / SR
    burst = np.sin(2 * np.pi * np.cumsum(90 / slow + 60 / slow * np.exp(-secs * 80 / slow)) / SR)
    burst *= np.exp(-secs * 60 / slow) * 0.6
    breath = _shaped_noise(n, [(650 / slow, 150 / slow, 1.0), (1200 / slow, 200 / slow, 0.5)], 11)
    breath *= np.clip((t - 0.7) / 0.1, 0, 1) * np.clip((1 - t) / 0.15, 0, 1)
    out = (voiced / max(1e-6, np.max(np.abs(voiced))) + burst
           + breath / max(1e-6, np.max(np.abs(breath))) * 0.35)
    return _finish(out, grit=1.4, room=0.25, slow=slow)


def goat(slow: float = 1.0) -> np.ndarray:
    """A goat's "meeeh" (the bleat: a fast wobble in pitch and loudness)."""
    n = int(SR * 0.55 * slow)
    t = np.arange(n) / n
    secs = np.arange(n) / SR
    wob = np.sin(2 * np.pi * 7.5 / slow * secs)
    f0 = (390 + 40 * np.exp(-t * 6) - 60 * t) * (1 + 0.11 * wob) / slow
    form = [(580 / slow + 0 * t, 130 / slow, 1.0), (1750 / slow + 0 * t, 220 / slow, 0.75),
            (2600 / slow + 0 * t, 260 / slow, 0.35)]
    out = _harmonics(f0, form, n, buzz=0.2)
    out *= _env(t, 0.05, 0.3) * (0.6 + 0.4 * wob)
    return _finish(out, grit=1.9, room=0.3, slow=slow)


def bonk(slow: float = 1.0) -> np.ndarray:
    """A cartoon bonk: a hollow wooden hit that drops in pitch."""
    n = int(SR * 0.28 * slow)
    secs = np.arange(n) / SR
    drop = 1 + 0.5 * np.exp(-secs * 40 / slow)
    fs = slow ** 0.7
    out = np.zeros(n)
    for f, a, dec in ((520, 1.0, 18), (1310, 0.55, 30), (2210, 0.25, 45)):
        out += a * np.sin(2 * np.pi * np.cumsum(f / fs * drop) / SR) * np.exp(-secs * dec / slow)
    click = np.random.default_rng(3).standard_normal(n) * np.exp(-secs * 900 / slow) * 0.6
    return _finish(out + click, grit=1.5 + 0.8 * slow, room=0.15, slow=slow)


def boing(slow: float = 1.0) -> np.ndarray:
    """A spring boing: a tone with a fast wobble that dies down."""
    n = int(SR * 0.45 * slow)
    t = np.arange(n) / n
    secs = np.arange(n) / SR
    f = (420 + 200 * t) * (1 + 0.22 * np.exp(-t * 4) * np.sin(2 * np.pi * 17 / slow * secs)) / slow ** 0.7
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * _env(t, 0.01, 0.5)
    return _finish(out, grit=1.3 + 1.2 * slow, room=0.2, slow=slow)   # the grit keeps it audible on a phone


MEME_MAKERS = {"demon_meow": demon_meow, "quack": quack, "huh": huh, "bruh": bruh, "goat": goat,
               "bonk": bonk, "boing": boing}


def meme_track(kind: str, hits: list, n: int) -> np.ndarray:
    """One meme sound per bounce, slower and deeper as the video goes on:
    plain at the start, ``MEMES[kind][0]`` times slower at the last bounce
    (the box is full by then). The gaps between them grow with the length."""
    out = np.zeros(n, dtype=np.float32)
    if not hits:
        return out
    most, gap0 = MEMES[kind]
    make = MEME_MAKERS[kind]
    end = max(1e-3, hits[-1][0])
    last, cache = -1e9, {}
    for t, vol in hits:
        p = min(1.0, max(0.0, t / end))
        slow = round(1.0 + (most - 1.0) * p ** 1.3, 2)
        slow = round(slow * 20) / 20                       # 0.05 steps, so they cache
        if t - last < gap0 * slow ** 0.6:
            continue
        w = cache.get(slow)
        if w is None:
            w = cache[slow] = make(slow)
        i = int(t * SR)
        seg = out[i:i + len(w)]
        seg += w[:len(seg)] * (0.5 + 0.5 * vol)
        last = t
    return out


def track(kind: str, hits: list, n: int) -> np.ndarray:
    """The bounce sound for a whole video: one meow / riff note / tap per
    hit ("none": silence)."""
    if kind in MEMES:
        return meme_track(kind, hits, n)
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
