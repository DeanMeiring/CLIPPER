"""Cursed merge noises (Dean, 19 Oct: "for the merging we move more to
cursed noises ... a deepened meow or a weird moan, something weird that
freaks people out but they're intrigued"). Made from nothing (numpy only),
so nothing can be claimed. Each one is a voice synthesised from harmonics
shaped by vowel formants (like ball_sounds.meow), then bent: slowed down,
reversed, tape-stopped, stacked out of tune, given a throaty rattle and a
room. Small merges get short ones, big merges deep, long, strange ones.

``merge_track`` lays them on a video's merges (the "pop" events); the
music (Dean's song) keeps playing under them."""
from __future__ import annotations

import numpy as np

from .ball_evolution import SR


# ---- building blocks ---------------------------------------------------------

def _rng(seed):
    return np.random.default_rng(seed)


def _voice(f0, f1, f2, f3=None, breath=0.08, rattle=0.0, rattle_hz=28.0, seed=0, bw=1.0):
    """A voice: harmonics of the pitch curve f0 weighted by formants f1/f2/f3
    (all arrays the same length). breath = noise mixed in; rattle = a throaty
    amplitude flutter (a purr, a fry)."""
    n = len(f0)
    r = _rng(seed)
    jitter = 1 + 0.012 * np.convolve(r.standard_normal(n), np.ones(400) / 400, mode="same")
    f0 = f0 * jitter
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    for h in range(1, 80):
        fh = f0 * h
        if np.min(fh) > SR / 2.3:
            break
        amp = (np.exp(-((fh - f1) / (110 * bw)) ** 2) + 0.7 * np.exp(-((fh - f2) / (170 * bw)) ** 2) + 0.1 / h)
        if f3 is not None:
            amp += 0.35 * np.exp(-((fh - f3) / (240 * bw)) ** 2)
        amp = np.where(fh < SR / 2.3, amp, 0)
        out += amp * np.sin(h * phase)
    if breath:
        noise = r.standard_normal(n)
        # breath shaped roughly like the second formant
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1 / SR)
        spec *= np.exp(-((freqs - np.mean(f2)) / 900) ** 2) + 0.3 * np.exp(-((freqs - np.mean(f1)) / 300) ** 2)
        noise = np.fft.irfft(spec, n)
        out += breath * noise / (np.max(np.abs(noise)) + 1e-9) * np.max(np.abs(out))
    if rattle:
        t = np.arange(n) / SR
        flutter = 0.5 + 0.5 * np.sin(2 * np.pi * rattle_hz * t + 2 * np.sin(2 * np.pi * 3.1 * t))
        out *= (1 - rattle) + rattle * flutter
    return out


def _env(n, attack, release):
    t = np.arange(n) / n
    return np.minimum(1, t / max(1e-4, attack)) * np.minimum(1, (1 - t) / max(1e-4, release))


def _slow(x, factor):
    """Play x `factor` times slower (pitch and formants drop together, like
    a slowed tape)."""
    n = int(len(x) * factor)
    return np.interp(np.arange(n) / factor, np.arange(len(x)), x)


def _tape_stop(x, start=0.45):
    """Play normally, then the tape slows to a stop from `start` (0..1)."""
    n = len(x)
    s = int(n * start)
    pos = list(np.arange(s, dtype=float))
    p, speed = float(s), 1.0
    while speed > 0.06 and p < n - 1:
        pos.append(p)
        p += speed
        speed *= 0.99994
    return np.interp(np.array(pos), np.arange(n), x)


def _reverb(x, sec=1.2, wet=0.35, seed=7, bright=0.5):
    """A room: convolution with decaying noise."""
    r = _rng(seed)
    m = int(SR * sec)
    ir = r.standard_normal(m) * np.exp(-np.arange(m) / m * 7)
    # darker tail
    k = int(2 + (1 - bright) * 20)
    ir = np.convolve(ir, np.ones(k) / k, mode="same")
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + m
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:L]
    dry = np.concatenate([x, np.zeros(m)])
    y = y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))
    return (1 - wet) * dry + wet * y


def _sub(x, mix=0.4):
    """An octave below, rough (a crude octave divider): demon weight."""
    sign = np.sign(x)
    flips = np.cumsum(np.abs(np.diff(sign, prepend=sign[0])) > 0) // 2 % 2
    sq = np.where(flips == 0, 1.0, -1.0) * np.abs(x)
    sq = np.convolve(sq, np.ones(30) / 30, mode="same")
    return x + mix * sq


def _norm(x, peak=0.5):
    return (x / (np.max(np.abs(x)) + 1e-9) * peak).astype(np.float32)


def _glide(n, *points):
    """A curve through (time 0..1, value) points."""
    t = np.arange(n) / max(1, n - 1)
    xs, ys = zip(*points)
    return np.interp(t, xs, ys)


# ---- the voices --------------------------------------------------------------

def meow_raw(dur=0.6, high=1.0, seed=0):
    """A plain meow ("mee-ow": i -> a -> u), the raw material."""
    n = int(SR * dur)
    t = np.arange(n) / n
    f0 = (430 + 330 * np.sin(np.pi * np.minimum(1, t * 1.4)) - 120 * t) * high
    k1 = np.clip(t * 2.2, 0, 1)
    k2 = np.clip(t * 2.2 - 1.1, 0, 1)
    f1 = 300 + 500 * k1 - 420 * k2
    f2 = 2300 - 1000 * k1 - 400 * k2
    out = _voice(f0, f1, f2, breath=0.05, seed=seed)
    out = np.tanh(out / (np.max(np.abs(out)) + 1e-9) * 1.8 * _env(n, 0.08, 0.25))
    return out


def demon_meow(depth=3.0, seed=0):
    """The meow slowed `depth` times, with an octave-down rumble and a big
    dark room."""
    x = _slow(meow_raw(0.55, seed=seed), depth)
    x = _sub(x, 0.35)
    return _norm(_reverb(x, 1.6, 0.4, seed, bright=0.2))


def kitten_moan(depth=1.8, seed=1):
    """A long, wavering "mrrraaoww" with a purring rattle, slowed: the
    moaning-cat meme kind of sound, but made here."""
    dur = 1.15
    n = int(SR * dur)
    t = np.arange(n) / n
    f0 = _glide(n, (0, 300), (0.15, 520), (0.45, 610), (0.7, 470), (1, 330))
    f0 *= 1 + 0.05 * np.sin(2 * np.pi * 6.5 * t * dur)               # the waver
    f1 = _glide(n, (0, 280), (0.2, 650), (0.6, 820), (1, 400))
    f2 = _glide(n, (0, 1900), (0.2, 1500), (0.6, 1250), (1, 850))
    x = _voice(f0, f1, f2, breath=0.12, rattle=0.55, rattle_hz=24, seed=seed)
    x = np.tanh(x / (np.max(np.abs(x)) + 1e-9) * 2.2 * _env(n, 0.1, 0.3))
    x = _slow(x, depth)
    return _norm(_reverb(x, 1.3, 0.33, seed, bright=0.3))


def moan(depth=1.4, seed=2):
    """A low human-ish "mmmm-ohhh-uhh", falling, with a slow wobble."""
    dur = 1.3
    n = int(SR * dur)
    t = np.arange(n) / n
    f0 = _glide(n, (0, 125), (0.3, 150), (1, 92)) * (1 + 0.03 * np.sin(2 * np.pi * 4.8 * t * dur))
    f1 = _glide(n, (0, 260), (0.25, 260), (0.45, 520), (0.8, 470), (1, 330))     # m -> o -> u
    f2 = _glide(n, (0, 1050), (0.25, 1050), (0.45, 880), (0.8, 900), (1, 760))
    f3 = _glide(n, (0, 2500), (1, 2400))
    x = _voice(f0, f1, f2, f3, breath=0.18, rattle=0.25, rattle_hz=11, seed=seed, bw=1.3)
    x = np.tanh(x / (np.max(np.abs(x)) + 1e-9) * 1.5 * _env(n, 0.22, 0.35))
    x = _slow(x, depth)
    return _norm(_reverb(x, 1.4, 0.38, seed, bright=0.35))


def reverse_wail(depth=2.2, seed=3):
    """A meow played backwards out of its own echo: it sucks in and cuts off."""
    x = meow_raw(0.7, high=0.9, seed=seed)
    x = _reverb(x, 1.8, 0.6, seed, bright=0.4)
    x = _slow(x[::-1], depth)
    return _norm(x * _env(len(x), 0.02, 0.01))


def dying_meow(seed=4):
    """A meow whose tape slows to a stop halfway."""
    x = meow_raw(0.9, high=1.0, seed=seed)
    x = _tape_stop(x, 0.3)
    return _norm(_reverb(x, 0.9, 0.3, seed))


def choir(depth=1.0, seed=5):
    """Four slightly out-of-tune voices singing "aaah" a semitone apart:
    uncanny."""
    dur = 1.8
    n = int(SR * dur)
    t = np.arange(n) / n
    out = np.zeros(n)
    for k, (semi, det) in enumerate([(0, 0.0), (1, 0.4), (-12, -0.3), (6, 0.2)]):
        base = 196 * 2 ** (semi / 12) * (1 + det / 100) / depth
        f0 = base * (1 + 0.012 * np.sin(2 * np.pi * (5 + k * 0.7) * t * dur + k))
        f0 *= _glide(n, (0, 1.0), (0.85, 0.97), (1, 0.85))           # sags at the end
        f1 = np.full(n, 760.0 / depth ** 0.5)
        f2 = np.full(n, 1180.0 / depth ** 0.5)
        out += _voice(f0, f1, f2, breath=0.1, seed=seed + k, bw=1.2)
    out *= _env(n, 0.35, 0.3)
    return _norm(_reverb(out, 2.2, 0.5, seed, bright=0.3))


def laugh(depth=2.6, seed=6):
    """A deep, slow "heh-heh-heh-heh"."""
    out = []
    for k in range(5):
        dur = 0.13
        n = int(SR * dur)
        f0 = _glide(n, (0, 260 - 12 * k), (1, 220 - 12 * k))
        f1 = np.full(n, 560.0)
        f2 = np.full(n, 1700.0)
        x = _voice(f0, f1, f2, breath=0.5, seed=seed + k)
        x = x / (np.max(np.abs(x)) + 1e-9) * _env(n, 0.05, 0.5) * (1 - 0.1 * k)
        out.append(np.concatenate([x, np.zeros(int(SR * 0.07))]))
    x = _slow(np.concatenate(out), depth)
    return _norm(_reverb(_sub(x, 0.3), 1.2, 0.35, seed, bright=0.25))


def growl(depth=1.0, seed=8):
    """A demon "uhhhh": a very low voice with vocal fry, distorted."""
    dur = 1.1
    n = int(SR * dur)
    f0 = _glide(n, (0, 70), (0.5, 62), (1, 48)) / depth
    f1 = _glide(n, (0, 420), (1, 360))
    f2 = _glide(n, (0, 900), (1, 800))
    x = _voice(f0, f1, f2, breath=0.3, rattle=0.7, rattle_hz=17, seed=seed, bw=1.5)
    x = np.tanh(x / (np.max(np.abs(x)) + 1e-9) * 4.0) * _env(n, 0.12, 0.3)
    return _norm(_reverb(_sub(x, 0.5), 1.3, 0.35, seed, bright=0.2))


def chirp_down(seed=9):
    """A tiny meow that drops an octave as it goes: for the small merges."""
    x = meow_raw(0.32, high=1.25, seed=seed)
    n = len(x)
    pos = np.cumsum(np.linspace(1.0, 0.45, n))
    pos = pos[pos < n - 1]
    x = np.interp(pos, np.arange(n), x)
    return _norm(_reverb(x, 0.5, 0.2, seed), 0.42)


SOUNDS = {
    "chirp_down": chirp_down,
    "dying_meow": dying_meow,
    "kitten_moan": kitten_moan,
    "demon_meow": demon_meow,
    "moan": moan,
    "reverse_wail": reverse_wail,
    "laugh": laugh,
    "choir": choir,
    "growl": growl,
}


def for_merge(tier: int, k: int, seed: int = 0) -> np.ndarray:
    """The noise for the k-th merge into `tier` (0 = smallest): small ones
    short, big ones deep and long; it rotates so the same one doesn't play
    twice in a row."""
    if tier <= 1:
        pool = [lambda s: chirp_down(s), lambda s: dying_meow(s)]
    elif tier <= 3:
        pool = [lambda s: kitten_moan(1.4 + 0.2 * tier, s), lambda s: demon_meow(2.0 + 0.3 * tier, s),
                lambda s: reverse_wail(1.6 + 0.2 * tier, s), lambda s: laugh(2.2, s)]
    else:
        pool = [lambda s: moan(1.2 + 0.1 * tier, s), lambda s: choir(1.0 + 0.08 * tier, s),
                lambda s: growl(1.0 + 0.05 * tier, s), lambda s: demon_meow(3.0 + 0.3 * tier, s)]
    return pool[(k + seed) % len(pool)](seed + k)


# small merges come several a second: space them out so they don't pile up
SMALL_GAP, MID_GAP = 0.30, 0.55
CAP = {0: 0.55, 1: 0.55, 2: 1.3, 3: 1.3}      # the longest a noise of that size may ring (s)
BIG_CAP = 2.8


def merge_track(merges: list, n: int, seed: int = 0) -> np.ndarray:
    """merges: [(time, tier)] -> the noises, n samples long."""
    out = np.zeros(n, dtype=np.float32)
    last_small = last_mid = -9.0
    k = 0
    cache: dict = {}
    for t, tier in merges:
        tier = int(tier)
        if tier <= 1:
            if t - last_small < SMALL_GAP or t - last_mid < 0.25:
                continue
            last_small = t
        elif tier <= 3:
            if t - last_mid < MID_GAP:
                continue
            last_mid = t
        key = (min(tier, 6), k % 8)
        w = cache.get(key)
        if w is None:
            w = for_merge(tier, k, seed)
            cap = int(SR * CAP.get(tier, BIG_CAP))
            if len(w) > cap:
                w = w[:cap].copy()
                f = int(SR * 0.25)
                w[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            cache[key] = w
        k += 1
        i = int(t * SR)
        seg = out[i:i + len(w)]
        seg += w[:len(seg)] * (0.7 if tier <= 1 else 1.0)
    return out
