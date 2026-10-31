"""Art looks for the non-fight ball videos (31 Oct).

Dean asked for new designs and 2D art on the 08:00 videos, rotating. From a
board of 8 he said "do it" to the suggested set: pixel 8-bit, sticker pop,
notebook doodle and arcade screen, with the current neon kept in the
rotation. Research found no study saying one art style wins; what holds is a
bold, high-contrast first frame, and YouTube's July 2026 rule against
channels whose videos all look like one template -- so the looks rotate.

Each look restyles a finished frame (``apply``), so every kind gets it
without its painter changing. ``recipe["art"]`` picks it; recipes without
it (every video made before) render exactly as before, and "neon" is the
frame as drawn.

* pixel: the frame at 1/4 size, 5 levels per channel, blown back up with
  hard pixels (on the plain black backdrop: a space nebula turned to blobs).
* arcade: green phosphor screen: brightness to green, a soft glow,
  scanlines, darker corners.
* sticker / notebook: the painter's background (its ``bg``: the backdrop it
  draws everything on) is swapped for yellow stripes / lined paper, and
  everything drawn on it gets a thick black (sticker) or thin ink
  (notebook) outline, which also keeps white lines and text visible on
  the paper (turning white to ink made white pictures, a skull, a ghost,
  black). What counts as drawn is any pixel that differs from the
  painter's background by more than ``FG_DIFF`` (dim glow and trail tails
  go with the background), worked out at half size for speed.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

ARTS = ("neon", "pixel", "sticker", "notebook", "arcade")
LABELS = {"neon": "Neon", "pixel": "Pixel 8-bit", "sticker": "Sticker pop", "notebook": "Notebook doodle",
          "arcade": "Arcade screen"}
# the kinds that get a look: everything but the fights (their own arenas)
# and Evolve (a light gumball machine, not on a dark backdrop)
KINDS = ("escape", "touch", "breakin", "merge", "center", "spikes", "shrink", "jaws", "spiral", "rays", "box",
         "zen", "trail", "survivor")
# the black hole of Hold the center is drawn in the backdrop's own black, so
# on paper / stripes it would vanish: those two looks skip it
SKIP = {"center": ("sticker", "notebook")}
FG_DIFF = 40
INK = (34, 34, 46)


def pick(recipe: dict, last: Optional[str], seed) -> Optional[str]:
    """A look for this recipe, never the one the last video had."""
    fmt = recipe.get("format")
    if fmt not in KINDS:
        return None
    rng = random.Random(f"art-{seed}")
    options = [a for a in ARTS if a != last and a not in SKIP.get(fmt, ())]
    return rng.choice(options or ["neon"])


class Look:
    """Restyles frames of one video (keeps the per-video pieces)."""

    def __init__(self, art: str, size: tuple, bg: Optional[Image.Image] = None):
        self.art = art
        self.w, self.h = size
        self.bg_small = None
        if bg is not None and bg.size == size:
            self.bg_small = np.asarray(bg.convert("RGB").resize((self.w // 2, self.h // 2), Image.BOX),
                                       dtype=np.int16)
        if art == "arcade":
            yy, xx = np.mgrid[0:self.h, 0:self.w]
            d = ((xx - self.w / 2) / (self.w / 2)) ** 2 + ((yy - self.h / 2) / (self.h / 2)) ** 2
            vig = np.clip(1.08 - 0.32 * d, 0.55, 1.0)
            scan = np.where((np.arange(self.h) % 3) == 2, 0.62, 1.0)[:, None]
            self.shade = (vig * scan).astype(np.float32)[..., None]
        if art in ("sticker", "notebook"):
            self.paper = np.asarray(self._backdrop(), dtype=np.uint8)
            import cv2
            self.cv2 = cv2
            r = 5 if art == "sticker" else 2           # outline width at half size
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    def _backdrop(self) -> Image.Image:
        im = Image.new("RGB", (self.w, self.h), (255, 214, 60) if self.art == "sticker" else (250, 247, 236))
        d = ImageDraw.Draw(im)
        if self.art == "sticker":
            for k in range(-self.h, self.w, 92):
                d.line([(k, self.h), (k + self.h, 0)], fill=(255, 200, 40), width=36)
        else:
            for y in range(150, self.h, 64):
                d.line([(0, y), (self.w, y)], fill=(176, 204, 236), width=3)
            d.line([(110, 0), (110, self.h)], fill=(236, 128, 128), width=3)
        return im

    def _fg(self, a: np.ndarray) -> np.ndarray:
        """Drawn-on pixels (half size, bool)."""
        small = np.asarray(Image.fromarray(a).resize((self.w // 2, self.h // 2), Image.BOX), dtype=np.int16)
        if self.bg_small is not None:
            return np.abs(small - self.bg_small).max(axis=2) > FG_DIFF
        return small.max(axis=2) > FG_DIFF + 10

    def apply(self, frame: Image.Image) -> Image.Image:
        if self.art == "neon":
            return frame
        if self.art == "pixel":
            s = frame.convert("RGB").resize((self.w // 4, self.h // 4), Image.BOX)
            a = np.asarray(s, dtype=np.float32)
            a = np.minimum(np.round(a / 64.0) * 64.0, 255.0)
            return Image.fromarray(a.astype(np.uint8)).resize((self.w, self.h), Image.NEAREST)
        a = np.asarray(frame.convert("RGB"))
        if self.art == "arcade":
            f = a.astype(np.float32)
            lum = 0.3 * f[..., 0] + 0.59 * f[..., 1] + 0.11 * f[..., 2]
            lum = np.clip(lum * 1.25, 0, 255)
            g = np.stack([lum * 0.32, lum, lum * 0.5], axis=2)
            glow = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((self.w // 8, self.h // 8), Image.BOX)
                              .resize((self.w, self.h), Image.BILINEAR), dtype=np.float32)
            out = (g + 0.55 * glow) * self.shade + np.array([2, 10, 6], np.float32)
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        # sticker / notebook: swap the background, outline what's drawn
        cv2 = self.cv2
        fg = self._fg(a).astype(np.uint8)
        ring = cv2.dilate(fg, self.kernel) & (1 - fg)
        fg_full = cv2.resize(fg, (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ring_full = cv2.resize(ring, (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out = self.paper.copy()
        out[fg_full] = a[fg_full]
        out[ring_full] = (20, 20, 30) if self.art == "sticker" else INK
        return Image.fromarray(out)
