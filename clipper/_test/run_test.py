import sys
sys.path.insert(0, "/home/claude/clipper")

from pathlib import Path
from clipper.transcribe import Word
from clipper.reframe import compute_crop_window
from clipper.captions import build_ass
from clipper.render import render_clip

video = Path("/home/claude/clipper/_test/test_source.mp4")
out_dir = Path("/home/claude/clipper/_test/out")
out_dir.mkdir(exist_ok=True)

# Fake a short transcript covering the first 8 seconds of the test video
sentence = "this is a test of the highlight clip captioning pipeline right now".split()
words = []
t = 0.5
for w in sentence:
    dur = 0.35 + 0.05 * len(w)
    words.append(Word(text=w, start=round(t, 2), end=round(t + dur, 2)))
    t += dur + 0.05

print("Words:", [(w.text, w.start, w.end) for w in words])

crop = compute_crop_window(video, start=0.0, end=8.0, target_ratio=(1080, 1920))
print("Crop window (no face expected -> fallback center):", crop)

ass_path = out_dir / "clip_01.ass"
build_ass(words, clip_start=0.0, output_path=ass_path)
print("ASS written:", ass_path, "-", len(ass_path.read_text().splitlines()), "lines")

out_path = out_dir / "clip_01.mp4"
render_clip(video, start=0.0, end=8.0, crop=crop, ass_path=ass_path, output_path=out_path)
print("Rendered:", out_path, out_path.stat().st_size, "bytes")
