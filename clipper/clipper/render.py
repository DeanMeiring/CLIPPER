"""Cut, crop, and burn captions into a single output clip via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .reframe import CropWindow


def _escape_for_filter(path: Path) -> str:
    """ffmpeg filtergraph args treat : and \\ specially -- this keeps
    Windows paths (C:\\Users\\...) working inside -vf/-filter_complex."""
    s = str(path).replace("\\", "/")
    s = s.replace(":", "\\:")
    return s


def render_clip(
    source_video: Path,
    start: float,
    end: float,
    crop: CropWindow,
    ass_path: Path,
    output_path: Path,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)

    vf = (
        f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
        f"scale={out_w}:{out_h},"
        f"ass='{_escape_for_filter(ass_path)}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source_video),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path.name}:\n{result.stderr[-2000:]}")
    return output_path
