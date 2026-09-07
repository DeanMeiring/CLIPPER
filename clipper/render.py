"""Cut, crop, and burn captions into a single output clip via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .reframe import CropWindow, Layout, SplitLayout


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
    layout: Layout,
    ass_path: Path,
    output_path: Path,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    ass = _escape_for_filter(ass_path)

    if isinstance(layout, SplitLayout):
        top, bottom = layout.top, layout.bottom
        filter_complex = (
            f"[0:v]crop={top.w}:{top.h}:{top.x}:{top.y},scale={out_w}:{layout.top_out_h}[top];"
            f"[0:v]crop={bottom.w}:{bottom.h}:{bottom.x}:{bottom.y},scale={out_w}:{layout.bottom_out_h}[bottom];"
            f"[top][bottom]vstack=inputs=2[stacked];"
            f"[stacked]ass='{ass}'[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        crop: CropWindow = layout
        vf = (
            f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
            f"scale={out_w}:{out_h},"
            f"ass='{ass}'"
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

    # A hard wall-clock ceiling: this runs on the web app's single sequential
    # worker thread, and the cancel signal is only checked between pipeline
    # steps, not inside a blocking subprocess call -- an ffmpeg hang (a
    # malformed/truncated source, an unusual codec, a filter stall) would
    # otherwise wedge that thread, and every future queued job, forever.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ffmpeg timed out after {e.timeout:.0f}s rendering {output_path.name} "
            "-- source video may be corrupt or an unusual codec"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path.name}:\n{result.stderr[-2000:]}")
    return output_path
