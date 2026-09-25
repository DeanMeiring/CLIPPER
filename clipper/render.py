"""Cut, crop, and burn captions into a single output clip via ffmpeg."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from .reframe import CropWindow, Layout, LetterboxLayout, MultiCamSplitLayout, SplitLayout


def _escape_for_filter(path: Path) -> str:
    """ffmpeg filtergraph args treat : and \\ specially -- this keeps
    Windows paths (C:\\Users\\...) working inside -vf/-filter_complex."""
    s = str(path).replace("\\", "/")
    s = s.replace(":", "\\:")
    return s


def _letterbox_scale_pad(w: int, h: int) -> str:
    """Scale a facecam crop to fit within w x h without distorting it,
    padding any leftover space with black bars instead of stretching to
    fill it exactly.

    A narrow multi-cam tile's aspect ratio (out_w split 2-3 ways against
    a fixed band height) doesn't match any real facecam window's -- the
    crop side of this used to force an exact-aspect match, which either
    pulled surrounding gameplay into the tile or squashed the person's
    face with a severe non-uniform stretch (measured on a real rejected
    render: height stretched 2.6x more than width). A small letterboxed
    thumbnail reads far better than either."""
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"


def render_clip(
    source_video: Path,
    start: float,
    end: float,
    layout: Layout,
    ass_path: Optional[Path],
    output_path: Path,
    out_w: int = 1080,
    out_h: int = 1920,
    crf: int = 20,
) -> Path:
    """ass_path None renders without captions -- the base for render_edited,
    which burns them in after its own cuts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    burn = f"ass='{_escape_for_filter(ass_path)}'" if ass_path else "null"

    if isinstance(layout, MultiCamSplitLayout):
        top = layout.top
        parts = [f"[0:v]crop={top.w}:{top.h}:{top.x}:{top.y},scale={out_w}:{layout.top_out_h}[top];"]
        tile_labels = []
        for i, (cam, tw) in enumerate(zip(layout.bottom_cams, layout.bottom_cam_out_widths)):
            label = f"cam{i}"
            parts.append(
                f"[0:v]crop={cam.w}:{cam.h}:{cam.x}:{cam.y},"
                f"{_letterbox_scale_pad(tw, layout.bottom_out_h)}[{label}];"
            )
            tile_labels.append(f"[{label}]")
        parts.append(f"{''.join(tile_labels)}hstack=inputs={len(tile_labels)}[bottom];")
        parts.append("[top][bottom]vstack=inputs=2[stacked];")
        parts.append(f"[stacked]{burn}[outv]")
        filter_complex = "".join(parts)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    elif isinstance(layout, SplitLayout):
        top, bottom = layout.top, layout.bottom
        filter_complex = (
            f"[0:v]crop={top.w}:{top.h}:{top.x}:{top.y},scale={out_w}:{layout.top_out_h}[top];"
            f"[0:v]crop={bottom.w}:{bottom.h}:{bottom.x}:{bottom.y},"
            f"{_letterbox_scale_pad(out_w, layout.bottom_out_h)}[bottom];"
            f"[top][bottom]vstack=inputs=2[stacked];"
            f"[stacked]{burn}[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    elif isinstance(layout, LetterboxLayout):
        # No crop at all -- the whole source frame, scaled to fit the
        # target width/height, with a blurred/zoomed copy of the same
        # frame filling whatever's left top/bottom instead of plain black
        # bars. `split` feeds the same input into two parallel chains: one
        # scaled UP to cover the full canvas then blurred (the backdrop),
        # one scaled DOWN to fit within it losslessly (the real shot),
        # composited centered on top. Both chains read the source's actual
        # dimensions themselves -- nothing here needs them precomputed.
        filter_complex = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},gblur=sigma=20[bg2];"
            f"[fg]scale={out_w}:-2:force_original_aspect_ratio=decrease[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,{burn}[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        crop: CropWindow = layout
        vf = (
            f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
            f"scale={out_w}:{out_h},"
            f"{burn}"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
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


def trim_clip(
    source_path: Path, output_path: Path, trim_start: float, trim_end: float, duration: float,
) -> Path:
    """Cut trim_start seconds off the front and trim_end seconds off the
    back of an ALREADY-RENDERED clip -- for trimming a beat or two before
    posting it, not part of the original render.

    Captions are burned directly into the pixels at render time (there's
    no separate subtitle track), so a plain cut of the finished video
    carries its captions along automatically -- nothing needs re-syncing
    the way it would if this touched the source video and layout instead."""
    new_duration = duration - trim_start - trim_end
    if trim_start < 0 or trim_end < 0:
        raise ValueError("Trim amounts can't be negative")
    if new_duration < 1.0:
        raise ValueError("That trim would leave less than a second of clip")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{trim_start:.3f}",
        "-i", str(source_path),
        "-t", f"{new_duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s trimming {source_path.name}") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed trimming {source_path.name}:\n{result.stderr[-2000:]}")
    return output_path


def overlay_hook_line(source_path: Path, ass_path: Path, output_path: Path) -> Path:
    """Burn a flash-hook .ass (captions.hook_line_ass) onto an ALREADY-
    RENDERED clip. New file out, audio stream copied untouched -- like
    trim_clip, this only composites a text layer onto the existing frames,
    it never re-times, re-cuts, or re-joins anything, so it carries none of
    the audio/video drift risk a concatenation-based intro card would."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass = _escape_for_filter(ass_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-vf", f"ass='{ass}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s adding the hook line to {source_path.name}") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed adding the hook line to {source_path.name}:\n{result.stderr[-2000:]}")
    return output_path


def _has_audio(path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0 and bool(json.loads(r.stdout or "{}").get("streams"))


def _frame_rate(path: Path) -> str:
    """The video's frame rate as an ffmpeg rational, sanity-checked --
    stream recordings can report nonsense (a 1/1000000 timebase read as
    1,000,000 fps), which x264 then refuses to encode sensibly."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        stream = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            num, _, den = str(stream.get(key) or "0/0").partition("/")
            if float(den or 0) > 0 and 10 <= float(num) / float(den) <= 120:
                return f"{num}/{den}"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return "30"


def apply_edits(base: Path, plan, ass_path: Path, output_path: Path, out_w: int = 1080, out_h: int = 1920) -> Path:
    """Cut, reorder and zoom an already-rendered, caption-less clip per the
    plan, then burn in captions timed to the edited timeline."""
    audio = _has_audio(base)
    fps = _frame_rate(base)
    parts, labels = [], []
    for i, piece in enumerate(plan.pieces):
        v = f"[0:v]trim=start={piece.start:.3f}:end={piece.end:.3f},setpts=PTS-STARTPTS"
        if piece.zoom > 1.0:
            zw, zh = int(round(out_w * piece.zoom / 2)) * 2, int(round(out_h * piece.zoom / 2)) * 2
            v += f",scale={zw}:{zh},crop={out_w}:{out_h}"
        # concat needs every piece's pixel aspect identical; a layout with a
        # not-quite-square SAR comes out of the zoom's scale rounded
        # differently, which concat rejects.
        v += ",setsar=1"
        parts.append(f"{v}[v{i}];")
        labels.append(f"[v{i}]")
        if audio:
            fade = min(0.012, piece.length / 4)
            parts.append(
                f"[0:a]atrim=start={piece.start:.3f}:end={piece.end:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, piece.length - fade):.3f}:d={fade:.3f}[a{i}];"
            )
            labels.append(f"[a{i}]")
    n = len(plan.pieces)
    parts.append(f"{''.join(labels)}concat=n={n}:v=1:a={1 if audio else 0}[cv]{'[ca]' if audio else ''};")
    parts.append(f"[cv]fps={fps},ass='{_escape_for_filter(ass_path)}'[outv]")
    cmd = [
        "ffmpeg", "-y", "-i", str(base),
        "-filter_complex", "".join(parts),
        "-map", "[outv]", *(["-map", "[ca]"] if audio else []),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", fps,
        *(["-c:a", "aac", "-b:a", "160k"] if audio else []),
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s editing {output_path.name}") from e
    if result.returncode != 0:
        # The actual error is usually early in the output, before ffmpeg's
        # progress lines -- keep both ends.
        err = result.stderr
        detail = err if len(err) <= 3000 else f"{err[:1500]}\n...\n{err[-1500:]}"
        raise RuntimeError(f"ffmpeg failed editing {output_path.name}:\n{detail}")
    return output_path


def render_edited(
    source_video: Path,
    start: float,
    end: float,
    layout: Layout,
    plan,
    ass_path: Path,
    output_path: Path,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    """Render a clip with its pacing edits (edit_plan.EditPlan): the layout
    rendered clean first, then apply_edits joins its kept pieces in plan
    order -- zoomed where the plan says -- and burns in the captions, already
    timed to the edited timeline, so they're never zoomed or cut mid-word.

    A trivial plan (nothing cut, zoomed or teased) is a single render_clip
    pass, exactly as before edits existed."""
    if plan is None or plan.is_trivial(end - start):
        return render_clip(source_video, start, end, layout, ass_path, output_path, out_w, out_h)

    base = output_path.with_name(f".{output_path.stem}.base{output_path.suffix}")
    try:
        # Near-lossless: this is an intermediate that gets encoded once more.
        render_clip(source_video, start, end, layout, None, base, out_w, out_h, crf=14)
        return apply_edits(base, plan, ass_path, output_path, out_w, out_h)
    finally:
        base.unlink(missing_ok=True)
