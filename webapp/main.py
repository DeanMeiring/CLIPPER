"""Web wrapper around the clipper CLI pipeline: submit a video, poll status,
download the rendered clips. One job runs at a time on a background worker
thread so a small Railway instance doesn't try to transcode multiple videos
at once.
"""
from __future__ import annotations

import datetime
import json
import os
import queue
import secrets
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from clipper.captions import build_ass
from clipper.download import _ffprobe_duration, download_video, is_url, probe_video
from clipper.long_vod import gather_candidates, is_long_vod, quick_probe_accessible, select_and_map
from clipper.loud_moments import find_loud_moments
from clipper.reframe import (
    MAX_COCAM_TILES,
    MultiCamSplitLayout,
    SplitLayout,
    center_crop_layout,
    compute_layout,
    layout_from_manual_boxes,
)
from clipper.render import render_clip, trim_clip
from clipper import facecam_vision
from clipper.select_moments import select_clips
from clipper.transcribe import Word, get_transcript
from clipper.trending import (
    get_trending_sections,
    search_creator,
    get_recommendation_candidates,
    get_top_twitch_clips,
    parse_twitch_duration,
)
from clipper.notify import send_telegram
from clipper.channel_insights import get_channel_snapshot, MAX_SHORT_SECONDS
from clipper import channel_strategy
from clipper.channel_strategy import get_ai_overview
from clipper import competitor_discovery
from clipper import competitor_content
from clipper import youtube_analytics
from clipper import youtube_oauth
from clipper import youtube_upload
from clipper import weekly_recap

BASE_DIR = Path(os.environ.get("CLIPPER_JOBS_DIR", "/tmp/clipper_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

JOB_META_NAME = "job.json"
TERMINAL_STATES = ("done", "error", "cancelled")

# Persists on the same volume job data lives on, so the connected YouTube
# account survives restarts/redeploys -- see clipper/youtube_oauth.py.
_youtube_token_store = youtube_oauth.TokenStore(BASE_DIR / "_youtube_oauth_token.json")
_channel_strategy_path = BASE_DIR / "_channel_strategy_history.json"
_competitor_channels_path = BASE_DIR / "_competitor_channels.json"
_recap_scheduler_state_path = BASE_DIR / "_recap_scheduler_state.json"


def _load_strategy_notes() -> Optional[str]:
    """The most recently saved AI channel-analysis overview, if any, for
    clip selection to use as guidance. None (not an error) if nothing's
    been saved yet -- callers should treat this exactly like the other
    optional signals (focus, loud moments): a hint when present, no
    behavior change when absent.

    Prefixed with how old the analysis is, because it otherwise steers
    every pick at full weight forever: notes written against a channel's
    numbers from two months ago read identically to ones written this
    morning, and only one of those deserves to override what the
    transcript itself says."""
    entry = channel_strategy.load_latest_overview(_channel_strategy_path)
    if not entry:
        return None
    age_days = max(0.0, (time.time() - entry["timestamp"]) / 86400)
    if age_days < 1:
        age = "generated today"
    elif age_days < 2:
        age = "generated yesterday"
    else:
        age = f"generated {int(age_days)} days ago"
    staleness = (
        " -- recent, weight it fully."
        if age_days <= 14
        else " -- this is old enough that the channel may have moved on; treat it"
             " as weaker evidence than what the transcript itself shows."
    )
    return f"(Channel analysis {age}{staleness})\n{entry['overview']}"
# CSRF state for the OAuth login flow: state -> issued_at. Short-lived and
# in-memory is fine -- a login round-trip through Google takes seconds, not
# something that needs to survive a restart.
_youtube_oauth_states: dict[str, float] = {}


class JobCancelled(Exception):
    """Raised (via a should_cancel callback) to unwind _run_job when the
    user hits the emergency stop button. Checked between pipeline steps,
    not inside them -- a step already in flight (an ffmpeg render, a
    download) still runs to its natural end."""

security = HTTPBasic(auto_error=False)


def require_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:
    """Gate every protected route behind APP_PASSWORD (any username works --
    only the password is checked). If APP_PASSWORD isn't set, auth is skipped
    entirely, so local dev works with no setup."""
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return
    if credentials is None or not secrets.compare_digest(credentials.password, password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Basic"},
        )


app = FastAPI(title="clipper")
protected = APIRouter(dependencies=[Depends(require_auth)])

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()
cancel_events: dict[str, threading.Event] = {}


class JobRequest(BaseModel):
    source: str
    focus: Optional[str] = None
    num_clips: int = 5
    min_len: float = 20.0
    max_len: float = 90.0
    # YouTube's auto-caption timestamps lag the actual audio noticeably --
    # Whisper aligns word timing to the audio itself, so default to it for
    # captions that don't look delayed. Slower, but accurate.
    whisper: bool = True


class RegenerateRequest(BaseModel):
    focus: Optional[str] = None
    num_clips: int = 3
    min_len: float = 20.0
    max_len: float = 90.0
    reset_used: bool = False


class FacecamBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FacecamBoxesRequest(BaseModel):
    boxes: List[FacecamBox]
    # Also re-render every other clip in the job that has no automatic
    # facecam with these same boxes. A collab stream's layout is fixed for
    # its whole length, so one placement normally fits every clip from it
    # -- without this, each rejected clip needs its own round of draw,
    # submit, wait.
    apply_to_all_missing: bool = False


def _job_public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("request", "pending_regenerate", "pending_manual_facecam")}


def _persist(job_id: str) -> None:
    """Write job state to disk alongside its clips, so a job's status and
    finished clips survive a container restart (Railway app-sleep, a
    redeploy, a crash) -- previously everything lived only in the `jobs`
    dict in RAM and was lost the moment the process restarted."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        data = _job_public(job)
    out_dir = BASE_DIR / job_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # _set() is called many times per job from the worker thread, and
        # can race with a request-handler thread persisting the same job
        # (cancel, regenerate). Writing straight to job.json (open truncates
        # then writes) lets two concurrent writers interleave into a
        # corrupt/partial file. Write to a per-writer temp file and rename
        # it into place instead -- an OS-level atomic op on POSIX -- so
        # concurrent writers only ever race on which write "wins" cleanly,
        # never on producing a half-written file.
        final_path = out_dir / JOB_META_NAME
        tmp_path = out_dir / f".{JOB_META_NAME}.tmp-{os.getpid()}-{threading.get_ident()}"
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        tmp_path.replace(final_path)
    except OSError:
        pass  # best-effort -- a disk hiccup here shouldn't take down the job


def _load_persisted_jobs() -> None:
    """Repopulate `jobs` from disk on startup. A job that wasn't in a
    terminal state when the process died can't be resumed (the pipeline
    has no checkpointing mid-stage), so it's marked as interrupted rather
    than left to hang forever in the UI."""
    if not BASE_DIR.exists():
        return
    for job_dir in BASE_DIR.iterdir():
        meta_path = job_dir / JOB_META_NAME
        if not job_dir.is_dir() or not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        job_id = data.get("id") or job_dir.name
        if data.get("state") not in TERMINAL_STATES:
            data["state"] = "error"
            data["error"] = "Interrupted -- the server restarted before this job finished. Please re-submit."
            data["message"] = "Interrupted by restart."
        jobs[job_id] = data
        try:
            meta_path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass


def _set(job_id: str, **kwargs) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            job.update(kwargs)
    _persist(job_id)


def _progress(job_id: str, value: float) -> None:
    _set(job_id, progress=max(0.0, min(1.0, value)))


def _check_cancel(job_id: str) -> None:
    event = cancel_events.get(job_id)
    if event is not None and event.is_set():
        raise JobCancelled()


# Rough, heuristic wall-clock estimates -- not measured per-job, just enough
# to give the user a "this'll take about N minutes" ballpark and a progress
# bar that moves at a believable rate. Downloading/transcoding speed varies
# a lot with source and instance load, so these are deliberately generous.
def _estimate_normal_seconds(duration: float, whisper: bool, num_clips: int) -> float:
    download = max(15.0, duration * 0.05)
    transcribe = duration * (0.35 if whisper else 0.05)
    select = 15.0
    render = num_clips * 25.0
    return download + transcribe + select + render


def _estimate_long_vod_seconds(num_candidates: int, num_clips: int) -> float:
    scan = 60.0
    # Measured from real Railway logs (job 7d6a980653fa): candidates 1-3 took
    # 128s, 147s, and 124s end-to-end (download + transcribe), bottlenecked
    # on Railway's outbound bandwidth (~1MiB/s on a ~110MiB window), not on
    # whisper. The original 25s/candidate guess was off by more than 5x.
    candidates = num_candidates * 135.0
    select = 20.0
    render = num_clips * 25.0
    return scan + candidates + select + render


def _estimate_recap_seconds(num_candidates: int) -> float:
    # Measured from real Railway logs on a live recap build: each Twitch-
    # clip candidate (download + Whisper transcription + facecam vision +
    # render) took roughly 20-35s end to end -- much less than a long-VOD
    # candidate window since a Twitch clip is short (usually well under
    # 60s) to begin with. 35s/candidate is the generous end of that.
    lookup = 10.0
    candidates = num_candidates * 35.0
    concat = 15.0
    return lookup + candidates + concat


def _run_job(job_id: str) -> None:
    with jobs_lock:
        pending_regenerate = jobs[job_id].pop("pending_regenerate", None)
        pending_manual_facecam = jobs[job_id].pop("pending_manual_facecam", None)
        pending_weekly_recap = jobs[job_id].pop("pending_weekly_recap", None)
    if pending_regenerate is not None:
        _run_regenerate(job_id, pending_regenerate)
        return
    if pending_manual_facecam is not None:
        _run_manual_facecam_render(job_id, pending_manual_facecam)
        return
    if pending_weekly_recap is not None:
        _run_weekly_recap_job(job_id)
        return

    req: JobRequest = jobs[job_id]["request"]
    # YouTube's Shorts feed itself allows up to 3 minutes, but a video
    # uploaded through the Data API only gets reliably auto-classified as a
    # Short up to MAX_SHORT_SECONDS (60s) -- past that it can silently land
    # as a regular video no matter what tag or aspect ratio it has. A clip
    # this app renders past that line can never actually become a Short via
    # the upload button, so clamp here rather than let a job silently
    # produce something that was never eligible.
    req.max_len = min(req.max_len, MAX_SHORT_SECONDS)
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_source"
    cancel = lambda: _check_cancel(job_id)  # noqa: E731

    _set(job_id, state="checking", message=f"Checking source: {req.source}")
    _progress(job_id, 0.02)
    cancel()
    info = None
    if is_url(req.source):
        try:
            info = probe_video(req.source)
        except Exception:
            info = None  # fall through to the normal download pipeline

    # Both pipelines below normalize to: source_title, and a list of
    # (video_path, words, pick) tuples ready for the shared render loop.
    if info and is_long_vod(info):
        source_title, duration = info.title, info.duration
        est_seconds = _estimate_long_vod_seconds(20, req.num_clips)
        _set(job_id, source_title=source_title, duration=duration,
             estimate_minutes=round(est_seconds / 60, 1),
             message=f"Long Twitch VOD ({duration / 3600:.1f}h) -- scanning chat highlights instead of downloading the whole stream")

        cancel()
        candidates = gather_candidates(
            req.source, info, raw_dir,
            on_progress=lambda msg: _set(job_id, state="scanning", message=msg),
            on_candidate_progress=lambda i, total: _progress(job_id, 0.05 + 0.45 * (i / max(total, 1))),
            should_cancel=cancel,
        )
        if not candidates:
            _set(job_id, state="error", error="No candidate highlight moments found in the chat replay.")
            return

        _cache_candidates(raw_dir, candidates)

        # Now that the real candidate count is known, refine the estimate.
        est_seconds = _estimate_long_vod_seconds(len(candidates), req.num_clips)
        _set(job_id, estimate_minutes=round(est_seconds / 60, 1))

        _progress(job_id, 0.5)
        cancel()
        _set(job_id, state="selecting", message=f"Asking Claude to pick up to {req.num_clips} moments from {len(candidates)} candidates...")
        mapped = select_and_map(
            candidates, n_clips=req.num_clips, min_len=req.min_len, max_len=req.max_len,
            focus=req.focus, api_key=None, source_title=source_title,
            strategy_notes=_load_strategy_notes(),
        )
        cand_words = {c["index"]: c["words"] for c in candidates}
        render_items = [(video_path, cand_words[pick.window_index], pick) for video_path, pick in mapped]
        render_base = 0.55
        _set(job_id, pipeline="long_vod", used_window_indices=[pick.window_index for _, pick in mapped])
    else:
        cancel()
        if info:
            est_seconds = _estimate_normal_seconds(info.duration, req.whisper, req.num_clips)
            _set(job_id, estimate_minutes=round(est_seconds / 60, 1))
        _set(job_id, state="downloading", message=f"Fetching source: {req.source}")
        _progress(job_id, 0.08)
        dl = download_video(req.source, raw_dir)
        source_title = dl.title
        est_seconds = _estimate_normal_seconds(dl.duration, req.whisper, req.num_clips)
        _set(job_id, source_title=dl.title, duration=dl.duration, estimate_minutes=round(est_seconds / 60, 1))

        cancel()
        _set(job_id, state="transcribing", message="Getting transcript...")
        _progress(job_id, 0.35)
        words = get_transcript(dl.video_path, dl.captions_path, prefer_whisper=req.whisper)
        if not words:
            _set(job_id, state="error", error="No speech/captions found -- nothing to clip.")
            return
        _cache_transcript(raw_dir, dl.video_path, dl.duration, words)

        cancel()
        _set(job_id, state="selecting", message="Scanning audio for loud/high-energy moments...")
        loud_moments = find_loud_moments(dl.video_path, dl.duration)

        cancel()
        _set(job_id, state="selecting", message=f"Asking Claude to pick up to {req.num_clips} moments...")
        _progress(job_id, 0.55)
        picks = select_clips(
            words, dl.duration,
            n_clips=req.num_clips, min_len=req.min_len, max_len=req.max_len,
            focus=req.focus, source_title=dl.title, loud_moments=loud_moments,
            strategy_notes=_load_strategy_notes(),
        )
        render_items = [(dl.video_path, words, pick) for pick in picks]
        render_base = 0.6
        _set(job_id, pipeline="short", used_ranges=[[pick.start, pick.end] for pick in picks])

    if not render_items:
        _set(job_id, state="error", error="Model returned no usable picks.")
        return

    clips_meta = _render_all(job_id, out_dir, render_items, render_base, [])
    _progress(job_id, 1.0)
    _set(job_id, state="done", message=f"Done. {len(clips_meta)} clip(s).")


def _render_atomic(
    video_path: Path, start: float, end: float, layout, ass_path: Path, out_path: Path,
    out_w: int = 1080, out_h: int = 1920,
) -> None:
    """Render to a temp file alongside the target, then move it into place
    in one step.

    ffmpeg writes its output progressively, and the clips endpoint serves
    straight out of this same directory while the job is still running --
    so rendering directly to the final path publishes a half-written mp4
    for as long as the encode takes. Anyone who opens the clip in that
    window gets a truncated file, which decodes into garbage rather than
    failing cleanly. The post-render fallback made it worse by rewriting
    an already-published clip in place, so a clip that was fine a moment
    ago would break under a reader mid-re-render. os.replace is atomic on
    POSIX, so a reader now sees either the previous complete file or the
    new one, never a partial."""
    tmp_path = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    try:
        render_clip(video_path, start, end, layout, ass_path, tmp_path, out_w=out_w, out_h=out_h)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _save_still(video_path: Path, out_path: Path) -> Optional[Path]:
    """Write one frame from partway through a clip as a JPEG beside it.

    A rejected facecam render is only useful if someone can actually look
    at it, and a 25MB mp4 is awkward to get off the server and past an
    upload limit. A still is a couple of hundred KB, opens straight in a
    browser, and shows the facecam band just as well as the video does."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if frames > 0 and fps > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, (frames / fps) * 1000 * 0.5)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return out_path if cv2.imwrite(str(out_path), frame) else None
    except Exception as e:  # noqa: BLE001 - a missing still must never fail a render
        print(f"[render] could not write a still for {out_path.name}: {e}", flush=True)
        return None


def _save_source_still(video_path: Path, start: float, end: float, out_path: Path) -> Optional[Path]:
    """Write one frame from partway through the clip's window in the SOURCE
    video (before any crop/tile) as a JPEG.

    _save_still's frame comes from the rendered/rejected clip, which only
    shows the crop that was already judged wrong -- no help for finding
    where the facecam(s) actually are. This one is the raw source frame a
    manual facecam fix gets drawn on top of."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + (end - start) / 2) * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return out_path if cv2.imwrite(str(out_path), frame) else None
    except Exception as e:  # noqa: BLE001 - a missing still must never fail a render
        print(f"[render] could not write a source still for {out_path.name}: {e}", flush=True)
        return None


def _remove_clip_files(out_dir: Path, filename: str) -> None:
    """Delete a clip and everything rendered alongside it: its caption
    file, any kept rejected-facecam render/still, and the source frame
    saved for manual facecam placement. A later regenerate reuses the
    same clip_NN numbering, so leaving these behind would let a stale
    still from a deleted clip sit beside its unrelated replacement."""
    stem = Path(filename).stem
    for name in (
        filename,
        f"_{stem}.ass",
        f"{stem}_rejected_facecam.mp4",
        f"{stem}_rejected_facecam.jpg",
        f"{stem}_source_frame.jpg",
    ):
        (out_dir / name).unlink(missing_ok=True)


def _render_all(job_id: str, out_dir: Path, render_items: list, render_base: float, clips_meta: list) -> list:
    """Render each (video_path, words, pick) item to clip_{n}.mp4, appending
    to clips_meta (already containing any earlier clips) and updating job
    progress/state as it goes. Shared by a fresh run and a regenerate run --
    they only differ in what render_items contains and whether clips_meta
    starts empty or with clips from a prior run."""
    render_span = 1.0 - render_base
    start_index = len(clips_meta)
    cancel = lambda: _check_cancel(job_id)  # noqa: E731
    for i, (video_path, words, pick) in enumerate(render_items, start=1):
        cancel()
        out_index = start_index + i
        _set(job_id, state="rendering",
             message=f'Rendering clip {out_index}/{start_index + len(render_items)}: "{pick.title}"')
        _progress(job_id, render_base + render_span * ((i - 1) / len(render_items)))
        clip_words = [w for w in words if w.start >= pick.start and w.end <= pick.end]
        layout = compute_layout(video_path, pick.start, pick.end, target_w=1080, target_h=1920)
        out_path = out_dir / f"clip_{out_index:02d}.mp4"
        ass_path = out_dir / f"_clip_{out_index:02d}.ass"
        build_ass(clip_words, pick.start, ass_path)
        _render_atomic(video_path, pick.start, pick.end, layout, ass_path, out_path)

        facecam_uncertain = False
        source_frame_name = None
        has_trusted_facecam = False
        if isinstance(layout, (SplitLayout, MultiCamSplitLayout)):
            has_trusted_facecam = True
            # This briefly ran on single-cam only, on the theory that a check
            # rejecting 100% of multi-cam renders couldn't be measuring
            # anything real. Turning it off proved the opposite: the renders
            # that then shipped had two tiles of gameplay and overlay border
            # around the people. The check has been right every time.
            # False (not None -- that means the check itself wasn't usable)
            # means vision confidently saw something wrong with the rendered
            # facecam band; re-render as a plain crop rather than ship a clip
            # with a broken-looking facecam.
            verified = facecam_vision.verify_rendered_facecam(out_path)
            if verified is False:
                print(f"[render] clip {out_index} failed post-render facecam check -- re-rendering as a plain crop", flush=True)
                # Keep what was rejected. Overwriting it in place meant a
                # rejected facecam render could never be looked at, so every
                # round of "still no facecam" came down to guessing at pixels
                # from box coordinates in a log. This costs one file per
                # rejection and makes the rejected version openable at
                # /api/jobs/{id}/clips/<name>, which settles in seconds what
                # otherwise takes a deploy and a regenerate to find out.
                try:
                    rejected_path = out_path.with_name(f"{out_path.stem}_rejected_facecam{out_path.suffix}")
                    shutil.copy2(out_path, rejected_path)
                    still = _save_still(rejected_path, rejected_path.with_suffix(".jpg"))
                    print(
                        f"[render] kept the rejected facecam render as {rejected_path.name}"
                        + (f" (still: {still.name})" if still else ""), flush=True,
                    )
                except OSError as e:
                    print(f"[render] could not keep the rejected render: {e}", flush=True)
                try:
                    fallback_layout = center_crop_layout(video_path, target_w=1080, target_h=1920)
                    _render_atomic(video_path, pick.start, pick.end, fallback_layout, ass_path, out_path)
                except Exception as e:
                    print(f"[render] fallback re-render also failed, keeping the original render: {e}", flush=True)
                facecam_uncertain = True
                has_trusted_facecam = False

        # Always keep a raw source frame so a person can place the facecam
        # by hand, even when the pipeline trusts its own placement -- the
        # post-render check catches an obviously broken facecam band, but
        # it isn't proof the placement is actually right (e.g. it can
        # confidently approve a frame that grabbed an on-screen overlay
        # graphic instead of an actual face). Without this, a clip the
        # check happened to approve had no way to fix a wrong placement
        # short of "generate more clips" and hoping for something different.
        source_frame_path = out_dir / f"clip_{out_index:02d}_source_frame.jpg"
        if _save_source_still(video_path, pick.start, pick.end, source_frame_path):
            source_frame_name = source_frame_path.name
            print(f"[render] saved the source frame for manual facecam placement as {source_frame_name}", flush=True)

        clips_meta.append({
            "file": out_path.name,
            "start": pick.start,
            "end": pick.end,
            "duration": round(pick.end - pick.start, 2),
            "title": pick.title,
            "hook_caption": pick.hook_caption,
            "upload_title": pick.upload_title,
            "description": pick.description,
            "reason": pick.reason,
            # Only WindowPick (long-VOD pipeline) has this -- carried along
            # so a later per-clip delete can free this candidate window
            # back up for "generate more" to reconsider, instead of
            # deleting the clip while permanently blocking whatever moment
            # it came from.
            "window_index": getattr(pick, "window_index", None),
            # The clip's own downloaded source file, so a later manual
            # facecam fix can re-open it without re-downloading anything.
            "source_video": video_path.name,
            # True when the post-render check rejected the automatic
            # facecam placement and this clip shipped as a plain crop
            # instead -- there IS a facecam here, detection just put it in
            # the wrong place, so the frontend prompts for a manual
            # placement as soon as the job finishes.
            "facecam_uncertain": facecam_uncertain,
            # True when the pipeline auto-placed and trusted a facecam here
            # (passed the post-render check, or the check wasn't usable) --
            # distinct from facecam_manual, so the frontend can label the
            # button "Adjust" (something's there, maybe wrong) rather than
            # "Add" (nothing's there) for a clip nobody has touched yet.
            "facecam_trusted": has_trusted_facecam,
            # The frame the manual box-picker draws on. Always saved now
            # (see above) so any clip's facecam can be overridden by hand,
            # not just ones the pipeline itself flagged as uncertain.
            "source_frame": source_frame_name,
        })
        _set(job_id, clips=list(clips_meta))
        _progress(job_id, render_base + render_span * (i / len(render_items)))
    return clips_meta


def _cache_candidates(raw_dir: Path, candidates: list) -> None:
    """Persist gathered long-VOD candidate windows (each already a small
    downloaded file) alongside their transcripts, so a later "generate more
    clips" pass can re-run selection over them without downloading or
    transcribing anything again."""
    try:
        data = [
            {
                "index": c["index"],
                "video_path": Path(c["video_path"]).name,
                "duration": c["duration"],
                "words": [asdict(w) for w in c["words"]],
                "signal": c["signal"],
            }
            for c in candidates
        ]
        (raw_dir / "candidates.json").write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # best-effort -- worst case a future regenerate falls back to re-transcribing


def _cache_transcript(raw_dir: Path, video_path: Path, duration: float, words: list) -> None:
    """Persist the full-video transcript so a later "generate more clips"
    pass can re-run selection over the same words without re-downloading or
    re-transcribing the source."""
    try:
        data = {
            "video_path": video_path.name,
            "duration": duration,
            "words": [asdict(w) for w in words],
        }
        (raw_dir / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def _load_candidates_for_regenerate(raw_dir: Path) -> Optional[list]:
    cache_path = raw_dir / "candidates.json"
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return [
                {
                    "index": c["index"],
                    "video_path": raw_dir / c["video_path"],
                    "duration": c["duration"],
                    "words": [Word(**w) for w in c["words"]],
                    "signal": c["signal"],
                }
                for c in raw
            ]
        except (OSError, ValueError, KeyError):
            pass
    # A job from before this cache existed: the candidate video files are
    # still on disk (only a fresh submit downloads or deletes them), so
    # reconstruct by re-transcribing each one -- skips the network download
    # (the slow, rate-limit-prone part) even though transcription reruns.
    cand_files = sorted(raw_dir.glob("cand_*.mp4"))
    if not cand_files:
        return None
    result = []
    for i, path in enumerate(cand_files):
        try:
            words = get_transcript(path, None, prefer_whisper=True)
            duration = _ffprobe_duration(path)
        except Exception:
            continue
        result.append({
            "index": i, "video_path": path, "duration": duration,
            "words": words, "signal": "previously downloaded candidate",
        })
    return result or None


def _load_transcript_for_regenerate(raw_dir: Path) -> Optional[dict]:
    cache_path = raw_dir / "transcript.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return {
                "video_path": raw_dir / data["video_path"],
                "duration": data["duration"],
                "words": [Word(**w) for w in data["words"]],
            }
        except (OSError, ValueError, KeyError):
            pass
    # A job from before this cache existed: fall back to the still-
    # downloaded video file itself (if unambiguous) and re-transcribe --
    # skips the download, even though transcription has to rerun.
    videos = [p for p in raw_dir.glob("*.mp4")]
    if len(videos) != 1:
        return None
    video_path = videos[0]
    try:
        words = get_transcript(video_path, None, prefer_whisper=True)
        duration = _ffprobe_duration(video_path)
    except Exception:
        return None
    return {"video_path": video_path, "duration": duration, "words": words}


def _run_regenerate(job_id: str, req: dict) -> None:
    """Pick and render additional clips reusing a job's already-downloaded
    source (or already-downloaded candidate windows, for a long VOD)
    instead of re-downloading anything."""
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_source"
    cancel = lambda: _check_cancel(job_id)  # noqa: E731

    num_clips = max(1, int(req.get("num_clips") or 3))
    focus = req.get("focus") or None
    min_len = float(req.get("min_len") or 20.0)
    max_len = min(float(req.get("max_len") or 90.0), MAX_SHORT_SECONDS)
    reset_used = bool(req.get("reset_used"))

    with jobs_lock:
        job = jobs[job_id]
        pipeline = job.get("pipeline")
        source_title = job.get("source_title") or ""
        source_duration = job.get("duration") or 0.0
        existing_clips = list(job.get("clips") or [])
        # reset_used ("start fresh") deliberately ignores which windows/
        # ranges earlier clips came from when SELECTING this batch, so it
        # can freely re-pick from the whole candidate pool -- the tradeoff
        # a tester explicitly asked for over the normal anti-duplicate
        # behavior, useful once a small candidate pool (long-VOD
        # chat-highlight windows especially) is mostly exhausted from
        # repeated regenerates. It only relaxes *selection*, though: the
        # original set is kept (original_used_*) and still merged into
        # what gets persisted below, so a still-kept older clip's window
        # isn't forgotten for the *next* (non-reset) regenerate just
        # because this one ignored it.
        original_used_ranges = [tuple(r) for r in (job.get("used_ranges") or [])]
        original_used_window_indices = set(job.get("used_window_indices") or [])
        used_ranges = [] if reset_used else list(original_used_ranges)
        used_window_indices = set() if reset_used else set(original_used_window_indices)
    if not pipeline:
        pipeline = "long_vod" if list(raw_dir.glob("cand_*.mp4")) else "short"

    # A regenerate reuses already-downloaded material, so it's much faster
    # than the original run -- recompute a realistic estimate instead of
    # leaving the old (much larger, download-inclusive) one on screen, and
    # restart the elapsed-time clock the UI measures the ETA against.
    if pipeline == "long_vod":
        has_cache = (raw_dir / "candidates.json").exists()
        n_uncached = 0 if has_cache else len(list(raw_dir.glob("cand_*.mp4")))
        est_seconds = 20.0 + num_clips * 25.0 + n_uncached * 40.0
    else:
        has_cache = (raw_dir / "transcript.json").exists()
        est_seconds = 20.0 + num_clips * 25.0 + (0.0 if has_cache else source_duration * 0.35)
    _set(job_id, created_at=time.time(), estimate_minutes=round(est_seconds / 60, 1))

    _set(job_id, state="selecting", message=f"Asking Claude to pick {num_clips} more moment(s)...")
    _progress(job_id, 0.1)
    cancel()

    if pipeline == "long_vod":
        candidates = _load_candidates_for_regenerate(raw_dir)
        if not candidates:
            _set(job_id, state="error", error="The downloaded candidate clips are gone -- resubmit the source URL instead.")
            return
        candidates = [c for c in candidates if c["index"] not in used_window_indices]
        if not candidates:
            _set(job_id, state="error", error="Every downloaded candidate moment has already been used in a clip.")
            return
        cancel()
        mapped = select_and_map(
            candidates, n_clips=num_clips, min_len=min_len, max_len=max_len,
            focus=focus, api_key=None, source_title=source_title,
            strategy_notes=_load_strategy_notes(),
        )
        cand_words = {c["index"]: c["words"] for c in candidates}
        render_items = [(video_path, cand_words[pick.window_index], pick) for video_path, pick in mapped]
        persisted_used_window_indices = original_used_window_indices | {pick.window_index for _, pick in mapped}
        _set(job_id, used_window_indices=list(persisted_used_window_indices))
    else:
        cached = _load_transcript_for_regenerate(raw_dir)
        if not cached:
            _set(job_id, state="error", error="The downloaded source video is gone -- resubmit the source URL instead.")
            return
        cancel()
        loud_moments = find_loud_moments(cached["video_path"], cached["duration"])
        cancel()
        picks = select_clips(
            cached["words"], cached["duration"],
            n_clips=num_clips + len(used_ranges), min_len=min_len, max_len=max_len,
            focus=focus, source_title=source_title, loud_moments=loud_moments,
            strategy_notes=_load_strategy_notes(),
        )

        def _overlaps_used(p) -> bool:
            return any(not (p.end <= u[0] or p.start >= u[1]) for u in used_ranges)

        picks = [p for p in picks if not _overlaps_used(p)][:num_clips]
        render_items = [(cached["video_path"], cached["words"], pick) for pick in picks]
        persisted_used_ranges = original_used_ranges + [(pick.start, pick.end) for pick in picks]
        _set(job_id, used_ranges=[list(r) for r in persisted_used_ranges])

    if not render_items:
        _set(job_id, state="error", error="Claude didn't return any new, non-overlapping moments this time -- try a different focus.")
        return

    clips_meta = _render_all(job_id, out_dir, render_items, 0.15, existing_clips)
    _progress(job_id, 1.0)
    _set(job_id, state="done", message=f"Done. {len(clips_meta)} clip(s) total.")


def _run_manual_facecam_render(job_id: str, req: dict) -> None:
    """Re-render one or more clips using facecam box(es) a human drew on a
    source frame, bypassing detection and the post-render check entirely
    -- the escape hatch for clips that shipped without a trusted facecam
    (see _render_all). Reuses each clip's already-downloaded source file
    and existing caption (.ass) file; only the layout and the rendered
    video change. One clip failing (source gone, ffmpeg error) is logged
    and skipped rather than losing the rest of the batch."""
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_source"
    cancel = lambda: _check_cancel(job_id)  # noqa: E731
    filenames = list(req.get("filenames") or [])
    boxes = [tuple(b) for b in req["boxes"]]

    with jobs_lock:
        by_file = {c.get("file"): c for c in (jobs[job_id].get("clips") or [])}

    updated, failed = [], []
    for i, filename in enumerate(filenames):
        cancel()
        clip = by_file.get(filename)
        label = (clip or {}).get("title") or filename
        _set(job_id, state="rendering",
             message=f'Re-rendering {i + 1}/{len(filenames)} with your facecam placement: "{label}"')
        _progress(job_id, i / max(len(filenames), 1))
        try:
            if clip is None:
                raise RuntimeError("no longer part of this job's clips")
            if not clip.get("source_video"):
                raise RuntimeError("predates manual facecam fixes -- use Generate more clips instead")
            video_path = raw_dir / clip["source_video"]
            if not video_path.exists():
                raise RuntimeError("its downloaded source is gone")
            ass_path = out_dir / f"_{Path(filename).stem}.ass"
            if not ass_path.exists():
                raise RuntimeError("its caption file is missing")
            layout = layout_from_manual_boxes(video_path, boxes, target_w=1080, target_h=1920)
            _render_atomic(video_path, clip["start"], clip["end"], layout, ass_path, out_dir / filename)
        except Exception as e:  # noqa: BLE001 - one clip failing shouldn't lose the rest of the batch
            print(f"[render] manual facecam re-render of {filename} failed: {e}", flush=True)
            failed.append(f"{filename}: {e}")
            continue
        updated.append(filename)
        with jobs_lock:
            for c in jobs[job_id].get("clips") or []:
                if c.get("file") == filename:
                    c["facecam_uncertain"] = False
                    c["facecam_manual"] = True
                    c["facecam_boxes"] = [list(b) for b in boxes]
        _persist(job_id)

    _progress(job_id, 1.0)
    if not updated:
        _set(job_id, state="error", error="Couldn't re-render with your facecam placement -- " + "; ".join(failed))
        return
    message = f"Done -- {len(updated)} clip(s) updated with your facecam placement."
    if failed:
        message += " Skipped " + "; ".join(failed)
    _set(job_id, state="done", message=message)


def _keepalive_loop(stop_event: threading.Event) -> None:
    """Railway's sleepApplication only watches HTTP traffic to the service
    -- a background job running with nobody polling looks idle to it even
    while it's actively downloading/transcoding, so the container gets
    slept mid-job. Pinging our own public healthz endpoint counts as real
    activity (confirmed by Railway's own docs: "outbound polling ... will
    keep the service awake"), so this keeps the container up for exactly
    as long as a job is in flight -- no manual toggle, no redeploy (which
    a sleepApplication config change would require anyway, killing the
    very job it's meant to protect). Idle again within ~10 minutes of the
    last job finishing, same as if this never ran."""
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if not domain:
        return
    import requests

    url = f"https://{domain}/healthz"
    while not stop_event.wait(240):  # well under Railway's ~10min idle window
        try:
            requests.get(url, timeout=10)
        except Exception:
            pass


def _notify_job_finished(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return
    label = job.get("source_title") or job.get("source_url") or job_id
    state = job.get("state")
    if state == "done":
        clips = job.get("clips") or []
        text = f'✅ Clipper done: "{label}" -- {len(clips)} clip(s) ready.'
        needs_placement = sum(1 for c in clips if c.get("facecam_uncertain"))
        if needs_placement:
            text += f"\n🎯 {needs_placement} clip(s) need you to place the facecam -- open the site to draw it."
    elif state == "cancelled":
        text = f'⏹ Clipper stopped: "{label}".'
    else:
        text = f'❌ Clipper failed: "{label}" -- {job.get("error") or "unknown error"}'
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if domain:
        text += f"\n\nhttps://{domain}"
    send_telegram(text)


def _worker() -> None:
    while True:
        job_id = job_queue.get()
        stop_keepalive = threading.Event()
        threading.Thread(target=_keepalive_loop, args=(stop_keepalive,), daemon=True).start()
        try:
            _run_job(job_id)
        except JobCancelled:
            _set(job_id, state="cancelled", message="Cancelled by user.")
        except Exception as e:  # noqa: BLE001 - surface any pipeline failure to the client
            # The UI only ever shows str(e) -- previously that was also
            # the *only* place a failure's detail existed at all, since
            # nothing here reached the server logs. Print the full
            # traceback too, so a job-runner exception can be diagnosed
            # from Railway logs instead of only from a screenshot of the
            # (much shorter) UI error message.
            print(f"[worker] job {job_id} failed:", flush=True)
            traceback.print_exc()
            _set(job_id, state="error", error=str(e))
        finally:
            stop_keepalive.set()
            cancel_events.pop(job_id, None)
            _notify_job_finished(job_id)
            job_queue.task_done()


_load_persisted_jobs()
threading.Thread(target=_worker, daemon=True).start()


@protected.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if not req.source or not req.source.strip():
        # Without this, an empty source silently resolves to the
        # container's own working directory in download_video() and
        # fails much later, mid-job, with a confusing ffprobe error --
        # reject it immediately instead with a message that actually
        # explains what's wrong.
        raise HTTPException(400, "Enter a video URL or file path first.")
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "source_url": req.source,
            "created_at": time.time(),
            "state": "queued",
            "message": "Queued",
            "progress": 0.0,
            "estimate_minutes": None,
            "clips": [],
            "error": None,
            "saved": False,
            "request": req,
        }
        cancel_events[job_id] = threading.Event()
    _persist(job_id)
    job_queue.put(job_id)
    return {"job_id": job_id}


@protected.get("/api/jobs")
def list_jobs() -> dict:
    """All non-deleted jobs (running, queued, done, or stopped-and-saved)
    -- backs the "Active & saved jobs" panel so a job stays reachable even
    after navigating away, and so a "Save progress" stop has somewhere to
    show up."""
    with jobs_lock:
        items = [_job_public(j) for j in jobs.values()]
    items.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    return {"jobs": items}


@protected.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Remove a finished job's clips and metadata from the volume -- the
    "I've downloaded these" button in the UI. Refuses a job that's still
    running so a click doesn't yank files out from under an active render."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- stop it first")
        jobs.pop(job_id, None)
        cancel_events.pop(job_id, None)
    shutil.rmtree(BASE_DIR / job_id, ignore_errors=True)
    return {"ok": True}


@protected.post("/api/jobs/{job_id}/regenerate")
def regenerate_job(job_id: str, req: RegenerateRequest) -> dict:
    """Pick and render additional clips from a finished job's already-
    downloaded source (or already-downloaded candidate windows, for a long
    VOD) without re-downloading anything -- the raw files a normal run
    leaves on the volume until the job is deleted."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- wait for it to finish first")
        raw_dir = BASE_DIR / job_id / "_source"
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            raise HTTPException(409, "the downloaded source is gone -- resubmit the URL instead")
        job["pending_regenerate"] = req.dict()
        job["state"] = "queued"
        job["message"] = "Queued -- reusing already-downloaded source"
        job["progress"] = 0.0
        # Clear the previous run's (much larger, download-inclusive) ETA
        # immediately -- _run_regenerate computes the real one once it
        # starts, but the UI shouldn't show the stale figure even briefly.
        job["created_at"] = time.time()
        job["estimate_minutes"] = None
        # Clear any error left over from an earlier failed attempt on this
        # same job -- the frontend appends job.error to the status line
        # whenever it's set, with no regard for the current state, so a
        # stale error here would show up glued onto this run's status even
        # after it finishes cleanly.
        job["error"] = None
        cancel_events[job_id] = threading.Event()
    _persist(job_id)
    job_queue.put(job_id)
    return {"ok": True}


_SOURCE_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".ts"}


def _infer_source_video(out_dir: Path) -> Optional[str]:
    """A clip from before source_video was tracked per-clip has no
    filename recorded for its downloaded source. The normal (non-long-VOD)
    pipeline downloads exactly one video into _source/ per job, shared by
    every clip in it -- if exactly one video file is sitting there, it's
    unambiguous which one this clip came from. A long-VOD job's _source/
    instead holds one small file per candidate window, so this correctly
    declines (returns None) rather than guessing wrong for those."""
    raw_dir = out_dir / "_source"
    if not raw_dir.is_dir():
        return None
    candidates = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in _SOURCE_VIDEO_EXTS]
    return candidates[0].name if len(candidates) == 1 else None


@protected.post("/api/jobs/{job_id}/clips/{filename}/ensure-source-frame")
def ensure_source_frame(job_id: str, filename: str) -> dict:
    """Return the clip's source-frame filename for the facecam picker to
    draw on, generating it on the spot if it's missing -- a clip rendered
    before source frames were saved for every clip (not just uncertain
    ones) has no source_frame in its stored metadata, but its downloaded
    source video is usually still sitting right there, so there's no need
    to make "Adjust facecam position" a dead end for it."""
    if Path(filename).name != filename:
        raise HTTPException(400, "bad filename")
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        clips = job.get("clips") or []
        clip = next((c for c in clips if c.get("file") == filename), None)
        if clip is None:
            raise HTTPException(404, "clip not found")
        existing = clip.get("source_frame")

    out_dir = BASE_DIR / job_id
    if existing and (out_dir / existing).is_file():
        return {"source_frame": existing}

    source_video = clip.get("source_video") or _infer_source_video(out_dir)
    if not source_video:
        raise HTTPException(409, "this clip predates manual facecam fixes -- try Generate more clips instead")
    video_path = out_dir / "_source" / source_video
    if not video_path.is_file():
        raise HTTPException(409, "the downloaded source is gone -- resubmit the URL instead")
    if not clip.get("source_video"):
        # Backfill so set_facecam_boxes (the actual re-render, triggered
        # next by "Re-render with these boxes") doesn't have to repeat
        # this inference -- once known, it's known for good.
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None:
                for c in job.get("clips") or []:
                    if c.get("file") == filename:
                        c["source_video"] = source_video
        _persist(job_id)

    source_frame_path = out_dir / f"{Path(filename).stem}_source_frame.jpg"
    if not _save_source_still(video_path, clip.get("start", 0.0), clip.get("end", 0.0), source_frame_path):
        raise HTTPException(500, "could not read a frame from the source video")

    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            for c in job.get("clips") or []:
                if c.get("file") == filename:
                    c["source_frame"] = source_frame_path.name
    _persist(job_id)
    return {"source_frame": source_frame_path.name}


@protected.post("/api/jobs/{job_id}/clips/{filename}/facecam-boxes")
def set_facecam_boxes(job_id: str, filename: str, req: FacecamBoxesRequest) -> dict:
    """Re-render one clip using facecam box(es) a human drew on its source
    frame -- the fix for a clip the automatic post-render check rejected
    (facecam_uncertain=True), where detection placed the facecam window
    wrong and it shipped as a plain crop instead. Skips detection and
    re-verification entirely: a human who looked at the actual frame and
    drew the box is more reliable than either."""
    if Path(filename).name != filename:
        raise HTTPException(400, "bad filename")
    if not req.boxes:
        raise HTTPException(400, "at least one facecam box is required")
    if len(req.boxes) > MAX_COCAM_TILES:
        raise HTTPException(400, f"at most {MAX_COCAM_TILES} facecam boxes are supported")
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- wait for it to finish first")
        clip = next((c for c in (job.get("clips") or []) if c.get("file") == filename), None)
        if clip is None:
            raise HTTPException(404, "clip not found")
        if not clip.get("source_video"):
            raise HTTPException(409, "this clip predates manual facecam fixes -- try Generate more clips instead")
        raw_dir = BASE_DIR / job_id / "_source"
        if not (raw_dir / clip["source_video"]).exists():
            raise HTTPException(409, "the downloaded source is gone -- resubmit the URL instead")
        filenames = [filename]
        if req.apply_to_all_missing:
            # Every other clip with a downloaded source, no trusted
            # automatic facecam, and no manual placement of its own yet.
            # Doesn't require a saved source_frame -- the re-render itself
            # only needs source_video, a preview still is only for showing
            # this clip's own frame in the picker.
            filenames += [
                c["file"] for c in (job.get("clips") or [])
                if c.get("file") != filename and c.get("source_video")
                and not c.get("facecam_trusted") and not c.get("facecam_manual")
            ]
        job["pending_manual_facecam"] = {
            "filenames": filenames,
            "boxes": [[b.x, b.y, b.w, b.h] for b in req.boxes],
        }
        job["state"] = "queued"
        job["message"] = f"Queued -- re-rendering {len(filenames)} clip(s) with your facecam placement"
        job["progress"] = 0.0
        job["error"] = None
        cancel_events[job_id] = threading.Event()
    _persist(job_id)
    job_queue.put(job_id)
    return {"ok": True}


@protected.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return {k: v for k, v in job.items() if k != "request"}


@protected.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, save: bool = False) -> dict:
    """Stop a running job. `save=true` keeps its files and marks it
    "saved" (shows up in the Active & saved jobs list, deleted only when
    the user explicitly deletes it); `save=false` is a plain stop -- the
    caller is expected to DELETE it once it reaches "cancelled" if they
    don't want it kept."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        event = cancel_events.get(job_id)
        if event is None or job["state"] in TERMINAL_STATES:
            return {"ok": True, "state": job["state"]}
        event.set()
        job["saved"] = save
        job["message"] = "Stopping (progress will be kept)..." if save else "Stopping..."
    _persist(job_id)
    return {"ok": True, "state": "cancelling"}


@protected.get("/api/jobs/{job_id}/clips/{filename}")
def get_clip(job_id: str, filename: str, download: bool = False) -> FileResponse:
    # Reject any filename that isn't a plain name, so a crafted path can't
    # walk out of the job directory and serve an arbitrary file off disk.
    if Path(filename).name != filename or filename.startswith("."):
        raise HTTPException(400, "bad filename")
    path = BASE_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    # A rejected render is saved alongside its clip as a .jpg still, and
    # labelling that video/mp4 makes a browser download it instead of just
    # showing it -- which defeats the point of having a still at all.
    media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "video/mp4"
    # Starlette only sends a Content-Disposition header at all when
    # `filename` is passed, and it defaults that header to "attachment" --
    # which some browsers take as a sign to refuse playing a <video src>
    # pointed at it and force a save dialog instead. So plain playback (the
    # in-page preview) omits `filename` entirely; only the explicit
    # "Download" button asks for ?download=1 and gets the Save-As behavior.
    if download:
        return FileResponse(path, media_type=media_type, filename=filename)
    return FileResponse(path, media_type=media_type)


class YouTubeUploadRequest(BaseModel):
    # Defaults to Unlisted rather than Public -- a wrong first click (the
    # wrong clip, a typo'd title before ever seeing this modal) shouldn't
    # be able to go live on the channel by accident. Public is one
    # deliberate radio-button choice away, not the default.
    privacy_status: str = "unlisted"
    # Optional: shave a beat off either end before posting, without
    # re-rendering or touching the kept copy on disk -- captions are
    # burned into the pixels already, so trimming the finished file
    # carries them along for free.
    trim_start: float = 0.0
    trim_end: float = 0.0


@protected.post("/api/jobs/{job_id}/clips/{filename}/upload-youtube")
def upload_clip_to_youtube(job_id: str, filename: str, req: YouTubeUploadRequest) -> dict:
    """Post one already-rendered, already-hand-picked clip straight to the
    connected YouTube channel -- the manual "I've decided this one's going
    up" action, never a bulk or automatic publish. Uses the clip's
    already-generated upload_title/description as-is."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- wait for it to finish first")
        clip = next((c for c in (job.get("clips") or []) if c.get("file") == filename), None)
        if clip is None:
            raise HTTPException(404, "clip not found")

    access_token = _youtube_token_store.get_valid_access_token()
    if not access_token:
        raise HTTPException(409, "Connect your YouTube account on the analytics page first.")

    path = BASE_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "clip file not found on disk")

    trimmed_path = None
    if req.trim_start > 0 or req.trim_end > 0:
        trimmed_path = path.with_name(f".{path.stem}.trimmed{path.suffix}")
        try:
            trim_clip(path, trimmed_path, req.trim_start, req.trim_end, float(clip.get("duration") or 0))
        except (RuntimeError, ValueError) as e:
            trimmed_path.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not trim the clip: {e}") from e
        upload_path = trimmed_path
    else:
        upload_path = path

    is_recap = bool(clip.get("is_recap"))
    try:
        video_id = youtube_upload.upload_video(
            access_token, upload_path,
            title=clip.get("upload_title") or clip.get("title") or filename,
            description=clip.get("description") or "",
            privacy_status=req.privacy_status,
            is_short=not is_recap,
        )
    except (youtube_upload.UploadError, ValueError) as e:
        raise HTTPException(502, str(e)) from e
    finally:
        if trimmed_path is not None:
            trimmed_path.unlink(missing_ok=True)

    return {"ok": True, "video_id": video_id, "url": f"https://youtu.be/{video_id}"}


_RECAP_OUT_W = 1920
_RECAP_OUT_H = 1080


def _render_twitch_clip_for_recap(video_path: Path, duration: float, words: list, out_path: Path) -> None:
    """Render one already-downloaded Twitch clip to landscape (1920x1080)
    with burned-in captions, for the recap's normal-video upload -- a raw
    Twitch clip download has no captions of its own, so those still need
    adding, but NOT the facecam-aware crop/split logic _render_all uses
    for a vertical Short.

    That logic exists specifically to carve a narrow vertical frame out
    of a wide landscape broadcast without losing either the gameplay or
    the facecam -- there's nothing to carve out here, since the target
    IS landscape, the same shape the clip was actually broadcast in, so
    a plain centered crop-to-16:9 (a no-op whenever the source is already
    16:9, which a Twitch clip almost always is) already shows everything
    the streamer's own layout composited, facecam included."""
    layout = center_crop_layout(video_path, target_w=_RECAP_OUT_W, target_h=_RECAP_OUT_H)
    ass_path = out_path.with_suffix(".ass")
    build_ass(words, 0.0, ass_path, play_res=(_RECAP_OUT_W, _RECAP_OUT_H))
    _render_atomic(video_path, 0.0, duration, layout, ass_path, out_path, out_w=_RECAP_OUT_W, out_h=_RECAP_OUT_H)
    ass_path.unlink(missing_ok=True)


def _run_weekly_recap_job(job_id: str) -> None:
    """Build this week's cross-streamer recap on the shared worker thread
    -- same async, progress-reporting flow as a normal clip job (the
    existing progress bar/poll UI just works for this job too), rather
    than blocking the request handler for however long it takes to
    download, transcribe, and render several Twitch clips back to back.
    The first version of this feature did exactly that and reliably
    outran the client/proxy's own timeout ("could not reach the server"
    on a real attempt) despite the work succeeding server-side -- moving
    it here is the actual fix, not just a nicer progress bar.

    Source material is each tracked streamer's own most-viewed Twitch
    clip(s) from the past week (trending.get_top_twitch_clips) -- Twitch's
    own curated highlight moments (made from the Clip button, by the
    creator or a viewer), available immediately with no dependency on
    this app having already rendered and uploaded something for that
    streamer first. Each chosen clip is downloaded and captioned (see
    _render_twitch_clip_for_recap) before being concatenated as landscape
    video -- normal-video shaped, not a Short, since a raw Twitch clip
    download is already the streamer's own landscape broadcast frame
    with no captions of its own.

    Ends in state "error" (not a raised exception) when there isn't
    enough to work with -- no tracked streamer had a clip this week, or
    fewer than 2 candidates could actually be downloaded and rendered --
    since that's a normal week, not a pipeline bug."""
    cancel = lambda: _check_cancel(job_id)  # noqa: E731
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_recap_source"
    out_dir.mkdir(parents=True, exist_ok=True)

    _set(job_id, state="checking", message="Looking up this week's top Twitch clips...")
    _progress(job_id, 0.05)
    cancel()
    twitch_logins = [l for l in os.environ.get("TRENDING_TWITCH_LOGINS", "").split(",") if l.strip()]
    clips = get_top_twitch_clips(twitch_logins, days=7.0, per_streamer=5)
    pools = weekly_recap.group_clips_by_streamer(clips)
    if not pools:
        _set(job_id, state="error", error="No Twitch clips found for your tracked streamers in the past week.")
        return

    per_streamer = weekly_recap.per_streamer_count(len(pools))
    total_candidates = sum(min(len(cs), per_streamer * 2) for cs in pools.values())  # rough, for the progress bar only
    # Reset created_at here (not when the job was queued) so the "time
    # remaining" math in the UI counts from when real work starts, not
    # from the brief Twitch-lookup step above -- same pattern _run_regenerate
    # uses once it knows enough to estimate.
    _set(job_id, created_at=time.time(), estimate_minutes=round(_estimate_recap_seconds(total_candidates) / 60, 1))

    # Each streamer's candidates are already ranked by view count -- walk
    # them in order and keep going past a download/transcription/render
    # failure rather than dropping that streamer from the recap entirely
    # just because their single top clip happened to fail.
    rendered = []
    processed = 0
    for login, candidates in pools.items():
        successes = 0
        for c in candidates:
            if successes >= per_streamer:
                break
            cancel()
            processed += 1
            display = weekly_recap.display_name({"streamer_login": login})
            _set(job_id, state="rendering", message=f'Clip {processed}: {display} -- "{c.get("title", "")}"')
            _progress(job_id, 0.05 + 0.85 * (processed / max(total_candidates, 1)))
            try:
                dl = download_video(c["url"], raw_dir)
            except Exception as e:
                print(f"[weekly_recap] could not download clip {c.get('id')} ({login}): {e}", flush=True)
                continue
            try:
                words = get_transcript(dl.video_path, dl.captions_path, prefer_whisper=True)
            except Exception as e:
                print(f"[weekly_recap] transcription failed for clip {c.get('id')} ({login}): {e}", flush=True)
                words = []
            rendered_path = out_dir / f"src_{len(rendered):02d}.mp4"
            try:
                _render_twitch_clip_for_recap(dl.video_path, dl.duration, words, rendered_path)
            except Exception as e:
                print(f"[weekly_recap] render failed for clip {c.get('id')} ({login}): {e}", flush=True)
                continue
            actual_duration = _ffprobe_duration(rendered_path) or dl.duration
            rendered.append({
                "streamer_login": login,
                "title": c.get("title") or "",
                "duration": actual_duration,
                "view_count": c.get("view_count") or 0,
                "path": rendered_path,
            })
            successes += 1

    shutil.rmtree(raw_dir, ignore_errors=True)  # downloaded source no longer needed once rendered

    if len(rendered) < 2:
        shutil.rmtree(out_dir, ignore_errors=True)
        _set(job_id, state="error", error=(
            "Not enough of this week's Twitch clips could be downloaded and rendered to build a recap "
            "(need at least 2). Check the deploy logs for why a specific clip failed."
        ))
        return

    cancel()
    _set(job_id, state="rendering", message="Combining clips into the recap...")
    _progress(job_id, 0.95)
    # Biggest hit first -- a compilation's opening clip is what decides
    # whether someone keeps watching, same as any other Short.
    rendered.sort(key=lambda r: r["view_count"], reverse=True)
    out_path = out_dir / "recap.mp4"
    try:
        weekly_recap.build_recap_video([r["path"] for r in rendered], out_path)
    except (RuntimeError, ValueError) as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        _set(job_id, state="error", error=f"Could not build the recap: {e}")
        return
    for r in rendered:
        r["path"].unlink(missing_ok=True)

    total_duration = round(sum(r["duration"] for r in rendered), 2)
    streamer_count = len({r["streamer_login"] for r in rendered})
    now = datetime.datetime.now(datetime.timezone.utc)
    week_label = f"{now - datetime.timedelta(days=7):%b %d}-{now:%b %d}"
    meta = weekly_recap.build_recap_metadata(rendered, week_label)

    message = f"Weekly recap ready -- {len(rendered)} clip(s) from {streamer_count} streamer(s)."

    _set(
        job_id,
        state="done",
        message=message,
        progress=1.0,
        source_title=meta["title"],
        clips=[{
            "file": out_path.name,
            "duration": total_duration,
            "title": meta["title"],
            "upload_title": meta["title"],
            "description": meta["description"],
            "hook_caption": None,
            "reason": None,
            "window_index": None,
            "source_video": None,
            # Landscape, not vertical -- see _render_twitch_clip_for_recap
            # -- and long-form on purpose: a recap is a compilation of
            # several clips, not itself meant to be classified as a Short
            # (see is_short=False on the upload endpoint below), so
            # nothing here needs to fit the 60s Shorts cap either.
            "facecam_uncertain": False,
            "facecam_trusted": False,
            "source_frame": None,
            "is_recap": True,
        }],
    )


def _queue_weekly_recap_job() -> str:
    """Creates and enqueues a weekly-recap job for the shared worker
    thread to pick up (see _run_job's pending_weekly_recap dispatch and
    _run_weekly_recap_job) -- shared by the manual button endpoint and
    the Monday scheduler below, so both go through the exact same async,
    progress-reporting path rather than one of them running the (multi-
    minute) build inline."""
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "source_url": None,
            "source_title": "Weekly recap",
            "created_at": time.time(),
            "state": "queued",
            "message": "Queued",
            "progress": 0.0,
            "estimate_minutes": None,
            "pipeline": "weekly_recap",
            "clips": [],
            "error": None,
            "saved": True,
            "pending_weekly_recap": True,
        }
        cancel_events[job_id] = threading.Event()
    _persist(job_id)
    job_queue.put(job_id)
    return job_id


@protected.post("/api/weekly-recap/generate")
def generate_weekly_recap() -> dict:
    """Queue this week's recap build as a background job -- the same
    "submit and watch the progress bar" flow as generating regular clips,
    since the actual build can take minutes (see _run_weekly_recap_job)
    and blocking the request for that long isn't reliable. Always
    produces a draft for review, never uploads on its own."""
    return {"job_id": _queue_weekly_recap_job()}


# How often the recap scheduler wakes up to check whether it's time --
# hourly is frequent enough to land within an hour of the target time
# without a dedicated cron mechanism, and cheap enough to just poll.
_RECAP_SCHEDULER_CHECK_SECONDS = 3600
_RECAP_SCHEDULE_WEEKDAY = 0  # Monday
_RECAP_SCHEDULE_HOUR_UTC = 9


def _weekly_recap_scheduler_loop() -> None:
    """Builds the week's recap as a draft automatically once a week, so
    it's just waiting for review rather than something the creator has to
    remember to click. Never uploads by itself -- see _run_weekly_recap_job.
    A persisted "last run" ISO week (not just a sleep timer) survives a
    Railway restart/redeploy without either skipping a week or firing
    twice for the same one."""
    while True:
        time.sleep(_RECAP_SCHEDULER_CHECK_SECONDS)
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            if now.weekday() != _RECAP_SCHEDULE_WEEKDAY or now.hour < _RECAP_SCHEDULE_HOUR_UTC:
                continue
            iso_week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
            state = {}
            if _recap_scheduler_state_path.exists():
                try:
                    state = json.loads(_recap_scheduler_state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
            if state.get("last_run_iso_week") == iso_week:
                continue
            job_id = _queue_weekly_recap_job()
            print(f"[weekly_recap] scheduled run for {iso_week}: queued job {job_id}", flush=True)
            state["last_run_iso_week"] = iso_week
            _recap_scheduler_state_path.write_text(json.dumps(state), encoding="utf-8")
        except Exception as e:
            # A missed or double-counted week is a minor annoyance; taking
            # the whole scheduler thread down over one bad week is worse.
            print(f"[weekly_recap] scheduler tick failed: {e}", flush=True)


threading.Thread(target=_weekly_recap_scheduler_loop, daemon=True).start()


@protected.delete("/api/jobs/{job_id}/clips/{filename}")
def delete_clip(job_id: str, filename: str) -> dict:
    """Drop a single clip from a finished job's results -- keep the rest.
    Only removes a filename that's actually in the job's own clips list
    (never an arbitrary path), and refuses while the job is still running
    so a click doesn't yank a file out from under an active render.

    Also frees the clip's source time range (or candidate window, for the
    long-VOD pipeline) back up in used_ranges/used_window_indices -- a
    clip that's been deleted clearly wasn't the moment the creator wanted
    kept, but a later "generate more" with a different focus (e.g.
    "funny ones") was still treating that time range as spoken for, so it
    could never reconsider it even though nothing kept was using it
    anymore -- exactly the moment most likely to actually match a new
    focus, permanently locked out."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- wait for it to finish first")
        clips = list(job.get("clips") or [])
        deleted = next((c for c in clips if c.get("file") == filename), None)
        if deleted is None:
            raise HTTPException(404, "clip not found")
        remaining = [c for c in clips if c.get("file") != filename]
        job["clips"] = remaining

        if deleted.get("window_index") is not None:
            used_window_indices = set(job.get("used_window_indices") or [])
            used_window_indices.discard(deleted["window_index"])
            job["used_window_indices"] = list(used_window_indices)
        else:
            used_ranges = [tuple(r) for r in (job.get("used_ranges") or [])]
            target = (deleted.get("start"), deleted.get("end"))
            used_ranges = [r for r in used_ranges if r != target]
            job["used_ranges"] = [list(r) for r in used_ranges]
    _persist(job_id)

    _remove_clip_files(BASE_DIR / job_id, filename)
    return {"ok": True, "clips": remaining}


@protected.delete("/api/jobs/{job_id}/clips")
def delete_all_clips(job_id: str) -> dict:
    """Drop every clip from a finished job at once -- keeps the job (and
    its already-downloaded source/candidates) around so "generate more"
    can immediately pick fresh ones without re-downloading anything.

    Also resets used_ranges/used_window_indices to empty, same reasoning
    as the single-clip delete above but for all of them at once: a
    creator who just wiped every clip clearly wants a genuinely fresh
    batch, not one still constrained by what an earlier, now-deleted
    round already picked -- including candidate windows (like a collab
    moment) that got used early and then never came up again."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- wait for it to finish first")
        clips = list(job.get("clips") or [])
        job["clips"] = []
        job["used_ranges"] = []
        job["used_window_indices"] = []
    _persist(job_id)

    out_dir = BASE_DIR / job_id
    for clip in clips:
        if clip.get("file"):
            _remove_clip_files(out_dir, clip["file"])
    return {"ok": True, "clips": []}


_trending_cache: dict = {"at": 0.0, "sections": {}}
_TRENDING_CACHE_SECONDS = 180.0


@protected.get("/api/trending")
def trending() -> dict:
    """Four rows for the UI: configured creators' latest YouTube upload,
    configured creators' latest Twitch VOD, Twitch's biggest live streams
    globally, and popular Twitch creators not already on the watchlist
    (see clipper/trending.py). Cached briefly so refreshing the page
    doesn't re-hit the Twitch/YouTube APIs (and YouTube's daily quota)
    every time."""
    now = time.time()
    if now - _trending_cache["at"] > _TRENDING_CACHE_SECONDS:
        try:
            sections = get_trending_sections()
        except Exception as e:
            print(f"[trending] lookup failed: {e}", flush=True)
            sections = _trending_cache["sections"]
        _trending_cache["sections"] = sections
        _trending_cache["at"] = now
    return {
        name: [vars(e) for e in entries]
        for name, entries in _trending_cache["sections"].items()
    }


@protected.get("/api/search-creator")
def search_creator_endpoint(q: str) -> dict:
    """On-demand lookup for one creator by name -- not limited to the
    configured watchlist rows. Not cached: a manual search is infrequent
    enough that hitting the Twitch/YouTube APIs live is fine."""
    try:
        results = search_creator(q)
    except Exception as e:
        print(f"[trending] search failed: {e}", flush=True)
        results = []
    return {"results": [vars(e) for e in results]}


_MAX_ACCESSIBILITY_PROBES = 4  # bound worst-case latency: stop checking further down the ranking


@protected.post("/api/recommend-vod")
def recommend_vod_endpoint() -> dict:
    """Which of the tracked streamers' recent VODs (TRENDING_TWITCH_LOGINS
    -- the same watchlist the trending rows use) is most worth downloading
    and clipping today. Not cached and only run on a button click, same as
    the AI overview: it's a real Claude API call, so it shouldn't fire on
    every page load.

    Twitch's video-list API can't tell us a VOD is subscriber-only,
    deleted-but-listed, or otherwise blocked -- that only shows up once
    something actually tries to download it. So before handing a pick back,
    this probes it for real accessibility and walks down Claude's ranking
    past any VOD that fails it, capped at _MAX_ACCESSIBILITY_PROBES
    candidates so one bad streak of inaccessible VODs can't make this
    endpoint hang. Uses quick_probe_accessible (a single fast attempt on a
    short window) rather than the job pipeline's careful multi-attempt
    probe_source_accessible -- this is a button click a person is waiting
    on, not a job already committed to one VOD, so speed matters more than
    certainty here; a wrongly-skipped VOD just falls through to the next
    ranked one instead of blocking the whole response."""
    import shutil
    import tempfile
    from datetime import datetime

    twitch_logins = os.environ.get("TRENDING_TWITCH_LOGINS", "").split(",")
    candidates = get_recommendation_candidates(twitch_logins)
    if not candidates:
        raise HTTPException(
            409,
            "No recent VODs found -- set TRENDING_TWITCH_LOGINS to the streamers you clip, "
            "or check that TWITCH_CLIENT_ID/TWITCH_CLIENT_SECRET are configured.",
        )

    candidate_dicts = []
    for c in candidates:
        d = vars(c).copy()
        d["_published_ts"] = None
        if c.published_at:
            try:
                d["_published_ts"] = datetime.fromisoformat(c.published_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        candidate_dicts.append(d)

    try:
        result = channel_strategy.recommend_vod(candidate_dicts, notes=_load_strategy_notes())
        ranking = result["ranking"]
        reasons = {ranking[0]: result["why_best"]} if ranking else {}
        if len(ranking) > 1:
            reasons[ranking[1]] = result["why_second"]
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e

    def _entry(index: int, reason: Optional[str]) -> dict:
        c = candidates[index]
        return {
            "name": c.name, "url": c.url, "title": c.title,
            "view_count": c.view_count, "duration": c.duration,
            "thumbnail": c.thumbnail, "published_at": c.published_at,
            "reason": reason,
        }

    picks: list = []
    skipped_inaccessible = 0
    probe_dir = Path(tempfile.mkdtemp(prefix="vod_probe_"))
    try:
        for index in ranking[:_MAX_ACCESSIBILITY_PROBES]:
            c = candidates[index]
            duration_seconds = parse_twitch_duration(c.duration or "")
            if duration_seconds and not quick_probe_accessible(c.url, duration_seconds, probe_dir):
                skipped_inaccessible += 1
                continue
            # No parseable duration -- can't pick a probe point, so take it
            # on trust rather than blocking the recommendation on that.
            reason = reasons.get(index)
            if reason is None:
                reason = (
                    f"Next best option after {skipped_inaccessible} higher-ranked VOD(s) "
                    "turned out to be inaccessible right now."
                )
            picks.append(_entry(index, reason))
            if len(picks) >= 2:
                break
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    if not picks:
        raise HTTPException(
            409,
            f"Checked the top {min(len(ranking), _MAX_ACCESSIBILITY_PROBES)} tracked VODs and none of them "
            "were downloadable right now (likely subscriber-only or otherwise restricted). Try again later.",
        )

    return {
        "pick": picks[0],
        "runner_up": picks[1] if len(picks) > 1 else None,
        "candidates_considered": len(candidates),
        "skipped_inaccessible": skipped_inaccessible,
    }


def _youtube_redirect_uri() -> str:
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if not domain:
        raise HTTPException(400, "RAILWAY_PUBLIC_DOMAIN isn't set -- can't build an OAuth redirect URI")
    return f"https://{domain}/auth/youtube/callback"


@protected.get("/auth/youtube/login")
def youtube_login() -> RedirectResponse:
    """Kick off the OAuth flow for the channel owner's own YouTube account
    (see clipper/youtube_oauth.py) -- reads real Analytics data instead of
    just the public Data API's view counts."""
    if not youtube_oauth.is_configured():
        raise HTTPException(
            400,
            "YOUTUBE_OAUTH_CLIENT_ID / YOUTUBE_OAUTH_CLIENT_SECRET aren't set -- "
            "see the README for how to create them in Google Cloud Console.",
        )
    redirect_uri = _youtube_redirect_uri()
    state = secrets.token_urlsafe(24)
    _youtube_oauth_states[state] = time.time()
    # Prune old, abandoned login attempts instead of growing forever --
    # this dict only ever holds a handful of entries for a single-tenant
    # app, so a plain sweep on every login is plenty.
    cutoff = time.time() - 600
    for s, issued_at in list(_youtube_oauth_states.items()):
        if issued_at < cutoff:
            _youtube_oauth_states.pop(s, None)
    return RedirectResponse(youtube_oauth.build_authorize_url(redirect_uri, state))


@protected.get("/auth/youtube/callback")
def youtube_callback(code: str = "", state: str = "", error: str = "") -> RedirectResponse:
    if error:
        return RedirectResponse(f"/analytics?youtube_error={error}")
    issued_at = _youtube_oauth_states.pop(state, None)
    if issued_at is None or time.time() - issued_at > 600:
        raise HTTPException(400, "invalid or expired OAuth login attempt -- try connecting again")
    token = youtube_oauth.exchange_code(code, _youtube_redirect_uri())
    _youtube_token_store.save(token)
    return RedirectResponse("/analytics?youtube_connected=1")


@protected.post("/api/youtube/disconnect")
def youtube_disconnect() -> dict:
    _youtube_token_store.clear()
    return {"ok": True}


def _gather_channel_insights_data() -> dict:
    """Shared by /api/channel-insights and /api/channel-insights/overview
    so the AI overview reasons over exactly the same numbers the panel
    shows, not a second, possibly-inconsistent fetch."""
    result: dict = {"oauth_configured": youtube_oauth.is_configured(), "oauth_connected": False}

    access_token = None
    channel_id = os.environ.get("YOUTUBE_OWN_CHANNEL") or None
    if youtube_oauth.is_configured():
        access_token = _youtube_token_store.get_valid_access_token()
        result["oauth_connected"] = access_token is not None
        if access_token:
            try:
                own = youtube_analytics.get_own_channel(access_token)
            except Exception as e:
                result["analytics_error"] = f"Could not read the connected channel: {e}"
                own = None
            if own:
                channel_id = own["id"]
                result["channel_title"] = own["title"]

    if channel_id:
        try:
            snapshot = get_channel_snapshot(channel_id)
        except Exception as e:
            snapshot = None
            result["heuristic_error"] = str(e)
        if snapshot:
            result["heuristic"] = snapshot
        elif "heuristic_error" not in result:
            # A None return (as opposed to a raised exception) means the
            # lookup itself succeeded but found no matching channel -- most
            # often YOUTUBE_OWN_CHANNEL missing the "@" prefix on a handle
            # (falls through to the legacy "username" lookup, which most
            # channels don't have) or a plain typo. Surface that instead of
            # silently showing nothing.
            result["heuristic_error"] = (
                f"No YouTube channel found for {channel_id!r}. If this is a handle, "
                "make sure it starts with \"@\" (e.g. @YourChannel), not just the name."
            )
    else:
        result["setup_needed"] = (
            "Set YOUTUBE_OWN_CHANNEL (your channel ID, @handle, or username) to see "
            "a heuristic snapshot without connecting an account, or connect your "
            "YouTube account below for real Analytics data."
        )

    if access_token and channel_id and result["oauth_connected"]:
        try:
            result["analytics"] = youtube_analytics.get_insights(access_token, channel_id)
        except Exception as e:
            result["analytics_error"] = str(e)

        # Per-video retention, merged onto each video in the heuristic
        # snapshot by id -- tells apart "nobody clicked it" (low views,
        # retention doesn't matter yet) from "people clicked but didn't
        # stick around" (decent views, weak retention), which raw view
        # counts alone can't distinguish.
        recent_videos = (result.get("heuristic") or {}).get("recent_videos")
        if recent_videos:
            try:
                retention_by_id = youtube_analytics.get_video_retention(access_token, channel_id)
            except Exception as e:
                result["retention_error"] = str(e)
                retention_by_id = {}
            for v in recent_videos:
                r = retention_by_id.get(v.get("id"))
                if r:
                    v["average_view_duration_seconds"] = r["average_view_duration_seconds"]
                    v["average_view_percentage"] = r["average_view_percentage"]

    saved = channel_strategy.load_latest_overview(_channel_strategy_path)
    if saved:
        result["saved_strategy_notes_at"] = saved["timestamp"]

    return result


@protected.get("/api/channel-insights")
def channel_insights() -> dict:
    """Best-day-to-post and channel-performance signals for the channel
    you're uploading clips to -- combines two independent sources:

    - `analytics`: real YouTube Analytics data (day-of-week views,
      retention, traffic sources) for the connected account, if OAuth is
      set up and connected. Most accurate, needs setup.
    - `heuristic`: a rough best-day guess from public view counts on
      recent uploads (normalized by video age), for whichever channel is
      configured -- works immediately with no OAuth, but noisier.
    """
    return _gather_channel_insights_data()


class OverviewRequest(BaseModel):
    focus: Optional[str] = None
    # Off by default: reads each saved competitor's top video's actual
    # transcript + loudness, not just its title -- meaningfully slower
    # (a caption + an audio-only fetch per video, capped below) than the
    # metadata-only overview, so it's an explicit opt-in rather than
    # something that silently makes every overview take longer.
    analyze_content: bool = False


# Worst-case latency ceiling for the opt-in content analysis: this many
# competitor videos, each a caption fetch + a short audio-only download,
# on top of the Claude call itself.
_MAX_CONTENT_ANALYSIS_VIDEOS = 5


@protected.post("/api/channel-insights/overview")
def channel_insights_overview(req: OverviewRequest) -> dict:
    """A plain-language strategy read from Claude over the same channel
    data the insights panel shows -- best day, what content is working,
    format notes, plus a competitor-pattern comparison if any competitor
    channels are saved. Costs a Claude API call, so this is its own
    on-demand endpoint (a button) rather than something the panel
    auto-loads."""
    data = _gather_channel_insights_data()
    snapshot = data.get("heuristic")
    if not snapshot:
        raise HTTPException(
            400,
            data.get("heuristic_error")
            or data.get("setup_needed")
            or "No channel data available yet -- set YOUTUBE_OWN_CHANNEL or connect your YouTube account first.",
        )
    competitor_snapshots = []
    for c in channel_strategy.load_competitors(_competitor_channels_path):
        channel_id = c.get("channel_id")
        if not channel_id:
            continue
        try:
            comp_snapshot = get_channel_snapshot(channel_id)
        except Exception as e:
            print(f"[channel_strategy] competitor lookup for {channel_id!r} failed, skipping: {e}", flush=True)
            continue
        if comp_snapshot:
            competitor_snapshots.append(comp_snapshot)

    content_analyses = []
    if req.analyze_content and competitor_snapshots:
        for comp in competitor_snapshots[:_MAX_CONTENT_ANALYSIS_VIDEOS]:
            videos = [v for v in (comp.get("recent_videos") or []) if not v.get("too_new_to_judge")]
            if not videos:
                continue
            top = max(videos, key=lambda v: v["views_per_day"])
            video_url = f"https://www.youtube.com/watch?v={top['id']}"
            try:
                content = competitor_content.analyze_video_content(video_url, top.get("duration_seconds") or 60.0)
            except Exception as e:
                print(f"[channel_strategy] content analysis for {video_url} failed, skipping: {e}", flush=True)
                continue
            if content:
                content_analyses.append({
                    "channel_title": comp.get("channel_title"),
                    "video_title": top.get("title"),
                    "transcript_text": content.transcript_text,
                    "loud_moments": [
                        {"start": m.start, "end": m.end, "peak_db": m.peak_db, "jump_db": m.jump_db}
                        for m in content.loud_moments
                    ],
                })

    try:
        overview = get_ai_overview(
            snapshot, data.get("analytics"), focus=req.focus,
            competitors=competitor_snapshots, content_analyses=content_analyses,
        )
    except Exception as e:
        raise HTTPException(502, f"Could not generate an overview: {e}") from e
    channel_strategy.save_overview(_channel_strategy_path, overview, channel_title=snapshot.get("channel_title"))
    return {"overview": overview}


@protected.delete("/api/channel-insights/overview")
def channel_insights_clear_overview() -> dict:
    """Wipe the saved AI-overview history, so a stale or noisy analysis
    stops influencing clip selection -- the next overview generated starts
    fresh instead of piling onto whatever's already saved."""
    channel_strategy.clear_history(_channel_strategy_path)
    return {"ok": True}


@protected.get("/api/competitor-search")
def competitor_search(streamer: str) -> dict:
    """Discover YouTube channels actively clipping `streamer`, for the
    competitor picker -- a creator clipping someone else's stream usually
    has no idea who else clips the same person, so this searches instead
    of asking them to type in channel names. The most expensive lookup
    this app makes (100 YouTube quota units), so it only ever runs from
    this explicit button, never automatically."""
    if not streamer or not streamer.strip():
        raise HTTPException(400, "enter a streamer name to search for")
    if not os.environ.get("YOUTUBE_API_KEY"):
        raise HTTPException(400, "YOUTUBE_API_KEY is not set on this deployment")
    try:
        candidates = competitor_discovery.search_clipping_channels(streamer)
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}") from e
    return {"channels": [
        {
            "channel_id": c.channel_id,
            "channel_title": c.channel_title,
            "thumbnail": c.thumbnail,
            "subscriber_count": c.subscriber_count,
            "sample_video_title": c.sample_video_title,
            "sample_video_views": c.sample_video_views,
        }
        for c in candidates
    ]}


class CompetitorChannel(BaseModel):
    channel_id: str
    channel_title: str


class CompetitorChannelsRequest(BaseModel):
    channels: List[CompetitorChannel]


@protected.get("/api/competitor-channels")
def get_competitor_channels() -> dict:
    return {"channels": channel_strategy.load_competitors(_competitor_channels_path)}


@protected.post("/api/competitor-channels")
def set_competitor_channels(req: CompetitorChannelsRequest) -> dict:
    """Replace the saved competitor list -- the frontend keeps the full
    set client-side (after an add or a remove) and always sends it whole,
    so this is a plain overwrite rather than incremental add/remove calls
    against the same file."""
    channels = [{"channel_id": c.channel_id, "channel_title": c.channel_title} for c in req.channels]
    channel_strategy.save_competitors(_competitor_channels_path, channels)
    return {"channels": channels}


@protected.get("/api/competitor-channels/insights")
def get_competitor_channels_insights() -> dict:
    """Recent-video snapshots for every saved competitor -- the same
    heuristic lookup used for the AI overview, but returned directly for
    the top-videos charts on the analytics page instead of feeding a
    Claude call. Cheap (a few YouTube Data API quota units per channel,
    not the 100-unit search), so unlike competitor-search this is safe to
    run on every page load. A channel whose lookup fails is skipped, not
    fatal -- one bad handle shouldn't blank out the rest of the page."""
    results = []
    for c in channel_strategy.load_competitors(_competitor_channels_path):
        channel_id = c.get("channel_id")
        if not channel_id:
            continue
        try:
            snapshot = get_channel_snapshot(channel_id)
        except Exception as e:
            print(f"[analytics] competitor insights lookup for {channel_id!r} failed, skipping: {e}", flush=True)
            continue
        if snapshot:
            results.append(snapshot)
    return {"channels": results}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@protected.post("/api/notify-test")
def notify_test() -> dict:
    """Send a one-off Telegram test message -- verify the bot token/chat
    setup works without waiting for a real job to finish."""
    ok = send_telegram("\U0001F914 Test message from clipper -- if you got this, notifications are working.")
    if not ok:
        raise HTTPException(503, "Could not send -- check CLIPPER_BOT_API is set and you've messaged the bot at least once")
    return {"ok": True}


@protected.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@protected.get("/analytics", response_class=HTMLResponse)
def analytics_page() -> str:
    return ANALYTICS_HTML


app.include_router(protected)


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f2f3f7;
    --card: #ffffff;
    --text: #1a1b1f;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #6d28d9;
    --accent2: #ec4899;
    --accent-text: #ffffff;
    --danger: #dc2626;
    --danger-hover: #b91c1c;
    --track: #e5e7eb;
    --shadow: 0 1px 2px rgba(16,24,40,0.04), 0 8px 24px rgba(16,24,40,0.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115;
      --card: #1a1c23;
      --text: #f2f3f7;
      --muted: #9aa0ac;
      --border: #2b2e37;
      --track: #2b2e37;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.4);
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 40px 16px;
  }
  .page { max-width: 640px; margin: 0 auto; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 28px 28px 32px;
  }
  .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
  .brand .logo {
    font-size: 1.4rem; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
  }
  h1 { font-size: 1.3rem; margin: 0; letter-spacing: -0.01em; }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin: 4px 0 24px; }
  label { display: block; margin-top: 16px; font-size: 0.82rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.02em; }
  input, textarea {
    width: 100%; padding: 10px 12px; margin-top: 6px; font-size: 0.95rem;
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input:focus, textarea:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
  }
  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }
  button {
    margin-top: 18px; padding: 11px 20px; font-size: 0.95rem; font-weight: 600;
    cursor: pointer; border: none; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: var(--accent-text);
    transition: opacity 0.15s, transform 0.05s;
  }
  button:hover:not(:disabled) { opacity: 0.92; }
  button:active:not(:disabled) { transform: scale(0.98); }
  button:disabled { opacity: 0.45; cursor: default; }
  #status { margin-top: 22px; white-space: pre-wrap; font-family: ui-monospace, "SF Mono", monospace; font-size: 0.82rem; color: var(--muted); }
  #progress-wrap { display: none; margin-top: 14px; }
  #progress-meta { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); margin-bottom: 6px; }
  #progress-track { background: var(--track); border-radius: 999px; height: 10px; overflow: hidden; }
  #progress-bar { background: linear-gradient(90deg, var(--accent), var(--accent2)); height: 100%; width: 0%; border-radius: 999px; transition: width 0.6s ease; }
  #cancel-btn { display: none; margin-top: 10px; margin-left: 10px; background: var(--danger); }
  #cancel-btn:hover:not(:disabled) { background: var(--danger-hover); opacity: 1; }
  #delete-btn { display: none; margin-top: 16px; width: 100%; background: transparent; color: var(--muted); border: 1px dashed var(--border); }
  #delete-btn:hover:not(:disabled) { color: var(--danger); border-color: var(--danger); opacity: 1; }
  #jobs-panel { margin-top: 20px; }
  #jobs-list { display: flex; flex-direction: column; gap: 8px; }
  .job-row { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 10px; }
  .job-row .job-info { flex: 1; min-width: 0; }
  .job-row .job-source { font-size: 0.85rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .job-row .job-meta { font-size: 0.76rem; color: var(--muted); margin-top: 2px; }
  .job-badge { font-size: 0.68rem; font-weight: 700; padding: 2px 7px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap; }
  .job-badge.running { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent); }
  .job-badge.saved { background: color-mix(in srgb, #d97706 18%, transparent); color: #d97706; }
  .job-badge.done { background: color-mix(in srgb, #16a34a 18%, transparent); color: #16a34a; }
  .job-badge.error { background: color-mix(in srgb, var(--danger) 18%, transparent); color: var(--danger); }
  .job-row button { margin-top: 0; padding: 6px 10px; font-size: 0.78rem; flex-shrink: 0; }
  .job-row .job-delete { background: transparent; color: var(--muted); border: 1px solid var(--border); }
  .job-row .job-delete:hover:not(:disabled) { color: var(--danger); border-color: var(--danger); opacity: 1; }
  #stop-modal-overlay, #mood-modal-overlay, #regen-modal-overlay, #facecam-modal-overlay, #youtube-upload-modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    align-items: center; justify-content: center; z-index: 100; padding: 16px;
  }
  #stop-modal-overlay.open, #mood-modal-overlay.open, #regen-modal-overlay.open, #facecam-modal-overlay.open, #youtube-upload-modal-overlay.open { display: flex; }
  .privacy-option {
    display: flex; align-items: flex-start; gap: 10px; margin-top: 10px; padding: 10px 12px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 10px; cursor: pointer;
    text-transform: none; letter-spacing: normal;
  }
  .privacy-option input { width: auto; margin-top: 3px; }
  .privacy-option .privacy-label { display: block; font-weight: 700; font-size: 0.9rem; color: var(--text); }
  .privacy-option .privacy-desc { font-size: 0.78rem; color: var(--muted); margin-top: 2px; font-weight: 400; }
  .modal { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px; max-width: 380px; box-shadow: var(--shadow); max-height: 92vh; overflow-y: auto; }
  .modal p { margin: 0 0 8px; font-size: 0.95rem; }
  .modal .hint { margin-bottom: 16px; }
  .modal-actions { display: flex; flex-direction: column; gap: 8px; }
  .modal-actions button { margin-top: 0; width: 100%; }
  .modal-actions button.ghost { background: transparent; color: var(--text); border: 1px solid var(--border); }
  .modal-actions button.ghost:hover:not(:disabled) { opacity: 1; border-color: var(--accent); }
  .modal label { margin-top: 12px; }
  .modal label:first-of-type { margin-top: 0; }
  #notify-test-btn { display: block; width: 100%; margin-top: 16px; background: transparent; color: var(--muted); border: 1px dashed var(--border); }
  #notify-test-btn:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); opacity: 1; }
  .mood-btn { background: var(--bg); color: var(--text); border: 1px solid var(--border); text-align: left; }
  .mood-btn:hover:not(:disabled) { opacity: 1; border-color: var(--accent); }
  .clip {
    margin-top: 12px; padding: 14px 16px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
  }
  .clip a { display: inline-block; margin-top: 8px; font-size: 0.88rem; color: var(--accent); text-decoration: none; font-weight: 600; }
  .clip a:hover { text-decoration: underline; }
  .clip strong { font-size: 0.98rem; }
  .clip em { color: var(--muted); font-size: 0.88rem; display: block; margin-top: 4px; font-style: italic; }
  .title-row { display: flex; gap: 6px; margin-top: 10px; align-items: center; }
  .title-row input { flex: 1; margin-top: 0; font-weight: 600; }
  .title-row textarea { flex: 1; margin-top: 0; font-family: inherit; font-size: 0.85rem; resize: vertical; align-self: stretch; }
  .title-row button { margin-top: 0; padding: 8px 12px; font-size: 0.82rem; align-self: flex-start; }
  .checkbox-row { display: flex; align-items: center; gap: 10px; margin-top: 18px; }
  .checkbox-row input { width: auto; margin-top: 0; accent-color: var(--accent); }
  .checkbox-row label { margin-top: 0; text-transform: none; font-weight: 600; color: var(--text); font-size: 0.9rem; }
  .hint { font-size: 0.78rem; color: var(--muted); font-weight: 400; margin-top: 2px; text-transform: none; letter-spacing: normal; }
  #trending-wrap { margin-bottom: 4px; }
  .trending-section { margin-bottom: 16px; }
  .trending-row { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
  .creator-card {
    flex: 0 0 auto; width: 128px; cursor: pointer;
    background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 8px; text-align: left; font: inherit; color: var(--text);
  }
  .creator-card:hover { border-color: var(--accent); }
  .creator-card img { width: 100%; height: 72px; object-fit: cover; border-radius: 6px; background: var(--track); display: block; }
  .creator-card .name { font-size: 0.78rem; font-weight: 700; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .creator-card .meta { font-size: 0.7rem; color: var(--muted); margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .live-badge { display: inline-block; background: var(--danger); color: #fff; font-size: 0.62rem; font-weight: 700; padding: 1px 5px; border-radius: 4px; margin-top: 6px; letter-spacing: 0.03em; }
  #recommend-vod-btn { padding: 8px 16px; }
  .recommend-card { display: flex; gap: 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin-bottom: 8px; }
  .recommend-card img { width: 110px; height: 62px; object-fit: cover; border-radius: 6px; background: var(--track); flex: 0 0 auto; }
  .recommend-card .rc-body { min-width: 0; flex: 1; }
  .recommend-card .rc-tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; color: var(--accent); text-transform: uppercase; }
  .recommend-card .rc-title { font-weight: 700; font-size: 0.88rem; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .recommend-card .rc-meta { font-size: 0.76rem; color: var(--muted); margin-top: 2px; }
  .recommend-card .rc-reason { font-size: 0.8rem; margin-top: 6px; }
  .recommend-card button { margin-top: 8px; padding: 5px 12px; font-size: 0.8rem; }
  .actions { display: flex; align-items: center; gap: 0; }
  #search-row { display: flex; gap: 8px; margin-top: 0; }
  #search-row input { flex: 1; margin-top: 0; }
  #search-row button { margin-top: 0; padding: 0 16px; white-space: nowrap; }
  #search-results-section { margin-bottom: 16px; display: none; }
  #facecam-stage { position: relative; display: inline-block; max-width: 100%; touch-action: none; cursor: crosshair; user-select: none; }
  #facecam-still-img { display: block; max-width: 100%; height: auto; border-radius: 8px; background: var(--track); }
  #facecam-box-layer { position: absolute; inset: 0; }
  .facecam-box { position: absolute; border: 2px solid var(--accent); background: color-mix(in srgb, var(--accent) 20%, transparent); box-sizing: border-box; }
  .facecam-box .rm-btn {
    position: absolute; top: -10px; right: -10px; width: 20px; height: 20px; border-radius: 50%;
    background: var(--danger); color: #fff; border: none; font-size: 12px; line-height: 20px;
    text-align: center; cursor: pointer; padding: 0; margin: 0;
  }
</style>
</head>
<body>
<div class="page">
<div class="card">

<div class="brand"><span class="logo">🎬</span><h1>clipper</h1></div>
<p class="subtitle">Paste a YouTube or Twitch link, get back short vertical highlight clips picked by Claude.</p>

<label style="margin-top:0">Search a creator</label>
<div id="search-row">
  <input id="search-input" placeholder="Twitch login or YouTube handle...">
  <button id="search-btn" type="button">Search</button>
</div>
<div class="trending-section" id="search-results-section">
  <div class="trending-row" id="search-results"></div>
  <div class="hint" id="search-status" style="display:none"></div>
</div>

<div class="trending-section" id="recommend-vod-section">
  <label style="margin-top:0">Recommended VOD to clip today</label>
  <button id="recommend-vod-btn" type="button">🎯 Recommend one</button>
  <div class="hint" id="recommend-vod-status" style="display:none"></div>
  <div id="recommend-vod-results" style="display:none; margin-top:10px"></div>
</div>

<div id="trending-wrap">
  <div class="trending-section" id="section-youtube_channels" style="display:none">
    <label style="margin-top:0">Latest uploads — YouTube</label>
    <div class="trending-row"></div>
  </div>
  <div class="trending-section" id="section-twitch_vods" style="display:none">
    <label>Latest VODs — Twitch</label>
    <div class="trending-row"></div>
  </div>
  <div class="trending-section" id="section-trending_live" style="display:none">
    <label>Trending live now — Twitch (100k+ viewers)</label>
    <div class="trending-row"></div>
  </div>
  <div class="trending-section" id="section-suggested_creators" style="display:none">
    <label>Suggested creators — Twitch (popular, not on your watchlist)</label>
    <div class="trending-row"></div>
  </div>
</div>

<label>Video URL</label>
<input id="source" placeholder="https://www.youtube.com/watch?v=...">

<label>Focus (optional)</label>
<input id="focus" placeholder="e.g. funniest moments">

<div class="row">
  <div>
    <label># clips</label>
    <input id="num_clips" type="number" value="5" min="1" max="10">
  </div>
  <div>
    <label>Min length (s)</label>
    <input id="min_len" type="number" value="20">
  </div>
  <div>
    <label>Max length (s)</label>
    <input id="max_len" type="number" value="60" max="60">
    <div class="hint">Capped at 60s -- past that, YouTube can silently upload it as a regular video instead of a Short.</div>
  </div>
</div>

<div class="checkbox-row">
  <input id="whisper" type="checkbox" checked>
  <label for="whisper">Accurate captions (Whisper)<div class="hint">Slower, but word timing is aligned to the audio. Uncheck to use YouTube's own captions instead (faster, but timing can lag the audio).</div></label>
</div>

<div class="actions">
  <button id="submit">Generate clips</button>
  <button id="cancel-btn" type="button">Emergency stop</button>
</div>

<div id="status"></div>
<div id="progress-wrap">
  <div id="progress-meta">
    <span id="progress-pct">0%</span>
    <span id="progress-eta"></span>
  </div>
  <div id="progress-track"><div id="progress-bar"></div></div>
</div>
<div id="clips"></div>
<button id="delete-btn" type="button">🗑 I've downloaded these — delete from server</button>

<div id="jobs-panel">
  <label style="margin-top:0">Active &amp; saved jobs</label>
  <div id="jobs-list"></div>
</div>

<button id="weekly-recap-btn" type="button" style="margin-top:10px">🗓 Generate this week's recap</button>
<button id="weekly-recap-view-btn" type="button" style="margin-top:10px;margin-left:8px" disabled>📺 No weekly recap yet</button>
<div class="hint">Pulls each tracked streamer's own most-viewed Twitch clip from the past week,
adds captions, and concatenates them into one landscape long-form draft (a normal
video, not a Short -- no 60s cap, no #Shorts tag) -- top 2 per streamer with 5 or
fewer streamers having a clip this week, top 1 each above that. A fresh one also
builds automatically every Monday. Never uploads on its own -- review, trim, and
hit Upload like any other clip.</div>
<div id="weekly-recap-status" class="hint"></div>

<button id="notify-test-btn" type="button">🔔 Test Telegram notification</button>

<button id="analytics-link-btn" type="button" style="margin-top:10px">📊 Analytics &amp; AI strategy</button>

</div>
</div>

<div id="stop-modal-overlay">
  <div class="modal">
    <p>Stop this job now?</p>
    <p class="hint">A step already in progress (a download, a render) finishes first -- this isn't instant.</p>
    <div class="modal-actions">
      <button id="stop-save-btn" type="button">Save progress</button>
      <button id="stop-yes-btn" type="button" class="danger">Yes, stop &amp; delete</button>
      <button id="stop-no-btn" type="button" class="ghost">No, continue</button>
    </div>
  </div>
</div>

<div id="mood-modal-overlay">
  <div class="modal">
    <p>What mood are you looking for?</p>
    <p class="hint">This becomes the instruction Claude uses when picking clips -- pick one, or skip to let it judge freely.</p>
    <div class="modal-actions">
      <button type="button" class="mood-btn" data-mood="the funniest moments -- genuine comedy, banter, or jokes that land">😂 Funny</button>
      <button type="button" class="mood-btn" data-mood="insane clutch plays -- high-pressure moments where they pull off something incredible at the last second">🔥 Insane clutch</button>
      <button type="button" class="mood-btn" data-mood="crazy, unexpected moments -- chaotic or jaw-dropping events that make you go &quot;no way&quot;">🤯 Crazy moment</button>
      <button type="button" class="mood-btn" data-mood="dark humor -- edgy or morbid jokes that get a shocked laugh">💀 Dark humor</button>
      <button id="mood-skip-btn" type="button" class="ghost">Skip -- no preference</button>
    </div>
  </div>
</div>

<div id="regen-modal-overlay">
  <div class="modal">
    <p>Generate more clips</p>
    <p class="hint">Reuses the already-downloaded source -- no re-download needed.</p>
    <label>Mood (optional)</label>
    <div class="modal-actions" style="margin-bottom:12px">
      <button type="button" class="mood-btn regen-mood-btn" data-mood="the funniest moments -- genuine comedy, banter, or jokes that land">😂 Funny</button>
      <button type="button" class="mood-btn regen-mood-btn" data-mood="insane clutch plays -- high-pressure moments where they pull off something incredible at the last second">🔥 Insane clutch</button>
      <button type="button" class="mood-btn regen-mood-btn" data-mood="crazy, unexpected moments -- chaotic or jaw-dropping events that make you go &quot;no way&quot;">🤯 Crazy moment</button>
      <button type="button" class="mood-btn regen-mood-btn" data-mood="dark humor -- edgy or morbid jokes that get a shocked laugh">💀 Dark humor</button>
    </div>
    <label>Focus (optional)</label>
    <input id="regen-focus" placeholder="e.g. funniest moments">
    <label>How many more clips?</label>
    <input id="regen-num-clips" type="number" value="3" min="1" max="10">
    <label style="margin-top:12px;display:flex;align-items:center;gap:8px;font-weight:normal">
      <input id="regen-reset-used" type="checkbox" style="width:auto">
      🔁 Start fresh (ignore previously-picked moments -- may repeat earlier clips)
    </label>
    <p class="hint">Off (default): only ever picks NEW moments, same as before. On: forgets what's already been
      picked so Claude can freely re-pick from everything again -- useful once repeated "generate more" calls
      have used up most of the available moments and it's only returning 1-2 clips.</p>
    <div class="modal-actions" style="margin-top:16px">
      <button id="regen-go-btn" type="button">Generate</button>
      <button id="regen-cancel-btn" type="button" class="ghost">Cancel</button>
    </div>
  </div>
</div>

<div id="facecam-modal-overlay">
  <div class="modal" style="max-width:720px;width:100%;max-height:92vh;overflow:auto">
    <p id="facecam-modal-title">Place the facecam</p>
    <p class="hint" id="facecam-modal-hint"></p>
    <div id="facecam-stage">
      <img id="facecam-still-img" draggable="false" alt="Source frame">
      <div id="facecam-box-layer"></div>
    </div>
    <p class="hint" id="facecam-count-hint"></p>
    <label id="facecam-apply-all-row" style="display:none;margin-top:10px;align-items:center;gap:8px;font-weight:normal;text-transform:none;letter-spacing:normal">
      <input id="facecam-apply-all" type="checkbox" style="width:auto;margin-top:0" checked>
      <span id="facecam-apply-all-text"></span>
    </label>
    <div class="modal-actions" style="margin-top:12px">
      <button id="facecam-go-btn" type="button">Re-render with these boxes</button>
      <button id="facecam-clear-btn" type="button" class="ghost">Clear boxes</button>
      <button id="facecam-cancel-btn" type="button" class="ghost">Skip for now</button>
    </div>
  </div>
</div>

<div id="youtube-upload-modal-overlay">
  <div class="modal" style="max-height:92vh;overflow:auto">
    <p>Upload to YouTube</p>
    <p class="hint" id="youtube-upload-title-hint"></p>
    <video id="youtube-upload-preview" controls preload="metadata" style="width:100%;max-height:40vh;border-radius:8px;background:var(--track);display:block;object-fit:contain"></video>
    <label class="privacy-option">
      <input type="radio" name="youtube-privacy" value="unlisted" checked>
      <span>
        <span class="privacy-label">Unlisted</span>
        <span class="privacy-desc" style="display:block">Only people with the link can see it -- good for a final check before going public.</span>
      </span>
    </label>
    <label class="privacy-option">
      <input type="radio" name="youtube-privacy" value="public">
      <span>
        <span class="privacy-label">Public</span>
        <span class="privacy-desc" style="display:block">Live immediately on your channel and in search/Shorts feed.</span>
      </span>
    </label>
    <label class="privacy-option">
      <input type="radio" name="youtube-privacy" value="private">
      <span>
        <span class="privacy-label">Private</span>
        <span class="privacy-desc" style="display:block">Only you can see it.</span>
      </span>
    </label>
    <label style="margin-top:16px">Trim before uploading (optional)</label>
    <p class="hint">Play the video above, pause where you want to cut, then use the buttons below -- or type seconds directly.</p>
    <div class="row">
      <div>
        <label style="margin-top:6px;font-size:0.7rem">Off the start (s)</label>
        <input id="youtube-upload-trim-start" type="number" value="0" min="0" step="0.5">
        <button id="youtube-upload-set-start-btn" type="button" style="margin-top:4px;width:100%">Set to current position</button>
      </div>
      <div>
        <label style="margin-top:6px;font-size:0.7rem">Off the end (s)</label>
        <input id="youtube-upload-trim-end" type="number" value="0" min="0" step="0.5">
        <button id="youtube-upload-set-end-btn" type="button" style="margin-top:4px;width:100%">Set to current position</button>
      </div>
    </div>
    <p class="hint" id="youtube-upload-trim-hint"></p>
    <div class="modal-actions" style="margin-top:16px">
      <button id="youtube-upload-go-btn" type="button">Upload</button>
      <button id="youtube-upload-cancel-btn" type="button" class="ghost">Cancel</button>
    </div>
  </div>
</div>

<script>
const statusEl = document.getElementById('status');
const clipsEl = document.getElementById('clips');
const deleteBtn = document.getElementById('delete-btn');
const submitBtn = document.getElementById('submit');
const cancelBtn = document.getElementById('cancel-btn');
const progressWrap = document.getElementById('progress-wrap');
const progressBar = document.getElementById('progress-bar');
const progressPct = document.getElementById('progress-pct');
const progressEta = document.getElementById('progress-eta');
const TRENDING_SECTIONS = ['youtube_channels', 'twitch_vods', 'trending_live', 'suggested_creators'];
const jobsListEl = document.getElementById('jobs-list');
const stopModal = document.getElementById('stop-modal-overlay');
let timer = null;
let jobsTimer = null;
let currentJobId = null;
let pendingDeleteOnCancel = false;

function formatViewers(n) {
  if (n >= 1000) return (n / 1000).toFixed(n >= 100000 ? 0 : 1) + 'K';
  return String(n);
}

function buildCreatorCard(c) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'creator-card';

  const img = document.createElement('img');
  if (c.thumbnail) img.src = c.thumbnail;
  card.appendChild(img);

  const name = document.createElement('div');
  name.className = 'name';
  name.textContent = c.name;
  card.appendChild(name);

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = c.title || '';
  card.appendChild(meta);

  if (c.live) {
    const badge = document.createElement('span');
    badge.className = 'live-badge';
    badge.textContent = c.viewers ? `LIVE · ${formatViewers(c.viewers)} viewers` : 'LIVE';
    card.appendChild(badge);
  }

  card.addEventListener('click', () => {
    document.getElementById('source').value = c.url;
    document.getElementById('source').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  return card;
}

async function loadTrending() {
  try {
    const resp = await fetch('/api/trending');
    if (!resp.ok) return;
    const sections = await resp.json();
    TRENDING_SECTIONS.forEach(key => {
      const entries = sections[key] || [];
      const sectionEl = document.getElementById(`section-${key}`);
      if (!sectionEl) return;
      const row = sectionEl.querySelector('.trending-row');
      row.innerHTML = '';
      entries.forEach(c => row.appendChild(buildCreatorCard(c)));
      sectionEl.style.display = entries.length ? 'block' : 'none';
    });
  } catch (e) {
    // trending is a nice-to-have -- never block the rest of the page on it
  }
}
loadTrending();

function formatDuration(d) {
  // Twitch's own format is already compact (e.g. "3h20m10s") -- just
  // space it out a bit for readability.
  if (!d) return '';
  return d.replace(/(\d+)h/, '$1h ').replace(/(\d+)m/, '$1m ').trim();
}

function buildRecommendCard(entry, tag) {
  const card = document.createElement('div');
  card.className = 'recommend-card';

  const img = document.createElement('img');
  if (entry.thumbnail) img.src = entry.thumbnail;
  card.appendChild(img);

  const body = document.createElement('div');
  body.className = 'rc-body';

  const tagEl = document.createElement('div');
  tagEl.className = 'rc-tag';
  tagEl.textContent = tag;
  body.appendChild(tagEl);

  const title = document.createElement('div');
  title.className = 'rc-title';
  title.title = entry.title || '';
  title.textContent = `${entry.name} — ${entry.title || ''}`;
  body.appendChild(title);

  const meta = document.createElement('div');
  meta.className = 'rc-meta';
  const metaParts = [];
  if (entry.view_count != null) metaParts.push(`${formatViewers(entry.view_count)} views`);
  if (entry.duration) metaParts.push(formatDuration(entry.duration));
  meta.textContent = metaParts.join(' · ');
  body.appendChild(meta);

  if (entry.reason) {
    const reason = document.createElement('div');
    reason.className = 'rc-reason';
    reason.textContent = entry.reason;
    body.appendChild(reason);
  }

  const useBtn = document.createElement('button');
  useBtn.type = 'button';
  useBtn.textContent = 'Use this VOD';
  useBtn.addEventListener('click', () => {
    document.getElementById('source').value = entry.url;
    document.getElementById('source').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  body.appendChild(useBtn);

  card.appendChild(body);
  return card;
}

const recommendVodBtn = document.getElementById('recommend-vod-btn');
const recommendVodStatus = document.getElementById('recommend-vod-status');
const recommendVodResults = document.getElementById('recommend-vod-results');

recommendVodBtn.addEventListener('click', async () => {
  recommendVodBtn.disabled = true;
  recommendVodResults.style.display = 'none';
  recommendVodResults.innerHTML = '';
  recommendVodStatus.style.display = 'block';
  recommendVodStatus.textContent = "Checking your tracked streamers' recent VODs -- this can take up to a minute since it double-checks each one is actually downloadable...";
  try {
    const resp = await fetch('/api/recommend-vod', { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      recommendVodStatus.textContent = data.detail || 'Could not get a recommendation.';
      return;
    }
    recommendVodStatus.style.display = 'none';
    if (data.pick) recommendVodResults.appendChild(buildRecommendCard(data.pick, 'Top pick'));
    if (data.runner_up) recommendVodResults.appendChild(buildRecommendCard(data.runner_up, 'Runner-up'));
    recommendVodResults.style.display = 'block';
  } catch (e) {
    recommendVodStatus.textContent = 'Could not get a recommendation -- try again.';
  } finally {
    recommendVodBtn.disabled = false;
  }
});

const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const searchResultsSection = document.getElementById('search-results-section');
const searchResultsRow = document.getElementById('search-results');
const searchStatus = document.getElementById('search-status');

async function runCreatorSearch() {
  const q = searchInput.value.trim();
  if (!q) return;
  searchResultsSection.style.display = 'block';
  searchResultsRow.innerHTML = '';
  searchStatus.style.display = 'block';
  searchStatus.textContent = 'Searching...';
  searchBtn.disabled = true;
  try {
    const resp = await fetch(`/api/search-creator?q=${encodeURIComponent(q)}`);
    const data = await resp.json();
    const results = data.results || [];
    if (!results.length) {
      searchStatus.textContent = `No creator found for "${q}".`;
    } else {
      searchStatus.style.display = 'none';
      results.forEach(c => searchResultsRow.appendChild(buildCreatorCard(c)));
    }
  } catch (e) {
    searchStatus.textContent = 'Search failed -- try again.';
  } finally {
    searchBtn.disabled = false;
  }
}
searchBtn.addEventListener('click', runCreatorSearch);
searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') runCreatorSearch();
});

function jobBadgeClass(job) {
  if (['queued', 'checking', 'downloading', 'scanning', 'transcribing', 'selecting', 'rendering'].includes(job.state)) return 'running';
  if (job.state === 'cancelled' && job.saved) return 'saved';
  if (job.state === 'done') return 'done';
  return 'error';
}

function jobBadgeText(job) {
  if (jobBadgeClass(job) === 'running') return 'Running';
  if (jobBadgeClass(job) === 'saved') return 'Saved';
  if (job.state === 'done') return 'Done';
  if (job.state === 'cancelled') return 'Stopped';
  return 'Error';
}

async function loadJobsList() {
  try {
    const resp = await fetch('/api/jobs');
    if (!resp.ok) return;
    const { jobs } = await resp.json();
    jobsListEl.innerHTML = '';
    jobs.forEach(job => {
      const row = document.createElement('div');
      row.className = 'job-row';

      const info = document.createElement('div');
      info.className = 'job-info';
      const source = document.createElement('div');
      source.className = 'job-source';
      source.textContent = job.source_title || job.source_url || job.id;
      info.appendChild(source);
      const meta = document.createElement('div');
      meta.className = 'job-meta';
      const pct = Math.round((job.progress || 0) * 100);
      const needFacecam = (job.clips || []).filter(c => c.facecam_uncertain && c.source_frame).length;
      meta.textContent = job.state === 'done'
        ? `${(job.clips || []).length} clip(s)` + (needFacecam ? ` · 🎯 ${needFacecam} need facecam placement` : '')
        : `${pct}% -- ${job.message || ''}`;
      info.appendChild(meta);
      row.appendChild(info);

      const badge = document.createElement('span');
      badge.className = 'job-badge ' + jobBadgeClass(job);
      badge.textContent = jobBadgeText(job);
      row.appendChild(badge);

      const running = jobBadgeClass(job) === 'running';
      const hasClips = (job.clips || []).length > 0;
      if (running || hasClips) {
        const viewBtn = document.createElement('button');
        viewBtn.type = 'button';
        viewBtn.textContent = running ? 'View' : 'View clips';
        viewBtn.addEventListener('click', () => attachToJob(job.id));
        row.appendChild(viewBtn);
      }
      if (!running && job.pipeline !== 'weekly_recap') {
        const regenBtn = document.createElement('button');
        regenBtn.type = 'button';
        regenBtn.textContent = 'Generate more clips';
        regenBtn.addEventListener('click', () => openRegenModal(job.id));
        row.appendChild(regenBtn);

        if (hasClips) {
          const clearBtn = document.createElement('button');
          clearBtn.type = 'button';
          clearBtn.textContent = '🗑 Clear clips & regenerate';
          clearBtn.addEventListener('click', async () => {
            if (!confirm('Delete all clips from this job? The downloaded source stays, so regenerating is still fast.')) return;
            clearBtn.disabled = true;
            try {
              const r = await fetch(`/api/jobs/${job.id}/clips`, { method: 'DELETE' });
              if (!r.ok) {
                const data = await r.json().catch(() => ({}));
                alert(data.detail || 'Could not clear clips.');
                return;
              }
              await loadJobsList();
              openRegenModal(job.id);
            } finally {
              clearBtn.disabled = false;
            }
          });
          row.appendChild(clearBtn);
        }

        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'job-delete';
        delBtn.textContent = 'Delete';
        delBtn.addEventListener('click', async () => {
          delBtn.disabled = true;
          await fetch(`/api/jobs/${job.id}`, { method: 'DELETE' });
          loadJobsList();
        });
        row.appendChild(delBtn);
      }

      jobsListEl.appendChild(row);
    });
  } catch (e) {
    // best-effort panel -- never block the rest of the page on it
  }
}
loadJobsList();
if (jobsTimer) clearInterval(jobsTimer);
jobsTimer = setInterval(loadJobsList, 5000);

async function attachToJob(jobId) {
  currentJobId = jobId;
  if (timer) clearInterval(timer);
  const resp = await fetch(`/api/jobs/${jobId}`);
  const stillRunning = resp.ok && !['done', 'error', 'cancelled'].includes((await resp.clone().json()).state);
  setRunning(stillRunning);
  if (stillRunning) timer = setInterval(() => poll(jobId), 2000);
  poll(jobId);
  clipsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setRunning(running) {
  submitBtn.disabled = running;
  cancelBtn.style.display = running ? 'inline-block' : 'none';
  progressWrap.style.display = running ? 'block' : 'none';
}

async function submitJob() {
  clipsEl.innerHTML = '';
  statusEl.textContent = 'Submitting...';
  setRunning(true);
  progressBar.style.width = '0%';
  progressPct.textContent = '0%';
  progressEta.textContent = '';
  const body = {
    source: document.getElementById('source').value,
    focus: document.getElementById('focus').value || null,
    num_clips: parseInt(document.getElementById('num_clips').value, 10),
    min_len: parseFloat(document.getElementById('min_len').value),
    max_len: parseFloat(document.getElementById('max_len').value),
    whisper: document.getElementById('whisper').checked,
  };
  const resp = await fetch('/api/jobs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    statusEl.textContent = 'Error submitting job: ' + (await resp.text());
    setRunning(false);
    return;
  }
  const { job_id } = await resp.json();
  pendingDeleteOnCancel = false;
  currentJobId = job_id;
  if (timer) clearInterval(timer);
  timer = setInterval(() => poll(job_id), 2000);
  poll(job_id);
  loadJobsList();
}

const moodModal = document.getElementById('mood-modal-overlay');

document.getElementById('submit').addEventListener('click', () => {
  if (!document.getElementById('focus').value.trim()) {
    moodModal.classList.add('open');
    return;
  }
  submitJob();
});

document.querySelectorAll('.mood-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('focus').value = btn.dataset.mood;
    moodModal.classList.remove('open');
    submitJob();
  });
});

document.getElementById('mood-skip-btn').addEventListener('click', () => {
  moodModal.classList.remove('open');
  submitJob();
});

const regenModal = document.getElementById('regen-modal-overlay');
const regenFocusInput = document.getElementById('regen-focus');
const regenNumClipsInput = document.getElementById('regen-num-clips');
const regenResetUsedInput = document.getElementById('regen-reset-used');
const regenGoBtn = document.getElementById('regen-go-btn');
let regenJobId = null;

function openRegenModal(jobId) {
  regenJobId = jobId;
  regenFocusInput.value = '';
  regenNumClipsInput.value = '3';
  regenResetUsedInput.checked = false;
  regenModal.classList.add('open');
}

document.querySelectorAll('.regen-mood-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    regenFocusInput.value = btn.dataset.mood;
  });
});

document.getElementById('regen-cancel-btn').addEventListener('click', () => {
  regenModal.classList.remove('open');
  regenJobId = null;
});

regenGoBtn.addEventListener('click', async () => {
  if (!regenJobId) return;
  const jobId = regenJobId;
  regenGoBtn.disabled = true;
  regenGoBtn.textContent = 'Starting...';
  try {
    const resp = await fetch(`/api/jobs/${jobId}/regenerate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        focus: regenFocusInput.value.trim() || null,
        num_clips: parseInt(regenNumClipsInput.value, 10) || 3,
        reset_used: regenResetUsedInput.checked,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || 'Could not start -- the downloaded source may be gone.');
      return;
    }
    regenModal.classList.remove('open');
    regenJobId = null;
    loadJobsList();
    attachToJob(jobId);
  } finally {
    regenGoBtn.disabled = false;
    regenGoBtn.textContent = 'Generate';
  }
});

const facecamModal = document.getElementById('facecam-modal-overlay');
const facecamStage = document.getElementById('facecam-stage');
const facecamImg = document.getElementById('facecam-still-img');
const facecamBoxLayer = document.getElementById('facecam-box-layer');
const facecamCountHint = document.getElementById('facecam-count-hint');
const facecamGoBtn = document.getElementById('facecam-go-btn');
const facecamModalTitle = document.getElementById('facecam-modal-title');
const facecamModalHint = document.getElementById('facecam-modal-hint');
const facecamApplyAllRow = document.getElementById('facecam-apply-all-row');
const facecamApplyAll = document.getElementById('facecam-apply-all');
const facecamApplyAllText = document.getElementById('facecam-apply-all-text');
const FACECAM_MAX_BOXES = 3;
let facecamJobId = null;
let facecamFilename = null;
let facecamBoxes = []; // {fx, fy, fw, fh} as fractions of the frame, so they survive resizes
let facecamDrawStart = null;
let facecamPendingSourceBoxes = null; // [[x,y,w,h]] in source pixels, applied once the frame's size is known
let lastFacecamSourceBoxes = null;    // the last placement submitted -- pre-fills the next clip's picker
const facecamPrompted = new Set();    // `${jobId}/${file}` already prompted for on this page load

function facecamOthersMissing(job, clip) {
  // Every clip with a downloaded source is eligible for the batch "apply
  // to others" option (a saved source_frame isn't required -- one gets
  // generated on demand if needed), but it must still only ever target
  // clips with no trusted placement yet -- otherwise it would offer to
  // stamp this clip's box position onto clips that already have a
  // perfectly good, differently-positioned facecam.
  return (job.clips || []).filter(c => c.file !== clip.file && c.source_video && !c.facecam_trusted && !c.facecam_manual).length;
}

async function openFacecamModal(jobId, clip, othersMissing) {
  facecamJobId = jobId;
  facecamFilename = clip.file;
  facecamBoxes = [];
  facecamDrawStart = null;
  facecamPendingSourceBoxes = clip.facecam_boxes || lastFacecamSourceBoxes;
  let why;
  if (clip.facecam_uncertain) {
    facecamModalTitle.textContent = `Fix the facecam position -- "${clip.title}"`;
    why = 'Detection found a facecam here but the automatic check rejected where it landed, so this clip shipped without one.';
  } else if (clip.facecam_manual) {
    facecamModalTitle.textContent = `Adjust the facecam position -- "${clip.title}"`;
    why = 'This clip uses the boxes you placed earlier.';
  } else if (clip.facecam_trusted) {
    facecamModalTitle.textContent = `Adjust the facecam position -- "${clip.title}"`;
    why = 'The automatic placement passed its check, but override it if it actually looks wrong.';
  } else {
    facecamModalTitle.textContent = `Add a facecam -- "${clip.title}"`;
    why = 'No facecam was detected in this clip.';
  }
  facecamModalHint.textContent = `${why} Click and drag on the frame to draw a box tightly around each facecam window (up to ${FACECAM_MAX_BOXES}), then re-render.`;
  facecamApplyAllRow.style.display = othersMissing > 0 ? 'flex' : 'none';
  facecamApplyAllText.textContent = `Also apply these boxes to the other ${othersMissing} clip(s) without an automatic facecam`;
  facecamApplyAll.checked = othersMissing > 0;
  facecamModal.classList.add('open');
  renderFacecamBoxes();
  let sourceFrame = clip.source_frame;
  if (!sourceFrame) {
    // An older clip rendered before every clip saved its own source frame
    // -- generate one now instead of leaving the picker with nothing to
    // draw on, as long as its downloaded source is still on disk.
    facecamModalHint.textContent = 'Loading the source frame...';
    try {
      const resp = await fetch(`/api/jobs/${jobId}/clips/${clip.file}/ensure-source-frame`, { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        facecamModalHint.textContent = data.detail || "Couldn't load a source frame for this clip.";
        return;
      }
      sourceFrame = data.source_frame;
      facecamModalHint.textContent = `${why} Click and drag on the frame to draw a box tightly around each facecam window (up to ${FACECAM_MAX_BOXES}), then re-render.`;
    } catch (e) {
      facecamModalHint.textContent = "Couldn't load a source frame for this clip -- try again.";
      return;
    }
  }
  facecamImg.src = `/api/jobs/${jobId}/clips/${sourceFrame}`;
  // A cached frame may already be complete before the load event queues
  // -- decode() resolves either way; if it rejects (the request changed
  // under it), the load listener below covers it.
  if (facecamImg.decode) facecamImg.decode().then(applyPendingSourceBoxes).catch(() => {});
}

function applyPendingSourceBoxes() {
  const W = facecamImg.naturalWidth, H = facecamImg.naturalHeight;
  if (!facecamPendingSourceBoxes || !W || !H) return;
  facecamBoxes = facecamPendingSourceBoxes.slice(0, FACECAM_MAX_BOXES)
    .map(b => ({ fx: b[0] / W, fy: b[1] / H, fw: b[2] / W, fh: b[3] / H }));
  facecamPendingSourceBoxes = null;
  renderFacecamBoxes();
}
facecamImg.addEventListener('load', applyPendingSourceBoxes);

function renderFacecamBoxes(draftBox) {
  facecamBoxLayer.innerHTML = '';
  const all = draftBox ? [...facecamBoxes, draftBox] : facecamBoxes;
  all.forEach((b, idx) => {
    const el = document.createElement('div');
    el.className = 'facecam-box';
    el.style.left = (b.fx * 100) + '%';
    el.style.top = (b.fy * 100) + '%';
    el.style.width = (b.fw * 100) + '%';
    el.style.height = (b.fh * 100) + '%';
    if (!b.draft) {
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.className = 'rm-btn';
      rm.textContent = '×';
      rm.addEventListener('click', (e) => {
        e.stopPropagation();
        facecamBoxes.splice(idx, 1);
        renderFacecamBoxes();
      });
      el.appendChild(rm);
    }
    facecamBoxLayer.appendChild(el);
  });
  facecamCountHint.textContent = facecamBoxes.length >= FACECAM_MAX_BOXES
    ? `Maximum ${FACECAM_MAX_BOXES} facecam boxes -- remove one (×) to redraw it.`
    : `${facecamBoxes.length} box(es) drawn -- click and drag on the frame to add ${facecamBoxes.length ? 'another' : 'one'} (up to ${FACECAM_MAX_BOXES}).`;
}

function facecamPoint(e) {
  const rect = facecamImg.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  return {
    fx: Math.max(0, Math.min((e.clientX - rect.left) / rect.width, 1)),
    fy: Math.max(0, Math.min((e.clientY - rect.top) / rect.height, 1)),
    rect,
  };
}

function facecamDraft(p) {
  return {
    fx: Math.min(p.fx, facecamDrawStart.fx),
    fy: Math.min(p.fy, facecamDrawStart.fy),
    fw: Math.abs(p.fx - facecamDrawStart.fx),
    fh: Math.abs(p.fy - facecamDrawStart.fy),
  };
}

facecamStage.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.rm-btn')) return;
  if (facecamBoxes.length >= FACECAM_MAX_BOXES) return;
  const p = facecamPoint(e);
  if (!p) return;
  facecamDrawStart = p;
  facecamStage.setPointerCapture(e.pointerId);
  e.preventDefault();
});

facecamStage.addEventListener('pointermove', (e) => {
  if (!facecamDrawStart) return;
  const p = facecamPoint(e);
  if (!p) return;
  renderFacecamBoxes({ ...facecamDraft(p), draft: true });
});

facecamStage.addEventListener('pointerup', (e) => {
  if (!facecamDrawStart) return;
  const p = facecamPoint(e);
  const box = p ? facecamDraft(p) : null;
  facecamDrawStart = null;
  if (box && box.fw * p.rect.width > 8 && box.fh * p.rect.height > 8 && facecamBoxes.length < FACECAM_MAX_BOXES) {
    facecamBoxes.push(box);
  }
  renderFacecamBoxes();
});

facecamStage.addEventListener('pointercancel', () => {
  facecamDrawStart = null;
  renderFacecamBoxes();
});

document.getElementById('facecam-clear-btn').addEventListener('click', () => {
  facecamBoxes = [];
  facecamPendingSourceBoxes = null;
  renderFacecamBoxes();
});

document.getElementById('facecam-cancel-btn').addEventListener('click', () => {
  facecamModal.classList.remove('open');
  facecamJobId = null;
  facecamFilename = null;
});

facecamGoBtn.addEventListener('click', async () => {
  if (!facecamJobId || !facecamFilename) return;
  if (!facecamBoxes.length) {
    alert('Draw at least one box around a facecam first.');
    return;
  }
  const W = facecamImg.naturalWidth, H = facecamImg.naturalHeight;
  if (!W || !H) {
    alert("The frame hasn't finished loading yet -- try again in a second.");
    return;
  }
  const jobId = facecamJobId;
  const filename = facecamFilename;
  const boxes = facecamBoxes.map(b => ({
    x: Math.round(b.fx * W), y: Math.round(b.fy * H),
    w: Math.round(b.fw * W), h: Math.round(b.fh * H),
  }));
  const applyAll = facecamApplyAllRow.style.display !== 'none' && facecamApplyAll.checked;
  facecamGoBtn.disabled = true;
  facecamGoBtn.textContent = 'Starting...';
  try {
    const resp = await fetch(`/api/jobs/${jobId}/clips/${filename}/facecam-boxes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ boxes, apply_to_all_missing: applyAll }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || 'Could not start the re-render.');
      return;
    }
    lastFacecamSourceBoxes = boxes.map(b => [b.x, b.y, b.w, b.h]);
    facecamModal.classList.remove('open');
    facecamJobId = null;
    facecamFilename = null;
    loadJobsList();
    attachToJob(jobId);
  } finally {
    facecamGoBtn.disabled = false;
    facecamGoBtn.textContent = 'Re-render with these boxes';
  }
});

const youtubeUploadModal = document.getElementById('youtube-upload-modal-overlay');
const youtubeUploadTitleHint = document.getElementById('youtube-upload-title-hint');
const youtubeUploadPreview = document.getElementById('youtube-upload-preview');
const youtubeUploadGoBtn = document.getElementById('youtube-upload-go-btn');
const youtubeUploadTrimStart = document.getElementById('youtube-upload-trim-start');
const youtubeUploadTrimEnd = document.getElementById('youtube-upload-trim-end');
const youtubeUploadTrimHint = document.getElementById('youtube-upload-trim-hint');
const youtubeUploadSetStartBtn = document.getElementById('youtube-upload-set-start-btn');
const youtubeUploadSetEndBtn = document.getElementById('youtube-upload-set-end-btn');
let youtubeUploadJobId = null;
let youtubeUploadFilename = null;
// Seeded from the clip's recorded duration so the hint has a number to
// show immediately, then overwritten by the video element's own
// loadedmetadata duration once the preview loads -- that's the ground
// truth for what ffmpeg will actually trim against, not whatever got
// rounded into the job's metadata at render time.
let youtubeUploadDuration = 0;

function refreshYoutubeUploadTrimHint() {
  const trimStart = Math.max(0, parseFloat(youtubeUploadTrimStart.value) || 0);
  const trimEnd = Math.max(0, parseFloat(youtubeUploadTrimEnd.value) || 0);
  const resultSeconds = youtubeUploadDuration - trimStart - trimEnd;
  if (resultSeconds < 1) {
    youtubeUploadTrimHint.textContent =
      `Clip is ${youtubeUploadDuration.toFixed(1)}s -- that trim leaves ${resultSeconds.toFixed(1)}s, too short. Leave at least 1s.`;
    youtubeUploadGoBtn.disabled = true;
  } else {
    youtubeUploadTrimHint.textContent =
      `Clip is ${youtubeUploadDuration.toFixed(1)}s -- uploading ${resultSeconds.toFixed(1)}s after this trim.`;
    youtubeUploadGoBtn.disabled = false;
  }
}
youtubeUploadTrimStart.addEventListener('input', refreshYoutubeUploadTrimHint);
youtubeUploadTrimEnd.addEventListener('input', refreshYoutubeUploadTrimHint);

youtubeUploadSetStartBtn.addEventListener('click', () => {
  youtubeUploadTrimStart.value = youtubeUploadPreview.currentTime.toFixed(1);
  refreshYoutubeUploadTrimHint();
});
youtubeUploadSetEndBtn.addEventListener('click', () => {
  const remaining = Math.max(0, youtubeUploadDuration - youtubeUploadPreview.currentTime);
  youtubeUploadTrimEnd.value = remaining.toFixed(1);
  refreshYoutubeUploadTrimHint();
});

function openYoutubeUploadModal(jobId, clip) {
  youtubeUploadJobId = jobId;
  youtubeUploadFilename = clip.file;
  youtubeUploadDuration = clip.duration || 0;
  youtubeUploadTitleHint.textContent = `"${clip.upload_title || clip.title}"`;
  youtubeUploadPreview.src = `/api/jobs/${jobId}/clips/${clip.file}`;
  youtubeUploadPreview.onloadedmetadata = () => {
    if (youtubeUploadPreview.duration && isFinite(youtubeUploadPreview.duration)) {
      youtubeUploadDuration = youtubeUploadPreview.duration;
    }
    refreshYoutubeUploadTrimHint();
  };
  youtubeUploadTrimStart.value = '0';
  youtubeUploadTrimEnd.value = '0';
  refreshYoutubeUploadTrimHint();
  document.querySelector('input[name="youtube-privacy"][value="unlisted"]').checked = true;
  youtubeUploadModal.classList.add('open');
}

document.getElementById('youtube-upload-cancel-btn').addEventListener('click', () => {
  youtubeUploadModal.classList.remove('open');
  youtubeUploadJobId = null;
  youtubeUploadFilename = null;
  youtubeUploadPreview.pause();
  youtubeUploadPreview.removeAttribute('src');
  youtubeUploadPreview.load();
});

youtubeUploadGoBtn.addEventListener('click', async () => {
  if (!youtubeUploadJobId || !youtubeUploadFilename) return;
  const jobId = youtubeUploadJobId;
  const filename = youtubeUploadFilename;
  const privacyInput = document.querySelector('input[name="youtube-privacy"]:checked');
  const privacyStatus = privacyInput ? privacyInput.value : 'unlisted';
  const trimStart = Math.max(0, parseFloat(youtubeUploadTrimStart.value) || 0);
  const trimEnd = Math.max(0, parseFloat(youtubeUploadTrimEnd.value) || 0);
  if (trimStart + trimEnd >= youtubeUploadDuration) {
    alert('That trim would cut the whole clip -- leave at least a second.');
    return;
  }
  youtubeUploadGoBtn.disabled = true;
  youtubeUploadGoBtn.textContent = (trimStart || trimEnd) ? 'Trimming & uploading...' : 'Uploading...';
  try {
    const resp = await fetch(`/api/jobs/${jobId}/clips/${filename}/upload-youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ privacy_status: privacyStatus, trim_start: trimStart, trim_end: trimEnd }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      alert(data.detail || 'Upload failed.');
      return;
    }
    youtubeUploadModal.classList.remove('open');
    youtubeUploadJobId = null;
    youtubeUploadFilename = null;
    youtubeUploadPreview.pause();
    youtubeUploadPreview.removeAttribute('src');
    youtubeUploadPreview.load();
    alert(`Uploaded -- ${data.url}`);
  } catch (e) {
    alert('Upload failed.');
  } finally {
    youtubeUploadGoBtn.disabled = false;
    youtubeUploadGoBtn.textContent = 'Upload';
  }
});

// The part that actually *asks*: once a job is done, if any clip's
// facecam got rejected, open the picker for it right away instead of
// leaving a button to be noticed. Each clip is offered once per page
// load, so "Skip for now" is respected -- the button stays on the clip.
function maybePromptFacecam(jobId, job) {
  if (document.querySelector('#stop-modal-overlay.open, #mood-modal-overlay.open, #regen-modal-overlay.open, #facecam-modal-overlay.open, #youtube-upload-modal-overlay.open')) return;
  const next = (job.clips || []).find(c => c.facecam_uncertain && c.source_frame && !facecamPrompted.has(`${jobId}/${c.file}`));
  if (!next) return;
  facecamPrompted.add(`${jobId}/${next.file}`);
  openFacecamModal(jobId, next, facecamOthersMissing(job, next));
}

cancelBtn.addEventListener('click', () => {
  if (!currentJobId) return;
  stopModal.classList.add('open');
});

document.getElementById('stop-no-btn').addEventListener('click', () => {
  stopModal.classList.remove('open');
});

async function doStop(save) {
  stopModal.classList.remove('open');
  if (!currentJobId) return;
  pendingDeleteOnCancel = !save;
  cancelBtn.disabled = true;
  cancelBtn.textContent = save ? 'Saving & stopping...' : 'Stopping...';
  await fetch(`/api/jobs/${currentJobId}/cancel?save=${save}`, { method: 'POST' });
}

document.getElementById('stop-save-btn').addEventListener('click', () => doStop(true));
document.getElementById('stop-yes-btn').addEventListener('click', () => doStop(false));

async function poll(jobId) {
  const resp = await fetch(`/api/jobs/${jobId}`);
  if (!resp.ok) return;
  const job = await resp.json();
  statusEl.textContent = `[${job.state}] ${job.message || ''}`;
  if (job.error) statusEl.textContent += `\\nError: ${job.error}`;

  const pct = Math.round((job.progress || 0) * 100);
  progressBar.style.width = pct + '%';
  progressPct.textContent = pct + '%';
  if (job.estimate_minutes && job.created_at) {
    const elapsedMin = (Date.now() - job.created_at * 1000) / 60000;
    const remainingMin = Math.max(0, job.estimate_minutes - elapsedMin);
    progressEta.textContent = job.state === 'done' || job.state === 'error' || job.state === 'cancelled'
      ? ''
      : `~${job.estimate_minutes} min total, ~${remainingMin.toFixed(1)} min left`;
  } else {
    progressEta.textContent = 'estimating...';
  }

  clipsEl.innerHTML = '';
  (job.clips || []).forEach(c => {
    const div = document.createElement('div');
    div.className = 'clip';

    const heading = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = c.title;
    heading.appendChild(strong);
    heading.appendChild(document.createTextNode(` (${c.duration}s)`));
    div.appendChild(heading);

    if (c.hook_caption) {
      const em = document.createElement('em');
      em.textContent = c.hook_caption;
      div.appendChild(em);
    }

    // Watchable right here -- downloading is now an extra, optional step
    // (the button below), not the only way to see what got rendered.
    const preview = document.createElement('video');
    preview.controls = true;
    preview.preload = 'metadata';
    preview.style.cssText = 'width:100%;max-width:360px;border-radius:8px;background:var(--track);display:block;margin:8px 0';
    preview.src = `/api/jobs/${jobId}/clips/${c.file}`;
    div.appendChild(preview);

    const titleRow = document.createElement('div');
    titleRow.className = 'title-row';
    const titleInput = document.createElement('input');
    titleInput.readOnly = true;
    titleInput.value = c.upload_title || c.title;
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.textContent = 'Copy title';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(titleInput.value).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy title'; }, 1500);
      });
    });
    titleRow.appendChild(titleInput);
    titleRow.appendChild(copyBtn);
    div.appendChild(titleRow);

    if (c.description) {
      const descRow = document.createElement('div');
      descRow.className = 'title-row';
      const descInput = document.createElement('textarea');
      descInput.readOnly = true;
      descInput.rows = 3;
      descInput.value = c.description;
      const descCopyBtn = document.createElement('button');
      descCopyBtn.type = 'button';
      descCopyBtn.textContent = 'Copy description';
      descCopyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(descInput.value).then(() => {
          descCopyBtn.textContent = 'Copied!';
          setTimeout(() => { descCopyBtn.textContent = 'Copy description'; }, 1500);
        });
      });
      descRow.appendChild(descInput);
      descRow.appendChild(descCopyBtn);
      div.appendChild(descRow);
    }

    const link = document.createElement('a');
    link.href = `/api/jobs/${jobId}/clips/${c.file}?download=1`;
    link.setAttribute('download', '');
    link.textContent = `Download ${c.file}`;
    div.appendChild(link);

    if (job.state === 'done' || job.state === 'error' || job.state === 'cancelled') {
      if (!c.is_recap) {
        // Always offered, even with neither field set -- a clip old
        // enough to predate source_video tracking entirely still has a
        // chance: ensure-source-frame can infer the source from the job's
        // download folder when there's only one candidate for it. If that
        // fails too, the modal says so instead of the button just not
        // being there with no explanation.
        // A weekly-recap clip (is_recap) has no single source video to
        // pull a facecam frame from -- it's a concatenation of several --
        // so this button is skipped for it entirely rather than opening
        // a modal that can only ever fail.
        const fixBtn = document.createElement('button');
        fixBtn.type = 'button';
        fixBtn.textContent = c.facecam_uncertain ? '🎯 Fix facecam position'
          : (c.facecam_manual || c.facecam_trusted) ? '🎯 Adjust facecam position' : '🎯 Add facecam manually';
        fixBtn.style.marginLeft = '8px';
        fixBtn.addEventListener('click', () => openFacecamModal(jobId, c, facecamOthersMissing(job, c)));
        div.appendChild(fixBtn);
      }

      const uploadBtn = document.createElement('button');
      uploadBtn.type = 'button';
      uploadBtn.textContent = '📤 Upload to YouTube';
      uploadBtn.style.marginLeft = '8px';
      uploadBtn.addEventListener('click', () => openYoutubeUploadModal(jobId, c));
      div.appendChild(uploadBtn);

      const delClipBtn = document.createElement('button');
      delClipBtn.type = 'button';
      delClipBtn.textContent = '🗑 Delete this clip';
      delClipBtn.style.marginLeft = '8px';
      delClipBtn.addEventListener('click', async () => {
        if (!confirm(`Delete ${c.file}? This can't be undone.`)) return;
        delClipBtn.disabled = true;
        delClipBtn.textContent = 'Deleting...';
        try {
          const r = await fetch(`/api/jobs/${jobId}/clips/${c.file}`, { method: 'DELETE' });
          if (!r.ok) {
            const data = await r.json().catch(() => ({}));
            alert(data.detail || 'Could not delete this clip.');
            delClipBtn.disabled = false;
            delClipBtn.textContent = '🗑 Delete this clip';
            return;
          }
          poll(jobId);
        } catch (e) {
          alert('Could not delete this clip.');
          delClipBtn.disabled = false;
          delClipBtn.textContent = '🗑 Delete this clip';
        }
      });
      div.appendChild(delClipBtn);
    }

    clipsEl.appendChild(div);
  });

  if (job.state === 'done' || job.state === 'error' || job.state === 'cancelled') {
    if (timer) clearInterval(timer);
    setRunning(false);
    cancelBtn.disabled = false;
    cancelBtn.textContent = 'Emergency stop';
    deleteBtn.style.display = (job.clips || []).length ? 'block' : 'none';
    if (job.state === 'done') maybePromptFacecam(jobId, job);
    if (job.state === 'cancelled' && pendingDeleteOnCancel) {
      pendingDeleteOnCancel = false;
      fetch(`/api/jobs/${jobId}`, { method: 'DELETE' }).then(loadJobsList);
    } else {
      loadJobsList();
    }
  } else {
    deleteBtn.style.display = 'none';
  }
}

deleteBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  if (!confirm('Delete these clips from the server? This can\\'t be undone.')) return;
  deleteBtn.disabled = true;
  deleteBtn.textContent = 'Deleting...';
  const resp = await fetch(`/api/jobs/${currentJobId}`, { method: 'DELETE' });
  if (resp.ok) {
    clipsEl.innerHTML = '';
    statusEl.textContent = 'Deleted.';
    deleteBtn.style.display = 'none';
    currentJobId = null;
    loadJobsList();
  } else {
    deleteBtn.textContent = "🗑 I've downloaded these — delete from server";
  }
  deleteBtn.disabled = false;
});

document.getElementById('analytics-link-btn').addEventListener('click', () => {
  window.location.href = '/analytics';
});

const weeklyRecapBtn = document.getElementById('weekly-recap-btn');
const weeklyRecapViewBtn = document.getElementById('weekly-recap-view-btn');
const weeklyRecapStatus = document.getElementById('weekly-recap-status');
let latestWeeklyRecapJobId = null;

// Finds the most recent weekly-recap job (if any) and updates the "Go to
// this week's recap" button for it -- so it's reachable any time the
// page is opened, not just right after clicking "Generate", including
// the recap the Monday scheduler builds on its own with nobody watching,
// and while one is still building (downloading/transcribing/rendering
// several Twitch clips can take minutes -- see webapp/main.py's
// _run_weekly_recap_job) so its live progress is always one click away.
// Left visible-but-disabled rather than hidden when none exists yet, so
// the feature itself is never invisible -- just says plainly there's
// nothing to jump to.
async function refreshWeeklyRecapViewBtn() {
  try {
    const resp = await fetch('/api/jobs');
    if (!resp.ok) return;
    const { jobs } = await resp.json();
    const latest = jobs.find(j => j.pipeline === 'weekly_recap');
    latestWeeklyRecapJobId = latest ? latest.id : null;
    weeklyRecapViewBtn.disabled = !latest;
    if (!latest) {
      weeklyRecapViewBtn.textContent = '📺 No weekly recap yet';
    } else if (latest.state === 'error') {
      weeklyRecapViewBtn.textContent = '⚠ Last recap attempt failed -- view details';
    } else if (['done', 'cancelled'].includes(latest.state)) {
      weeklyRecapViewBtn.textContent = "📺 Go to this week's recap";
    } else {
      weeklyRecapViewBtn.textContent = '⏳ Recap building -- view progress';
    }
  } catch (e) {
    // leave the button as-is -- a failed check here shouldn't reset an
    // already-known recap or spam an error for a background refresh
  }
}
weeklyRecapViewBtn.addEventListener('click', () => {
  if (latestWeeklyRecapJobId) attachToJob(latestWeeklyRecapJobId);
});
refreshWeeklyRecapViewBtn();
setInterval(refreshWeeklyRecapViewBtn, 5000);

weeklyRecapBtn.addEventListener('click', async () => {
  weeklyRecapBtn.disabled = true;
  weeklyRecapStatus.textContent = '';
  try {
    const resp = await fetch('/api/weekly-recap/generate', { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      weeklyRecapStatus.textContent = data.detail || 'Could not queue the recap.';
    } else {
      // Queuing is near-instant -- the actual build runs as a background
      // job (same as a normal clip job), so jump straight to its live
      // progress rather than waiting here for it to finish.
      await loadJobsList();
      await refreshWeeklyRecapViewBtn();
      attachToJob(data.job_id);
    }
  } catch (e) {
    weeklyRecapStatus.textContent = 'Could not reach the server.';
  } finally {
    weeklyRecapBtn.disabled = false;
  }
});

const notifyTestBtn = document.getElementById('notify-test-btn');
notifyTestBtn.addEventListener('click', async () => {
  notifyTestBtn.disabled = true;
  notifyTestBtn.textContent = 'Sending...';
  const resp = await fetch('/api/notify-test', { method: 'POST' });
  notifyTestBtn.textContent = resp.ok
    ? '✅ Sent -- check Telegram'
    : '❌ Failed -- check CLIPPER_BOT_API is set and message the bot first';
  setTimeout(() => {
    notifyTestBtn.textContent = '🔔 Test Telegram notification';
    notifyTestBtn.disabled = false;
  }, 3000);
});
</script>
</body>
</html>
"""


ANALYTICS_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper — analytics</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f2f3f7;
    --card: #ffffff;
    --text: #1a1b1f;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #6d28d9;
    --accent2: #ec4899;
    --accent-text: #ffffff;
    --danger: #dc2626;
    --danger-hover: #b91c1c;
    --track: #e5e7eb;
    --shadow: 0 1px 2px rgba(16,24,40,0.04), 0 8px 24px rgba(16,24,40,0.06);
    /* Chart series colors -- "your channel" gets the app's own accent
       (it's not a competitor among competitors, so it sits outside the
       categorical rotation); competitor channels are assigned blue, aqua,
       yellow, magenta in that order, skipping orange deliberately -- orange
       next to yellow is the one adjacent pair in this palette that fails
       colorblind-safety, so a 5th competitor folds into a repeat rather
       than ever seat orange beside yellow. */
    --chart-you: #4a3aa7;
    --chart-c1: #2a78d6;
    --chart-c2: #1baf7a;
    --chart-c3: #eda100;
    --chart-c4: #e87ba4;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115;
      --card: #1a1c23;
      --text: #f2f3f7;
      --muted: #9aa0ac;
      --border: #2b2e37;
      --track: #2b2e37;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.4);
      --chart-you: #9085e9;
      --chart-c1: #3987e5;
      --chart-c2: #199e70;
      --chart-c3: #c98500;
      --chart-c4: #d55181;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 40px 16px;
  }
  .page { max-width: 640px; margin: 0 auto; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 28px 28px 32px;
  }
  .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
  .brand .logo {
    font-size: 1.4rem; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
  }
  h1 { font-size: 1.3rem; margin: 0; letter-spacing: -0.01em; }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin: 4px 0 24px; }
  label { display: block; margin-top: 16px; font-size: 0.82rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.02em; }
  input, textarea {
    width: 100%; padding: 10px 12px; margin-top: 6px; font-size: 0.95rem;
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input:focus, textarea:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
  }
  button {
    margin-top: 18px; padding: 11px 20px; font-size: 0.95rem; font-weight: 600;
    cursor: pointer; border: none; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: var(--accent-text);
    transition: opacity 0.15s, transform 0.05s;
  }
  button:hover:not(:disabled) { opacity: 0.92; }
  button:active:not(:disabled) { transform: scale(0.98); }
  button:disabled { opacity: 0.45; cursor: default; }
  .hint { font-size: 0.78rem; color: var(--muted); font-weight: 400; margin-top: 2px; text-transform: none; letter-spacing: normal; }
  .back-link { display: inline-block; margin-bottom: 4px; color: var(--muted); font-size: 0.85rem; text-decoration: none; }
  .back-link:hover { color: var(--accent); }
  .section { margin-top: 28px; padding-top: 20px; border-top: 1px solid var(--border); }
  .section:first-of-type { margin-top: 20px; padding-top: 0; border-top: none; }
  #insights-body > div { margin-top: 6px; }
  #insights-body > button { margin-top: 12px; }
  #competitor-search-row { display: flex; gap: 8px; margin-top: 6px; }
  #competitor-search-row input { flex: 1; margin-top: 0; }
  #competitor-search-row button { margin-top: 0; padding: 0 16px; white-space: nowrap; }
  .competitor-card {
    display: flex; align-items: center; gap: 12px; padding: 10px 12px; margin-top: 8px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
  }
  .competitor-card img { width: 56px; height: 56px; border-radius: 8px; object-fit: cover; background: var(--track); flex-shrink: 0; }
  .competitor-card .info { flex: 1; min-width: 0; }
  .competitor-card .title { font-size: 0.88rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .competitor-card .meta { font-size: 0.75rem; color: var(--muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .competitor-card button { margin-top: 0; padding: 6px 12px; font-size: 0.8rem; flex-shrink: 0; }
  .competitor-card button.remove-btn { background: transparent; color: var(--muted); border: 1px solid var(--border); }
  .competitor-card button.remove-btn:hover:not(:disabled) { color: var(--danger); border-color: var(--danger); opacity: 1; }
  #ai-overview-body { margin-top: 10px; }
  .chart-block { margin-top: 18px; }
  .chart-block:first-child { margin-top: 8px; }
  .chart-title { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; font-weight: 700; }
  .chart-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .chart-title .hint { font-weight: 400; margin: 0; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .bar-label { flex: 0 0 42%; min-width: 0; }
  .bar-label .bar-title { font-size: 0.78rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-label .bar-sub { font-size: 0.68rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-track { flex: 1; height: 20px; background: var(--track); border-radius: 4px; }
  .bar-fill { height: 16px; margin-top: 2px; border-radius: 0 4px 4px 0; min-width: 3px; }
  .bar-value { flex: 0 0 auto; min-width: 46px; text-align: right; font-size: 0.75rem; color: var(--muted); font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div class="page">
<div class="card">

<a href="/" class="back-link">&larr; Back to clipper</a>
<div class="brand"><span class="logo">📊</span><h1>Analytics &amp; AI strategy</h1></div>
<p class="subtitle">Best day to post, what's working, and how you compare to channels clipping the same streamers.</p>

<div class="section">
  <label style="margin-top:0">📈 Channel insights — best day to post</label>
  <div id="insights-body"><div class="hint">Loading...</div></div>
</div>

<div class="section">
  <label style="margin-top:0">🏆 Your top 10 videos</label>
  <div id="own-top-videos"><div class="hint">Loading...</div></div>
</div>

<div class="section">
  <label style="margin-top:0">🔍 Compare against other clipping channels</label>
  <p class="hint">Search a streamer you clip to find channels already posting clips of them, then add a few as
    comparison points -- the AI overview below will contrast your top titles against theirs.</p>
  <div id="competitor-search-row">
    <input id="competitor-search-input" placeholder="Streamer name, e.g. jynxzi">
    <button id="competitor-search-btn" type="button">Search</button>
  </div>
  <div class="hint" id="competitor-search-status" style="display:none"></div>
  <div id="competitor-search-results"></div>
  <label style="margin-top:16px">Comparing against</label>
  <div id="competitor-saved-list"></div>
  <div id="competitor-top-videos"></div>
</div>

<div class="section">
  <label style="margin-top:0">🤖 AI strategy overview</label>
  <label style="margin-top:10px;display:flex;align-items:center;gap:8px;font-weight:normal;text-transform:none;letter-spacing:normal">
    <input id="ai-overview-analyze-content" type="checkbox" style="width:auto">
    🔬 Also read competitor clips' actual content (transcript + loudness), not just titles -- slower
  </label>
  <button id="ai-overview-btn" type="button">🤖 Get AI strategy overview</button>
  <button id="ai-overview-clear-btn" type="button" style="margin-left:8px">🗑 Clear saved analysis</button>
  <div id="ai-overview-body"></div>
</div>

</div>
</div>

<script>
function formatSeconds(s) {
  s = Math.round(s || 0);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

function el(tag, opts) {
  const node = document.createElement(tag);
  if (opts) {
    if (opts.text) node.textContent = opts.text;
    if (opts.className) node.className = opts.className;
    if (opts.href) node.href = opts.href;
  }
  return node;
}

function formatCompact(n) {
  n = n || 0;
  if (n >= 1000000) return (n / 1000000).toFixed(n >= 10000000 ? 0 : 1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'K';
  return String(n);
}

const DAY_NAMES_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

// Builds one labeled bar-chart block (title + up to 10 ranked bars) and
// returns it -- callers own placing it (one channel per block, so a page
// with several competitor channels appends several of these into one
// container). Single series per block, so per the dataviz color rules this
// needs no legend box -- the title + colored dot next to it already say
// what's plotted. A flat single hue per channel (not a sequential ramp) is
// enough: bar LENGTH already carries the magnitude, so color here only
// needs to carry which channel this block belongs to.
function buildTopVideosBlock(label, colorVar, videos, subtitle) {
  const block = document.createElement('div');
  block.className = 'chart-block';

  const titleRow = document.createElement('div');
  titleRow.className = 'chart-title';
  const dot = document.createElement('span');
  dot.className = 'chart-dot';
  dot.style.background = colorVar;
  titleRow.appendChild(dot);
  titleRow.appendChild(el('span', { text: label }));
  if (subtitle) titleRow.appendChild(el('span', { className: 'hint', text: subtitle }));
  block.appendChild(titleRow);

  // Too-new-to-judge videos are excluded here the same way they're excluded
  // from every other performance comparison in this app -- a video posted
  // hours ago hasn't earned its views yet, and ranking it by raw view count
  // against settled uploads would bury it near the bottom for being new,
  // not for underperforming.
  const top = (videos || [])
    .filter(v => !v.too_new_to_judge)
    .slice()
    .sort((a, b) => b.views - a.views)
    .slice(0, 10);

  if (!top.length) {
    block.appendChild(el('div', { className: 'hint', text: 'No settled uploads yet to rank.' }));
    return block;
  }

  const maxViews = top[0].views || 1;
  top.forEach(v => {
    const row = document.createElement('div');
    row.className = 'bar-row';

    const labelCol = document.createElement('div');
    labelCol.className = 'bar-label';
    labelCol.appendChild(el('div', { className: 'bar-title', text: v.title }));
    labelCol.appendChild(el('div', { className: 'bar-sub', text: `posted ${DAY_NAMES_SHORT[v.weekday]} · ${v.views_per_day}/day` }));
    row.appendChild(labelCol);

    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = Math.max(3, Math.round((v.views / maxViews) * 100)) + '%';
    fill.style.background = colorVar;
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el('div', { className: 'bar-value', text: formatCompact(v.views) }));
    block.appendChild(row);
  });
  return block;
}

const CHART_COMPETITOR_COLORS = ['var(--chart-c1)', 'var(--chart-c2)', 'var(--chart-c3)', 'var(--chart-c4)'];

async function loadChannelInsights() {
  const body = document.getElementById('insights-body');
  body.innerHTML = '';
  let data;
  try {
    const resp = await fetch('/api/channel-insights');
    data = await resp.json();
  } catch (e) {
    body.appendChild(el('div', { className: 'hint', text: 'Could not load channel insights.' }));
    return;
  }

  if (data.channel_title) {
    body.appendChild(el('div', { text: `Channel: ${data.channel_title}` }));
  }

  if (data.analytics) {
    const a = data.analytics;
    body.appendChild(el('div', {
      text: a.best_day
        ? `📅 Best day to post (last ${a.lookback_days}d, real Analytics data): ${a.best_day}`
        : `Not enough Analytics data yet over the last ${a.lookback_days} days.`,
    }));
    body.appendChild(el('div', { className: 'hint', text: `Views by day: ${Object.entries(a.views_by_day).map(([d, v]) => `${d} ${v}`).join(' · ')}` }));
    body.appendChild(el('div', { className: 'hint', text: `Avg view duration: ${formatSeconds(a.average_view_duration_seconds)} (${(a.average_view_percentage || 0).toFixed(0)}% of video) · Subs gained: ${a.subscribers_gained} · lost: ${a.subscribers_lost}` }));
    if (a.traffic_sources && a.traffic_sources.length) {
      body.appendChild(el('div', { className: 'hint', text: `Top traffic sources: ${a.traffic_sources.map(t => `${t.source} (${t.views})`).join(', ')}` }));
    }
  } else if (data.analytics_error) {
    body.appendChild(el('div', { className: 'hint', text: `Analytics: ${data.analytics_error}` }));
  }

  if (data.heuristic) {
    const h = data.heuristic;
    if (!data.analytics) {
      body.appendChild(el('div', {
        text: h.best_day_heuristic
          ? `📅 Best day to post (heuristic, from public view counts): ${h.best_day_heuristic}`
          : 'Not enough recent uploads yet to guess a best day.',
      }));
    }
    const subs = h.subscriber_count === null ? 'hidden' : h.subscriber_count;
    body.appendChild(el('div', { className: 'hint', text: `${h.video_count} videos · ${subs} subscribers · sampled ${h.recent_videos_sampled} recent upload(s)` }));
    body.appendChild(el('div', { className: 'hint', text: h.note }));
  } else if (data.heuristic_error) {
    body.appendChild(el('div', { className: 'hint', text: `Heuristic: ${data.heuristic_error}` }));
  }

  if (data.setup_needed) {
    body.appendChild(el('div', { className: 'hint', text: data.setup_needed }));
  }

  if (data.saved_strategy_notes_at) {
    const when = new Date(data.saved_strategy_notes_at * 1000).toLocaleDateString();
    body.appendChild(el('div', {
      className: 'hint',
      text: `🧠 Using saved strategy notes from ${when} to help pick clips -- generate a fresh overview below to update them.`,
    }));
  }

  if (data.oauth_configured) {
    const btn = el('button', { text: data.oauth_connected ? '🔌 Disconnect YouTube account' : '🔗 Connect YouTube account for real Analytics' });
    btn.type = 'button';
    if (data.oauth_connected) {
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        await fetch('/api/youtube/disconnect', { method: 'POST' });
        loadChannelInsights();
      });
    } else {
      btn.addEventListener('click', () => { window.location.href = '/auth/youtube/login'; });
    }
    body.appendChild(btn);
  } else {
    body.appendChild(el('div', { className: 'hint', text: 'YouTube OAuth isn\\'t configured on this deployment -- see the README for setup steps to enable real Analytics data.' }));
  }

  const ownTopVideos = document.getElementById('own-top-videos');
  ownTopVideos.innerHTML = '';
  ownTopVideos.appendChild(buildTopVideosBlock(
    'Your channel', 'var(--chart-you)', (data.heuristic || {}).recent_videos || [],
  ));
}
loadChannelInsights();

const aiOverviewBtn = document.getElementById('ai-overview-btn');
const aiOverviewBody = document.getElementById('ai-overview-body');
const aiOverviewAnalyzeContent = document.getElementById('ai-overview-analyze-content');
aiOverviewBtn.addEventListener('click', async () => {
  const analyzeContent = aiOverviewAnalyzeContent.checked;
  aiOverviewBtn.disabled = true;
  aiOverviewBtn.textContent = analyzeContent ? 'Reading competitor clips (this takes longer)...' : 'Thinking...';
  aiOverviewBody.innerHTML = '';
  try {
    const resp = await fetch('/api/channel-insights/overview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analyze_content: analyzeContent }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      aiOverviewBody.appendChild(el('div', { className: 'hint', text: data.detail || 'Could not generate an overview.' }));
    } else if (!data.overview || !data.overview.trim()) {
      aiOverviewBody.appendChild(el('div', { className: 'hint', text: 'Got an empty response -- try again.' }));
    } else {
      const pre = el('div', { text: data.overview });
      pre.style.whiteSpace = 'pre-wrap';
      pre.style.marginTop = '10px';
      pre.style.padding = '12px';
      pre.style.border = '1px solid var(--border)';
      pre.style.borderRadius = '8px';
      aiOverviewBody.appendChild(pre);
    }
    aiOverviewBody.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    aiOverviewBody.appendChild(el('div', { className: 'hint', text: 'Could not generate an overview.' }));
  } finally {
    aiOverviewBtn.disabled = false;
    aiOverviewBtn.textContent = '🤖 Get AI strategy overview';
  }
});

const aiOverviewClearBtn = document.getElementById('ai-overview-clear-btn');
aiOverviewClearBtn.addEventListener('click', async () => {
  if (!confirm('Clear the saved AI analysis? Future clip picks will stop using it until you generate a new one.')) return;
  aiOverviewClearBtn.disabled = true;
  try {
    await fetch('/api/channel-insights/overview', { method: 'DELETE' });
    aiOverviewBody.innerHTML = '';
    loadChannelInsights();
  } finally {
    aiOverviewClearBtn.disabled = false;
  }
});

(function handleYoutubeOAuthRedirect() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('youtube_connected') || params.has('youtube_error')) {
    if (params.has('youtube_error')) {
      alert('YouTube connection failed: ' + params.get('youtube_error'));
    }
    const url = new URL(window.location.href);
    url.searchParams.delete('youtube_connected');
    url.searchParams.delete('youtube_error');
    window.history.replaceState({}, '', url.toString());
  }
})();

const competitorSearchInput = document.getElementById('competitor-search-input');
const competitorSearchBtn = document.getElementById('competitor-search-btn');
const competitorSearchStatus = document.getElementById('competitor-search-status');
const competitorSearchResults = document.getElementById('competitor-search-results');
const competitorSavedList = document.getElementById('competitor-saved-list');
let savedCompetitors = [];

function buildCompetitorCard(c, opts) {
  const card = document.createElement('div');
  card.className = 'competitor-card';

  const img = document.createElement('img');
  if (c.thumbnail) img.src = c.thumbnail;
  card.appendChild(img);

  const info = document.createElement('div');
  info.className = 'info';
  const title = document.createElement('div');
  title.className = 'title';
  title.textContent = c.channel_title;
  info.appendChild(title);
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = opts.metaText;
  info.appendChild(meta);
  card.appendChild(info);

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = opts.btnText;
  if (opts.btnClass) btn.className = opts.btnClass;
  btn.disabled = !!opts.btnDisabled;
  btn.addEventListener('click', opts.onClick);
  card.appendChild(btn);

  return card;
}

function renderSavedCompetitors() {
  competitorSavedList.innerHTML = '';
  if (!savedCompetitors.length) {
    competitorSavedList.appendChild(el('div', { className: 'hint', text: 'No competitor channels added yet -- search a streamer above.' }));
    return;
  }
  savedCompetitors.forEach(c => {
    competitorSavedList.appendChild(buildCompetitorCard(
      { channel_title: c.channel_title, thumbnail: null },
      {
        metaText: 'Included in the AI overview comparison',
        btnText: 'Remove',
        btnClass: 'remove-btn',
        onClick: () => {
          savedCompetitors = savedCompetitors.filter(x => x.channel_id !== c.channel_id);
          persistCompetitors();
        },
      },
    ));
  });
}

async function persistCompetitors() {
  renderSavedCompetitors();
  try {
    await fetch('/api/competitor-channels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channels: savedCompetitors.map(c => ({ channel_id: c.channel_id, channel_title: c.channel_title })) }),
    });
  } catch (e) {
    // best-effort -- the list still reflects the change in this tab even if the save failed
  }
  loadCompetitorTopVideos();
}

async function loadCompetitorTopVideos() {
  const container = document.getElementById('competitor-top-videos');
  if (!savedCompetitors.length) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = '';
  container.appendChild(el('div', { className: 'hint', text: 'Loading competitor videos...' }));
  try {
    const resp = await fetch('/api/competitor-channels/insights');
    const data = await resp.json();
    const channels = data.channels || [];
    container.innerHTML = '';
    if (!channels.length) {
      container.appendChild(el('div', { className: 'hint', text: 'Could not load video data for the saved competitor channel(s).' }));
      return;
    }
    channels.forEach((snap, idx) => {
      const color = CHART_COMPETITOR_COLORS[idx % CHART_COMPETITOR_COLORS.length];
      container.appendChild(buildTopVideosBlock(
        snap.channel_title, color, snap.recent_videos || [],
      ));
    });
  } catch (e) {
    container.innerHTML = '';
    container.appendChild(el('div', { className: 'hint', text: 'Could not load competitor videos.' }));
  }
}

async function loadSavedCompetitors() {
  try {
    const resp = await fetch('/api/competitor-channels');
    const data = await resp.json();
    savedCompetitors = data.channels || [];
  } catch (e) {
    savedCompetitors = [];
  }
  renderSavedCompetitors();
  loadCompetitorTopVideos();
}
loadSavedCompetitors();

async function runCompetitorSearch() {
  const streamer = competitorSearchInput.value.trim();
  if (!streamer) return;
  competitorSearchResults.innerHTML = '';
  competitorSearchStatus.style.display = 'block';
  competitorSearchStatus.textContent = 'Searching YouTube...';
  competitorSearchBtn.disabled = true;
  try {
    const resp = await fetch(`/api/competitor-search?streamer=${encodeURIComponent(streamer)}`);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      competitorSearchStatus.textContent = data.detail || 'Search failed.';
      return;
    }
    const results = data.channels || [];
    if (!results.length) {
      competitorSearchStatus.textContent = `No clipping channels found for "${streamer}".`;
      return;
    }
    competitorSearchStatus.style.display = 'none';
    results.forEach(c => {
      const alreadyAdded = savedCompetitors.some(x => x.channel_id === c.channel_id);
      const subsText = c.subscriber_count != null ? `${c.subscriber_count} subs · ` : '';
      competitorSearchResults.appendChild(buildCompetitorCard(c, {
        metaText: `${subsText}top clip: "${c.sample_video_title}" (${c.sample_video_views} views)`,
        btnText: alreadyAdded ? 'Added' : '+ Add',
        btnDisabled: alreadyAdded,
        onClick: (e) => {
          if (savedCompetitors.some(x => x.channel_id === c.channel_id)) return;
          savedCompetitors.push({ channel_id: c.channel_id, channel_title: c.channel_title });
          persistCompetitors();
          e.currentTarget.textContent = 'Added';
          e.currentTarget.disabled = true;
        },
      }));
    });
  } catch (e) {
    competitorSearchStatus.textContent = 'Search failed -- try again.';
  } finally {
    competitorSearchBtn.disabled = false;
  }
}
competitorSearchBtn.addEventListener('click', runCompetitorSearch);
competitorSearchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') runCompetitorSearch();
});
</script>
</body>
</html>
"""
