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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from clipper.captions import build_ass, clean_hook_text, hook_line_ass
from clipper.download import _ffprobe_duration, download_video, is_url, probe_video
from clipper import hook_line
from clipper import longform
from clipper import longform_video
from clipper.highlights import fetch_vod_clips
from clipper.long_vod import gather_candidates, is_long_vod, quick_probe_accessible, select_and_map
from clipper.loud_moments import find_loud_moments
from clipper.reframe import (
    MAX_COCAM_TILES,
    LetterboxLayout,
    MultiCamSplitLayout,
    SplitLayout,
    center_crop_layout,
    compute_layout,
    layout_from_manual_boxes,
)
from clipper.render import render_clip, render_edited, trim_clip, overlay_hook_line
from clipper.edit_plan import EditPlan, Piece, clip_levels, plan_edit, remap_words
from clipper import facecam_vision
from clipper.select_moments import select_clips
from clipper.transcribe import Word, get_transcript
from clipper.trending import (
    get_trending_sections,
    search_creator,
    get_recommendation_candidates,
    parse_twitch_duration,
)
from clipper.notify import send_telegram
from clipper.channel_insights import get_channel_snapshot, MAX_SHORT_SECONDS
from clipper import channel_strategy
from clipper import clip_features
from clipper import clip_performance
from clipper import clip_registry
from clipper.channel_strategy import get_ai_overview
from clipper import competitor_discovery
from clipper import competitor_content
from clipper import youtube_analytics
from clipper import youtube_oauth
from clipper import youtube_upload
from clipper import thumbnail

BASE_DIR = Path(os.environ.get("CLIPPER_JOBS_DIR", "/tmp/clipper_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

JOB_META_NAME = "job.json"
TERMINAL_STATES = ("done", "error", "cancelled")

# A "channel profile" is one clip channel -- its tracked streamers, YouTube
# channel and on-clip branding. Only the main channel exists today (a
# Spanish channel was tried and removed); the plumbing stays so another
# channel can be added later. Adding one is just another entry here: a
# Twitch-logins env var, a display name to stamp on clips, an accent colour
# for the mascot so two channels' clips don't look identical, and its own
# YouTube OAuth token file so it can be connected independently.
CHANNEL_PROFILES: dict = {
    "main": {
        "label": "Main",
        "twitch_env": "TRENDING_TWITCH_LOGINS",
        "brand_name": os.environ.get("CLIPPER_BRAND_NAME", "Caught On Stream"),
        "mascot_accent": "00CCFF",
        "token_file": "_youtube_oauth_token.json",
        "page_path": "/",
        # Language Claude writes clip titles/hook text/descriptions in --
        # None leaves the selection prompt exactly as it always was.
        "output_language": None,
    },
}
DEFAULT_CHANNEL_PROFILE = "main"


def _profile_logins(profile: str) -> List[str]:
    cfg = CHANNEL_PROFILES.get(profile) or CHANNEL_PROFILES[DEFAULT_CHANNEL_PROFILE]
    return [l.strip().lower() for l in os.environ.get(cfg["twitch_env"], "").split(",") if l.strip()]


def _profile_or_default(profile: Optional[str]) -> str:
    return profile if profile in CHANNEL_PROFILES else DEFAULT_CHANNEL_PROFILE


def _profile_output_language(profile: Optional[str]) -> Optional[str]:
    return CHANNEL_PROFILES[_profile_or_default(profile)]["output_language"]


# Persists on the same volume job data lives on, so each connected YouTube
# account survives restarts/redeploys -- see clipper/youtube_oauth.py. One
# TokenStore per channel profile, each its own file, so connecting one
# channel's YouTube account never touches another's token.
_youtube_token_stores: dict = {
    profile: youtube_oauth.TokenStore(BASE_DIR / cfg["token_file"])
    for profile, cfg in CHANNEL_PROFILES.items()
}
_youtube_token_store = _youtube_token_stores[DEFAULT_CHANNEL_PROFILE]  # back-compat alias for the main profile
_channel_strategy_path = BASE_DIR / "_channel_strategy_history.json"
_competitor_channels_path = BASE_DIR / "_competitor_channels.json"
_reminder_scheduler_state_path = BASE_DIR / "_reminder_scheduler_state.json"
# Outlives the jobs themselves -- see clipper/clip_registry.py.
_clip_registry_path = BASE_DIR / "_clip_registry.json"
_retention_curves_path = BASE_DIR / "_retention_curves.json"


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


def _load_performance_notes() -> Optional[str]:
    """The measured comparisons from the Analyze page's "What your Shorts
    actually do" panel (cached, so usually free), for the AI to cite. None
    when they're unavailable -- everything that uses them works without."""
    try:
        performance = _gather_clip_performance()
    except Exception as e:  # noqa: BLE001 - never worth failing a job or an overview over
        print(f"[clip_performance] unavailable: {e}", flush=True)
        return None
    if not performance.get("available") or not performance.get("summary", {}).get("shorts"):
        return None
    return clip_performance.render_prompt_text(performance)


# CSRF state for the OAuth login flow: state -> (issued_at, channel_profile).
# Short-lived and in-memory is fine -- a login round-trip through Google
# takes seconds, not something that needs to survive a restart.
_youtube_oauth_states: dict[str, tuple] = {}


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
    min_len: float = 15.0
    max_len: float = 90.0
    # YouTube's auto-caption timestamps lag the actual audio noticeably --
    # Whisper aligns word timing to the audio itself, so default to it for
    # captions that don't look delayed. Slower, but accurate.
    whisper: bool = True
    # Burn each clip's hook_caption into its first few seconds (see
    # captions.build_ass). Kept per job so "generate more" matches.
    hook_text: bool = True
    # Cut quiet pauses and punch in on audio spikes (see clipper/edit_plan.py).
    pacing: bool = True
    # Open each clip on a ~1.4s flash of its loudest late moment.
    teaser: bool = False
    # The channel mascot in the corner and the channel name stamped beside it
    # for the first few seconds (captions.brand_dialogues).
    branding: bool = True
    # Render every clip as the whole scene (reframe.LetterboxLayout) instead
    # of auto-detecting a facecam. Any clip can still be switched to a
    # facecam split afterwards with the manual box-picker.
    irl_layout: bool = True
    # Which tracked-streamer list / YouTube account / on-clip brand this job
    # belongs to (see CHANNEL_PROFILES). Defaults to the original channel so
    # every existing client that doesn't send this keeps working unchanged.
    channel_profile: str = DEFAULT_CHANNEL_PROFILE


class RegenerateRequest(BaseModel):
    focus: Optional[str] = None
    num_clips: int = 3
    min_len: float = 15.0
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
    return {k: v for k, v in job.items() if k not in ("request", "pending_regenerate", "pending_manual_facecam", "pending_manual_letterbox")}


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


# How many of a Twitch VOD's viewer-made clips the job page shows,
# most-viewed first.
TWITCH_VIEWER_CLIPS_SHOWN = 10


def _twitch_viewer_clips(info) -> list:
    """The most-viewed clips Twitch viewers made (Clip button) of this exact
    VOD, slimmed to what the job page needs. Empty for
    anything that isn't a Twitch VOD, or when the lookup fails -- this is
    extra signal, never a reason for a job to fail."""
    extractor = (info.extractor or "") if info else ""
    if "twitch" not in extractor or "vod" not in extractor or not info.broadcaster_login or not info.id:
        return []
    try:
        clips = fetch_vod_clips(info.broadcaster_login, info.id, created_at=info.created_at, duration=info.duration)
    except Exception as e:
        print(f"[job] Twitch viewer clips lookup failed: {e}", flush=True)
        return []
    return [
        {
            "title": c.get("title") or "",
            "url": c.get("url") or "",
            "view_count": int(c.get("view_count") or 0),
            "duration": float(c.get("duration") or 0.0),
            "vod_offset": float(c["vod_offset"]),
            "creator_name": c.get("creator_name") or "",
            "thumbnail_url": c.get("thumbnail_url") or "",
        }
        for c in clips[:TWITCH_VIEWER_CLIPS_SHOWN]
    ]


def _run_job(job_id: str) -> None:
    with jobs_lock:
        pending_regenerate = jobs[job_id].pop("pending_regenerate", None)
        pending_manual_facecam = jobs[job_id].pop("pending_manual_facecam", None)
        pending_manual_letterbox = jobs[job_id].pop("pending_manual_letterbox", None)
    if pending_regenerate is not None:
        _run_regenerate(job_id, pending_regenerate)
        return
    if pending_manual_facecam is not None:
        _run_manual_facecam_render(job_id, pending_manual_facecam)
        return
    if pending_manual_letterbox is not None:
        _run_manual_letterbox_render(job_id, pending_manual_letterbox)
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
    profile = _profile_or_default(req.channel_profile)
    _set(job_id, hook_text=req.hook_text, pacing=req.pacing, teaser=req.teaser, branding=req.branding,
         channel_profile=profile, irl_layout=req.irl_layout)
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

    # Clips Twitch viewers already made of this VOD, shown on the job page
    # for reference only -- clip picking doesn't use this list.
    twitch_clips = _twitch_viewer_clips(info)
    if twitch_clips:
        _set(job_id, twitch_clips=twitch_clips)

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
            strategy_notes=_load_strategy_notes(), performance_notes=_load_performance_notes(),
            output_language=_profile_output_language(profile),
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
            strategy_notes=_load_strategy_notes(), performance_notes=_load_performance_notes(),
            output_language=_profile_output_language(profile),
        )
        render_items = [(dl.video_path, words, pick) for pick in picks]
        render_base = 0.6
        _set(job_id, pipeline="short", used_ranges=[[pick.start, pick.end] for pick in picks])

    if not render_items:
        _set(job_id, state="error", error="Model returned no usable picks.")
        return

    clips_meta = _render_all(
        job_id, out_dir, render_items, render_base, [],
        hook_text=req.hook_text, pacing=req.pacing, teaser=req.teaser, branding=req.branding,
        channel_profile=profile, irl_layout=req.irl_layout,
    )
    _progress(job_id, 1.0)
    _set(job_id, state="done", message=f"Done. {len(clips_meta)} clip(s).")


def _render_atomic(
    video_path: Path, start: float, end: float, layout, ass_path: Path, out_path: Path,
    out_w: int = 1080, out_h: int = 1920, plan: Optional[EditPlan] = None,
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
        # With a plan, the .ass was written for the edited timeline, so the
        # edits must be applied again on every re-render of this clip.
        render_edited(video_path, start, end, layout, plan, ass_path, tmp_path, out_w=out_w, out_h=out_h)
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
        f"hookline_{filename}",
        f"_hookline_{stem}.ass",
    ):
        (out_dir / name).unlink(missing_ok=True)


def _plan_edit(video_path: Path, pick, clip_words: List[Word], pacing: bool, teaser: bool) -> Optional[EditPlan]:
    """This clip's pacing edits, or None to render it straight through.
    Never fails a render: any problem planning just means no edits."""
    if not pacing and not teaser:
        return None
    try:
        levels = clip_levels(video_path, pick.start, pick.end)
        plan = plan_edit(clip_words, pick.start, pick.end, levels, teaser=teaser, max_len=MAX_SHORT_SECONDS)
        if not pacing:
            # Teaser only: the clip itself plays through uncut and unzoomed.
            whole = Piece(0.0, round(pick.end - pick.start, 3))
            plan = EditPlan(([plan.pieces[0]] if plan.teaser else []) + [whole], teaser=plan.teaser)
    except Exception as e:  # noqa: BLE001 - a clip without edits beats no clip
        print(f"[edit_plan] planning failed, rendering without edits: {e}", flush=True)
        return None
    if plan.is_trivial(pick.end - pick.start):
        return None
    print(
        f"[edit_plan] {pick.start:.1f}-{pick.end:.1f}s: cut {plan.cut_seconds:.1f}s of pauses "
        f"({len(plan.cuts)} cut(s)), {plan.zooms} zoom(s), teaser={'yes' if plan.teaser else 'no'} "
        f"-> {plan.duration:.1f}s", flush=True,
    )
    return plan


def _render_all(
    job_id: str, out_dir: Path, render_items: list, render_base: float, clips_meta: list,
    hook_text: bool = True, pacing: bool = True, teaser: bool = False, branding: bool = True,
    channel_profile: str = DEFAULT_CHANNEL_PROFILE, irl_layout: bool = True,
) -> list:
    """Render each (video_path, words, pick) item to clip_{n}.mp4, appending
    to clips_meta (already containing any earlier clips) and updating job
    progress/state as it goes. Shared by a fresh run and a regenerate run --
    they only differ in what render_items contains and whether clips_meta
    starts empty or with clips from a prior run. hook_text burns each pick's
    hook_caption into the top of its first few seconds; pacing and teaser
    control the edits in clipper/edit_plan.py; branding adds the channel
    mascot and name stamp -- channel_profile picks *which* channel's name/
    mascot colour (see CHANNEL_PROFILES). irl_layout renders every clip
    as the whole scene; off, each clip's facecam is auto-detected."""
    render_span = 1.0 - render_base
    start_index = len(clips_meta)
    profile_cfg = CHANNEL_PROFILES.get(channel_profile) or CHANNEL_PROFILES[DEFAULT_CHANNEL_PROFILE]
    brand_name = profile_cfg["brand_name"]
    mascot_accent = profile_cfg["mascot_accent"]
    cancel = lambda: _check_cancel(job_id)  # noqa: E731
    for i, (video_path, words, pick) in enumerate(render_items, start=1):
        cancel()
        out_index = start_index + i
        _set(job_id, state="rendering",
             message=f'Rendering clip {out_index}/{start_index + len(render_items)}: "{pick.title}"')
        _progress(job_id, render_base + render_span * ((i - 1) / len(render_items)))
        clip_words = [w for w in words if w.start >= pick.start and w.end <= pick.end]
        layout = (
            LetterboxLayout() if irl_layout
            else compute_layout(video_path, pick.start, pick.end, target_w=1080, target_h=1920)
        )
        out_path = out_dir / f"clip_{out_index:02d}.mp4"
        ass_path = out_dir / f"_clip_{out_index:02d}.ass"
        burned_hook = clean_hook_text(pick.hook_caption) if hook_text else ""
        plan = _plan_edit(video_path, pick, clip_words, pacing, teaser)
        if plan is not None:
            caption_words, caption_start, out_duration = remap_words(clip_words, pick.start, plan), 0.0, plan.duration
        else:
            caption_words, caption_start, out_duration = clip_words, pick.start, pick.end - pick.start
        build_ass(caption_words, caption_start, ass_path, hook_text=burned_hook, punchy=True, brand=branding,
                  brand_name=brand_name, mascot_accent=mascot_accent)
        try:
            _render_atomic(video_path, pick.start, pick.end, layout, ass_path, out_path, plan=plan)
        except RuntimeError as e:
            if plan is None:
                raise
            # The edits are a bonus; a failed edit pass costs this clip its
            # edits, never the clip -- or the rest of the job.
            print(f"[edit_plan] edited render of {out_path.name} failed, rendering it without edits: {e}", flush=True)
            plan = None
            caption_words, caption_start, out_duration = clip_words, pick.start, pick.end - pick.start
            build_ass(caption_words, caption_start, ass_path, hook_text=burned_hook, punchy=True, brand=branding,
                      brand_name=brand_name, mascot_accent=mascot_accent)
            _render_atomic(video_path, pick.start, pick.end, layout, ass_path, out_path)

        facecam_uncertain = False
        source_frame_name = None
        has_trusted_facecam = False
        final_layout = layout
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
                # A rejected facecam band is itself evidence there may be no
                # real facecam here at all -- exactly what happens when
                # detect_facecams mistakes a person in a continuous wide IRL
                # shot for a corner overlay (see reframe.compute_layout).
                # Falling back to a plain crop + asking the user to draw a
                # facecam box in that case makes no sense -- there's nothing
                # to draw a box around. Ask once more, specifically: is this
                # actually a wide multi-person scene? If so, letterbox it
                # instead and skip the manual-placement prompt entirely,
                # since there's now real evidence (a failed facecam render)
                # backing that call, not just the original pre-render guess.
                is_wide_scene = None
                try:
                    is_wide_scene = facecam_vision.detect_wide_scene(video_path, pick.start, pick.end)
                except Exception as e:
                    print(f"[render] wide-scene re-check after facecam rejection errored: {e}", flush=True)
                try:
                    fallback_layout = (
                        LetterboxLayout() if is_wide_scene else center_crop_layout(video_path, target_w=1080, target_h=1920)
                    )
                    _render_atomic(video_path, pick.start, pick.end, fallback_layout, ass_path, out_path, plan=plan)
                    final_layout = fallback_layout
                except Exception as e:
                    print(f"[render] fallback re-render also failed, keeping the original render: {e}", flush=True)
                facecam_uncertain = not is_wide_scene
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

        _record_clip(
            job_id, pick, caption_words, caption_start, out_duration, out_path, final_layout, burned_hook, plan,
            branding,
        )

        clips_meta.append({
            "file": out_path.name,
            "start": pick.start,
            "end": pick.end,
            # Length of the finished video -- shorter than end - start once
            # pauses are cut, a little longer with a teaser.
            "duration": round(out_duration, 2),
            "score": getattr(pick, "score", None),
            "moment_type": getattr(pick, "moment_type", None),
            "subscores": getattr(pick, "subscores", None),
            # Kept so a facecam/IRL re-render applies the same edits the
            # clip's caption file was timed for.
            "edit_plan": plan.to_dict() if plan else None,
            "title": pick.title,
            "hook_caption": pick.hook_caption,
            # What's actually burned into the video's opening, if anything.
            "hook_text": burned_hook or None,
            "branding": branding,
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
            # Rendered as the whole scene -- by default, or because a wide
            # multi-person shot was detected. The UI offers "switch to
            # facecam" for these.
            "is_irl_scene": isinstance(final_layout, LetterboxLayout),
            # The frame the manual box-picker draws on. Always saved now
            # (see above) so any clip's facecam can be overridden by hand,
            # not just ones the pipeline itself flagged as uncertain.
            "source_frame": source_frame_name,
        })
        _set(job_id, clips=list(clips_meta))
        _progress(job_id, render_base + render_span * (i / len(render_items)))
    return clips_meta


def _layout_name(layout) -> str:
    if isinstance(layout, MultiCamSplitLayout):
        return "multicam"
    if isinstance(layout, SplitLayout):
        return "split"
    if isinstance(layout, LetterboxLayout):
        return "letterbox"
    return "crop"


def _registry_id(job_id: str, clip: dict) -> str:
    return clip_registry.clip_id_for(job_id, clip["start"], clip["end"], clip.get("window_index"))


def _record_clip(
    job_id: str, pick, words: List[Word], words_start: float, out_duration: float, out_path: Path, layout,
    hook_text: str = "", plan: Optional[EditPlan] = None, branding: bool = False,
) -> None:
    """Save what this clip looks like to the clip registry, so its posted
    video's performance can later be compared against it. Never allowed to
    fail the render it's recording -- a clip with no record just sits out
    the comparisons."""
    try:
        with jobs_lock:
            job = jobs.get(job_id) or {}
            source_title, pipeline = job.get("source_title"), job.get("pipeline")
        clip_registry.upsert(_clip_registry_path, {
            "clip_id": clip_registry.clip_id_for(job_id, pick.start, pick.end, getattr(pick, "window_index", None)),
            "job_id": job_id,
            "file": out_path.name,
            "created_at": time.time(),
            "source_title": source_title,
            "pipeline": pipeline,
            "start": pick.start,
            "end": pick.end,
            "duration": round(out_duration, 2),
            "title": pick.title,
            "hook_caption": pick.hook_caption,
            "upload_title": pick.upload_title,
            "reason": pick.reason,
            "layout": _layout_name(layout),
            "hook_text": hook_text or None,
            "branding": branding,
            "score": getattr(pick, "score", None),
            "moment_type": getattr(pick, "moment_type", None),
            "subscores": getattr(pick, "subscores", None),
            "edits": (
                {"cut_seconds": round(plan.cut_seconds, 2), "zooms": plan.zooms, "teaser": plan.teaser}
                if plan else None
            ),
            # Measured on the finished video (edited timeline), since that's
            # what viewers actually see.
            "features": clip_features.measure_clip(words, words_start, words_start + out_duration, out_path),
        })
    except Exception as e:  # noqa: BLE001 - recording must never fail a render
        print(f"[clip_registry] could not record {out_path.name}: {e}", flush=True)


def _cached_words(raw_dir: Path) -> dict:
    """Word lists straight from a job's transcript caches, keyed by
    candidate-window index (None for the single-transcript pipeline).
    Cache files only -- the regenerate loaders fall back to re-running
    Whisper when a cache is missing, far too slow to do across old jobs."""
    result: dict = {}
    try:
        transcript = raw_dir / "transcript.json"
        if transcript.exists():
            data = json.loads(transcript.read_text(encoding="utf-8"))
            result[None] = [Word(**w) for w in data["words"]]
        candidates = raw_dir / "candidates.json"
        if candidates.exists():
            for c in json.loads(candidates.read_text(encoding="utf-8")):
                result[c["index"]] = [Word(**w) for w in c["words"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[clip_registry] unreadable transcript cache in {raw_dir}: {e}", flush=True)
    return result


def _backfill_clip_record(job_id: str, job: dict, clip: dict, words_by_window: Optional[dict] = None) -> None:
    """A registry record for a clip rendered before the registry existed,
    built from what its job still has on disk. Speech timing comes from the
    cached transcript; audio is measured later by _measure_missing_audio,
    and only for clips that actually got posted."""
    if words_by_window is None:
        words_by_window = _cached_words(BASE_DIR / job_id / "_source")
    words = words_by_window.get(clip.get("window_index"))
    features = {}
    if words is not None:
        clip_words = [w for w in words if w.start >= clip["start"] and w.end <= clip["end"]]
        features = clip_features.speech_features(clip_words, clip["start"], clip["end"])
    clip_registry.upsert(_clip_registry_path, {
        "clip_id": _registry_id(job_id, clip),
        "job_id": job_id,
        "file": clip.get("file"),
        # The real render time was never stored, and job.created_at resets
        # on every "generate more" -- unknown beats a wrong timestamp that
        # could rule out a genuine title match.
        "created_at": None,
        "source_title": job.get("source_title"),
        "pipeline": job.get("pipeline"),
        "start": clip["start"],
        "end": clip["end"],
        "duration": clip.get("duration") or round(clip["end"] - clip["start"], 2),
        "title": clip.get("title"),
        "hook_caption": clip.get("hook_caption"),
        "upload_title": clip.get("upload_title"),
        "reason": clip.get("reason"),
        # Crop vs. auto-letterbox was never stored per clip, so this can't
        # be reconstructed -- these clips just sit out the layout comparison.
        "layout": None,
        "features": features,
    })


def _backfill_clip_registry() -> None:
    known = {r["clip_id"] for r in clip_registry.all_records(_clip_registry_path)}
    with jobs_lock:
        job_items = [(job_id, dict(job), list(job.get("clips") or [])) for job_id, job in jobs.items()]
    for job_id, job, clips in job_items:
        missing = [
            c for c in clips
            if not c.get("is_recap") and c.get("start") is not None and c.get("end") is not None
            and _registry_id(job_id, c) not in known
        ]
        if not missing:
            continue
        words_by_window = _cached_words(BASE_DIR / job_id / "_source")
        for c in missing:
            _backfill_clip_record(job_id, job, c, words_by_window)


def _measure_missing_audio(records: list) -> None:
    """Audio features for posted clips that don't have them yet (clips
    backfilled from before the registry existed). Posted clips only: they're
    the only ones a measurement can be compared against, and each costs an
    ffmpeg pass. A clip whose audio can't be read is marked so it isn't
    retried on every page load."""
    for r in records:
        features = r.get("features") or {}
        if not r.get("video_id") or "opening_vs_median_db" in features or features.get("audio_unavailable"):
            continue
        path = BASE_DIR / (r.get("job_id") or "") / (r.get("file") or "")
        if not r.get("job_id") or not r.get("file") or not path.is_file():
            continue
        audio = clip_features.audio_features(path, r.get("duration") or 0.0)
        clip_registry.update_fields(
            _clip_registry_path, r["clip_id"],
            features={**features, **(audio or {"audio_unavailable": True})},
        )


def _published_ts(published_at: Optional[str]) -> Optional[float]:
    if not published_at:
        return None
    try:
        return datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _load_retention_curves(access_token: str, channel_id: str, videos: list) -> dict:
    """Retention curves for these videos, reusing the on-disk copy while
    it's fresh -- a Short's curve barely moves once it's a week old, so
    re-asking YouTube for all ~50 on every page load would only add seconds
    of waiting."""
    try:
        cache = json.loads(_retention_curves_path.read_text(encoding="utf-8")) if _retention_curves_path.exists() else {}
    except (OSError, ValueError):
        cache = {}
    now = time.time()

    def is_stale(v: dict) -> bool:
        entry = cache.get(v["id"])
        if entry is None:
            return True
        max_age = 6 * 3600 if (v.get("age_days") or 0) < 7 else 3 * 86400
        return now - entry.get("fetched_at", 0) > max_age

    def fetch(v: dict):
        # A day early: Analytics days aren't UTC days, and a start date
        # after the real publish date would cut off the first day's views.
        published = (_published_ts(v.get("published_at")) or now) - 86400
        start_date = datetime.date.fromtimestamp(published).isoformat()
        try:
            return v["id"], youtube_analytics.get_retention_curve(access_token, channel_id, v["id"], start_date)
        except Exception as e:  # noqa: BLE001 - one missing curve shouldn't blank the rest
            print(f"[clip_performance] retention curve for {v['id']} failed: {e}", flush=True)
            return v["id"], None

    todo = [v for v in videos if is_stale(v)]
    if todo:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for video_id, curve in pool.map(fetch, todo):
                if curve is not None:
                    cache[video_id] = {"fetched_at": now, "curve": curve}
        wanted = {v["id"] for v in videos}
        cache = {video_id: entry for video_id, entry in cache.items() if video_id in wanted}
        try:
            tmp = _retention_curves_path.with_name(f".{_retention_curves_path.name}.tmp-{threading.get_ident()}")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(_retention_curves_path)
        except OSError as e:
            print(f"[clip_performance] could not save the retention cache: {e}", flush=True)
    return {video_id: entry["curve"] for video_id, entry in cache.items()}


_clip_performance_cache: dict = {}
# Long enough that the Analyze page and the AI overview button share one
# fetch, short enough that a clip posted a few minutes ago shows up.
_CLIP_PERFORMANCE_TTL_SECONDS = 600


def _gather_clip_performance(refresh: bool = False) -> dict:
    """The connected channel's recent Shorts with their real retention
    curves, matched back to the clips this app made -- see
    clipper/clip_performance.py for what gets compared."""
    cached = _clip_performance_cache.get("data")
    if not refresh and cached and time.time() - _clip_performance_cache.get("at", 0) < _CLIP_PERFORMANCE_TTL_SECONDS:
        return cached
    if not youtube_oauth.is_configured():
        return {"available": False, "reason": "YouTube OAuth isn't configured on this deployment -- retention curves only come from a connected channel's own Analytics."}
    access_token = _youtube_token_store.get_valid_access_token()
    if not access_token:
        return {"available": False, "reason": "Connect your YouTube account (Channel insights, above) -- retention curves only exist for your own channel's videos."}
    own = youtube_analytics.get_own_channel(access_token)
    if not own:
        return {"available": False, "reason": "The connected Google account has no YouTube channel."}
    snapshot = get_channel_snapshot(own["id"], sample_size=50)
    if not snapshot:
        return {"available": False, "reason": "Couldn't list your channel's uploads -- is YOUTUBE_API_KEY set?"}
    videos = snapshot.get("recent_videos") or []

    _backfill_clip_registry()
    match_input = [
        {
            "id": v["id"], "title": v["title"], "duration_seconds": v.get("duration_seconds"),
            "published_ts": _published_ts(v.get("published_at")),
        }
        for v in videos
    ]
    for clip_id, video_id, posted in clip_registry.match_uploads(clip_registry.all_records(_clip_registry_path), match_input):
        clip_registry.link_video(_clip_registry_path, clip_id, video_id, "title_match", posted_duration=posted)
    _measure_missing_audio(clip_registry.all_records(_clip_registry_path))
    records = clip_registry.all_records(_clip_registry_path)

    metrics = youtube_analytics.get_video_retention(access_token, own["id"], lookback_days=365, max_videos=200)
    curves = _load_retention_curves(access_token, own["id"], videos)
    stats = clip_performance.build_stats(videos, metrics, curves, records)
    try:
        daily = youtube_analytics.get_daily_totals(access_token, own["id"])
        trend = clip_performance.build_trend(daily, stats["videos"])
    except Exception as e:  # noqa: BLE001 - the trend is extra; the rest of the panel stands without it
        print(f"[clip_performance] weekly trend unavailable: {e}", flush=True)
        trend = None
    data = {
        "available": True,
        "channel_title": own["title"],
        "clips_recorded": len(records),
        "clips_linked": sum(1 for r in records if r.get("video_id")),
        "generated_at": time.time(),
        **stats,
        "trend": trend,
    }
    _clip_performance_cache.update(at=time.time(), data=data)
    return data


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


def _clip_transcript_words(job_id: str, clip: dict) -> List[Word]:
    """The word-level transcript that falls inside one specific finished
    clip's start/end window -- recovered from the same _source/ cache a
    normal render already leaves behind (_cache_transcript for the short
    pipeline, _cache_candidates for long-VOD), reusing the exact loaders
    "generate more clips" uses. Powers the Hook Line page without
    re-downloading or re-transcribing anything. Returns [] if the source
    was deleted or never cached (e.g. a job from before this existed)."""
    raw_dir = BASE_DIR / job_id / "_source"
    window_index = clip.get("window_index")
    if window_index is not None:
        candidates = _load_candidates_for_regenerate(raw_dir)
        cand = next((c for c in (candidates or []) if c["index"] == window_index), None)
        words = cand["words"] if cand else []
    else:
        data = _load_transcript_for_regenerate(raw_dir)
        words = data["words"] if data else []
    return [w for w in words if w.start >= clip["start"] and w.end <= clip["end"]]


def _run_regenerate(job_id: str, req: dict) -> None:
    """Pick and render additional clips reusing a job's already-downloaded
    source (or already-downloaded candidate windows, for a long VOD)
    instead of re-downloading anything."""
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_source"
    cancel = lambda: _check_cancel(job_id)  # noqa: E731

    num_clips = max(1, int(req.get("num_clips") or 3))
    focus = req.get("focus") or None
    min_len = float(req.get("min_len") or 15.0)
    max_len = min(float(req.get("max_len") or 90.0), MAX_SHORT_SECONDS)
    reset_used = bool(req.get("reset_used"))

    with jobs_lock:
        job = jobs[job_id]
        pipeline = job.get("pipeline")
        source_title = job.get("source_title") or ""
        source_duration = job.get("duration") or 0.0
        existing_clips = list(job.get("clips") or [])
        hook_text = job.get("hook_text", True)
        pacing = job.get("pacing", True)
        teaser = job.get("teaser", False)
        # Jobs from before branding existed get it on their new clips too:
        # it's the channel's look now, not a per-job experiment.
        branding = job.get("branding", True)
        channel_profile = job.get("channel_profile", DEFAULT_CHANNEL_PROFILE)
        # Older jobs predate the flag; their new clips follow the current
        # default rather than the old auto-detected facecam behaviour.
        irl_layout = job.get("irl_layout", True)
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
            strategy_notes=_load_strategy_notes(), performance_notes=_load_performance_notes(),
            output_language=_profile_output_language(channel_profile),
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
            strategy_notes=_load_strategy_notes(), performance_notes=_load_performance_notes(),
            output_language=_profile_output_language(channel_profile),
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

    clips_meta = _render_all(
        job_id, out_dir, render_items, 0.15, existing_clips,
        hook_text=hook_text, pacing=pacing, teaser=teaser, branding=branding,
        channel_profile=channel_profile, irl_layout=irl_layout,
    )
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
            _render_atomic(video_path, clip["start"], clip["end"], layout, ass_path, out_dir / filename,
                           plan=EditPlan.from_dict(clip.get("edit_plan")))
        except Exception as e:  # noqa: BLE001 - one clip failing shouldn't lose the rest of the batch
            print(f"[render] manual facecam re-render of {filename} failed: {e}", flush=True)
            failed.append(f"{filename}: {e}")
            continue
        updated.append(filename)
        clip_registry.update_fields(_clip_registry_path, _registry_id(job_id, clip), layout=_layout_name(layout))
        with jobs_lock:
            for c in jobs[job_id].get("clips") or []:
                if c.get("file") == filename:
                    c["facecam_uncertain"] = False
                    c["facecam_manual"] = True
                    c["facecam_boxes"] = [list(b) for b in boxes]
                    c["is_irl_scene"] = False
        _persist(job_id)

    _progress(job_id, 1.0)
    if not updated:
        _set(job_id, state="error", error="Couldn't re-render with your facecam placement -- " + "; ".join(failed))
        return
    message = f"Done -- {len(updated)} clip(s) updated with your facecam placement."
    if failed:
        message += " Skipped " + "; ".join(failed)
    _set(job_id, state="done", message=message)


def _run_manual_letterbox_render(job_id: str, req: dict) -> None:
    """Re-render one clip as a plain letterboxed wide shot (see
    reframe.LetterboxLayout) -- the escape hatch for a clip auto-detection
    still got wrong: it thought there was a facecam here (see _render_all's
    post-render rejection path) when there really isn't, e.g. a genuine
    multi-person IRL scene it misread. Skips compute_layout/vision entirely
    -- a human who looked at the clip and said "this isn't a facecam" is
    more reliable than another detection pass would be. Reuses the clip's
    already-downloaded source and existing caption (.ass) file, same as
    _run_manual_facecam_render; only the layout and rendered video change."""
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_source"
    cancel = lambda: _check_cancel(job_id)  # noqa: E731
    filenames = list(req.get("filenames") or [])

    with jobs_lock:
        by_file = {c.get("file"): c for c in (jobs[job_id].get("clips") or [])}

    updated, failed = [], []
    for i, filename in enumerate(filenames):
        cancel()
        clip = by_file.get(filename)
        label = (clip or {}).get("title") or filename
        _set(job_id, state="rendering", message=f'Re-rendering as an IRL scene: "{label}"')
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
            _render_atomic(video_path, clip["start"], clip["end"], LetterboxLayout(), ass_path, out_dir / filename,
                           plan=EditPlan.from_dict(clip.get("edit_plan")))
        except Exception as e:  # noqa: BLE001 - one clip failing shouldn't lose the rest of the batch
            print(f"[render] manual IRL re-render of {filename} failed: {e}", flush=True)
            failed.append(f"{filename}: {e}")
            continue
        updated.append(filename)
        clip_registry.update_fields(_clip_registry_path, _registry_id(job_id, clip), layout="letterbox")
        with jobs_lock:
            for c in jobs[job_id].get("clips") or []:
                if c.get("file") == filename:
                    c["facecam_uncertain"] = False
                    c["facecam_trusted"] = False
                    c["facecam_manual"] = False
                    c["is_irl_scene"] = True
        _persist(job_id)

    _progress(job_id, 1.0)
    if not updated:
        _set(job_id, state="error", error="Couldn't re-render as an IRL scene -- " + "; ".join(failed))
        return
    message = f"Done -- {len(updated)} clip(s) re-rendered as an IRL scene."
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


def _cleanup_empty_recap_stubs() -> None:
    """One-time startup sweep for weekly-recap (and game-recap) job husks
    left behind by a clip deletion that predates delete_clip's own cleanup
    (see delete_clip) -- an empty "0 clip(s), Done" entry with nothing
    useful left to do with it, stuck in the jobs list until removed by
    hand. Runs once after _load_persisted_jobs so any stub already on disk
    clears itself on the next deploy instead of needing a manual
    "I've downloaded these" click per stub."""
    with jobs_lock:
        stale_ids = [
            job_id for job_id, job in jobs.items()
            if job.get("pipeline") in ("weekly_recap", "game_recap")
            and job.get("state") in TERMINAL_STATES
            and not (job.get("clips") or [])
        ]
        for job_id in stale_ids:
            jobs.pop(job_id, None)
    for job_id in stale_ids:
        shutil.rmtree(BASE_DIR / job_id, ignore_errors=True)


_load_persisted_jobs()
_cleanup_empty_recap_stubs()
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
            "channel_profile": _profile_or_default(req.channel_profile),
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
        if job.get("pipeline") in ("weekly_recap", "game_recap"):
            # Recaps were removed; an old recap job left on the volume can
            # still be viewed and deleted, but not rebuilt.
            raise HTTPException(409, "recaps were removed -- this old recap job can only be downloaded or deleted")
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


@protected.post("/api/jobs/{job_id}/clips/{filename}/mark-irl")
def mark_clip_irl(job_id: str, filename: str) -> dict:
    """Re-render one clip as a plain letterboxed wide shot -- the override
    for when auto-detection (see reframe.compute_layout,
    facecam_vision.detect_wide_scene) still thought there was a facecam
    here and there really isn't, e.g. a genuine multi-person IRL scene it
    misread. No boxes needed, unlike facecam-boxes -- there's nothing to
    draw a box around. Only ever targets the one clip clicked: unlike
    facecam placement, "this is IRL" doesn't generalize to other clips in
    the same job the way one fixed camera layout does."""
    if Path(filename).name != filename:
        raise HTTPException(400, "bad filename")
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
        job["pending_manual_letterbox"] = {"filenames": [filename]}
        job["state"] = "queued"
        job["message"] = "Queued -- re-rendering as an IRL scene"
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
    # Claude's generated title/description, as edited in the upload modal --
    # None (not sent, e.g. an older client) falls back to the clip's stored
    # values; an empty description is a deliberate clear, not "unset".
    title: Optional[str] = None
    description: Optional[str] = None


@protected.post("/api/jobs/{job_id}/clips/{filename}/upload-youtube")
def upload_clip_to_youtube(job_id: str, filename: str, req: YouTubeUploadRequest) -> dict:
    """Post one already-rendered, already-hand-picked clip straight to the
    connected YouTube channel -- the manual "I've decided this one's going
    up" action, never a bulk or automatic publish. Uses the clip's
    already-generated upload_title/description, or the edited versions from
    the upload modal if the creator changed them before posting."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["state"] not in TERMINAL_STATES:
            raise HTTPException(409, "job is still running -- wait for it to finish first")
        clip = next((c for c in (job.get("clips") or []) if c.get("file") == filename), None)
        if clip is None:
            raise HTTPException(404, "clip not found")
        channel_profile = job.get("channel_profile", DEFAULT_CHANNEL_PROFILE)
    if channel_profile not in CHANNEL_PROFILES:
        # e.g. an old job from the removed Spanish channel -- never let its
        # clips fall through to the main channel's YouTube account.
        raise HTTPException(409, "This clip was made for a channel that no longer exists -- download it instead.")

    # A job's clips always go to the same channel they were generated for,
    # through that channel's own connected YouTube account.
    access_token = _youtube_token_stores[_profile_or_default(channel_profile)].get_valid_access_token()
    if not access_token:
        label = CHANNEL_PROFILES[_profile_or_default(channel_profile)]["label"]
        raise HTTPException(409, f"Connect the {label} YouTube account first.")

    path = BASE_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "clip file not found on disk")

    title = (req.title if req.title is not None else clip.get("upload_title") or clip.get("title") or filename).strip()
    if not title:
        raise HTTPException(400, "Title cannot be empty.")
    description = req.description if req.description is not None else (clip.get("description") or "")

    is_recap = bool(clip.get("is_recap"))
    posted_duration = round(float(clip.get("duration") or 0) - req.trim_start - req.trim_end, 2)
    if not is_recap and posted_duration > MAX_SHORT_SECONDS:
        raise HTTPException(
            400,
            f"This clip would upload at {posted_duration:.1f}s. Uploads from the app over {MAX_SHORT_SECONDS}s "
            f"land as regular videos, not Shorts -- trim at least {posted_duration - MAX_SHORT_SECONDS:.1f}s off, "
            "or download it and post it from your phone.",
        )

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

    try:
        video_id = youtube_upload.upload_video(
            access_token, upload_path,
            title=title,
            description=description,
            privacy_status=req.privacy_status,
            is_short=not is_recap,
        )
    except (youtube_upload.UploadError, ValueError) as e:
        raise HTTPException(502, str(e)) from e
    finally:
        if trimmed_path is not None:
            trimmed_path.unlink(missing_ok=True)

    # Link the posted video back to this clip so its real retention can be
    # compared against what the clip looked like (see /api/clip-performance).
    if not is_recap:
        clip_id = _registry_id(job_id, clip)
        if clip_registry.get_record(_clip_registry_path, clip_id) is None:
            _backfill_clip_record(job_id, job, clip)
        clip_registry.link_video(_clip_registry_path, clip_id, video_id, "upload", posted_duration=posted_duration)
    with jobs_lock:
        for c in (jobs.get(job_id) or {}).get("clips") or []:
            if c.get("file") == filename:
                c["youtube_video_id"] = video_id
                # Reflect what's actually live -- the modal may have edited
                # these from what Claude originally generated.
                c["upload_title"] = title
                c["description"] = description
    _persist(job_id)

    # A youtu.be link can open a Short in the regular player, which looks
    # just like a failed Shorts upload -- link Shorts to the Shorts player.
    url = f"https://youtu.be/{video_id}" if is_recap else f"https://youtube.com/shorts/{video_id}"
    return {"ok": True, "video_id": video_id, "url": url}


def _thumbnail_default_text(clip: dict) -> str:
    text = (clip.get("hook_caption") or clip.get("upload_title") or clip.get("title") or "").strip()
    return text.upper()


def _get_job_clip(job_id: str, filename: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        clip = next((c for c in (job.get("clips") or []) if c.get("file") == filename), None)
        if clip is None:
            raise HTTPException(404, "clip not found")
        return clip


@protected.get("/api/hook-line/clips")
def list_hook_line_clips() -> dict:
    """Every clip from a finished job that still has its source transcript
    cached on disk -- the pick list for the Hook Line page. Deliberately
    separate from the main /api/jobs listing: this is a picker over
    ALREADY-RENDERED clips, not a job-status view. Newest jobs first."""
    with jobs_lock:
        snapshot = [dict(j) for j in jobs.values() if j.get("state") == "done"]
    snapshot.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    items = []
    for job in snapshot:
        if not (BASE_DIR / job["id"] / "_source").is_dir():
            continue
        for clip in job.get("clips") or []:
            items.append({
                "job_id": job["id"],
                "source_title": job.get("source_title"),
                "file": clip.get("file"),
                "title": clip.get("title"),
                "hook_caption": clip.get("hook_caption"),
                "duration": clip.get("duration"),
            })
    return {"clips": items}


class HookLineGenerateRequest(BaseModel):
    job_id: str
    filename: str


@protected.post("/api/hook-line/generate")
def generate_hook_line_text(req: HookLineGenerateRequest) -> dict:
    """Ask Claude to write a spoiler-style flash-hook line from this clip's
    own (already-cached) transcript -- text only, doesn't render anything."""
    clip = _get_job_clip(req.job_id, req.filename)
    words = _clip_transcript_words(req.job_id, clip)
    if not words:
        raise HTTPException(409, "No transcript available for this clip -- the source may have been deleted.")
    try:
        text = hook_line.generate_hook_line(words, clip.get("title") or "")
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    return {"hook_text": text}


class HookLineRenderRequest(BaseModel):
    job_id: str
    filename: str
    hook_text: str


@protected.post("/api/hook-line/render")
def render_hook_line(req: HookLineRenderRequest) -> dict:
    """Burn the (possibly user-edited) hook_text onto the already-rendered
    clip as a second ffmpeg pass -- see clipper.render.overlay_hook_line.
    Never touches the original clip file; writes a new hookline_{file}
    alongside it, served by the existing clip-download route."""
    clip = _get_job_clip(req.job_id, req.filename)
    hook_text = req.hook_text.strip()
    if not hook_text:
        raise HTTPException(400, "hook_text is required")

    out_dir = BASE_DIR / req.job_id
    source_path = out_dir / clip["file"]
    if not source_path.is_file():
        raise HTTPException(404, "source clip not found on disk")

    out_name = f"hookline_{clip['file']}"
    out_path = out_dir / out_name
    ass_path = out_dir / f"_hookline_{Path(clip['file']).stem}.ass"
    dims = hook_line.clip_dimensions(source_path)
    ass_path.write_text(hook_line_ass(hook_text, hook_line.FLASH_SECONDS, play_res=dims), encoding="utf-8")

    tmp_path = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    try:
        overlay_hook_line(source_path, ass_path, tmp_path)
        os.replace(tmp_path, out_path)
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    with jobs_lock:
        job = jobs.get(req.job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        renders = [r for r in (job.get("hook_line_renders") or []) if r.get("source_file") != clip["file"]]
        renders.append({"file": out_name, "source_file": clip["file"], "hook_text": hook_text, "created_at": time.time()})
        job["hook_line_renders"] = renders
    _persist(req.job_id)
    return {"ok": True, "file": out_name}


@protected.post("/api/jobs/{job_id}/clips/{filename}/thumbnails")
def generate_thumbnails(job_id: str, filename: str) -> dict:
    """Render a handful of candidate downloadable thumbnails for one
    already-picked clip -- a real frame from the clip itself (not an
    AI-generated image) with one bold auto-written line of text burned
    over it. Candidate frames are picked from the clip's own loudest
    moments (a proxy for "the exciting part"), so the person reviewing
    clips gets a few genuinely different options to choose from rather
    than just whatever the midpoint happens to be."""
    clip = _get_job_clip(job_id, filename)
    path = BASE_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "clip file not found on disk")

    duration = float(clip.get("duration") or 0)
    if duration <= 0:
        raise HTTPException(400, "clip has no known duration")

    text = _thumbnail_default_text(clip)
    stem = path.stem
    frame_times = thumbnail.pick_thumbnail_frame_times(path, duration, n=4)
    thumbs = []
    for i, frame_time in enumerate(frame_times, start=1):
        out_path = path.with_name(f"{stem}_thumb{i}.jpg")
        try:
            thumbnail.render_thumbnail(path, frame_time, text, out_path)
        except RuntimeError as e:
            print(f"[thumbnail] candidate {i} failed for {filename}: {e}", flush=True)
            continue
        thumbs.append({"index": i, "frame_time": frame_time, "url": f"/api/jobs/{job_id}/clips/{filename}/thumbnails/{i}"})

    if not thumbs:
        raise HTTPException(500, "could not render any thumbnail candidates")
    return {"ok": True, "text": text, "thumbnails": thumbs}


class ThumbnailEditRequest(BaseModel):
    text: str
    frame_time: float


@protected.post("/api/jobs/{job_id}/clips/{filename}/thumbnails/{index}")
def regenerate_thumbnail(job_id: str, filename: str, index: int, req: ThumbnailEditRequest) -> dict:
    """Re-burn one already-picked candidate's frame with edited text --
    the user has chosen this one out of the batch generate_thumbnails
    made and wants to tweak the wording before downloading it."""
    _get_job_clip(job_id, filename)  # 404s if the job/clip don't exist
    path = BASE_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "clip file not found on disk")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text cannot be empty")

    out_path = path.with_name(f"{path.stem}_thumb{index}.jpg")
    try:
        thumbnail.render_thumbnail(path, req.frame_time, text, out_path)
    except RuntimeError as e:
        raise HTTPException(500, f"could not render thumbnail: {e}") from e
    return {"ok": True, "text": text, "url": f"/api/jobs/{job_id}/clips/{filename}/thumbnails/{index}"}


@protected.get("/api/jobs/{job_id}/clips/{filename}/thumbnails/{index}")
def get_thumbnail(job_id: str, filename: str, index: int, download: bool = False) -> FileResponse:
    if Path(filename).name != filename or filename.startswith("."):
        raise HTTPException(400, "bad filename")
    path = BASE_DIR / job_id / Path(filename).with_name(f"{Path(filename).stem}_thumb{index}.jpg")
    if not path.is_file():
        raise HTTPException(404, "thumbnail not found -- generate it first")
    if download:
        dl_name = f"{Path(filename).stem}_thumbnail.jpg"
        return FileResponse(path, media_type="image/jpeg", filename=dl_name)
    return FileResponse(path, media_type="image/jpeg")


# How often the Sunday reminder scheduler wakes up to check whether it's
# time -- hourly is frequent enough to land within an hour of the target
# time without a dedicated cron mechanism, and cheap enough to just poll.
_SCHEDULER_CHECK_SECONDS = 3600


# Sunday evening, UTC -- adjust _REMINDER_SCHEDULE_HOUR_UTC if this lands
# at an inconvenient local time.
_REMINDER_SCHEDULE_WEEKDAY = 6  # Sunday
_REMINDER_SCHEDULE_HOUR_UTC = 18


def _weekly_reminder_scheduler_loop() -> None:
    """Sends a plain Telegram nudge every Sunday to upload the week's
    clips -- a reminder for the creator's own regular posting habit. A
    persisted "last sent" ISO week (not just a sleep timer) survives a
    Railway restart/redeploy without either skipping a week or sending
    the reminder twice. Independent of
    whether a Telegram bot is actually configured -- send_telegram() is
    itself silent/best-effort on a missing token, so this thread doesn't
    need to check first."""
    while True:
        time.sleep(_SCHEDULER_CHECK_SECONDS)
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            if now.weekday() != _REMINDER_SCHEDULE_WEEKDAY or now.hour < _REMINDER_SCHEDULE_HOUR_UTC:
                continue
            iso_week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
            state = {}
            if _reminder_scheduler_state_path.exists():
                try:
                    state = json.loads(_reminder_scheduler_state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
            if state.get("last_sent_iso_week") == iso_week:
                continue
            send_telegram("\U0001F4C5 Sunday reminder -- remember to upload this week's clips!")
            state["last_sent_iso_week"] = iso_week
            _reminder_scheduler_state_path.write_text(json.dumps(state), encoding="utf-8")
        except Exception as e:
            print(f"[reminder] scheduler tick failed: {e}", flush=True)


threading.Thread(target=_weekly_reminder_scheduler_loop, daemon=True).start()


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
    focus, permanently locked out.

    A weekly-recap or game-recap job has exactly one "clip" -- the whole
    compilation -- so deleting it leaves nothing else in that job worth
    keeping (there's no source video or request to regenerate from, unlike
    a normal job). Removes the whole job in that case instead of leaving
    an empty "0 clip(s), Done" husk behind in the jobs list forever."""
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

        if not remaining and job.get("pipeline") in ("weekly_recap", "game_recap"):
            jobs.pop(job_id, None)
            cancel_events.pop(job_id, None)
            job_deleted = True
        else:
            job["clips"] = remaining
            job_deleted = False
            job["hook_line_renders"] = [
                r for r in (job.get("hook_line_renders") or []) if r.get("source_file") != filename
            ]
            if deleted.get("window_index") is not None:
                used_window_indices = set(job.get("used_window_indices") or [])
                used_window_indices.discard(deleted["window_index"])
                job["used_window_indices"] = list(used_window_indices)
            else:
                used_ranges = [tuple(r) for r in (job.get("used_ranges") or [])]
                target = (deleted.get("start"), deleted.get("end"))
                used_ranges = [r for r in used_ranges if r != target]
                job["used_ranges"] = [list(r) for r in used_ranges]

    if job_deleted:
        shutil.rmtree(BASE_DIR / job_id, ignore_errors=True)
    else:
        _persist(job_id)
        _remove_clip_files(BASE_DIR / job_id, filename)
    return {"ok": True, "clips": remaining, "job_deleted": job_deleted}


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


_trending_cache: dict = {}  # profile -> {"at": float, "sections": dict}
_TRENDING_CACHE_SECONDS = 180.0


@protected.get("/api/profiles")
def list_channel_profiles() -> dict:
    """Which channel profiles exist (see CHANNEL_PROFILES), and whether each
    one's YouTube account is currently connected -- backs the channel
    switcher in the UI."""
    return {
        "profiles": [
            {
                "id": profile,
                "label": cfg["label"],
                "brand_name": cfg["brand_name"],
                "twitch_logins": _profile_logins(profile),
                "youtube_connected": _youtube_token_stores[profile].is_connected(),
            }
            for profile, cfg in CHANNEL_PROFILES.items()
        ],
        "default": DEFAULT_CHANNEL_PROFILE,
    }


@protected.get("/api/trending")
def trending(profile: str = DEFAULT_CHANNEL_PROFILE) -> dict:
    """Four rows for the UI: configured creators' latest YouTube upload,
    configured creators' latest Twitch VOD, Twitch's biggest live streams
    globally, and popular Twitch creators not already on the watchlist
    (see clipper/trending.py). profile picks which channel's watchlist to
    use (see CHANNEL_PROFILES) -- defaults to the main channel. Cached
    briefly per profile so refreshing the page doesn't re-hit the Twitch/
    YouTube APIs (and YouTube's daily quota) every time."""
    profile = _profile_or_default(profile)
    cache = _trending_cache.setdefault(profile, {"at": 0.0, "sections": {}})
    now = time.time()
    if now - cache["at"] > _TRENDING_CACHE_SECONDS:
        try:
            # Only the main profile has a configured YouTube watchlist row
            # today -- a second profile just shows its own Twitch rows.
            youtube_channels = None if profile == DEFAULT_CHANNEL_PROFILE else []
            sections = get_trending_sections(twitch_logins=_profile_logins(profile), youtube_channels=youtube_channels)
        except Exception as e:
            print(f"[trending] lookup failed for profile {profile!r}: {e}", flush=True)
            sections = cache["sections"]
        cache["sections"] = sections
        cache["at"] = now
    return {
        name: [vars(e) for e in entries]
        for name, entries in cache["sections"].items()
    }


@protected.get("/api/vods/yesterday")
def yesterday_vods(profile: str = DEFAULT_CHANNEL_PROFILE) -> dict:
    """Yesterday's VODs (by UTC calendar date) from a channel profile's
    tracked streamers -- for when the streamers are big enough that
    browsing "what did they stream yesterday" and picking one by hand is the actual workflow, rather than the single
    latest-VOD row /api/trending shows. Reuses get_recommendation_candidates
    (the same Twitch lookup /api/recommend-vod uses) with a wider per-
    streamer pull and no AI ranking -- this just lists what's there for a
    human to pick from and feed into the normal /api/jobs flow, exactly
    like any other source URL."""
    from datetime import datetime, timezone, timedelta

    profile = _profile_or_default(profile)
    logins = _profile_logins(profile)
    if not logins:
        raise HTTPException(
            409,
            f"No streamers configured for this profile -- set {CHANNEL_PROFILES[profile]['twitch_env']} first.",
        )

    today_utc = datetime.now(timezone.utc).date()
    yesterday_utc = today_utc - timedelta(days=1)

    try:
        candidates = get_recommendation_candidates(logins, per_streamer=10, max_age_days=2.5)
    except Exception as e:
        raise HTTPException(502, f"Twitch lookup failed: {e}") from e

    results = []
    for c in candidates:
        if not c.published_at:
            continue
        try:
            published = datetime.fromisoformat(c.published_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published.astimezone(timezone.utc).date() != yesterday_utc:
            continue
        results.append(vars(c).copy())

    results.sort(key=lambda c: c.get("view_count") or 0, reverse=True)
    return {"date": yesterday_utc.isoformat(), "results": results}


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
def youtube_login(profile: str = DEFAULT_CHANNEL_PROFILE) -> RedirectResponse:
    """Kick off the OAuth flow for one channel profile's YouTube account
    (see clipper/youtube_oauth.py and CHANNEL_PROFILES) -- reads real
    Analytics data instead of just the public Data API's view counts. Both
    profiles share one registered Google OAuth client (just a different
    Google account consenting each time); which profile this login is for
    is carried through the round trip in _youtube_oauth_states, keyed by
    the CSRF state, since Google's redirect back doesn't let us pass our
    own query params through untouched."""
    profile = _profile_or_default(profile)
    if not youtube_oauth.is_configured():
        raise HTTPException(
            400,
            "YOUTUBE_OAUTH_CLIENT_ID / YOUTUBE_OAUTH_CLIENT_SECRET aren't set -- "
            "see the README for how to create them in Google Cloud Console.",
        )
    redirect_uri = _youtube_redirect_uri()
    state = secrets.token_urlsafe(24)
    _youtube_oauth_states[state] = (time.time(), profile)
    # Prune old, abandoned login attempts instead of growing forever --
    # this dict only ever holds a handful of entries for a single-tenant
    # app, so a plain sweep on every login is plenty.
    cutoff = time.time() - 600
    for s, (issued_at, _profile) in list(_youtube_oauth_states.items()):
        if issued_at < cutoff:
            _youtube_oauth_states.pop(s, None)
    return RedirectResponse(youtube_oauth.build_authorize_url(redirect_uri, state))


@protected.get("/auth/youtube/callback")
def youtube_callback(code: str = "", state: str = "", error: str = "") -> RedirectResponse:
    if error:
        return RedirectResponse(f"/analytics?youtube_error={error}")
    issued = _youtube_oauth_states.pop(state, None)
    if issued is None or time.time() - issued[0] > 600:
        raise HTTPException(400, "invalid or expired OAuth login attempt -- try connecting again")
    profile = _profile_or_default(issued[1])
    token = youtube_oauth.exchange_code(code, _youtube_redirect_uri())
    _youtube_token_stores[profile].save(token)
    # The main channel's connect flow also lives on the analytics page (it
    # reads real Analytics data there); every profile's own channel page
    # (see CHANNEL_PROFILES' page_path) shows connect status too, and is
    # where a second profile's connect button sends you from.
    if profile == DEFAULT_CHANNEL_PROFILE:
        return RedirectResponse("/analytics?youtube_connected=1")
    return RedirectResponse(f"{CHANNEL_PROFILES[profile]['page_path']}?youtube_connected=1")


@protected.post("/api/youtube/disconnect")
def youtube_disconnect(profile: str = DEFAULT_CHANNEL_PROFILE) -> dict:
    _youtube_token_stores[_profile_or_default(profile)].clear()
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
        # counts alone can't distinguish. get_video_retention isn't
        # duration-filtered, so the same lookup covers long-form videos
        # too -- arguably where retention matters most, since a long-form
        # video's whole draw is holding attention past the first few
        # seconds a Short lives or dies on.
        heuristic = result.get("heuristic") or {}
        recent_videos = heuristic.get("recent_videos")
        recent_long_form = heuristic.get("recent_long_form_videos")
        if recent_videos or recent_long_form:
            try:
                retention_by_id = youtube_analytics.get_video_retention(access_token, channel_id)
            except Exception as e:
                result["retention_error"] = str(e)
                retention_by_id = {}
            for v in (recent_videos or []) + (recent_long_form or []):
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

    # The overview has to cite these numbers rather than form its own
    # impression of the channel.
    performance_text = _load_performance_notes()

    try:
        overview = get_ai_overview(
            snapshot, data.get("analytics"), focus=req.focus,
            competitors=competitor_snapshots, content_analyses=content_analyses,
            performance_text=performance_text,
        )
    except Exception as e:
        raise HTTPException(502, f"Could not generate an overview: {e}") from e
    channel_strategy.save_overview(_channel_strategy_path, overview, channel_title=snapshot.get("channel_title"))
    return {"overview": overview}


@protected.get("/api/clip-performance")
def clip_performance_report(refresh: bool = False) -> dict:
    """What the channel's posted Shorts actually did -- real retention
    curves, and how uploads with different measured traits compare. Backs
    the "What your Shorts actually do" panel and feeds the AI overview."""
    try:
        return _gather_clip_performance(refresh=refresh)
    except Exception as e:  # noqa: BLE001 - show the reason on the page instead of a bare 500
        traceback.print_exc()
        return {"available": False, "reason": f"Could not load clip performance: {e}"}


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


@protected.get("/hook-line", response_class=HTMLResponse)
def hook_line_page() -> str:
    return HOOK_LINE_HTML


@protected.get("/long-form", response_class=HTMLResponse)
def long_form_page() -> str:
    return LONGFORM_HTML


# ---- Long-form videos (second channel) -------------------------------------
# One project per video: incident -> Claude script -> scene-by-scene recording
# with a misread check -> joined narration. See clipper/longform.py.
_longform_store = longform.ProjectStore(BASE_DIR / "_longform")
_longform_writing: set = set()  # project ids with a script being written right now
_longform_writing_lock = threading.Lock()


def _longform_project(pid: str) -> dict:
    try:
        project = _longform_store.load(pid)
    except (KeyError, OSError, ValueError):
        raise HTTPException(404, "long-form video not found")
    if project.get("status") == "writing":
        with _longform_writing_lock:
            alive = pid in _longform_writing
        if not alive:
            # The server restarted mid-write; the thread that would have
            # finished it is gone.
            project = _longform_store.update(pid, lambda pr: pr.update(
                status="error", error="Interrupted by a restart -- click Rewrite script."))
    if project.get("visuals_status") == "planning" and not _longform_is_busy(pid, "visuals"):
        project = _longform_store.update(pid, lambda pr: pr.update(
            visuals_status="error", visuals_error="Interrupted by a restart -- click Plan visuals again.", visuals_message=None))
    if (project.get("render") or {}).get("status") == "rendering" and not _longform_is_busy(pid, "render"):
        project = _longform_store.update(pid, lambda pr: pr.update(
            render={"status": "error", "error": "Interrupted by a restart -- click Render again.", "progress": 0}))
    return project


def _longform_write_script(pid: str, report_pdf: Optional[bytes] = None) -> None:
    """Background: get the report text (downloading the PDF if needed), then
    have Claude write the scene-by-scene script."""
    try:
        d = _longform_store.path(pid)
        text_path = d / "report.txt"
        if not text_path.exists():
            if report_pdf is None:
                url = (_longform_store.load(pid).get("incident") or {}).get("report_url")
                if not url:
                    raise RuntimeError("No report to write from -- upload the report PDF.")
                _longform_store.update(pid, lambda pr: pr.update(message="Downloading the NTSB report..."))
                report_pdf = longform.download_report(url)
            _longform_store.update(pid, lambda pr: pr.update(message="Reading the report..."))
            (d / "report.pdf").write_bytes(report_pdf)
            text = longform.pdf_text(report_pdf)
            if len(text) < 2000:
                raise RuntimeError("Couldn't read enough text from that report (it may be a scanned PDF).")
            text_path.write_text(text, encoding="utf-8")
        text = text_path.read_text(encoding="utf-8")
        title = (_longform_store.load(pid).get("incident") or {}).get("title") or "an aviation incident"
        _longform_store.update(pid, lambda pr: pr.update(message="Claude is writing the script (about a minute)..."))
        scenes = longform.write_script(text, title)

        def done(pr: dict) -> None:
            pr.update(status="script_ready", error=None, message=None, scenes=scenes, narration=None,
                      visuals_status=None, visuals_error=None, render=None, publish=None)
        _longform_store.update(pid, done)
    except Exception as e:
        print(f"[longform] script for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _longform_store.update(pid, lambda pr: pr.update(status="error", error=err, message=None))
        except Exception:
            pass
    finally:
        with _longform_writing_lock:
            _longform_writing.discard(pid)


def _longform_start_script(pid: str, report_pdf: Optional[bytes] = None) -> None:
    with _longform_writing_lock:
        if pid in _longform_writing:
            raise HTTPException(409, "the script is already being written")
        _longform_writing.add(pid)
    _longform_store.update(pid, lambda pr: pr.update(status="writing", error=None, message="Starting..."))
    threading.Thread(target=_longform_write_script, args=(pid, report_pdf), daemon=True).start()


class LongformCreateRequest(BaseModel):
    incident_id: Optional[str] = None
    report_url: Optional[str] = None
    title: Optional[str] = None


class LongformScene(BaseModel):
    narration: str
    visual: str = "stock"
    visual_note: str = ""


class LongformScenesRequest(BaseModel):
    scenes: List[LongformScene]


@protected.get("/api/longform/incidents")
def longform_incidents() -> dict:
    return {"incidents": longform.INCIDENTS}


@protected.get("/api/longform/projects")
def longform_projects() -> dict:
    return {"projects": [longform.summary(p) for p in _longform_store.list()]}


@protected.post("/api/longform/projects")
def longform_create(req: LongformCreateRequest) -> dict:
    if req.incident_id:
        match = next((i for i in longform.INCIDENTS if i["id"] == req.incident_id), None)
        if not match:
            raise HTTPException(404, "unknown incident")
        incident = {k: match[k] for k in ("id", "title", "subtitle", "report_url")}
    elif req.report_url:
        url = req.report_url.strip()
        if not url.startswith("https://") or not url.lower().split("?")[0].endswith(".pdf"):
            raise HTTPException(400, "paste a direct https link to the report PDF")
        incident = {"id": None, "title": (req.title or "").strip() or "Untitled incident",
                    "subtitle": url, "report_url": url}
    else:
        raise HTTPException(400, "pick an incident or paste a report link")
    project = _longform_store.create(incident)
    _longform_start_script(project["id"])
    return {"id": project["id"]}


@protected.post("/api/longform/projects/upload-report")
async def longform_create_from_pdf(request: Request, title: str = "") -> dict:
    data = await request.body()
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "that file isn't a PDF")
    if len(data) > longform.MAX_REPORT_BYTES:
        raise HTTPException(413, "that report is too large")
    project = _longform_store.create({"id": None, "title": title.strip() or "Untitled incident",
                                      "subtitle": "Uploaded report", "report_url": None})
    _longform_start_script(project["id"], report_pdf=data)
    return {"id": project["id"]}


@protected.get("/api/longform/projects/{pid}")
def longform_get(pid: str) -> dict:
    return _longform_project(pid)


@protected.delete("/api/longform/projects/{pid}")
def longform_delete(pid: str) -> dict:
    _longform_project(pid)
    _longform_store.delete(pid)
    return {"ok": True}


@protected.post("/api/longform/projects/{pid}/rewrite")
def longform_rewrite(pid: str) -> dict:
    _longform_project(pid)
    _longform_start_script(pid)
    return {"ok": True}


@protected.put("/api/longform/projects/{pid}/scenes")
def longform_save_scenes(pid: str, req: LongformScenesRequest) -> dict:
    """Save script edits. A scene whose words changed loses its recorded
    take (it no longer matches what was read); unchanged scenes keep theirs."""
    project = _longform_project(pid)
    if project.get("status") == "writing":
        raise HTTPException(409, "wait for the script to finish writing")
    new = []
    for sc in req.scenes:
        narration = " ".join(sc.narration.split())
        if narration:
            new.append({"narration": narration,
                        "visual": sc.visual if sc.visual in longform.VISUAL_TYPES else "stock",
                        "visual_note": " ".join(sc.visual_note.split())})
    if not new:
        raise HTTPException(400, "the script can't be empty")

    def apply(pr: dict) -> None:
        old = {s["narration"]: s for s in pr.get("scenes") or []}
        for sc in new:
            prev = old.get(sc["narration"]) or {}
            sc["take"] = prev.get("take")
            if prev.get("spec"):
                sc["spec"], sc["planned"] = prev["spec"], prev.get("planned")
        pr["scenes"] = new
        pr["narration"] = None
        pr["status"] = "script_ready"
    return _longform_store.update(pid, apply)


_TAKE_EXTS = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
              "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/aac": ".aac", "video/webm": ".webm"}


@protected.post("/api/longform/projects/{pid}/scenes/{index}/take")
async def longform_take(pid: str, index: int, request: Request) -> dict:
    """One recorded take of one scene: convert, transcribe, compare with the
    script. A take that matches is accepted; a flagged one is kept aside
    until it's re-read (or kept anyway)."""
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not 0 <= index < len(scenes):
        raise HTTPException(404, "scene not found")
    data = await request.body()
    if len(data) < 1000:
        raise HTTPException(400, "that recording is empty -- check your mic")
    if len(data) > longform.MAX_TAKE_BYTES:
        raise HTTPException(413, "that recording is too long for one scene")
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    ext = _TAKE_EXTS.get(ctype, ".webm")
    script = scenes[index]["narration"]
    takes_dir = _longform_store.path(pid) / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    raw = takes_dir / f"scene{index:02d}_{stamp}{ext}"
    wav = takes_dir / f"scene{index:02d}_{stamp}.wav"

    def work() -> dict:
        raw.write_bytes(data)
        try:
            longform.to_wav(raw, wav)
        finally:
            raw.unlink(missing_ok=True)
        duration = longform.audio_duration(wav)
        heard = longform.transcribe_take(wav)
        return {"duration": round(duration, 2), "heard": " ".join(heard).strip(),
                **longform.check_take(script, heard)}

    try:
        result = await run_in_threadpool(work)
    except Exception as e:
        wav.unlink(missing_ok=True)
        print(f"[longform] take check failed for {pid} scene {index}: {e}", flush=True)
        raise HTTPException(500, f"Couldn't check that take: {e}")

    take = {"file": wav.name, "recorded_at": time.time(), "kept": False, **result}
    replaced = []

    def apply(pr: dict) -> None:
        sc = (pr.get("scenes") or [])[index] if index < len(pr.get("scenes") or []) else None
        if sc is None or sc["narration"] != script:
            raise HTTPException(409, "the script changed while this take was being checked -- record it again")
        old = sc.get("take") or {}
        if old.get("file") and old["file"] != wav.name:
            replaced.append(old["file"])
        sc["take"] = take
        pr["narration"] = None
    try:
        _longform_store.update(pid, apply)
    except HTTPException:
        wav.unlink(missing_ok=True)
        raise
    for name in replaced:
        (takes_dir / name).unlink(missing_ok=True)
    return {"take": take}


@protected.post("/api/longform/projects/{pid}/scenes/{index}/keep")
def longform_keep_take(pid: str, index: int) -> dict:
    """Accept a flagged take as it is (e.g. Whisper misheard, not you)."""
    def apply(pr: dict) -> None:
        scenes = pr.get("scenes") or []
        if not 0 <= index < len(scenes) or not (scenes[index].get("take") or {}).get("file"):
            raise HTTPException(404, "no take recorded for that scene")
        scenes[index]["take"]["kept"] = True
    _longform_project(pid)
    return _longform_store.update(pid, apply)


@protected.get("/api/longform/projects/{pid}/scenes/{index}/take")
def longform_take_audio(pid: str, index: int) -> FileResponse:
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    take = (scenes[index].get("take") if 0 <= index < len(scenes) else None) or {}
    path = _longform_store.path(pid) / "takes" / (take.get("file") or "_none")
    if not take.get("file") or not path.is_file():
        raise HTTPException(404, "no take recorded for that scene")
    return FileResponse(path, media_type="audio/wav")


@protected.post("/api/longform/projects/{pid}/narration")
def longform_build_narration(pid: str) -> dict:
    """Join every scene's accepted take into one narration track, and record
    where each scene starts in it (what the visuals will be timed to)."""
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not scenes:
        raise HTTPException(409, "there's no script yet")
    missing = [i + 1 for i, s in enumerate(scenes) if not longform.scene_ready(s)]
    if missing:
        raise HTTPException(409, "record these scenes first: " + ", ".join(map(str, missing)))
    d = _longform_store.path(pid)
    wavs = [d / "takes" / s["take"]["file"] for s in scenes]
    starts, t = [], 0.0
    for i, s in enumerate(scenes):
        starts.append(round(t, 2))
        t += float(s["take"].get("duration") or 0.0) + longform.SCENE_GAP_SECONDS
    try:
        longform.join_takes(wavs, d / "narration.wav", d / "narration.m4a")
        duration = longform.audio_duration(d / "narration.wav")
    except Exception as e:
        raise HTTPException(500, f"Couldn't join the takes: {e}")
    narration = {"file": "narration.m4a", "duration": round(duration, 2), "scene_starts": starts,
                 "built_at": time.time()}
    return _longform_store.update(pid, lambda pr: pr.update(narration=narration))


@protected.get("/api/longform/projects/{pid}/narration")
def longform_narration_audio(pid: str) -> FileResponse:
    project = _longform_project(pid)
    path = _longform_store.path(pid) / "narration.m4a"
    if not project.get("narration") or not path.is_file():
        raise HTTPException(404, "build the narration first")
    title = (project.get("incident") or {}).get("title") or "narration"
    safe = "".join(ch if ch.isalnum() or ch in " -_" else "" for ch in title).strip()[:60] or "narration"
    return FileResponse(path, media_type="audio/mp4", filename=f"{safe} - narration.m4a")


# ---- Long-form visuals and render --------------------------------------------
# Claude plans each scene's visual from the report (plan), the creator
# adjusts any of them (the /visuals/{index} edits), then the whole video is
# rendered in the background, timed to the joined narration. See
# clipper/longform_video.py.
_longform_busy: set = set()  # (project id, "visuals" | "render") running right now
_longform_cache_dir = BASE_DIR / "_longform_cache"  # map tiles + stock downloads, shared across videos


def _longform_claim(pid: str, kind: str) -> None:
    with _longform_writing_lock:
        if (pid, kind) in _longform_busy:
            raise HTTPException(409, "already running -- wait for it to finish")
        _longform_busy.add((pid, kind))


def _longform_release(pid: str, kind: str) -> None:
    with _longform_writing_lock:
        _longform_busy.discard((pid, kind))


def _longform_is_busy(pid: str, kind: str) -> bool:
    with _longform_writing_lock:
        return (pid, kind) in _longform_busy


def _longform_report_images(pid: str, project: dict) -> list:
    """Pull the photos/diagrams out of the report PDF once (downloading the
    PDF again for videos started before it was kept)."""
    if project.get("report_images") is not None:
        return project["report_images"]
    d = _longform_store.path(pid)
    pdf_path = d / "report.pdf"
    if not pdf_path.exists():
        url = (project.get("incident") or {}).get("report_url")
        if not url:
            return []
        pdf_path.write_bytes(longform.download_report(url))
    images = longform_video.extract_report_images(pdf_path.read_bytes(), d / "report_images")
    _longform_store.update(pid, lambda pr: pr.update(report_images=images))
    return images


def _longform_plan_visuals(pid: str) -> None:
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        _longform_store.update(pid, lambda pr: pr.update(visuals_message="Pulling photos and diagrams out of the report..."))
        try:
            images = _longform_report_images(pid, project)
        except Exception as e:
            print(f"[longform] report images for {pid} failed: {e}", flush=True)
            images = []
        _longform_store.update(pid, lambda pr: pr.update(visuals_message="Claude is planning the visuals (about a minute)..."))
        text = (d / "report.txt").read_text(encoding="utf-8")
        title = (project.get("incident") or {}).get("title") or "an aviation incident"
        specs = longform_video.plan_visuals(text, title, project["scenes"])
        # Hand out the report's images in order, so report scenes don't all
        # show the same one.
        next_img = 0
        for spec in specs:
            if spec["visual"] in ("report", "stock") and images:
                spec["image_index"] = next_img % len(images)
                next_img += 1
            spec["rev"] = 1

        def done(pr: dict) -> None:
            for sc, spec in zip(pr.get("scenes") or [], specs):
                sc["spec"] = spec
                sc["planned"] = dict(spec)
            pr.update(visuals_status="ready", visuals_error=None, visuals_message=None, render=None)
        _longform_store.update(pid, done)
    except Exception as e:
        print(f"[longform] visuals for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _longform_store.update(pid, lambda pr: pr.update(visuals_status="error", visuals_error=err, visuals_message=None))
        except Exception:
            pass
    finally:
        _longform_release(pid, "visuals")


@protected.get("/api/longform/settings")
def longform_settings() -> dict:
    return {"stock_enabled": bool(os.environ.get("PEXELS_API_KEY")), "brand": longform_video.BRAND}


@protected.post("/api/longform/projects/{pid}/visuals/plan")
def longform_plan(pid: str) -> dict:
    project = _longform_project(pid)
    if not project.get("scenes") or project.get("status") != "script_ready":
        raise HTTPException(409, "write the script first")
    _longform_claim(pid, "visuals")
    _longform_store.update(pid, lambda pr: pr.update(visuals_status="planning", visuals_error=None, visuals_message="Starting..."))
    threading.Thread(target=_longform_plan_visuals, args=(pid,), daemon=True).start()
    return {"ok": True}


class LongformVisualEdit(BaseModel):
    visual: Optional[str] = None
    caption: Optional[str] = None
    lines: Optional[str] = None  # cockpit lines, one "SPEAKER: text" per line
    stock_query: Optional[str] = None
    image_step: int = 0
    stock_step: int = 0


def _scene_durations(project: dict) -> list:
    scenes = project.get("scenes") or []
    narr = project.get("narration") or {}
    starts = narr.get("scene_starts")
    if starts and len(starts) == len(scenes):
        bounds = list(starts) + [float(narr.get("duration") or starts[-1] + 20) + 0.8]
        return [max(1.0, bounds[i + 1] - bounds[i]) for i in range(len(scenes))]
    return [max(4.0, len(s["narration"].split()) / 2.5) for s in scenes]


@protected.put("/api/longform/projects/{pid}/visuals/{index}")
def longform_edit_visual(pid: str, index: int, req: LongformVisualEdit) -> dict:
    """Adjust one scene's visual: its type, caption, cockpit lines, which
    report image, or which stock clip."""
    project = _longform_project(pid)
    n_images = len(project.get("report_images") or [])

    def apply(pr: dict) -> None:
        scenes = pr.get("scenes") or []
        if not 0 <= index < len(scenes) or not scenes[index].get("spec"):
            raise HTTPException(404, "plan the visuals first")
        sc = scenes[index]
        spec = dict(sc["spec"])
        planned = sc.get("planned") or {}
        if req.visual and req.visual != spec.get("visual"):
            v = req.visual
            if v == planned.get("visual"):
                spec = dict(planned)
            elif v == "report":
                spec = {"visual": "report", "caption": spec.get("caption", ""), "report_hint": "", "image_index": index % max(1, n_images)}
            elif v == "stock":
                spec = {"visual": "stock", "caption": spec.get("caption", ""), "stock_query": sc.get("visual_note") or "airliner flying", "stock_index": 0,
                        "image_index": index % max(1, n_images)}
            elif v == "cockpit":
                spec = {"visual": "cockpit", "caption": spec.get("caption", ""), "time": "", "lines": []}
            else:
                raise HTTPException(400, "that visual needs data from the report -- use Re-plan visuals")
        if req.caption is not None:
            spec["caption"] = " ".join(req.caption.split())[:60]
        if req.lines is not None and spec.get("visual") == "cockpit":
            lines = []
            for raw in req.lines.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                speaker, _, text = raw.partition(":")
                if not text:
                    speaker, text = "", raw
                lines.append({"speaker": speaker.strip().upper()[:24], "text": text.strip().strip('"“”')[:220]})
            spec["lines"] = lines[:5]
        if req.stock_query is not None and spec.get("visual") == "stock":
            spec["stock_query"] = " ".join(req.stock_query.split())[:60] or spec.get("stock_query")
            spec["stock_index"] = 0
        if req.image_step and n_images:
            spec["image_index"] = ((spec.get("image_index") or 0) + req.image_step) % n_images
        if req.stock_step and spec.get("visual") == "stock":
            spec["stock_index"] = max(0, int(spec.get("stock_index") or 0) + req.stock_step)
        if spec.get("visual") == "cockpit" and not spec.get("lines"):
            spec.setdefault("lines", [])
        spec["rev"] = int(sc["spec"].get("rev") or 0) + 1
        sc["spec"] = spec
    return _longform_store.update(pid, apply)


@protected.get("/api/longform/projects/{pid}/visuals/{index}/preview")
async def longform_preview(pid: str, index: int) -> FileResponse:
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not 0 <= index < len(scenes) or not scenes[index].get("spec"):
        raise HTTPException(404, "plan the visuals first")
    spec = dict(scenes[index]["spec"])
    if spec.get("visual") == "cockpit" and not spec.get("lines"):
        spec["lines"] = [{"speaker": "", "text": "(add the cockpit lines for this scene)"}]
    dur = _scene_durations(project)[index]
    d = _longform_store.path(pid)
    out = d / "render" / "previews" / f"scene{index:02d}_{longform_video.spec_hash(spec, dur)}.jpg"
    if not out.exists():
        ctx = longform_video.SceneContext(d, (project.get("incident") or {}).get("title") or "", project.get("report_images") or [],
                                          _longform_cache_dir)
        try:
            await run_in_threadpool(longform_video.preview_still, spec, dur, ctx, index, out)
        except Exception as e:
            print(f"[longform] preview {pid}/{index} failed: {e}", flush=True)
            raise HTTPException(500, f"Couldn't draw this visual: {e}")
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


_MUSIC_EXTS = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mp4": ".m4a",
               "audio/x-m4a": ".m4a", "audio/aac": ".aac", "audio/ogg": ".ogg", "audio/flac": ".flac"}


@protected.post("/api/longform/projects/{pid}/music")
async def longform_upload_music(pid: str, request: Request, name: str = "") -> dict:
    """Optional background music (e.g. from the YouTube Audio Library), mixed
    quietly under the narration and ducked while you talk."""
    _longform_project(pid)
    data = await request.body()
    if len(data) < 1000:
        raise HTTPException(400, "that file is empty")
    if len(data) > 60 * 1024 * 1024:
        raise HTTPException(413, "that music file is too large")
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    ext = _MUSIC_EXTS.get(ctype) or (Path(name).suffix.lower() if Path(name).suffix.lower() in _MUSIC_EXTS.values() else ".mp3")
    d = _longform_store.path(pid)
    for old in d.glob("music.*"):
        old.unlink(missing_ok=True)
    (d / f"music{ext}").write_bytes(data)
    label = " ".join(Path(name).name.split())[:80] or f"music{ext}"
    return _longform_store.update(pid, lambda pr: pr.update(music={"file": f"music{ext}", "name": label}))


@protected.delete("/api/longform/projects/{pid}/music")
def longform_remove_music(pid: str) -> dict:
    _longform_project(pid)
    for old in _longform_store.path(pid).glob("music.*"):
        old.unlink(missing_ok=True)
    return _longform_store.update(pid, lambda pr: pr.update(music=None))


def _longform_render(pid: str) -> None:
    started = time.time()

    def progress(p: float, msg: str) -> None:
        _longform_store.update(pid, lambda pr: pr.update(render={**(pr.get("render") or {}), "status": "rendering",
                                                                 "progress": round(p, 3), "message": msg}))
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        narr = project["narration"]
        music = (project.get("music") or {}).get("file")
        final, credits = longform_video.render_video(
            d, project["scenes"], narr["scene_starts"], d / "narration.wav", float(narr["duration"]),
            (project.get("incident") or {}).get("title") or "", project.get("report_images") or [],
            d / music if music else None, _longform_cache_dir, on_progress=progress,
        )
        length = longform.audio_duration(final)
        _longform_store.update(pid, lambda pr: pr.update(render={
            "status": "done", "progress": 1.0, "message": None, "error": None, "built_at": time.time(),
            "duration": round(length, 2), "credits": credits, "took_seconds": round(time.time() - started),
        }))
    except Exception as e:
        print(f"[longform] render for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _longform_store.update(pid, lambda pr: pr.update(render={"status": "error", "error": err, "progress": 0}))
        except Exception:
            pass
    finally:
        _longform_release(pid, "render")


@protected.post("/api/longform/projects/{pid}/render")
def longform_start_render(pid: str) -> dict:
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not project.get("narration"):
        raise HTTPException(409, "join the narration first")
    if not scenes or not all(sc.get("spec") for sc in scenes):
        raise HTTPException(409, "plan the visuals first")
    _longform_claim(pid, "render")
    _longform_store.update(pid, lambda pr: pr.update(render={"status": "rendering", "progress": 0.0, "message": "Starting..."}))
    threading.Thread(target=_longform_render, args=(pid,), daemon=True).start()
    return {"ok": True}


@protected.get("/api/longform/projects/{pid}/video")
def longform_video_file(pid: str) -> FileResponse:
    project = _longform_project(pid)
    path = _longform_store.path(pid) / "final.mp4"
    if (project.get("render") or {}).get("status") != "done" or not path.is_file():
        raise HTTPException(404, "render the video first")
    title = (project.get("incident") or {}).get("title") or "video"
    safe = "".join(ch if ch.isalnum() or ch in " -_" else "" for ch in title).strip()[:60] or "video"
    return FileResponse(path, media_type="video/mp4", filename=f"{safe}.mp4")


@protected.post("/api/longform/projects/{pid}/publish-text")
async def longform_publish_text(pid: str) -> dict:
    """Three title options and a description with chapters and credits."""
    project = _longform_project(pid)
    narr = project.get("narration") or {}
    scenes = project.get("scenes") or []
    if not narr.get("scene_starts"):
        raise HTTPException(409, "join the narration first")
    incident = project.get("incident") or {}
    ref = incident.get("subtitle", "").split("·")[-1].strip() if incident.get("report_url") else ""
    credits = (project.get("render") or {}).get("credits") or []
    try:
        text = await run_in_threadpool(longform_video.write_publish_text, incident.get("title") or "", scenes,
                                       narr["scene_starts"], credits, ref)
    except Exception as e:
        raise HTTPException(500, f"Couldn't write the title and description: {e}")
    return _longform_store.update(pid, lambda pr: pr.update(publish=text))


app.include_router(protected)


_CHANNEL_HOME_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper — __PROFILE_LABEL__</title>
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
  .topnav {
    display: flex; flex-wrap: wrap; gap: 4px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px; margin-bottom: 16px; box-shadow: var(--shadow);
  }
  .topnav a {
    flex: 1 1 auto; text-align: center; padding: 9px 10px; border-radius: 9px;
    font-size: 0.84rem; font-weight: 600; color: var(--muted); text-decoration: none;
  }
  .topnav a:hover { color: var(--text); }
  .topnav a.active { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: var(--accent-text); }
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
  .longform-link {
    display: flex; justify-content: space-between; align-items: center; margin: -8px 0 22px; padding: 12px 16px;
    border: 1px solid var(--border); border-radius: 12px; background: var(--bg); color: var(--text);
    font-weight: 700; font-size: 0.92rem; text-decoration: none;
  }
  .longform-link:hover { border-color: var(--accent); color: var(--accent); }
  label { display: block; margin-top: 16px; font-size: 0.82rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.02em; }
  input, textarea, select {
    width: 100%; padding: 10px 12px; margin-top: 6px; font-size: 0.95rem;
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input:focus, textarea:focus, select:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
  }
  #profile-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 16px; }
  #profile-row .hint { margin: 0; }
  #profile-row button { width: auto; white-space: nowrap; padding: 8px 14px; margin-top: 0; font-size: 0.85rem; }
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
  #stop-modal-overlay, #mood-modal-overlay, #regen-modal-overlay, #facecam-modal-overlay, #youtube-upload-modal-overlay, #thumbnail-modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    align-items: center; justify-content: center; z-index: 100; padding: 16px;
  }
  #stop-modal-overlay.open, #mood-modal-overlay.open, #regen-modal-overlay.open, #facecam-modal-overlay.open, #youtube-upload-modal-overlay.open, #thumbnail-modal-overlay.open { display: flex; }
  #thumbnail-gallery { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 10px 0; }
  #thumbnail-gallery img { width: 100%; border-radius: 8px; display: block; cursor: pointer; border: 3px solid transparent; background: var(--track); }
  #thumbnail-gallery img.selected { border-color: var(--accent); }
  #thumbnail-gallery .thumb-loading { aspect-ratio: 9/16; background: var(--track); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 0.8rem; }
  #thumbnail-edit-panel img { width: 100%; max-width: 280px; border-radius: 8px; display: block; margin: 0 auto 10px; background: var(--track); }
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
  #twitch-clips { margin-top: 20px; }
  #twitch-clips .hint { margin: 4px 0 4px; }
  .twitch-clip {
    display: flex; gap: 12px; align-items: center; margin-top: 10px; padding: 10px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
    color: var(--text); text-decoration: none;
  }
  .twitch-clip:hover { border-color: var(--accent); }
  .twitch-clip img { width: 128px; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 8px; flex-shrink: 0; background: var(--border); }
  .twitch-clip-info { min-width: 0; }
  .twitch-clip-title { font-weight: 700; font-size: 0.92rem; overflow-wrap: anywhere; }
  .twitch-clip-meta { color: var(--muted); font-size: 0.8rem; margin-top: 3px; }
  .twitch-clip-rank { color: var(--accent); font-weight: 800; margin-right: 4px; }
  @media (max-width: 480px) { .twitch-clip img { width: 96px; } }
  .clip-primary-row { display: flex; gap: 8px; margin-top: 12px; }
  .clip-primary-row button { flex: 1; margin-top: 0; padding: 10px 14px; font-size: 0.86rem; }
  .clip-secondary-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
  .clip-secondary-row button {
    flex: 1; margin-top: 0; padding: 8px 10px; font-size: 0.78rem; font-weight: 600;
    background: transparent; color: var(--muted); border: 1px solid var(--border);
  }
  .clip-secondary-row button:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); opacity: 1; }
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
  .score-badge { display: inline-block; margin-left: 8px; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; background: var(--bg); border: 1px solid var(--border); vertical-align: middle; }
  .score-breakdown { display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 6px 0 2px; font-size: 0.78rem; opacity: 0.85; }
  .score-breakdown .overall { font-weight: 700; opacity: 1; }
  .score-badge.best { background: #fff4cc; border-color: #f2c200; color: #5c4700; }
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
__NAV_LINKS__
<div class="card">

<div class="brand"><span class="logo">🎬</span><h1>clipper — __PROFILE_LABEL__</h1></div>
<p class="subtitle">Paste a YouTube or Twitch link, get back short vertical highlight clips picked by Claude.</p>
<a class="longform-link" href="/long-form"><span>🛫 Go to long-form videos</span><span>→</span></a>

<div id="profile-row">
  <div class="hint" id="profile-youtube-status" style="margin-top:0"></div>
  <button id="profile-youtube-btn" type="button">Connect YouTube</button>
</div>

<div class="trending-section" id="yesterday-vods-section" style="display:none">
  <label style="margin-top:0">Yesterday's VODs</label>
  <div class="hint" id="yesterday-vods-status" style="display:none"></div>
  <div id="yesterday-vods-results"></div>
</div>

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
    <input id="min_len" type="number" value="15">
  </div>
  <div>
    <label>Max length (s)</label>
    <input id="max_len" type="number" value="45" max="60">
    <div class="hint">Capped at 60s -- past that, YouTube can silently upload it as a regular video instead of a Short.</div>
  </div>
</div>

<div class="checkbox-row">
  <input id="whisper" type="checkbox" checked>
  <label for="whisper">Accurate captions (Whisper)<div class="hint">Slower, but word timing is aligned to the audio. Uncheck to use YouTube's own captions instead (faster, but timing can lag the audio).</div></label>
</div>

<div class="checkbox-row">
  <input id="pacing" type="checkbox" checked>
  <label for="pacing">Tighten pacing<div class="hint">Cuts quiet pauses out of each clip so it never goes slow.</div></label>
</div>

<div class="checkbox-row">
  <input id="teaser" type="checkbox">
  <label for="teaser">Payoff teaser<div class="hint">Opens each clip on a 1-second flash of its biggest moment, then plays it from the start. A strong hook, but try it on a few clips before making it a habit.</div></label>
</div>

<div class="checkbox-row">
  <input id="branding" type="checkbox" checked>
  <label for="branding">Channel mascot + name<div class="hint">Puts the channel's mascot in the top-left corner of every clip, with the channel name next to it for the first 3 seconds, so viewers start to recognise the channel.</div></label>
</div>

<div class="checkbox-row">
  <input id="irl_layout" type="checkbox" checked>
  <label for="irl_layout">IRL layout (whole scene)<div class="hint">Shows the whole stream in every clip instead of splitting out the facecam. Any clip can be switched to facecam afterwards with its Switch to facecam button. Untick to auto-detect the facecam instead.</div></label>
</div>

<div class="checkbox-row">
  <input id="hook_text" type="checkbox" checked>
  <label for="hook_text">Hook text on screen<div class="hint">Puts a short line at the top of each clip for its first 3 seconds, saying why to keep watching. That's when viewers decide whether to swipe away.</div></label>
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
<div id="twitch-clips"></div>

<div id="jobs-panel">
  <label style="margin-top:0">Active &amp; saved jobs</label>
  <div id="jobs-list"></div>
</div>

<button id="notify-test-btn" type="button">🔔 Test Telegram notification</button>

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
      <button id="facecam-irl-btn" type="button" class="ghost">🎬 Use the IRL layout (whole scene)</button>
      <button id="facecam-cancel-btn" type="button" class="ghost">Skip for now</button>
    </div>
  </div>
</div>

<div id="youtube-upload-modal-overlay">
  <div class="modal" style="max-height:92vh;overflow:auto">
    <p>Upload to YouTube</p>
    <video id="youtube-upload-preview" controls preload="metadata" style="width:100%;max-height:40vh;border-radius:8px;background:var(--track);display:block;object-fit:contain"></video>
    <label>Title</label>
    <input id="youtube-upload-title-input" type="text" maxlength="100">
    <p class="hint" id="youtube-upload-title-counter"></p>
    <label>Description</label>
    <textarea id="youtube-upload-description-input" rows="3"></textarea>
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

<div id="thumbnail-modal-overlay">
  <div class="modal" style="max-width:520px;width:100%;max-height:92vh;overflow:auto">
    <p>Choose a thumbnail</p>
    <p class="hint">A real frame from the clip -- not AI-generated -- with one bold line of
      text burned over it. Pick one below, then edit the wording if you want.</p>
    <div id="thumbnail-gallery-panel">
      <div id="thumbnail-gallery"></div>
      <p class="hint" id="thumbnail-status-hint"></p>
    </div>
    <div id="thumbnail-edit-panel" style="display:none">
      <img id="thumbnail-edit-preview" alt="Selected thumbnail">
      <label>Text</label>
      <input id="thumbnail-edit-text" type="text" maxlength="80">
      <div class="modal-actions" style="margin-top:12px">
        <button id="thumbnail-regen-btn" type="button">Regenerate with this text</button>
        <button id="thumbnail-back-btn" type="button" class="ghost">&larr; Back to choices</button>
      </div>
      <a id="thumbnail-download-link" href="#" download style="display:inline-block;margin-top:12px">Download thumbnail</a>
    </div>
    <div class="modal-actions" style="margin-top:16px">
      <button id="thumbnail-close-btn" type="button" class="ghost">Close</button>
    </div>
  </div>
</div>

<script>
const statusEl = document.getElementById('status');
const clipsEl = document.getElementById('clips');
const deleteBtn = document.getElementById('delete-btn');
const twitchClipsEl = document.getElementById('twitch-clips');
let twitchClipsKey = '';

function fmtVodTime(sec) {
  const t = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), r = t % 60;
  const pad = n => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
}

function fmtViews(n) {
  n = n || 0;
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'K';
  return String(n);
}

// Twitch's own clips of the job's source VOD (made by viewers with the Clip
// button), most-viewed first. Rebuilt only when the list changes, not on
// every 2s poll, so the thumbnails don't flicker.
function renderTwitchClips(jobId, job) {
  const list = (job && job.twitch_clips) || [];
  const key = list.length ? `${jobId}:${list.map(c => c.url).join(',')}` : '';
  if (key === twitchClipsKey) return;
  twitchClipsKey = key;
  twitchClipsEl.innerHTML = '';
  if (!list.length) return;
  const heading = document.createElement('label');
  heading.textContent = '🔥 Most-viewed Twitch clips of this stream';
  twitchClipsEl.appendChild(heading);
  const hint = document.createElement('div');
  hint.className = 'hint';
  hint.textContent = 'Clips Twitch viewers made themselves with the Clip button.';
  twitchClipsEl.appendChild(hint);
  list.forEach((c, i) => {
    const a = document.createElement('a');
    a.className = 'twitch-clip';
    a.href = c.url;
    a.target = '_blank';
    a.rel = 'noopener';
    if (c.thumbnail_url) {
      const img = document.createElement('img');
      img.src = c.thumbnail_url;
      img.alt = '';
      img.loading = 'lazy';
      a.appendChild(img);
    }
    const info = document.createElement('div');
    info.className = 'twitch-clip-info';
    const title = document.createElement('div');
    title.className = 'twitch-clip-title';
    const rank = document.createElement('span');
    rank.className = 'twitch-clip-rank';
    rank.textContent = `#${i + 1}`;
    title.appendChild(rank);
    title.appendChild(document.createTextNode(c.title || 'Untitled clip'));
    info.appendChild(title);
    const meta = document.createElement('div');
    meta.className = 'twitch-clip-meta';
    const parts = [
      `👁 ${fmtViews(c.view_count)} views`,
      `at ${fmtVodTime(c.vod_offset)} in the stream`,
      `${Math.round(c.duration || 0)}s`,
    ];
    if (c.creator_name) parts.push(`clipped by ${c.creator_name}`);
    meta.textContent = parts.join(' · ');
    info.appendChild(meta);
    a.appendChild(info);
    twitchClipsEl.appendChild(a);
  });
}
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
const currentProfile = '__PROFILE_ID__';

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
    const resp = await fetch(`/api/trending?profile=${encodeURIComponent(currentProfile)}`);
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

const yesterdayVodsSection = document.getElementById('yesterday-vods-section');
const yesterdayVodsStatus = document.getElementById('yesterday-vods-status');
const yesterdayVodsResults = document.getElementById('yesterday-vods-results');

async function loadYesterdayVods() {
  yesterdayVodsResults.innerHTML = '';
  yesterdayVodsStatus.style.display = 'none';
  try {
    const resp = await fetch(`/api/vods/yesterday?profile=${encodeURIComponent(currentProfile)}`);
    const data = await resp.json();
    if (!resp.ok) {
      // No streamers configured for this profile yet -- just hide the
      // section instead of showing an error for something that isn't a
      // real problem (e.g. the main profile, which uses Recommend VOD instead).
      yesterdayVodsSection.style.display = 'none';
      return;
    }
    const results = data.results || [];
    if (!results.length) {
      yesterdayVodsSection.style.display = 'block';
      yesterdayVodsStatus.style.display = 'block';
      yesterdayVodsStatus.textContent = `No VODs found from yesterday (${data.date}) for this channel's tracked streamers.`;
      return;
    }
    yesterdayVodsSection.style.display = 'block';
    results.forEach(c => yesterdayVodsResults.appendChild(buildRecommendCard(c, c.name)));
  } catch (e) {
    yesterdayVodsSection.style.display = 'none';
  }
}

const profileYoutubeBtn = document.getElementById('profile-youtube-btn');
const profileYoutubeStatus = document.getElementById('profile-youtube-status');
let profileYoutubeConnected = false;

function renderProfileYoutubeStatus(label) {
  if (profileYoutubeConnected) {
    profileYoutubeStatus.textContent = `YouTube connected for ${label}.`;
    profileYoutubeBtn.textContent = 'Disconnect YouTube';
  } else {
    profileYoutubeStatus.textContent = `YouTube not connected for ${label} -- uploads for this channel won't work until it is.`;
    profileYoutubeBtn.textContent = 'Connect YouTube';
  }
}

profileYoutubeBtn.addEventListener('click', async () => {
  if (profileYoutubeConnected) {
    if (!confirm('Disconnect the __PROFILE_LABEL__ YouTube account?')) return;
    await fetch(`/api/youtube/disconnect?profile=${encodeURIComponent(currentProfile)}`, { method: 'POST' });
    await loadProfileYoutubeStatus();
  } else {
    window.location.href = `/auth/youtube/login?profile=${encodeURIComponent(currentProfile)}`;
  }
});

async function loadProfileYoutubeStatus() {
  try {
    const resp = await fetch('/api/profiles');
    if (!resp.ok) return;
    const data = await resp.json();
    const p = (data.profiles || []).find(p => p.id === currentProfile);
    if (!p) return;
    profileYoutubeConnected = p.youtube_connected;
    renderProfileYoutubeStatus(p.label);
  } catch (e) {
    // connect status is a nice-to-have -- generating/uploading still work
  }
}

loadProfileYoutubeStatus();
loadTrending();
loadYesterdayVods();

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
    // Every job shows here, including any left over from the removed
    // Spanish channel, so old ones can still be downloaded or deleted.
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
      if (!running && !['weekly_recap', 'game_recap'].includes(job.pipeline)) {
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
  renderTwitchClips(null, null);
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
    hook_text: document.getElementById('hook_text').checked,
    pacing: document.getElementById('pacing').checked,
    teaser: document.getElementById('teaser').checked,
    branding: document.getElementById('branding').checked,
    irl_layout: document.getElementById('irl_layout').checked,
    channel_profile: currentProfile,
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
  facecamIrlBtn.style.display = clip.is_irl_scene ? 'none' : '';
  if (clip.is_irl_scene) {
    facecamModalTitle.textContent = `Switch to facecam -- "${clip.title}"`;
    why = "This clip shows the whole scene (IRL layout). To split out a facecam instead, draw a box around it below.";
  } else if (clip.facecam_uncertain) {
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

const facecamIrlBtn = document.getElementById('facecam-irl-btn');
facecamIrlBtn.addEventListener('click', async () => {
  if (!facecamJobId || !facecamFilename) return;
  const jobId = facecamJobId;
  const filename = facecamFilename;
  facecamIrlBtn.disabled = true;
  facecamIrlBtn.textContent = 'Starting...';
  try {
    const resp = await fetch(`/api/jobs/${jobId}/clips/${filename}/mark-irl`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || 'Could not start the re-render.');
      return;
    }
    facecamModal.classList.remove('open');
    facecamJobId = null;
    facecamFilename = null;
    loadJobsList();
    attachToJob(jobId);
  } finally {
    facecamIrlBtn.disabled = false;
    facecamIrlBtn.textContent = "🎬 Use the IRL layout (whole scene)";
  }
});

const youtubeUploadModal = document.getElementById('youtube-upload-modal-overlay');
const youtubeUploadTitleInput = document.getElementById('youtube-upload-title-input');
const youtubeUploadTitleCounter = document.getElementById('youtube-upload-title-counter');
const youtubeUploadDescriptionInput = document.getElementById('youtube-upload-description-input');
const youtubeUploadPreview = document.getElementById('youtube-upload-preview');
const youtubeUploadGoBtn = document.getElementById('youtube-upload-go-btn');
const youtubeUploadTrimStart = document.getElementById('youtube-upload-trim-start');
const youtubeUploadTrimEnd = document.getElementById('youtube-upload-trim-end');
const thumbnailModal = document.getElementById('thumbnail-modal-overlay');
const thumbnailGalleryPanel = document.getElementById('thumbnail-gallery-panel');
const thumbnailGallery = document.getElementById('thumbnail-gallery');
const thumbnailStatusHint = document.getElementById('thumbnail-status-hint');
const thumbnailEditPanel = document.getElementById('thumbnail-edit-panel');
const thumbnailEditPreview = document.getElementById('thumbnail-edit-preview');
const thumbnailEditText = document.getElementById('thumbnail-edit-text');
const thumbnailRegenBtn = document.getElementById('thumbnail-regen-btn');
const thumbnailDownloadLink = document.getElementById('thumbnail-download-link');
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
let youtubeUploadIsShort = true;
let thumbnailJobId = null;
let thumbnailFilename = null;
let thumbnailText = '';
let thumbnailSelectedIndex = null;
let thumbnailFrameTimes = {};

// Matches channel_insights.MAX_SHORT_SECONDS -- the server refuses longer
// Shorts uploads too; this just says so before the click.
const SHORTS_MAX_SECONDS = 60;

function refreshYoutubeUploadTrimHint() {
  const trimStart = Math.max(0, parseFloat(youtubeUploadTrimStart.value) || 0);
  const trimEnd = Math.max(0, parseFloat(youtubeUploadTrimEnd.value) || 0);
  const resultSeconds = youtubeUploadDuration - trimStart - trimEnd;
  if (resultSeconds < 1) {
    youtubeUploadTrimHint.textContent =
      `Clip is ${youtubeUploadDuration.toFixed(1)}s -- that trim leaves ${resultSeconds.toFixed(1)}s, too short. Leave at least 1s.`;
    youtubeUploadGoBtn.disabled = true;
  } else if (youtubeUploadIsShort && resultSeconds > SHORTS_MAX_SECONDS + 0.05) {
    youtubeUploadTrimHint.textContent =
      `Clip is ${youtubeUploadDuration.toFixed(1)}s -- uploads from the app over ${SHORTS_MAX_SECONDS}s land as regular videos, `
      + `not Shorts. Trim at least ${(resultSeconds - SHORTS_MAX_SECONDS).toFixed(1)}s more.`;
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

function refreshYoutubeUploadTitleCounter() {
  const len = youtubeUploadTitleInput.value.length;
  youtubeUploadTitleCounter.textContent = `${len}/100`;
}
youtubeUploadTitleInput.addEventListener('input', refreshYoutubeUploadTitleCounter);

// Claude's generated title/description are sometimes wrong (misheard name,
// wrong game, an awkward hook) -- editable here so a bad one doesn't have
// to be caught after it's already live, or force posting by hand instead.
function openYoutubeUploadModal(jobId, clip) {
  youtubeUploadJobId = jobId;
  youtubeUploadFilename = clip.file;
  youtubeUploadDuration = clip.duration || 0;
  youtubeUploadIsShort = !clip.is_recap;
  youtubeUploadTitleInput.value = clip.upload_title || clip.title || '';
  youtubeUploadDescriptionInput.value = clip.description || '';
  refreshYoutubeUploadTitleCounter();
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
  const title = youtubeUploadTitleInput.value.trim();
  if (!title) {
    alert('Title cannot be empty.');
    return;
  }
  youtubeUploadGoBtn.disabled = true;
  youtubeUploadGoBtn.textContent = (trimStart || trimEnd) ? 'Trimming & uploading...' : 'Uploading...';
  try {
    const resp = await fetch(`/api/jobs/${jobId}/clips/${filename}/upload-youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        privacy_status: privacyStatus, trim_start: trimStart, trim_end: trimEnd,
        title, description: youtubeUploadDescriptionInput.value,
      }),
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

function closeThumbnailModal() {
  thumbnailModal.classList.remove('open');
  thumbnailJobId = null;
  thumbnailFilename = null;
  thumbnailSelectedIndex = null;
  thumbnailFrameTimes = {};
  thumbnailGallery.innerHTML = '';
  thumbnailEditPanel.style.display = 'none';
  thumbnailGalleryPanel.style.display = '';
}

function showThumbnailChoices() {
  thumbnailEditPanel.style.display = 'none';
  thumbnailGalleryPanel.style.display = '';
  thumbnailSelectedIndex = null;
}

function selectThumbnail(index) {
  thumbnailSelectedIndex = index;
  thumbnailGallery.querySelectorAll('img').forEach(img => {
    img.classList.toggle('selected', Number(img.dataset.index) === index);
  });
  thumbnailEditText.value = thumbnailText;
  const url = `/api/jobs/${thumbnailJobId}/clips/${thumbnailFilename}/thumbnails/${index}?t=${Date.now()}`;
  thumbnailEditPreview.src = url;
  thumbnailDownloadLink.href = `/api/jobs/${thumbnailJobId}/clips/${thumbnailFilename}/thumbnails/${index}?download=1&t=${Date.now()}`;
  thumbnailGalleryPanel.style.display = 'none';
  thumbnailEditPanel.style.display = '';
}

async function openThumbnailModal(jobId, clip) {
  thumbnailJobId = jobId;
  thumbnailFilename = clip.file;
  thumbnailSelectedIndex = null;
  thumbnailFrameTimes = {};
  thumbnailGallery.innerHTML = '';
  thumbnailEditPanel.style.display = 'none';
  thumbnailGalleryPanel.style.display = '';
  thumbnailStatusHint.textContent = 'Generating thumbnail options...';
  thumbnailModal.classList.add('open');

  try {
    const resp = await fetch(`/api/jobs/${jobId}/clips/${clip.file}/thumbnails`, { method: 'POST' });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      thumbnailStatusHint.textContent = data.detail || 'Could not generate thumbnails.';
      return;
    }
    thumbnailText = data.text || '';
    thumbnailStatusHint.textContent = 'Click one to pick it, then edit the text if you want.';
    thumbnailGallery.innerHTML = '';
    (data.thumbnails || []).forEach(t => {
      thumbnailFrameTimes[t.index] = t.frame_time;
      const img = document.createElement('img');
      img.src = `${t.url}?t=${Date.now()}`;
      img.dataset.index = t.index;
      img.alt = `Thumbnail option ${t.index}`;
      img.addEventListener('click', () => selectThumbnail(t.index));
      thumbnailGallery.appendChild(img);
    });
    if (!(data.thumbnails || []).length) {
      thumbnailStatusHint.textContent = 'No thumbnail candidates could be generated for this clip.';
    }
  } catch (e) {
    thumbnailStatusHint.textContent = 'Could not generate thumbnails.';
  }
}

document.getElementById('thumbnail-back-btn').addEventListener('click', showThumbnailChoices);
document.getElementById('thumbnail-close-btn').addEventListener('click', closeThumbnailModal);

thumbnailRegenBtn.addEventListener('click', async () => {
  if (!thumbnailJobId || !thumbnailFilename || thumbnailSelectedIndex === null) return;
  const text = thumbnailEditText.value.trim();
  if (!text) {
    alert('Text cannot be empty.');
    return;
  }
  const frameTime = thumbnailFrameTimes[thumbnailSelectedIndex];
  thumbnailRegenBtn.disabled = true;
  thumbnailRegenBtn.textContent = 'Regenerating...';
  try {
    const resp = await fetch(
      `/api/jobs/${thumbnailJobId}/clips/${thumbnailFilename}/thumbnails/${thumbnailSelectedIndex}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, frame_time: frameTime }),
      },
    );
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      alert(data.detail || 'Could not regenerate thumbnail.');
      return;
    }
    thumbnailText = data.text;
    const bust = `?t=${Date.now()}`;
    thumbnailEditPreview.src = `${data.url}${bust}`;
    thumbnailDownloadLink.href = `${data.url}?download=1&t=${Date.now()}`;
    const galleryImg = thumbnailGallery.querySelector(`img[data-index="${thumbnailSelectedIndex}"]`);
    if (galleryImg) galleryImg.src = `${data.url}${bust}`;
  } catch (e) {
    alert('Could not regenerate thumbnail.');
  } finally {
    thumbnailRegenBtn.disabled = false;
    thumbnailRegenBtn.textContent = 'Regenerate with this text';
  }
});

// The part that actually *asks*: once a job is done, if any clip's
// facecam got rejected, open the picker for it right away instead of
// leaving a button to be noticed. Each clip is offered once per page
// load, so "Skip for now" is respected -- the button stays on the clip.
function maybePromptFacecam(jobId, job) {
  if (document.querySelector('#stop-modal-overlay.open, #mood-modal-overlay.open, #regen-modal-overlay.open, #facecam-modal-overlay.open, #youtube-upload-modal-overlay.open, #thumbnail-modal-overlay.open')) return;
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

  renderTwitchClips(jobId, job);
  clipsEl.innerHTML = '';
  const scored = (job.clips || []).filter(c => typeof c.score === 'number' && !c.is_recap);
  const bestScore = scored.length > 1 ? Math.max(...scored.map(c => c.score)) : null;
  (job.clips || []).forEach(c => {
    const div = document.createElement('div');
    div.className = 'clip';

    const heading = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = c.title;
    heading.appendChild(strong);
    heading.appendChild(document.createTextNode(` (${c.duration}s)`));
    if (typeof c.score === 'number') {
      const best = bestScore !== null && c.score === bestScore;
      const badge = document.createElement('span');
      badge.className = 'score-badge' + (best ? ' best' : '');
      const kind = c.moment_type ? `${c.moment_type} · ` : '';
      badge.textContent = best ? `⭐ Post this one first · ${kind}${c.score}/10` : `${kind}${c.score}/10`;
      badge.title = c.reason || '';
      heading.appendChild(badge);
    }
    div.appendChild(heading);
    if (c.subscores && Object.keys(c.subscores).length) {
      const labels = [
        ['hook', 'Hook'], ['controversy', 'Controversy'], ['reaction', 'Reaction'],
        ['payoff', 'Payoff'], ['standalone', 'Makes sense alone'],
      ];
      const grid = document.createElement('div');
      grid.className = 'score-breakdown';
      labels.forEach(([key, label]) => {
        if (typeof c.subscores[key] !== 'number') return;
        const row = document.createElement('span');
        row.textContent = `${label} ${c.subscores[key]}/10`;
        grid.appendChild(row);
      });
      if (typeof c.score === 'number') {
        const overall = document.createElement('span');
        overall.className = 'overall';
        overall.textContent = `Overall ${c.score}/10`;
        grid.appendChild(overall);
      }
      div.appendChild(grid);
    }

    if (c.hook_caption) {
      const em = document.createElement('em');
      em.textContent = c.hook_caption;
      div.appendChild(em);
    }

    if (c.youtube_video_id) {
      const posted = document.createElement('a');
      posted.href = c.is_recap
        ? `https://youtu.be/${encodeURIComponent(c.youtube_video_id)}`
        : `https://youtube.com/shorts/${encodeURIComponent(c.youtube_video_id)}`;
      posted.target = '_blank';
      posted.rel = 'noopener';
      posted.textContent = '✅ Posted to YouTube';
      posted.style.display = 'block';
      div.appendChild(posted);
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
      // One clear primary action (Upload) plus a quieter secondary row for
      // everything else -- four identical-weight buttons in a run-on line
      // made it hard to tell at a glance which one actually ships the clip.
      const primaryRow = document.createElement('div');
      primaryRow.className = 'clip-primary-row';
      const uploadBtn = document.createElement('button');
      uploadBtn.type = 'button';
      uploadBtn.textContent = '📤 Upload to YouTube';
      uploadBtn.addEventListener('click', () => openYoutubeUploadModal(jobId, c));
      primaryRow.appendChild(uploadBtn);
      div.appendChild(primaryRow);

      const secondaryRow = document.createElement('div');
      secondaryRow.className = 'clip-secondary-row';

      const thumbBtn = document.createElement('button');
      thumbBtn.type = 'button';
      thumbBtn.textContent = '🖼 Thumbnail';
      thumbBtn.addEventListener('click', () => openThumbnailModal(jobId, c));
      secondaryRow.appendChild(thumbBtn);

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
        fixBtn.textContent = c.is_irl_scene ? '🎯 Switch to facecam'
          : c.facecam_uncertain ? '🎯 Fix facecam'
          : (c.facecam_manual || c.facecam_trusted) ? '🎯 Adjust facecam' : '🎯 Add facecam';
        fixBtn.addEventListener('click', () => openFacecamModal(jobId, c, facecamOthersMissing(job, c)));
        secondaryRow.appendChild(fixBtn);
      }

      const delClipBtn = document.createElement('button');
      delClipBtn.type = 'button';
      delClipBtn.textContent = '🗑 Delete';
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
            delClipBtn.textContent = '🗑 Delete';
            return;
          }
          const data = await r.json().catch(() => ({}));
          if (data.job_deleted) {
            // A weekly-recap job has only this one clip -- the whole job
            // is gone now too (see delete_clip), so there's nothing left
            // for poll(jobId) to fetch. Clear the view instead of leaving
            // the just-deleted clip on screen until a manual refresh.
            clipsEl.innerHTML = '';
            statusEl.textContent = '';
            renderTwitchClips(null, null);
            progressWrap.style.display = 'none';
            await loadJobsList();
          } else {
            poll(jobId);
          }
        } catch (e) {
          alert('Could not delete this clip.');
          delClipBtn.disabled = false;
          delClipBtn.textContent = '🗑 Delete';
        }
      });
      secondaryRow.appendChild(delClipBtn);
      div.appendChild(secondaryRow);
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
    renderTwitchClips(null, null);
    deleteBtn.style.display = 'none';
    currentJobId = null;
    loadJobsList();
  } else {
    deleteBtn.textContent = "🗑 I've downloaded these — delete from server";
  }
  deleteBtn.disabled = false;
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


def _nav_links(active_path: str) -> str:
    """The shared topnav, repeated on every page -- active_path marks which
    link (by href) gets the highlighted style."""
    links = [("Home", "/"), ("Analytics", "/analytics"), ("Hook Line", "/hook-line")]

    def _link(label: str, href: str) -> str:
        active_attr = ' class="active"' if href == active_path else ""
        return f'<a href="{href}"{active_attr}>{label}</a>'

    rows = "\n  ".join(_link(label, href) for label, href in links)
    return f'<div class="topnav">\n  {rows}\n</div>'


def _render_channel_home(profile: str) -> str:
    """The home page template with one channel profile baked in (see
    CHANNEL_PROFILES) -- a channel's own URL, "Active & saved jobs" list
    and Connect YouTube button."""
    cfg = CHANNEL_PROFILES[profile]
    page = _CHANNEL_HOME_TEMPLATE.replace("__NAV_LINKS__", _nav_links(cfg["page_path"]))
    return (
        page
        .replace("__PROFILE_ID__", profile)
        .replace("__PROFILE_LABEL__", cfg["label"])
    )


INDEX_HTML = _render_channel_home(DEFAULT_CHANNEL_PROFILE)


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
  .topnav {
    display: flex; flex-wrap: wrap; gap: 4px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px; margin-bottom: 16px; box-shadow: var(--shadow);
  }
  .topnav a {
    flex: 1 1 auto; text-align: center; padding: 9px 10px; border-radius: 9px;
    font-size: 0.84rem; font-weight: 600; color: var(--muted); text-decoration: none;
  }
  .topnav a:hover { color: var(--text); }
  .topnav a.active { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: var(--accent-text); }
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
  .stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; }
  .stat-tile { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
  .stat-label { font-size: 0.72rem; color: var(--muted); }
  .stat-value { font-size: 1.35rem; font-weight: 650; margin-top: 2px; }
  .stat-sub { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }
  .table-scroll { overflow-x: auto; margin-top: 10px; }
  .perf-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .perf-table th, .perf-table td {
    padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--border);
    white-space: nowrap; font-variant-numeric: tabular-nums;
  }
  .perf-table th:first-child, .perf-table td:first-child { text-align: left; white-space: normal; }
  .perf-table thead th { color: var(--muted); font-weight: 600; font-size: 0.72rem; }
  .perf-table tr.group-name th { text-align: left; padding-top: 14px; font-weight: 700; color: var(--text); }
  .perf-table tr.thin td { color: var(--muted); }
  @media (max-width: 480px) {
    .perf-table th, .perf-table td { padding: 5px 4px; }
  }
  details.perf-videos { margin-top: 16px; }
  .bar-row.latest .bar-title { font-weight: 700; }
  details.perf-videos summary { cursor: pointer; font-size: 0.85rem; font-weight: 600; }
</style>
</head>
<body>
<div class="page">
<div class="topnav">
  <a href="/">Home</a>
  <a href="/analytics" class="active">Analytics</a>
  <a href="/hook-line">Hook Line</a>
</div>
<div class="card">

<div class="brand"><span class="logo">📊</span><h1>Analytics &amp; AI strategy</h1></div>
<p class="subtitle">Best day to post, what's working, and how you compare to channels clipping the same streamers.</p>

<div class="section">
  <label style="margin-top:0">📈 Channel insights — best day to post</label>
  <div id="insights-body"><div class="hint">Loading...</div></div>
</div>

<div class="section">
  <label style="margin-top:0">🚀 Is it growing?</label>
  <p class="hint">Views and new subscribers week by week, and how many views the Shorts you posted each week got.
    Updates with the refresh button below.</p>
  <div id="growth-body"><div class="hint">Loading...</div></div>
</div>

<div class="section">
  <label style="margin-top:0">🎯 What your Shorts actually do</label>
  <p class="hint">Real retention from YouTube Analytics for your recent Shorts, and how clips with different
    traits compare. Clips made here are matched to the video you posted, by the upload button or by title.</p>
  <div id="clip-perf-body"><div class="hint">Loading...</div></div>
  <button id="clip-perf-refresh-btn" type="button">↻ Refresh from YouTube</button>
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

function fmtFraction(f) { return f == null ? '–' : `${Math.round(f * 100)}%`; }
function fmtPercent(p) { return p == null ? '–' : `${Math.round(p)}%`; }
function fmtCount(n) { return n == null ? '–' : Number(n).toLocaleString(); }

function statTile(label, value, sub) {
  const tile = el('div', { className: 'stat-tile' });
  tile.appendChild(el('div', { className: 'stat-label', text: label }));
  tile.appendChild(el('div', { className: 'stat-value', text: value }));
  if (sub) tile.appendChild(el('div', { className: 'stat-sub', text: sub }));
  return tile;
}

// Share of views still watching at four checkpoints -- one series, so one
// hue and no legend; every bar carries its value at the tip, and the title
// attribute spells out what the number means on hover.
function buildDropOffBlock(s) {
  const block = el('div', { className: 'chart-block' });
  const titleRow = el('div', { className: 'chart-title' });
  titleRow.style.flexWrap = 'wrap';
  titleRow.appendChild(el('span', { text: 'Where viewers leave' }));
  titleRow.appendChild(el('span', { className: 'hint', text: 'share of views still watching, averaged over your Shorts' }));
  block.appendChild(titleRow);
  const points = [
    ['At 1 second', s.watch_1s],
    ['At 3 seconds', s.watch_3s],
    ['Halfway', s.watch_mid],
    ['Near the end', s.watch_end],
  ];
  const scaleMax = Math.max(1, ...points.map(p => p[1] || 0));
  points.forEach(([label, value]) => {
    const row = el('div', { className: 'bar-row' });
    row.title = `${fmtFraction(value)} of views were still watching ${label.toLowerCase()}`;
    const labelCol = el('div', { className: 'bar-label' });
    labelCol.appendChild(el('div', { className: 'bar-title', text: label }));
    row.appendChild(labelCol);
    const track = el('div', { className: 'bar-track' });
    const fill = el('div', { className: 'bar-fill' });
    fill.style.width = Math.max(1, Math.round(((value || 0) / scaleMax) * 100)) + '%';
    fill.style.background = 'var(--chart-you)';
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el('div', { className: 'bar-value', text: fmtFraction(value) }));
    block.appendChild(row);
  });
  return block;
}

function fmtChange(pct) {
  if (pct == null) return null;
  return `${pct > 0 ? '+' : ''}${pct}% vs the week before`;
}
function fmtSigned(n) { return n == null ? '–' : (n > 0 ? `+${n}` : String(n)); }

// One measure per chart, one row per week (oldest at the top, the latest
// week in bold), the value printed at the tip and spelled out on hover.
function buildWeeklyBars(title, hint, weeks, valueOf, fmt, subOf) {
  const block = el('div', { className: 'chart-block' });
  const titleRow = el('div', { className: 'chart-title' });
  titleRow.style.flexWrap = 'wrap';
  titleRow.appendChild(el('span', { text: title }));
  if (hint) titleRow.appendChild(el('span', { className: 'hint', text: hint }));
  block.appendChild(titleRow);
  const scaleMax = Math.max(1, ...weeks.map(w => valueOf(w) || 0));
  weeks.forEach((w, i) => {
    const value = valueOf(w);
    const row = el('div', { className: 'bar-row' + (i === weeks.length - 1 ? ' latest' : '') });
    row.title = `${w.label}: ${value == null ? 'no data' : fmt(value)}${subOf ? ' (' + subOf(w) + ')' : ''}`;
    const labelCol = el('div', { className: 'bar-label' });
    labelCol.appendChild(el('div', { className: 'bar-title', text: w.label }));
    if (subOf) labelCol.appendChild(el('div', { className: 'bar-sub', text: subOf(w) }));
    row.appendChild(labelCol);
    const track = el('div', { className: 'bar-track' });
    const fill = el('div', { className: 'bar-fill' });
    fill.style.width = Math.max(0, Math.round((Math.max(0, value || 0) / scaleMax) * 100)) + '%';
    fill.style.background = 'var(--chart-you)';
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el('div', { className: 'bar-value', text: value == null ? '–' : fmt(value) }));
    block.appendChild(row);
  });
  return block;
}

function renderGrowth(data) {
  const body = document.getElementById('growth-body');
  body.innerHTML = '';
  if (!data.available) {
    body.appendChild(el('div', { className: 'hint', text: data.reason || 'Not available.' }));
    return;
  }
  const t = data.trend;
  if (!t || !t.weeks || !t.weeks.length) {
    body.appendChild(el('div', { className: 'hint', text: 'No week-by-week numbers from YouTube yet.' }));
    return;
  }
  const w = t.last_week;
  const prev = t.weeks.length > 1 ? t.weeks[t.weeks.length - 2] : null;
  const tiles = el('div', { className: 'stat-row' });
  tiles.appendChild(statTile('Views, latest week', fmtCount(w.views), fmtChange(t.views_change_pct)));
  tiles.appendChild(statTile('New subscribers, latest week', fmtSigned(w.subscribers_net),
    prev ? `${fmtSigned(prev.subscribers_net)} the week before` : null));
  tiles.appendChild(statTile('Median views per Short posted', fmtCount(w.median_views_per_short),
    fmtChange(t.median_views_change_pct) || `${w.shorts_posted} Shorts posted that week`));
  tiles.appendChild(statTile('Subscribers per 1,000 views', w.subs_per_1k == null ? '–' : w.subs_per_1k.toFixed(1),
    'latest week'));
  body.appendChild(tiles);
  const through = new Date(t.data_through + 'T00:00:00').toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
  const note = el('div', { className: 'hint', text: `Weeks are the 7 days up to ${through}, the latest day YouTube has numbers for (they run about 2 days behind Studio).` });
  note.style.marginTop = '8px';
  body.appendChild(note);
  body.appendChild(buildWeeklyBars('Views per week', 'whole channel', t.weeks, x => x.views, fmtCount));
  body.appendChild(buildWeeklyBars('New subscribers per week', 'gained minus lost', t.weeks, x => x.subscribers_net, fmtSigned));
  body.appendChild(buildWeeklyBars(
    'Median views per Short posted that week', 'newer weeks are still collecting views',
    t.weeks, x => x.median_views_per_short, fmtCount,
    x => `${x.shorts_posted} posted, ${x.made_here_posted} made here`,
  ));
}

function buildComparisonTable(groups) {
  const wrap = el('div', { className: 'table-scroll' });
  const table = el('table', { className: 'perf-table' });
  const head = el('thead');
  const headRow = el('tr');
  ['', 'Shorts', 'At 3s', 'Watched', 'Median views', 'Subs / 1k views'].forEach(h => headRow.appendChild(el('th', { text: h })));
  head.appendChild(headRow);
  table.appendChild(head);
  const body = el('tbody');
  groups.forEach(g => {
    const nameRow = el('tr', { className: 'group-name' });
    const nameCell = el('th', { text: g.name + (g.made_here_only ? ' (clips made here only)' : '') });
    nameCell.colSpan = 6;
    nameRow.appendChild(nameCell);
    body.appendChild(nameRow);
    g.buckets.forEach(b => {
      const row = el('tr', { className: b.enough ? '' : 'thin' });
      row.appendChild(el('td', { text: b.label + (b.enough ? '' : ' *') }));
      row.appendChild(el('td', { text: String(b.n) }));
      row.appendChild(el('td', { text: fmtFraction(b.watch_3s) }));
      row.appendChild(el('td', { text: fmtPercent(b.avg_view_pct) }));
      row.appendChild(el('td', { text: fmtCount(b.median_views) }));
      row.appendChild(el('td', { text: b.subs_per_1k == null ? '–' : b.subs_per_1k.toFixed(1) }));
      body.appendChild(row);
    });
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function buildVideoTable(videos) {
  const details = el('details', { className: 'perf-videos' });
  details.appendChild(el('summary', { text: `Every Short in this analysis (${videos.length})` }));
  const wrap = el('div', { className: 'table-scroll' });
  const table = el('table', { className: 'perf-table' });
  const head = el('thead');
  const headRow = el('tr');
  ['Title', 'At 3s', 'Watched', 'Views', 'Subs', 'Length', 'Posted', 'Made here'].forEach(h => headRow.appendChild(el('th', { text: h })));
  head.appendChild(headRow);
  table.appendChild(head);
  const body = el('tbody');
  videos.forEach(v => {
    const row = el('tr', { className: v.too_new_to_judge ? 'thin' : '' });
    const titleCell = el('td');
    const link = el('a', { text: v.title, href: `https://youtube.com/shorts/${encodeURIComponent(v.id)}` });
    link.target = '_blank';
    link.rel = 'noopener';
    titleCell.appendChild(link);
    if (v.too_new_to_judge) titleCell.appendChild(el('span', { className: 'hint', text: ' (too new to judge)' }));
    row.appendChild(titleCell);
    row.appendChild(el('td', { text: fmtFraction(v.retention ? v.retention.watch_3s : null) }));
    row.appendChild(el('td', { text: fmtPercent(v.avg_view_pct) }));
    row.appendChild(el('td', { text: fmtCount(v.views) }));
    row.appendChild(el('td', { text: v.subs_gained == null ? '–' : fmtSigned(v.subs_gained) }));
    row.appendChild(el('td', { text: v.duration != null ? `${Math.round(v.duration)}s` : '–' }));
    row.appendChild(el('td', { text: v.published_at ? new Date(v.published_at).toLocaleDateString() : '–' }));
    const madeHere = !v.made_here ? '–' : (v.clip && v.clip.link_method === 'upload' ? '✓ uploaded here' : '✓ title match');
    row.appendChild(el('td', { text: madeHere }));
    body.appendChild(row);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  details.appendChild(wrap);
  return details;
}

async function loadClipPerformance(refresh) {
  const body = document.getElementById('clip-perf-body');
  const refreshBtn = document.getElementById('clip-perf-refresh-btn');
  refreshBtn.disabled = true;
  // Keep the previous numbers on screen (dimmed) while refetching instead
  // of blanking the panel.
  body.style.opacity = '0.5';
  let data;
  try {
    const resp = await fetch('/api/clip-performance' + (refresh ? '?refresh=true' : ''));
    data = await resp.json();
  } catch (e) {
    data = { available: false, reason: 'Could not load clip performance.' };
  }
  body.style.opacity = '';
  refreshBtn.disabled = false;
  renderGrowth(data);
  body.innerHTML = '';
  if (!data.available) {
    body.appendChild(el('div', { className: 'hint', text: data.reason || 'Not available.' }));
    return;
  }
  const s = data.summary;
  body.appendChild(el('div', {
    className: 'hint',
    text: `${s.shorts} Shorts with settled numbers (${s.with_retention} with a retention curve, ${s.too_new} too new to judge) · `
      + `${data.clips_linked} of ${data.clips_recorded} clips made here matched to a posted video`,
  }));
  if (!s.shorts) {
    body.appendChild(el('div', { className: 'hint', text: 'Nothing settled to analyze yet -- Shorts need about 2 days before their numbers mean much.' }));
    return;
  }
  const tiles = el('div', { className: 'stat-row' });
  tiles.appendChild(statTile('Median views', fmtCount(s.median_views)));
  tiles.appendChild(statTile('Watched on average', fmtPercent(s.avg_view_pct), 'of each Short, replays included'));
  tiles.appendChild(statTile('Still watching at 3s', fmtFraction(s.watch_3s), 'per view'));
  if (s.subs_per_1k != null) {
    tiles.appendChild(statTile('Subscribers per 1,000 views', s.subs_per_1k.toFixed(1), 'across these Shorts'));
  }
  if (s.opening_vs_similar != null) {
    tiles.appendChild(statTile('First 3s vs similar videos', s.opening_vs_similar.toFixed(2), '0.5 = typical for the length'));
  }
  body.appendChild(tiles);
  if (s.watch_1s != null) {
    body.appendChild(buildDropOffBlock(s));
    if (s.typical_steepest_drop_at != null) {
      body.appendChild(el('div', { className: 'hint', text: `The single biggest drop usually lands ${s.typical_steepest_drop_at}s in.` }));
    }
  }
  if (data.groups && data.groups.length) {
    const note = el('div', {
      className: 'hint',
      text: 'Rows marked * (greyed out) have fewer than 5 Shorts behind them -- too few to read anything into yet.',
    });
    note.style.marginTop = '14px';
    body.appendChild(note);
    body.appendChild(buildComparisonTable(data.groups));
  }
  if (data.videos && data.videos.length) body.appendChild(buildVideoTable(data.videos));
}
loadClipPerformance(false);
document.getElementById('clip-perf-refresh-btn').addEventListener('click', () => loadClipPerformance(true));

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

HOOK_LINE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper — hook line</title>
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
  .topnav {
    display: flex; flex-wrap: wrap; gap: 4px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px; margin-bottom: 16px; box-shadow: var(--shadow);
  }
  .topnav a {
    flex: 1 1 auto; text-align: center; padding: 9px 10px; border-radius: 9px;
    font-size: 0.84rem; font-weight: 600; color: var(--muted); text-decoration: none;
  }
  .topnav a:hover { color: var(--text); }
  .topnav a.active { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: var(--accent-text); }
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
  input, select, textarea {
    width: 100%; padding: 10px 12px; margin-top: 6px; font-size: 0.95rem;
    background: var(--bg); color: var(--text); font-family: inherit;
    border: 1px solid var(--border); border-radius: 10px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input:focus, select:focus, textarea:focus {
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
  button.secondary { background: transparent; color: var(--muted); border: 1px solid var(--border); }
  button:hover:not(:disabled) { opacity: 0.92; }
  button:active:not(:disabled) { transform: scale(0.98); }
  button:disabled { opacity: 0.45; cursor: default; }
  .hint { font-size: 0.78rem; color: var(--muted); font-weight: 400; margin-top: 6px; text-transform: none; letter-spacing: normal; }
  .section { margin-top: 28px; padding-top: 20px; border-top: 1px solid var(--border); }
  .row { display: flex; gap: 10px; }
  .row > * { flex: 1; }
  #preview-video, #result-video { width: 100%; max-width: 320px; border-radius: 10px; margin-top: 10px; background: #000; display: block; }
  #error-msg { color: var(--danger); font-size: 0.88rem; margin-top: 10px; }
  #result-block { margin-top: 16px; display: none; }
  #result-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  #result-actions a, #result-actions button { margin-top: 0; }
  #result-actions a.dl-link {
    display: inline-flex; align-items: center; padding: 11px 20px; font-size: 0.95rem; font-weight: 600;
    border-radius: 10px; border: 1px solid var(--border); color: var(--text); text-decoration: none;
  }
  #empty-msg { color: var(--muted); font-size: 0.9rem; margin-top: 16px; display: none; }
</style>
</head>
<body>
<div class="page">
  <div class="topnav">
    <a href="/">Home</a>
    <a href="/analytics">Analytics</a>
    <a href="/hook-line" class="active">Hook Line</a>
  </div>
  <div class="card">
    <div class="brand"><span class="logo">⚡</span><h1>Hook line</h1></div>
    <div class="subtitle">
      Pick a clip you've already rendered on the normal Clipping tab, then flash a spoiler line
      ("MARLON ALMOST KNOCKS OUT JASON") dead-center for under a second right as it starts, before
      the moment actually happens -- the curiosity-gap trick clip channels use to hook a scroll.
      This is a separate pass over the finished clip: nothing here touches the original render, its
      audio, or its captions.
    </div>

    <label for="clip-select">Clip</label>
    <select id="clip-select"><option value="">Loading clips...</option></select>
    <div class="hint">Only clips from finished jobs still on the server show up here.</div>
    <div id="empty-msg">No finished clips available yet -- render some on the Clipping tab first.</div>

    <video id="preview-video" controls preload="metadata" style="display:none"></video>

    <div class="section">
      <label for="hook-text-input">Hook line text</label>
      <textarea id="hook-text-input" rows="2" placeholder="Generate one, or write your own"></textarea>
      <div class="row">
        <button id="generate-btn" class="secondary" disabled>Generate with Claude</button>
        <button id="render-btn" disabled>Render preview</button>
      </div>
      <div class="hint">Flashes for well under a second, then disappears -- edit the text above before rendering if you want something different.</div>
    </div>

    <div id="error-msg"></div>

    <div id="result-block">
      <div class="section">
        <label style="margin-top:0">Result</label>
        <video id="result-video" controls preload="metadata"></video>
        <div id="result-actions">
          <a id="result-download" class="dl-link" href="#">Download</a>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const clipSelect = document.getElementById('clip-select');
const emptyMsg = document.getElementById('empty-msg');
const previewVideo = document.getElementById('preview-video');
const hookTextInput = document.getElementById('hook-text-input');
const generateBtn = document.getElementById('generate-btn');
const renderBtn = document.getElementById('render-btn');
const errorMsg = document.getElementById('error-msg');
const resultBlock = document.getElementById('result-block');
const resultVideo = document.getElementById('result-video');
const resultDownload = document.getElementById('result-download');

let clips = [];

function showError(msg) {
  errorMsg.textContent = msg || '';
}

function selectedClip() {
  const idx = clipSelect.value;
  return idx === '' ? null : clips[Number(idx)];
}

function onClipChange() {
  const clip = selectedClip();
  resultBlock.style.display = 'none';
  showError('');
  if (!clip) {
    previewVideo.style.display = 'none';
    generateBtn.disabled = true;
    renderBtn.disabled = true;
    return;
  }
  previewVideo.src = `/api/jobs/${clip.job_id}/clips/${clip.file}`;
  previewVideo.style.display = 'block';
  hookTextInput.value = clip.hook_caption || '';
  generateBtn.disabled = false;
  renderBtn.disabled = false;
}

async function loadClips() {
  try {
    const resp = await fetch('/api/hook-line/clips');
    const data = await resp.json();
    clips = data.clips || [];
  } catch (e) {
    clips = [];
  }
  clipSelect.innerHTML = '';
  if (!clips.length) {
    clipSelect.innerHTML = '<option value="">No clips available</option>';
    emptyMsg.style.display = 'block';
    return;
  }
  emptyMsg.style.display = 'none';
  clipSelect.appendChild(new Option('Choose a clip...', ''));
  clips.forEach((c, i) => {
    const label = `${c.source_title || 'Untitled source'} — ${c.title || c.file} (${c.duration}s)`;
    clipSelect.appendChild(new Option(label, String(i)));
  });
}

clipSelect.addEventListener('change', onClipChange);

generateBtn.addEventListener('click', async () => {
  const clip = selectedClip();
  if (!clip) return;
  showError('');
  generateBtn.disabled = true;
  generateBtn.textContent = 'Generating...';
  try {
    const resp = await fetch('/api/hook-line/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: clip.job_id, filename: clip.file }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showError(data.detail || 'Could not generate a hook line.');
      return;
    }
    hookTextInput.value = data.hook_text;
  } catch (e) {
    showError('Could not generate a hook line.');
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = 'Generate with Claude';
  }
});

renderBtn.addEventListener('click', async () => {
  const clip = selectedClip();
  const hookText = hookTextInput.value.trim();
  if (!clip) return;
  if (!hookText) {
    showError('Write or generate a hook line first.');
    return;
  }
  showError('');
  renderBtn.disabled = true;
  renderBtn.textContent = 'Rendering...';
  try {
    const resp = await fetch('/api/hook-line/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: clip.job_id, filename: clip.file, hook_text: hookText }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showError(data.detail || 'Could not render the hook line.');
      return;
    }
    const url = `/api/jobs/${clip.job_id}/clips/${data.file}`;
    resultVideo.src = `${url}?t=${Date.now()}`;
    resultDownload.href = `${url}?download=1`;
    resultBlock.style.display = 'block';
  } catch (e) {
    showError('Could not render the hook line.');
  } finally {
    renderBtn.disabled = false;
    renderBtn.textContent = 'Render preview';
  }
});

loadClips();
</script>
</body>
</html>
"""


# The /long-form page -- see clipper/longform.py and the /api/longform routes.
LONGFORM_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper — Long-form videos</title>
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
  .page { max-width: 720px; margin: 0 auto; }
  .topnav {
    display: flex; flex-wrap: wrap; gap: 4px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px; margin-bottom: 16px; box-shadow: var(--shadow);
  }
  .topnav a {
    flex: 1 1 auto; text-align: center; padding: 9px 10px; border-radius: 9px;
    font-size: 0.84rem; font-weight: 600; color: var(--muted); text-decoration: none;
  }
  .topnav a:hover { color: var(--text); }
  .topnav a.active { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: var(--accent-text); }
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
  input, select, textarea {
    width: 100%; padding: 10px 12px; margin-top: 6px; font-size: 0.95rem;
    background: var(--bg); color: var(--text); font-family: inherit;
    border: 1px solid var(--border); border-radius: 10px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input:focus, select:focus, textarea:focus {
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
  button.secondary { background: transparent; color: var(--muted); border: 1px solid var(--border); }
  button:hover:not(:disabled) { opacity: 0.92; }
  button:active:not(:disabled) { transform: scale(0.98); }
  button:disabled { opacity: 0.45; cursor: default; }
  .hint { font-size: 0.78rem; color: var(--muted); font-weight: 400; margin-top: 6px; text-transform: none; letter-spacing: normal; }
  .section { margin-top: 28px; padding-top: 20px; border-top: 1px solid var(--border); }
  .row { display: flex; gap: 10px; }
  .row > * { flex: 1; }
  .back { display: inline-block; margin-bottom: 12px; color: var(--accent); font-weight: 700; text-decoration: none; font-size: 0.9rem; }
  h2 { font-size: 1.15rem; margin: 10px 0 0; }
  h3 { font-size: 1.02rem; margin: 0 0 4px; display: flex; align-items: center; gap: 8px; }
  h3 .n { display: inline-flex; width: 24px; height: 24px; border-radius: 50%; align-items: center; justify-content: center;
    font-size: 0.78rem; background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; }
  button.linkish { background: none; border: none; color: var(--accent); padding: 0; margin: 0; font-weight: 700; font-size: 0.88rem; }
  button.danger-link { background: none; border: 1px solid var(--border); color: var(--danger); margin-top: 0; font-size: 0.85rem; padding: 8px 14px; }
  .incident, .project { display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; margin-top: 10px; border: 1px solid var(--border);
    border-radius: 12px; background: var(--bg); cursor: pointer; width: 100%; text-align: left; color: var(--text); font-weight: 400; }
  .incident:hover, .project:hover { border-color: var(--accent); opacity: 1; }
  .incident .ico { font-size: 1.5rem; line-height: 1; }
  .incident .t, .project .t { font-weight: 700; font-size: 0.95rem; }
  .incident .m, .project .m { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
  .tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 7px; }
  .tag { font-size: 0.7rem; font-weight: 700; padding: 3px 8px; border-radius: 999px; background: var(--card); border: 1px solid var(--border); color: var(--muted); }
  .tag.good { color: #059669; }
  .tag.warn { color: #b45309; }
  details { margin-top: 16px; }
  summary { cursor: pointer; font-weight: 700; font-size: 0.9rem; color: var(--accent); }
  .steps { display: flex; gap: 6px; margin: 14px 0 4px; flex-wrap: wrap; }
  .steps span { font-size: 0.74rem; font-weight: 700; padding: 5px 10px; border-radius: 999px; background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
  .steps span.done { color: var(--accent); border-color: var(--accent); }
  .steps span.on { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; border-color: transparent; }
  .status-box { margin-top: 16px; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border); background: var(--bg); font-size: 0.9rem; }
  .status-box.err { border-color: var(--danger); color: var(--danger); }
  .meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted); margin: 8px 0 4px; }
  .meta b { color: var(--text); }
  .scene { display: grid; grid-template-columns: 30px 1fr; gap: 10px; padding: 12px 0; border-bottom: 1px solid var(--border); }
  .scene .num { font-weight: 800; color: var(--accent); padding-top: 12px; }
  .scene textarea { margin-top: 0; min-height: 90px; resize: vertical; line-height: 1.45; }
  .scene .vis-row { display: flex; gap: 8px; align-items: center; margin-top: 6px; }
  .scene select { width: auto; margin-top: 0; padding: 6px 8px; font-size: 0.82rem; }
  .scene .vnote { font-size: 0.78rem; color: var(--muted); }
  .scene .rec-badge { font-size: 0.72rem; font-weight: 700; margin-left: auto; white-space: nowrap; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .actions button { margin-top: 12px; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
  .chips button { margin-top: 0; padding: 6px 0; width: 36px; font-size: 0.8rem; background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
  .chips button.ready { background: color-mix(in srgb, #059669 18%, var(--bg)); color: #059669; border-color: #059669; }
  .chips button.flag { background: color-mix(in srgb, #f59e0b 18%, var(--bg)); color: #b45309; border-color: #f59e0b; }
  .chips button.cur { outline: 3px solid var(--accent); outline-offset: 1px; }
  .prompter { background: #0b0c10; color: #f3f4f6; border-radius: 14px; padding: 22px; margin-top: 12px; font-size: 1.35rem; line-height: 1.65; }
  .prompter .miss { background: rgba(239, 68, 68, 0.4); color: #fff; border-radius: 5px; padding: 0 3px; }
  .prompter .label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 700; }
  .rec-row { display: flex; gap: 12px; align-items: center; }
  #rec-btn { font-size: 1.05rem; padding: 14px 22px; }
  #rec-btn.recording { background: var(--danger); }
  .rec-timer { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--danger); margin-top: 18px; }
  .take-result { margin-top: 12px; padding: 12px 14px; border-radius: 12px; font-size: 0.92rem; font-weight: 600; }
  .take-result.ok { background: color-mix(in srgb, #059669 14%, var(--bg)); color: #047857; }
  .take-result.bad { background: color-mix(in srgb, #f59e0b 16%, var(--bg)); color: #92400e; }
  .take-result.wait { background: var(--bg); color: var(--muted); }
  .take-result .heard { display: block; font-weight: 400; font-size: 0.8rem; margin-top: 6px; color: var(--muted); }
  .vgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; margin-top: 14px; }
  .vcard { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--bg); }
  .vcard img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #0b0c10; }
  .vcard .vbody { padding: 10px 12px 12px; }
  .vcard .vhead { display: flex; gap: 8px; align-items: center; }
  .vcard .vhead b { font-size: 0.86rem; white-space: nowrap; }
  .vcard select, .vcard input, .vcard textarea { margin-top: 6px; padding: 7px 9px; font-size: 0.84rem; }
  .vcard .vhead select { margin-top: 0; width: auto; flex: 1; }
  .vcard .vtext { font-size: 0.78rem; color: var(--muted); margin-top: 6px; line-height: 1.35; }
  .vcard .vbtns { display: flex; gap: 6px; }
  .vcard .vbtns button { margin-top: 8px; padding: 6px 10px; font-size: 0.78rem; flex: 1; }
  .music-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .music-row input { flex: 1; min-width: 200px; }
  .music-row button { margin-top: 6px; }
  .bar { height: 10px; border-radius: 99px; background: var(--track); margin-top: 14px; overflow: hidden; }
  .bar div { height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.4s; }
  .title-opt { display: flex; gap: 8px; align-items: center; margin-top: 8px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg); font-size: 0.9rem; }
  .title-opt span { flex: 1; }
  .title-opt button { margin-top: 0; padding: 6px 10px; font-size: 0.78rem; }
  a.dl-link { display: inline-flex; margin-top: 10px; padding: 10px 16px; border-radius: 10px; border: 1px solid var(--border);
    color: var(--text); text-decoration: none; font-weight: 600; font-size: 0.9rem; }
  @media (max-width: 560px) {
    .card { padding: 20px 16px 24px; }
    .prompter { font-size: 1.2rem; padding: 18px; }
    .scene { grid-template-columns: 22px 1fr; }
  }
</style>
</head>
<body>
<div class="page">
<a class="back" href="/">← Back to clips</a>
<div class="card">
  <div class="brand"><span class="logo">🛫</span><h1>Long-form videos</h1></div>
  <p class="subtitle">True aviation stories for your second channel. Claude writes the script from the official report, you read it here one scene at a time, and the app checks every take.</p>

  <div id="list-view">
    <div id="projects-wrap" style="display:none">
      <label style="margin-top:0">Your videos</label>
      <div id="projects"></div>
    </div>

    <label>Start a new video</label>
    <div class="hint">Pick an incident. Claude reads its NTSB report and writes a 10–12 minute script, which takes about a minute.</div>
    <div id="incidents"></div>

    <details id="other-report">
      <summary>Or use another NTSB report</summary>
      <label>Title</label>
      <input id="custom-title" placeholder="e.g. United 232: the DC-10 that lost all its hydraulics">
      <label>Report link (PDF)</label>
      <input id="custom-url" placeholder="https://www.ntsb.gov/investigations/AccidentReports/Reports/....pdf">
      <button id="custom-url-btn" type="button">Write script from link</button>
      <label>Or upload the report PDF</label>
      <input id="custom-pdf" type="file" accept="application/pdf">
      <button id="custom-pdf-btn" type="button" class="secondary">Upload report and write script</button>
      <div class="hint" id="custom-status"></div>
    </details>
  </div>

  <div id="project-view" style="display:none">
    <button id="to-list" type="button" class="linkish">← All long-form videos</button>
    <h2 id="p-title"></h2>
    <div class="hint" id="p-sub" style="margin-top:2px"></div>
    <div class="steps" id="steps"></div>
    <div id="p-status" class="status-box" style="display:none"></div>

    <div class="section" id="script-section" style="display:none">
      <h3><span class="n">1</span>Script</h3>
      <div class="hint">Edit any line before you record it. Changing a scene you've already recorded means reading that scene again.</div>
      <div class="meta" id="script-meta"></div>
      <details id="script-details">
        <summary id="script-summary">Show and edit the script</summary>
        <div id="scenes"></div>
        <div class="actions">
          <button id="save-script" type="button" class="secondary" disabled>💾 Save changes</button>
          <button id="rewrite-script" type="button" class="secondary">🔄 Rewrite script</button>
        </div>
      </details>
    </div>

    <div class="section" id="record-section" style="display:none">
      <h3><span class="n">2</span>Record</h3>
      <div class="hint">Read the scene, then tap Stop. The app listens back, and if you skipped or fluffed something it asks you to read that scene again. Use headphones or a quiet room.</div>
      <div class="chips" id="chips"></div>
      <div class="prompter" id="prompter"></div>
      <div class="rec-row">
        <button id="rec-btn" type="button">🎙 Record scene 1</button>
        <span id="rec-timer" class="rec-timer"></span>
      </div>
      <div id="take-result" class="take-result" style="display:none"></div>
      <div class="actions" id="take-actions">
        <button id="play-take" type="button" class="secondary" style="display:none">▶ Play my take</button>
        <button id="keep-take" type="button" class="secondary" style="display:none">Keep anyway</button>
        <button id="prev-scene" type="button" class="secondary">← Previous</button>
        <button id="next-scene" type="button" class="secondary">Next →</button>
      </div>
      <audio id="take-audio" style="display:none"></audio>
    </div>

    <div class="section" id="narration-section" style="display:none">
      <h3><span class="n">3</span>Narration</h3>
      <div class="hint" id="narration-hint"></div>
      <button id="build-narration" type="button">🎧 Join into one narration</button>
      <div id="narration-out" style="display:none">
        <audio id="narration-audio" controls style="width:100%;margin-top:12px"></audio>
        <a id="narration-dl" class="dl-link" href="#">⬇ Download narration</a>
      </div>
    </div>

    <div class="section" id="visuals-section" style="display:none">
      <h3><span class="n">4</span>Visuals</h3>
      <div class="hint">Claude picks a visual for every scene from the report: route maps, the real cockpit transcript, charts of the flight data, and the report's own photos and diagrams. Change any of them below.</div>
      <div id="stock-note" class="hint" style="display:none">🎞 Stock footage is off until a free Pexels API key is added in Railway as <b>PEXELS_API_KEY</b>. Until then, stock scenes use photos from the report.</div>
      <div id="visuals-status" class="status-box" style="display:none"></div>
      <button id="plan-visuals" type="button">🎨 Plan visuals</button>
      <div id="visual-grid" class="vgrid"></div>
    </div>

    <div class="section" id="render-section" style="display:none">
      <h3><span class="n">5</span>Render</h3>
      <label style="margin-top:4px">Background music (optional)</label>
      <div class="hint">Download a track from the YouTube Audio Library (free to use on monetised videos) and upload it here. It plays quietly and dips while you talk.</div>
      <div id="music-current" class="hint" style="display:none"></div>
      <div class="music-row">
        <input id="music-file" type="file" accept="audio/*">
        <button id="music-upload" type="button" class="secondary">Upload music</button>
        <button id="music-remove" type="button" class="secondary" style="display:none">Remove</button>
      </div>
      <div class="hint" id="render-hint" style="margin-top:14px"></div>
      <button id="render-btn" type="button">🎬 Render video (1080p)</button>
      <div id="render-progress" style="display:none">
        <div class="bar"><div id="render-bar"></div></div>
        <div class="hint" id="render-msg"></div>
      </div>
      <div id="render-error" class="status-box err" style="display:none"></div>
      <div id="render-out" style="display:none">
        <video id="final-video" controls preload="metadata" style="width:100%;border-radius:12px;margin-top:12px;background:#000"></video>
        <div class="actions">
          <a id="video-dl" class="dl-link" href="#">⬇ Download video</a>
          <button id="publish-btn" type="button" class="secondary">✍️ Write title &amp; description</button>
        </div>
        <div class="hint" id="render-info"></div>
        <div id="publish-out" style="display:none">
          <label>Title options</label>
          <div id="title-list"></div>
          <label>Description (with chapters and credits)</label>
          <textarea id="desc-text" rows="12"></textarea>
          <button id="copy-desc" type="button" class="secondary">📋 Copy description</button>
        </div>
      </div>
    </div>

    <div class="section">
      <button id="delete-project" type="button" class="danger-link">🗑 Delete this video</button>
    </div>
  </div>
</div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const VISUALS = [['map', '🗺 Map'], ['cockpit', '💬 Cockpit card'], ['chart', '📈 Chart'], ['stock', '🎞 Stock footage'], ['report', '📄 Report image']];
let project = null;
let recIndex = 0;
let pollTimer = null;
let recorder = null;
let recStream = null;
let recChunks = [];
let recStarted = 0;
let recTick = null;
let scriptDirty = false;
let lastResult = null;

function sceneReady(s) { const t = s.take || {}; return !!t.file && !!(t.ok || t.kept); }
function sceneFlagged(s) { const t = s.take || {}; return !!t.file && !t.ok && !t.kept; }
function fmtTime(sec) { sec = Math.max(0, Math.round(sec || 0)); return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`; }

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
  return data;
}

// ---------- list view ----------
async function showList() {
  stopPolling();
  project = null;
  history.replaceState(null, '', '/long-form');
  $('project-view').style.display = 'none';
  $('list-view').style.display = 'block';
  const [{ projects }, { incidents }] = await Promise.all([api('/api/longform/projects'), api('/api/longform/incidents')]);
  $('projects-wrap').style.display = projects.length ? 'block' : 'none';
  const pl = $('projects');
  pl.innerHTML = '';
  projects.forEach(p => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'project';
    const info = document.createElement('div');
    const t = document.createElement('div'); t.className = 't'; t.textContent = p.title;
    const m = document.createElement('div'); m.className = 'm';
    const state = p.status === 'writing' ? 'Writing script…'
      : p.status === 'error' ? 'Needs attention'
      : p.has_narration ? 'Narration ready'
      : `${p.recorded}/${p.scenes} scenes recorded`;
    m.textContent = `${state} · ~${p.minutes} min`;
    info.appendChild(t); info.appendChild(m); b.appendChild(info);
    b.addEventListener('click', () => openProject(p.id));
    pl.appendChild(b);
  });
  const il = $('incidents');
  il.innerHTML = '';
  incidents.forEach(inc => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'incident';
    const ico = document.createElement('div'); ico.className = 'ico'; ico.textContent = inc.icon;
    const info = document.createElement('div');
    const t = document.createElement('div'); t.className = 't'; t.textContent = inc.title;
    const m = document.createElement('div'); m.className = 'm'; m.textContent = inc.subtitle;
    const tags = document.createElement('div'); tags.className = 'tags';
    inc.tags.forEach(([text, kind]) => { const s = document.createElement('span'); s.className = 'tag ' + kind; s.textContent = text; tags.appendChild(s); });
    info.appendChild(t); info.appendChild(m); info.appendChild(tags);
    b.appendChild(ico); b.appendChild(info);
    b.addEventListener('click', async () => {
      if (!confirm(`Start a video on "${inc.title}"? Claude will write the script from the NTSB report.`)) return;
      b.disabled = true;
      try {
        const { id } = await api('/api/longform/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ incident_id: inc.id }) });
        openProject(id);
      } catch (e) { alert(e.message); } finally { b.disabled = false; }
    });
    il.appendChild(b);
  });
}

$('custom-url-btn').addEventListener('click', async () => {
  const url = $('custom-url').value.trim();
  if (!url) { $('custom-status').textContent = 'Paste the link to the report PDF first.'; return; }
  try {
    const { id } = await api('/api/longform/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ report_url: url, title: $('custom-title').value }) });
    openProject(id);
  } catch (e) { $('custom-status').textContent = e.message; }
});

$('custom-pdf-btn').addEventListener('click', async () => {
  const f = $('custom-pdf').files[0];
  if (!f) { $('custom-status').textContent = 'Choose the report PDF first.'; return; }
  $('custom-status').textContent = 'Uploading…';
  try {
    const { id } = await api(`/api/longform/projects/upload-report?title=${encodeURIComponent($('custom-title').value)}`,
      { method: 'POST', headers: { 'Content-Type': 'application/pdf' }, body: f });
    $('custom-status').textContent = '';
    openProject(id);
  } catch (e) { $('custom-status').textContent = e.message; }
});

// ---------- project view ----------
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function openProject(id) {
  history.replaceState(null, '', `/long-form?v=${encodeURIComponent(id)}`);
  $('list-view').style.display = 'none';
  $('project-view').style.display = 'block';
  scriptDirty = false;
  lastResult = null;
  visualsKey = '';
  await loadProject(id, true);
  // Open the script for reading/editing until recording starts; after
  // that the recorder is what matters, so keep the long script folded.
  if (project) $('script-details').open = !(project.scenes || []).some(s => s.take);
}

async function loadProject(id, resetIndex) {
  try {
    project = await api(`/api/longform/projects/${encodeURIComponent(id)}`);
  } catch (e) { alert(e.message); showList(); return; }
  if (resetIndex) {
    const firstTodo = (project.scenes || []).findIndex(s => !sceneReady(s));
    recIndex = firstTodo === -1 ? 0 : firstTodo;
  }
  render();
  stopPolling();
  if (project.status === 'writing') pollTimer = setInterval(() => loadProject(project.id, true), 3000);
  else if (project.visuals_status === 'planning' || (project.render || {}).status === 'rendering') {
    pollTimer = setInterval(() => loadProject(project.id, false), 3000);
  }
}

function render() {
  const p = project;
  const scenes = p.scenes || [];
  $('p-title').textContent = (p.incident || {}).title || 'Untitled';
  $('p-sub').textContent = (p.incident || {}).subtitle || '';
  const ready = scenes.filter(sceneReady).length;
  const hasScript = scenes.length > 0 && p.status !== 'writing';
  const steps = [
    ['1 Script', hasScript ? 'done' : 'on'],
    ['2 Record', !hasScript ? '' : ready === scenes.length ? 'done' : 'on'],
    ['3 Narration', p.narration ? 'done' : (hasScript && ready === scenes.length ? 'on' : '')],
    ['4 Visuals', scenes.length && scenes.every(s => s.spec) ? 'done' : (p.narration ? 'on' : '')],
    ['5 Render', (p.render || {}).status === 'done' ? 'done' : (p.narration && scenes.every(s => s.spec) ? 'on' : '')],
  ];
  $('steps').innerHTML = '';
  steps.forEach(([t, cls]) => { const s = document.createElement('span'); s.textContent = t + (cls === 'done' ? ' ✓' : ''); if (cls) s.className = cls; $('steps').appendChild(s); });

  const st = $('p-status');
  if (p.status === 'writing') {
    st.style.display = 'block'; st.className = 'status-box';
    st.textContent = '⏳ ' + (p.message || 'Writing the script…');
  } else if (p.status === 'error') {
    st.style.display = 'block'; st.className = 'status-box err';
    st.textContent = '⚠️ ' + (p.error || 'Something went wrong.') + ' Use Rewrite script to try again.';
  } else { st.style.display = 'none'; }

  $('script-section').style.display = (hasScript || p.status === 'error') ? 'block' : 'none';
  $('record-section').style.display = hasScript ? 'block' : 'none';
  $('narration-section').style.display = hasScript ? 'block' : 'none';
  if (!scriptDirty) renderScript();
  $('visuals-section').style.display = hasScript ? 'block' : 'none';
  $('render-section').style.display = hasScript ? 'block' : 'none';
  if (hasScript) { renderRecorder(); renderNarration(); renderVisuals(); renderRender(); }
}

function renderScript() {
  const scenes = project.scenes || [];
  const words = scenes.reduce((n, s) => n + s.narration.split(' ').length, 0);
  $('script-meta').innerHTML = '';
  [[`~${(words / 150).toFixed(1)} min`, ' read'], [String(words), ' words'], [String(scenes.length), ' scenes']].forEach(([b, t]) => {
    const s = document.createElement('span'); const bb = document.createElement('b'); bb.textContent = b; s.appendChild(bb); s.appendChild(document.createTextNode(t)); $('script-meta').appendChild(s);
  });
  const wrap = $('scenes');
  wrap.innerHTML = '';
  scenes.forEach((s, i) => {
    const row = document.createElement('div'); row.className = 'scene';
    const num = document.createElement('div'); num.className = 'num'; num.textContent = i + 1;
    const body = document.createElement('div');
    const ta = document.createElement('textarea'); ta.value = s.narration; ta.dataset.i = i;
    ta.addEventListener('input', () => { scriptDirty = true; $('save-script').disabled = false; });
    const vr = document.createElement('div'); vr.className = 'vis-row';
    const sel = document.createElement('select'); sel.dataset.i = i;
    VISUALS.forEach(([v, label]) => { const o = document.createElement('option'); o.value = v; o.textContent = label; if (v === s.visual) o.selected = true; sel.appendChild(o); });
    sel.addEventListener('change', () => { scriptDirty = true; $('save-script').disabled = false; });
    const note = document.createElement('span'); note.className = 'vnote'; note.textContent = s.visual_note || ''; note.dataset.note = s.visual_note || '';
    const badge = document.createElement('span'); badge.className = 'rec-badge';
    badge.textContent = sceneReady(s) ? '✅ recorded' : sceneFlagged(s) ? '⚠️ retake' : '';
    vr.appendChild(sel); vr.appendChild(note); vr.appendChild(badge);
    body.appendChild(ta); body.appendChild(vr);
    row.appendChild(num); row.appendChild(body);
    wrap.appendChild(row);
  });
  $('save-script').disabled = true;
}

$('save-script').addEventListener('click', async () => {
  const rows = [...$('scenes').querySelectorAll('.scene')];
  const scenes = rows.map(r => ({
    narration: r.querySelector('textarea').value,
    visual: r.querySelector('select').value,
    visual_note: r.querySelector('.vnote').dataset.note || '',
  }));
  $('save-script').disabled = true;
  try {
    project = await api(`/api/longform/projects/${project.id}/scenes`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenes }) });
    scriptDirty = false;
    if (recIndex >= project.scenes.length) recIndex = 0;
    render();
  } catch (e) { alert(e.message); $('save-script').disabled = false; }
});

$('rewrite-script').addEventListener('click', async () => {
  if (!confirm('Have Claude write a fresh script? Your edits and any recorded scenes will be replaced.')) return;
  try {
    await api(`/api/longform/projects/${project.id}/rewrite`, { method: 'POST' });
    scriptDirty = false;
    loadProject(project.id, true);
  } catch (e) { alert(e.message); }
});

// ---------- recorder ----------
function renderRecorder() {
  const scenes = project.scenes;
  if (recIndex >= scenes.length) recIndex = scenes.length - 1;
  const chips = $('chips');
  chips.innerHTML = '';
  scenes.forEach((s, i) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = String(i + 1);
    b.className = (sceneReady(s) ? 'ready' : sceneFlagged(s) ? 'flag' : '') + (i === recIndex ? ' cur' : '');
    b.title = sceneReady(s) ? 'Recorded' : sceneFlagged(s) ? 'Needs a retake' : 'Not recorded yet';
    b.addEventListener('click', () => { if (recorder) return; recIndex = i; lastResult = null; renderRecorder(); });
    chips.appendChild(b);
  });
  const s = scenes[recIndex];
  const take = s.take || {};
  const missed = new Set(sceneFlagged(s) ? (take.missed || []) : []);
  const pr = $('prompter');
  pr.innerHTML = '';
  const label = document.createElement('div'); label.className = 'label';
  label.textContent = `Scene ${recIndex + 1} of ${scenes.length}`;
  pr.appendChild(label);
  s.narration.split(' ').forEach((w, i) => {
    const span = document.createElement('span');
    span.textContent = w + ' ';
    if (missed.has(i)) span.className = 'miss';
    pr.appendChild(span);
  });
  const rb = $('rec-btn');
  if (!recorder) {
    rb.textContent = sceneFlagged(s) ? `🎙 Read scene ${recIndex + 1} again` : sceneReady(s) ? `🎙 Re-record scene ${recIndex + 1}` : `🎙 Record scene ${recIndex + 1}`;
    rb.className = '';
  }
  const res = $('take-result');
  const show = lastResult || (take.file ? take : null);
  if (show && !recorder) {
    res.style.display = 'block';
    const good = show.ok || show.kept;
    res.className = 'take-result ' + (good ? 'ok' : 'bad');
    res.textContent = good ? (show.kept && !show.ok ? '✅ Kept as recorded.' : '✅ ' + (show.message || 'Sounds right.')) : '⚠️ ' + show.message;
    if (show.heard) { const h = document.createElement('span'); h.className = 'heard'; h.textContent = 'Heard: ' + show.heard; res.appendChild(h); }
  } else if (!recorder) { res.style.display = 'none'; }
  $('play-take').style.display = take.file ? 'inline-block' : 'none';
  $('keep-take').style.display = sceneFlagged(s) ? 'inline-block' : 'none';
  $('prev-scene').disabled = recIndex === 0 || !!recorder;
  $('next-scene').disabled = recIndex === scenes.length - 1 || !!recorder;
}

function pickMime() {
  const opts = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
  return opts.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || '';
}

async function startRecording() {
  if (!navigator.mediaDevices || !window.MediaRecorder) { alert("This browser can’t record audio. Try Chrome or Safari."); return; }
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (e) { alert('Microphone access was blocked. Allow it for this site and try again.'); return; }
  const mime = pickMime();
  recorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined);
  recChunks = [];
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
  recorder.onstop = uploadTake;
  recorder.start();
  recStarted = Date.now();
  lastResult = null;
  $('take-result').style.display = 'none';
  const rb = $('rec-btn');
  rb.textContent = '⏹ Stop and check';
  rb.className = 'recording';
  $('rec-timer').textContent = '0:00';
  recTick = setInterval(() => { $('rec-timer').textContent = fmtTime((Date.now() - recStarted) / 1000); }, 500);
  renderRecorder();
}

function stopRecording() {
  if (!recorder) return;
  clearInterval(recTick);
  recorder.stop();
  recStream.getTracks().forEach(t => t.stop());
}

async function uploadTake() {
  const idx = recIndex;
  const type = (recorder && recorder.mimeType) || recChunks[0]?.type || 'audio/webm';
  const blob = new Blob(recChunks, { type });
  recorder = null;
  $('rec-timer').textContent = '';
  const rb = $('rec-btn');
  rb.disabled = true;
  rb.className = '';
  rb.textContent = 'Checking…';
  const res = $('take-result');
  res.style.display = 'block';
  res.className = 'take-result wait';
  res.textContent = '👂 Listening back to your take…';
  try {
    const { take } = await api(`/api/longform/projects/${project.id}/scenes/${idx}/take`, {
      method: 'POST', headers: { 'Content-Type': type.split(';')[0] }, body: blob,
    });
    lastResult = take;
    project.scenes[idx].take = take;
    project.narration = null;
    rb.disabled = false;
    render();
    if (take.ok) {
      const next = project.scenes.findIndex((s, i) => i > idx && !sceneReady(s));
      if (next !== -1) setTimeout(() => { if (!recorder && recIndex === idx) { recIndex = next; lastResult = null; renderRecorder(); } }, 1500);
    }
  } catch (e) {
    rb.disabled = false;
    lastResult = null;
    renderRecorder();
    res.style.display = 'block';
    res.className = 'take-result bad';
    res.textContent = '⚠️ ' + e.message;
  }
}

$('rec-btn').addEventListener('click', () => { if (recorder) stopRecording(); else startRecording(); });
$('prev-scene').addEventListener('click', () => { if (recIndex > 0) { recIndex--; lastResult = null; renderRecorder(); } });
$('next-scene').addEventListener('click', () => { if (recIndex < project.scenes.length - 1) { recIndex++; lastResult = null; renderRecorder(); } });
$('play-take').addEventListener('click', () => {
  const a = $('take-audio');
  a.src = `/api/longform/projects/${project.id}/scenes/${recIndex}/take?t=${Date.now()}`;
  a.play();
});
$('keep-take').addEventListener('click', async () => {
  try {
    project = await api(`/api/longform/projects/${project.id}/scenes/${recIndex}/keep`, { method: 'POST' });
    lastResult = null;
    render();
  } catch (e) { alert(e.message); }
});

// ---------- narration ----------
function renderNarration() {
  const scenes = project.scenes;
  const ready = scenes.filter(sceneReady).length;
  const all = ready === scenes.length;
  $('build-narration').disabled = !all;
  $('narration-hint').textContent = all
    ? (project.narration ? `Narration ready: ${fmtTime(project.narration.duration)} long.` : 'Every scene is recorded. Join them into one narration track.')
    : `${ready} of ${scenes.length} scenes recorded. Record the rest to continue.`;
  const out = $('narration-out');
  if (project.narration) {
    out.style.display = 'block';
    const src = `/api/longform/projects/${project.id}/narration?t=${Math.round(project.narration.built_at || 0)}`;
    if ($('narration-audio').getAttribute('src') !== src) $('narration-audio').setAttribute('src', src);
    $('narration-dl').href = src;
    $('build-narration').textContent = '🎧 Join again';
  } else {
    out.style.display = 'none';
    $('build-narration').textContent = '🎧 Join into one narration';
  }
}

$('build-narration').addEventListener('click', async () => {
  const b = $('build-narration');
  b.disabled = true;
  b.textContent = 'Joining…';
  try {
    project = await api(`/api/longform/projects/${project.id}/narration`, { method: 'POST' });
    render();
  } catch (e) { alert(e.message); render(); }
});

$('delete-project').addEventListener('click', async () => {
  if (!confirm("Delete this video, its script and every recorded take? This can’t be undone.")) return;
  try { await api(`/api/longform/projects/${project.id}`, { method: 'DELETE' }); showList(); } catch (e) { alert(e.message); }
});
$('to-list').addEventListener('click', () => { if (recorder) stopRecording(); showList(); });

// ---------- visuals ----------
let settings = { stock_enabled: false };
api('/api/longform/settings').then(s => { settings = s; if (project) render(); }).catch(() => {});
const VIS_LABELS = { map: '🗺 Map', cockpit: '💬 Cockpit card', chart: '📈 Chart', stock: '🎞 Stock footage', report: '📄 Report image' };
let visualsKey = '';

async function editVisual(i, body) {
  try {
    project = await api(`/api/longform/projects/${project.id}/visuals/${i}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    render();
  } catch (e) { alert(e.message); }
}

function renderVisuals() {
  const p = project;
  const scenes = p.scenes || [];
  const planned = scenes.length && scenes.every(s => s.spec);
  const planning = p.visuals_status === 'planning';
  const rendering = (p.render || {}).status === 'rendering';
  $('stock-note').style.display = settings.stock_enabled ? 'none' : 'block';
  const st = $('visuals-status');
  if (planning) { st.style.display = 'block'; st.className = 'status-box'; st.textContent = '⏳ ' + (p.visuals_message || 'Planning…'); }
  else if (p.visuals_status === 'error') { st.style.display = 'block'; st.className = 'status-box err'; st.textContent = '⚠️ ' + (p.visuals_error || 'Planning failed.'); }
  else st.style.display = 'none';
  const pb = $('plan-visuals');
  pb.disabled = planning || rendering;
  pb.textContent = planned ? '🔄 Re-plan all visuals' : '🎨 Plan visuals';
  pb.className = planned ? 'secondary' : '';

  const narrKey = p.narration ? Math.round(p.narration.built_at || 0) : 0;
  const key = JSON.stringify([planned, narrKey, rendering, settings.stock_enabled, (p.report_images || []).length, scenes.map(s => s.spec && s.spec.rev)]);
  if (key === visualsKey) return;
  visualsKey = key;
  const grid = $('visual-grid');
  grid.innerHTML = '';
  if (!planned) return;
  scenes.forEach((s, i) => {
    const spec = s.spec;
    const card = document.createElement('div'); card.className = 'vcard';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = `Scene ${i + 1} visual`;
    img.src = `/api/longform/projects/${p.id}/visuals/${i}/preview?rev=${spec.rev || 0}&n=${narrKey}`;
    card.appendChild(img);
    const body = document.createElement('div'); body.className = 'vbody';
    const head = document.createElement('div'); head.className = 'vhead';
    const b = document.createElement('b'); b.textContent = `Scene ${i + 1}`;
    const sel = document.createElement('select');
    const types = ['report', 'stock', 'cockpit'];
    const pv = (s.planned || {}).visual;
    if (pv && !types.includes(pv)) types.unshift(pv);
    if (!types.includes(spec.visual)) types.unshift(spec.visual);
    types.forEach(t => { const o = document.createElement('option'); o.value = t; o.textContent = VIS_LABELS[t] || t; if (t === spec.visual) o.selected = true; sel.appendChild(o); });
    sel.disabled = rendering;
    sel.addEventListener('change', () => editVisual(i, { visual: sel.value }));
    head.appendChild(b); head.appendChild(sel);
    body.appendChild(head);
    const cap = document.createElement('input');
    cap.placeholder = 'Caption (optional)'; cap.value = spec.caption || ''; cap.disabled = rendering;
    cap.addEventListener('change', () => editVisual(i, { caption: cap.value }));
    body.appendChild(cap);
    const btns = document.createElement('div'); btns.className = 'vbtns';
    const usesImage = spec.visual === 'report' || (spec.visual === 'stock' && !settings.stock_enabled);
    if (usesImage && (p.report_images || []).length > 1) {
      const prev = document.createElement('button'); prev.type = 'button'; prev.className = 'secondary'; prev.textContent = '◀ Image';
      const next = document.createElement('button'); next.type = 'button'; next.className = 'secondary'; next.textContent = 'Image ▶';
      prev.disabled = next.disabled = rendering;
      prev.addEventListener('click', () => editVisual(i, { image_step: -1 }));
      next.addEventListener('click', () => editVisual(i, { image_step: 1 }));
      btns.appendChild(prev); btns.appendChild(next);
    }
    if (spec.visual === 'stock' && settings.stock_enabled) {
      const q = document.createElement('input'); q.value = spec.stock_query || ''; q.placeholder = 'Stock search words'; q.disabled = rendering;
      q.addEventListener('change', () => editVisual(i, { stock_query: q.value }));
      body.appendChild(q);
      const nx = document.createElement('button'); nx.type = 'button'; nx.className = 'secondary'; nx.textContent = 'Next clip ▶'; nx.disabled = rendering;
      nx.addEventListener('click', () => editVisual(i, { stock_step: 1 }));
      btns.appendChild(nx);
    }
    if (spec.visual === 'cockpit') {
      const ta = document.createElement('textarea'); ta.rows = 3; ta.disabled = rendering;
      ta.placeholder = 'One line each, like  CAPTAIN: my aircraft.';
      ta.value = (spec.lines || []).map(l => (l.speaker ? l.speaker + ': ' : '') + l.text).join(String.fromCharCode(10));
      ta.addEventListener('change', () => editVisual(i, { lines: ta.value }));
      body.appendChild(ta);
    }
    if (btns.children.length) body.appendChild(btns);
    const tx = document.createElement('div'); tx.className = 'vtext';
    tx.textContent = s.narration.length > 110 ? s.narration.slice(0, 110) + '…' : s.narration;
    body.appendChild(tx);
    card.appendChild(body);
    grid.appendChild(card);
  });
}

$('plan-visuals').addEventListener('click', async () => {
  const planned = (project.scenes || []).every(s => s.spec);
  if (planned && !confirm('Re-plan every scene? Your visual changes will be replaced.')) return;
  try {
    await api(`/api/longform/projects/${project.id}/visuals/plan`, { method: 'POST' });
    visualsKey = '';
    loadProject(project.id, false);
  } catch (e) { alert(e.message); }
});

// ---------- render ----------
function renderRender() {
  const p = project;
  const scenes = p.scenes || [];
  const planned = scenes.length && scenes.every(s => s.spec);
  const r = p.render || {};
  const rendering = r.status === 'rendering';
  const music = p.music;
  $('music-current').style.display = music ? 'block' : 'none';
  $('music-current').textContent = music ? `🎵 ${music.name}` : '';
  $('music-remove').style.display = music ? 'inline-block' : 'none';
  $('music-upload').disabled = $('music-remove').disabled = rendering;
  const rb = $('render-btn');
  rb.disabled = rendering || !p.narration || !planned;
  rb.textContent = r.status === 'done' ? '🎬 Render again' : '🎬 Render video (1080p)';
  $('render-hint').textContent = !p.narration ? 'Join the narration (step 3) first.'
    : !planned ? 'Plan the visuals (step 4) first.'
    : rendering ? '' : `About ${Math.max(1, Math.round((p.narration.duration || 0) / 60))} min of video. Rendering takes roughly as long as the video, and you can leave this page while it runs.`;
  $('render-progress').style.display = rendering ? 'block' : 'none';
  if (rendering) {
    $('render-bar').style.width = Math.round((r.progress || 0) * 100) + '%';
    $('render-msg').textContent = r.message || 'Rendering…';
  }
  $('render-error').style.display = r.status === 'error' ? 'block' : 'none';
  $('render-error').textContent = r.status === 'error' ? '⚠️ ' + (r.error || 'Render failed.') : '';
  const out = $('render-out');
  if (r.status === 'done') {
    out.style.display = 'block';
    const src = `/api/longform/projects/${p.id}/video?t=${Math.round(r.built_at || 0)}`;
    if ($('final-video').getAttribute('src') !== src) $('final-video').setAttribute('src', src);
    $('video-dl').href = src;
    $('render-info').textContent = `${fmtTime(r.duration)} long · rendered in ${Math.max(1, Math.round((r.took_seconds || 0) / 60))} min.`;
  } else out.style.display = 'none';
  const pub = p.publish;
  $('publish-out').style.display = pub ? 'block' : 'none';
  if (pub) {
    const tl = $('title-list');
    tl.innerHTML = '';
    (pub.titles || []).forEach(t => {
      const row = document.createElement('div'); row.className = 'title-opt';
      const sp = document.createElement('span'); sp.textContent = t;
      const cp = document.createElement('button'); cp.type = 'button'; cp.className = 'secondary'; cp.textContent = '📋 Copy';
      cp.addEventListener('click', () => navigator.clipboard && navigator.clipboard.writeText(t));
      row.appendChild(sp); row.appendChild(cp); tl.appendChild(row);
    });
    if (document.activeElement !== $('desc-text')) $('desc-text').value = pub.description || '';
  }
}

$('render-btn').addEventListener('click', async () => {
  try {
    await api(`/api/longform/projects/${project.id}/render`, { method: 'POST' });
    loadProject(project.id, false);
  } catch (e) { alert(e.message); }
});

$('music-upload').addEventListener('click', async () => {
  const f = $('music-file').files[0];
  if (!f) { alert('Choose a music file first.'); return; }
  $('music-upload').disabled = true;
  try {
    project = await api(`/api/longform/projects/${project.id}/music?name=${encodeURIComponent(f.name)}`,
      { method: 'POST', headers: { 'Content-Type': f.type || 'audio/mpeg' }, body: f });
    $('music-file').value = '';
    render();
  } catch (e) { alert(e.message); } finally { $('music-upload').disabled = false; }
});
$('music-remove').addEventListener('click', async () => {
  try { project = await api(`/api/longform/projects/${project.id}/music`, { method: 'DELETE' }); render(); } catch (e) { alert(e.message); }
});

$('publish-btn').addEventListener('click', async () => {
  const b = $('publish-btn');
  b.disabled = true; b.textContent = 'Writing…';
  try { project = await api(`/api/longform/projects/${project.id}/publish-text`, { method: 'POST' }); render(); }
  catch (e) { alert(e.message); }
  finally { b.disabled = false; b.textContent = '✍️ Write title & description'; }
});
$('copy-desc').addEventListener('click', () => navigator.clipboard && navigator.clipboard.writeText($('desc-text').value));

const startId = new URLSearchParams(location.search).get('v');
if (startId) openProject(startId); else showList();
</script>
</body>
</html>
"""
