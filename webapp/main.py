"""Web wrapper around the clipper CLI pipeline: submit a video, poll status,
download the rendered clips. One job runs at a time on a background worker
thread so a small Railway instance doesn't try to transcode multiple videos
at once.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
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
from clipper import documentary
from clipper import longform
from clipper import longform_video
from clipper import voice_clone
from clipper import clip_search
from clipper import visual_sources
from clipper import longform_lessons
from clipper import longform_analytics
from clipper import explainer
from clipper import longform_promo
from clipper import longform_thumbnail
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
# YouTube accounts that aren't Shorts channel profiles: Caught On Code (the
# long-form explainers, see clipper/explainer.py) has no clip pipeline or
# Twitch watchlist, so it isn't in CHANNEL_PROFILES (that would add it to
# Home's channel switcher) -- it only needs its own connected account.
EXTRA_YOUTUBE_ACCOUNTS: dict = {
    explainer.PROFILE: {"label": explainer.CHANNEL, "token_file": "_youtube_oauth_token_code.json",
                        "page_path": "/caught-on-code"},
}
for _acct, _cfg in EXTRA_YOUTUBE_ACCOUNTS.items():
    _youtube_token_stores[_acct] = youtube_oauth.TokenStore(BASE_DIR / _cfg["token_file"])


def _oauth_account(profile: Optional[str]) -> str:
    """A channel profile or an extra YouTube account; anything else means
    the main channel."""
    return profile if profile in EXTRA_YOUTUBE_ACCOUNTS else _profile_or_default(profile)
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
    try:
        for vid, extra in youtube_analytics.get_video_engagement(access_token, own["id"]).items():
            metrics.setdefault(vid, {}).update(extra)
    except Exception as e:  # noqa: BLE001 - engagement is extra; retention and views stand without it
        print(f"[clip_performance] engagement unavailable: {e}", flush=True)
    curves = _load_retention_curves(access_token, own["id"], videos)
    stats = clip_performance.build_stats(videos, metrics, curves, records,
                                         streamers=_profile_logins(DEFAULT_CHANNEL_PROFILE))
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
            synthetic_media=bool(clip.get("synthetic_voice")),
        )
    except (youtube_upload.UploadError, ValueError) as e:
        raise HTTPException(502, str(e)) from e
    finally:
        if trimmed_path is not None:
            trimmed_path.unlink(missing_ok=True)

    # Link the posted video back to this clip so its real retention can be
    # compared against what the clip looked like (see /api/clip-performance).
    # Not for a long-form episode's cliffhanger promo: it wasn't picked by
    # the clip pipeline, so it would skew what that learns.
    if not is_recap and not clip.get("promo"):
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
    profile = _oauth_account(profile)
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
    profile = _oauth_account(issued[1])
    token = youtube_oauth.exchange_code(code, _youtube_redirect_uri())
    _youtube_token_stores[profile].save(token)
    if profile in EXTRA_YOUTUBE_ACCOUNTS:
        return RedirectResponse(f"{EXTRA_YOUTUBE_ACCOUNTS[profile]['page_path']}?youtube_connected=1")
    # The main channel's connect flow also lives on the analytics page (it
    # reads real Analytics data there); every profile's own channel page
    # (see CHANNEL_PROFILES' page_path) shows connect status too, and is
    # where a second profile's connect button sends you from.
    if profile == DEFAULT_CHANNEL_PROFILE:
        return RedirectResponse("/analytics?youtube_connected=1")
    return RedirectResponse(f"{CHANNEL_PROFILES[profile]['page_path']}?youtube_connected=1")


@protected.post("/api/youtube/disconnect")
def youtube_disconnect(profile: str = DEFAULT_CHANNEL_PROFILE) -> dict:
    _youtube_token_stores[_oauth_account(profile)].clear()
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


_longform_analytics_cache: dict = {}


def _longform_episode_analytics(access_token: str, channel_id: str, project: dict) -> dict:
    when = project.get("uploaded_at") or project.get("created_at") or time.time()
    start = (datetime.date.fromtimestamp(when) - datetime.timedelta(days=1)).isoformat()
    vid = project["youtube_video_id"]
    stats = youtube_analytics.get_video_stats(access_token, channel_id, vid, start)
    extras = {}
    for name, fn in (("curve", youtube_analytics.get_retention_curve), ("daily", youtube_analytics.get_video_daily),
                     ("traffic", youtube_analytics.get_video_traffic)):
        try:
            extras[name] = fn(access_token, channel_id, vid, start)
        except Exception as e:  # noqa: BLE001 - one missing part shouldn't hide the rest
            print(f"[longform-analytics] {name} for {vid} unavailable: {e}", flush=True)
            extras[name] = []
    shorts = []
    with jobs_lock:
        job = jobs.get(project.get("promo_job_id") or "") or {}
        posted = [dict(c) for c in job.get("clips") or [] if c.get("youtube_video_id")]
    for c in posted:
        row = {"title": c.get("upload_title") or c.get("title") or "", "video_id": c["youtube_video_id"],
               "url": f"https://youtube.com/shorts/{c['youtube_video_id']}"}
        try:
            row.update(youtube_analytics.get_video_stats(access_token, channel_id, c["youtube_video_id"], start))
        except Exception as e:  # noqa: BLE001
            print(f"[longform-analytics] Short {c['youtube_video_id']} unavailable: {e}", flush=True)
        shorts.append(row)
    return longform_analytics.build_episode(project, stats, extras["curve"], extras["daily"], extras["traffic"], shorts)


@protected.get("/api/longform-analytics")
def longform_analytics_report(refresh: bool = False) -> dict:
    """How each posted "Story Of" episode is doing: views, watch time,
    retention laid over its scenes and chapters, traffic sources, and its
    cliffhanger Shorts. Backs the long-form panel on the Analytics page."""
    cached = _longform_analytics_cache.get("data")
    if not refresh and cached and time.time() - _longform_analytics_cache.get("at", 0) < _CLIP_PERFORMANCE_TTL_SECONDS:
        return cached
    posted = [p for p in _longform_store.list() if p.get("youtube_video_id")]
    if not posted:
        return {"available": False, "reason": "No long-form episode is on YouTube yet. Once you upload one from the long-form page, its numbers show up here."}
    if not youtube_oauth.is_configured():
        return {"available": False, "reason": "YouTube OAuth isn't configured on this deployment."}
    accounts: dict = {}  # account -> (token, channel) or an error string

    def account(p: dict):
        key = _longform_account(p)
        if key not in accounts:
            token = _youtube_token_stores[key].get_valid_access_token()
            if not token:
                accounts[key] = (f"Connect the {_longform_brand(p)} YouTube account to see this episode's numbers."
                                 if key != DEFAULT_CHANNEL_PROFILE else
                                 "Connect your YouTube account (Channel insights, above) to see how your episodes are doing.")
            else:
                try:
                    own = youtube_analytics.get_own_channel(token)
                    accounts[key] = (token, own) if own else "The connected Google account has no YouTube channel."
                except Exception as e:  # noqa: BLE001
                    accounts[key] = f"Couldn't reach YouTube Analytics: {e}"
        return accounts[key]

    if all(isinstance(account(p), str) for p in posted):
        return {"available": False, "reason": account(posted[0])}
    episodes = []
    for p in posted:
        acct = account(p)
        if isinstance(acct, str):
            episodes.append({"id": p["id"], "title": p.get("title") or "Untitled", "url": p.get("youtube_url"), "error": acct,
                             "channel": _longform_brand(p)})
            continue
        try:
            episodes.append({**_longform_episode_analytics(acct[0], acct[1]["id"], p), "channel": _longform_brand(p)})
        except Exception as e:  # noqa: BLE001 - show the reason next to that episode instead of failing the panel
            traceback.print_exc()
            episodes.append({"id": p["id"], "title": p.get("title") or "Untitled", "url": p.get("youtube_url"),
                             "error": f"Couldn't load its numbers: {e}"})
    data = {"available": True, "episodes": episodes, "generated_at": time.time()}
    _longform_analytics_cache.update(at=time.time(), data=data)
    return data


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


@protected.get("/caught-on-code", response_class=HTMLResponse)
def caught_on_code_page() -> str:
    return EXPLAINER_HTML


@protected.get("/voices", response_class=HTMLResponse)
def voices_page() -> str:
    return VOICES_HTML.replace("__NAV_LINKS__", _nav_links("/voices"))


# ---- Long-form: "The Story Of" streamer documentaries ------------------------
# A bi-weekly series on the main (Caught On Stream) channel, a different
# streamer each episode. Research -> story -> record -> render -> post.
# See clipper/documentary.py (research + story), clipper/longform.py
# (recording + storage) and clipper/longform_video.py (render).
_longform_store = longform.ProjectStore(BASE_DIR / "_longform")
_longform_lessons = longform_lessons.LessonStore(BASE_DIR / "_longform" / "_lessons.json")
_longform_series_path = BASE_DIR / "_longform" / "_series.json"
_longform_busy: set = set()  # (project id, "research" | "write" | "render") running right now
_longform_lock = threading.Lock()


def _longform_claim(pid: str, kind: str) -> None:
    with _longform_lock:
        if (pid, kind) in _longform_busy:
            raise HTTPException(409, "already running -- wait for it to finish")
        _longform_busy.add((pid, kind))


def _longform_release(pid: str, kind: str) -> None:
    with _longform_lock:
        _longform_busy.discard((pid, kind))


def _longform_is_busy(pid: str, kind: str) -> bool:
    with _longform_lock:
        return (pid, kind) in _longform_busy


def _longform_brand(project: Optional[dict] = None) -> str:
    if project and project.get("kind") == explainer.KIND:
        return explainer.CHANNEL
    return CHANNEL_PROFILES[DEFAULT_CHANNEL_PROFILE]["brand_name"]


def _is_explainer(project: dict) -> bool:
    return project.get("kind") == explainer.KIND


def _longform_account(project: dict) -> str:
    """Which YouTube account an episode goes to: Caught On Code for the
    explainers, Caught On Stream for the documentaries."""
    return explainer.PROFILE if _is_explainer(project) else DEFAULT_CHANNEL_PROFILE


def _longform_access_token(project: dict) -> str:
    token = _youtube_token_stores[_longform_account(project)].get_valid_access_token()
    if not token:
        where = "on the Caught On Code page" if _is_explainer(project) else "on the Home page"
        raise HTTPException(409, f"Connect the {_longform_brand(project)} YouTube account first ({where}).")
    return token


def _longform_project(pid: str) -> dict:
    try:
        project = _longform_store.load(pid)
    except (KeyError, OSError, ValueError):
        raise HTTPException(404, "long-form video not found")
    # Work that was running when the server restarted can't finish -- say so
    # instead of leaving the page waiting forever.
    for status, kind, msg in (("researching", "research", "Interrupted by a restart -- click Run research again."),
                              ("writing", "write", "Interrupted by a restart -- click Write the story again.")):
        if project.get("status") == status and not _longform_is_busy(pid, kind):
            fallback = "research_ready" if kind == "write" and project.get("library") else "error"
            project = _longform_store.update(pid, lambda pr, f=fallback, m=msg: pr.update(status=f, error=m, message=None))
    if (project.get("visuals") or {}).get("status") == "planning" and not _longform_is_busy(pid, "visuals"):
        project = _longform_store.update(pid, lambda pr: pr.update(
            visuals={"status": "error", "error": "Interrupted by a restart -- click Plan visuals again."}))
    if (project.get("clip_search") or {}).get("status") == "searching" and not _longform_is_busy(pid, "clips"):
        project = _longform_store.update(pid, lambda pr: pr.update(clip_search={
            **pr["clip_search"], "status": "error", "message": None,
            "error": "Interrupted by a restart -- press Find clips again."}))
    if (project.get("ai_all") or {}).get("status") == "running" and not _longform_is_busy(pid, "ai_all"):
        project = _longform_store.update(pid, lambda pr: pr.update(ai_all={
            **pr["ai_all"], "status": "error", "current": None,
            "error": "Interrupted by a restart -- press it again; the scenes it already read are kept."}))
    if (project.get("render") or {}).get("status") == "rendering" and not _longform_is_busy(pid, "render"):
        project = _longform_store.update(pid, lambda pr: pr.update(
            render={"status": "error", "error": "Interrupted by a restart -- click Render again.", "progress": 0}))
    return project


def _longform_library(project: dict) -> dict:
    return {c["id"]: c for c in project.get("library") or []}


def _longform_research(pid: str) -> None:
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        say = lambda m: _longform_store.update(pid, lambda pr: pr.update(message=m))  # noqa: E731
        if _is_explainer(project):
            dossier = explainer.build_dossier(project["topic"], project.get("notes") or "", project.get("links") or [], on_progress=say)
            (d / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
            _longform_store.update(pid, lambda pr: pr.update(
                status="research_ready", error=None, message=None,
                sources={"pages": explainer.sources(dossier), "failed_links": dossier.get("failed_links") or []}))
            # Look for streamer clips of it in the background, the first time.
            if os.environ.get("TWITCH_CLIENT_ID") and not project.get("clip_search"):
                try:
                    _longform_start_clip_search(pid, [])
                except HTTPException:
                    pass  # one is already running
            return
        dossier = documentary.build_dossier(project["login"], project.get("notes") or "", project.get("links") or [], on_progress=say)
        (d / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
        profile = dossier["profile"]
        _longform_store.update(pid, lambda pr: pr.update(
            streamer=profile, title=f"The Story of {profile['display_name']}",
            sources={"wikipedia": (dossier.get("wikipedia") or {}).get("url"),
                     "articles": [{"url": a["url"], "title": a["title"]} for a in dossier.get("articles") or []],
                     "failed_links": dossier.get("failed_links") or []}))
        library = documentary.build_library(
            dossier["clips"], d / "clips",
            on_progress=lambda i, n: say(f"Downloading and transcribing clip {min(i + 1, n)} of {n} (a few minutes)..."),
        )
        if not library:
            raise RuntimeError("None of their clips could be downloaded -- try again in a bit.")
        _longform_store.update(pid, lambda pr: pr.update(status="research_ready", error=None, message=None, library=library))
    except Exception as e:
        print(f"[longform] research for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _longform_store.update(pid, lambda pr: pr.update(status="error", error=err, message=None))
        except Exception:
            pass
    finally:
        _longform_release(pid, "research")


def _longform_write(pid: str) -> None:
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        dossier = json.loads((d / "dossier.json").read_text(encoding="utf-8"))
        if _is_explainer(project):
            scenes = explainer.write_script(dossier, library=project.get("library") or [], project_dir=d)
        else:
            scenes = documentary.write_script(dossier, project["library"], d, _longform_brand(), _longform_lessons.texts())
        _longform_store.update(pid, lambda pr: pr.update(status="script_ready", error=None, message=None, scenes=scenes,
                                                         render=None, publish=None, visuals=None))
    except Exception as e:
        print(f"[longform] story for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _longform_store.update(pid, lambda pr: pr.update(status="research_ready", error=err, message=None))
        except Exception:
            pass
    finally:
        _longform_release(pid, "write")
    # The story is in: plan its keyword visuals straight away (in the
    # background -- the story can be read and recorded meanwhile). An
    # explainer's diagrams come with its script.
    try:
        done = _longform_store.load(pid)
        if done.get("status") == "script_ready" and not _is_explainer(done):
            _longform_claim(pid, "visuals")
            _longform_store.update(pid, lambda pr: pr.update(visuals={"status": "planning", "message": "Planning the visuals..."}))
            threading.Thread(target=_longform_visuals, args=(pid, True), daemon=True).start()
    except Exception as e:
        print(f"[longform] couldn't start visuals for {pid}: {e}", flush=True)


def _longform_visuals(pid: str, replan: bool) -> None:
    """Plan the keyword visuals (Claude) and download the free footage and
    photos they need. Runs after the story is written, and again from the
    "Plan visuals again" button. Only scenes whose words haven't changed
    meanwhile get the new visuals."""
    say = lambda m: _longform_store.update(pid, lambda pr: pr.update(visuals={"status": "planning", "message": m}))  # noqa: E731
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        dossier = json.loads((d / "dossier.json").read_text(encoding="utf-8"))
        scenes = project.get("scenes") or []
        if replan:
            say("Claude is picking the key words and their visuals...")
            scenes = documentary.plan_visuals(dossier, project.get("library") or [], scenes, _longform_lessons.texts())
        scenes = documentary.fetch_visuals(scenes, dossier, d, on_progress=say)
        planned = {i: sc.get("cues") or [] for i, sc in enumerate(scenes) if sc.get("kind") == "narrate"}
        texts = {i: sc.get("narration") for i, sc in enumerate(scenes)}

        def apply(pr: dict) -> None:
            for i, sc in enumerate(pr.get("scenes") or []):
                if i in planned and sc.get("narration") == texts.get(i):
                    sc["cues"] = planned[i]
            n = sum(len(c) for c in planned.values())
            pr["visuals"] = {"status": "done", "message": None, "error": None, "count": n, "at": time.time()}
        _longform_store.update(pid, apply)
    except Exception as e:
        print(f"[longform] visuals for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _longform_store.update(pid, lambda pr: pr.update(visuals={"status": "error", "error": err, "message": None}))
        except Exception:
            pass
    finally:
        _longform_release(pid, "visuals")


def _longform_start(pid: str, kind: str, target, status: str, message: str) -> None:
    _longform_claim(pid, kind)
    _longform_store.update(pid, lambda pr: pr.update(status=status, error=None, message=message))
    threading.Thread(target=target, args=(pid,), daemon=True).start()


_LOGIN_RE = re.compile(r"^[a-z0-9_]{3,25}$")


def _clean_login(login: str) -> str:
    login = (login or "").strip().lower().lstrip("@")
    login = re.sub(r"^https?://(www\.)?twitch\.tv/", "", login).strip("/")
    if not _LOGIN_RE.match(login):
        raise HTTPException(400, "enter the streamer's Twitch login, e.g. stableronaldo")
    return login


def _clean_links(links: List[str]) -> List[str]:
    return [u.strip() for u in links if u.strip().startswith(("http://", "https://"))][:5]


class LongformCreateRequest(BaseModel):
    login: str
    notes: str = ""
    links: List[str] = []
    slot: Optional[str] = None  # the series date this episode is for


class LongformResearchRequest(BaseModel):
    notes: Optional[str] = None
    links: Optional[List[str]] = None


class LongformScenesRequest(BaseModel):
    scenes: List[dict]


def _load_series() -> dict:
    try:
        data = json.loads(_longform_series_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"start": data.get("start") or datetime.date.today().isoformat(), "plan": data.get("plan") or {}}


def _save_series(series: dict) -> None:
    _longform_series_path.parent.mkdir(parents=True, exist_ok=True)
    _longform_series_path.write_text(json.dumps(series), encoding="utf-8")


@protected.get("/api/longform/series")
def longform_series() -> dict:
    """The bi-weekly schedule: the next release dates, which streamer each
    is for, and the episode (if started) for each."""
    series = _load_series()
    projects = {p["id"]: p for p in _longform_store.list()}
    slots = []
    for date in documentary.series_slots(series["start"]):
        entry = series["plan"].get(date) or {}
        pr = projects.get(entry.get("project_id") or "")
        slots.append({"date": date, "login": entry.get("login") or "",
                      "project": longform.summary(pr) if pr else None})
    return {"start": series["start"], "every_days": documentary.SERIES_EVERY_DAYS, "slots": slots,
            "suggested": _profile_logins(DEFAULT_CHANNEL_PROFILE)}


class LongformSeriesRequest(BaseModel):
    start: str
    plan: dict  # date -> streamer login


@protected.put("/api/longform/series")
def longform_save_series(req: LongformSeriesRequest) -> dict:
    try:
        datetime.date.fromisoformat(req.start)
    except ValueError:
        raise HTTPException(400, "pick a start date")
    series = _load_series()
    series["start"] = req.start
    for date, login in req.plan.items():
        entry = dict(series["plan"].get(date) or {})
        entry["login"] = (login or "").strip().lower().lstrip("@")[:25]
        series["plan"][date] = entry
    _save_series(series)
    return longform_series()


@protected.get("/api/longform/lessons")
def longform_lessons_list() -> dict:
    """What the series has learned (YouTube's reviews, Dean's notes); every
    new story and visuals plan is written with these."""
    return {"lessons": _longform_lessons.load()["lessons"]}


class LongformLearnRequest(BaseModel):
    text: str
    source: str = ""


@protected.post("/api/longform/lessons")
async def longform_lessons_learn(req: LongformLearnRequest) -> dict:
    text = req.text.strip()
    if len(text) < 20:
        raise HTTPException(400, "paste the feedback first")
    source = " ".join(req.source.split())[:120] or f"Feedback added {datetime.date.today():%d %b %Y}"
    try:
        data = await run_in_threadpool(_longform_lessons.learn, text, source)
    except Exception as e:
        raise HTTPException(500, f"Couldn't learn from that: {e}")
    return {"lessons": data["lessons"]}


@protected.delete("/api/longform/lessons/{index}")
def longform_lessons_remove(index: int) -> dict:
    try:
        return {"lessons": _longform_lessons.remove(index)["lessons"]}
    except IndexError:
        raise HTTPException(404, "no such lesson")


@protected.get("/api/longform/projects")
def longform_projects(series: str = "") -> dict:
    """series=documentary: the Story Of episodes (and old tests);
    series=explainer: Caught On Code; empty: everything."""
    projects = _longform_store.list()
    if series == explainer.KIND:
        projects = [p for p in projects if _is_explainer(p)]
    elif series:
        projects = [p for p in projects if not _is_explainer(p)]
    return {"projects": [longform.summary(p) for p in projects]}


@protected.post("/api/longform/projects")
def longform_create(req: LongformCreateRequest) -> dict:
    login = _clean_login(req.login)
    project = _longform_store.create({"kind": "documentary", "login": login, "title": f"The Story of {login}",
                                      "notes": req.notes.strip()[:6000], "links": _clean_links(req.links),
                                      "slot": req.slot})
    if req.slot:
        series = _load_series()
        series["plan"][req.slot] = {"login": login, "project_id": project["id"]}
        _save_series(series)
    _longform_start(project["id"], "research", _longform_research, "researching", "Starting research...")
    return {"id": project["id"]}


# ---- Streamer clips for Caught On Code (clipper/clip_search.py): real Twitch
# moments of what an episode explains, played as "moment" scenes. Found by
# topic in the background (also right after the first research), or pasted
# as links. They join project["library"] like a documentary's clips, with
# "use" (Dean's tick; starts as Claude's fits/not-fits check) and "what".

def _clips_set(pid: str, **changes) -> dict:
    return _longform_store.update(pid, lambda pr: pr.update(clip_search={**(pr.get("clip_search") or {}), **changes}))


def _longform_clip_search(pid: str, links: List[str]) -> None:
    say = lambda m: _clips_set(pid, message=m)  # noqa: E731
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        lib = project.get("library") or []
        have = {c.get("twitch_id") for c in lib}
        # YouTube search uses the episode's channel connection (read-only
        # scope is enough); without one only Twitch is searched.
        yt_token = None
        for account in dict.fromkeys([_longform_account(project), DEFAULT_CHANNEL_PROFILE]):
            try:
                yt_token = _youtube_token_stores[account].get_valid_access_token()
            except Exception:
                yt_token = None
            if yt_token:
                break
        pasted = []
        unresolved = 0
        if links:
            say("Looking up your links...")
            got_links = clip_search.resolve_links(links, yt_token or "")
            unresolved = max(0, len(links) - len(got_links))
            pasted = [dict(c, pasted=True) for c in got_links if c["twitch_id"] not in have]
            have |= {c["twitch_id"] for c in pasted}
        res = clip_search.find(project.get("topic") or project.get("title") or "", on_progress=say, yt_token=yt_token or "")
        new = pasted + [c for c in res["picked"] if c["twitch_id"] not in have]
        n0 = max([int(c["id"][1:]) for c in lib if str(c.get("id", ""))[1:].isdigit()] + [0])
        for i, c in enumerate(new, start=n0 + 1):
            c["id"] = f"C{i:02d}"
        got = documentary.build_library(
            new, d / "clips", on_progress=lambda i, n: say(f"Downloading and transcribing clip {min(i + 1, n)} of {n}..."))
        if got:
            say("Checking what happens in each clip...")
        verdict = clip_search.check(project.get("topic") or "", got, d) if got else {}
        for c in got:
            v = verdict.get(c["id"], {})
            c["what"] = v.get("what") or c.get("why") or ""
            if v:
                c["fits"] = bool(v.get("fits"))
            c["use"] = bool(c.get("pasted")) or bool(v.get("fits"))
        _longform_store.update(pid, lambda pr: pr.update(
            library=(pr.get("library") or []) + got,
            clip_search={"status": "done", "message": None, "error": None, "games": res["games"], "keywords": res["keywords"],
                         "youtube": bool(yt_token and res.get("youtube")), "unresolved": unresolved,
                         "scanned": res["scanned"], "added": len(got), "finished_at": time.time()}))
    except Exception as e:
        print(f"[longform] clip search for {pid} failed: {e}", flush=True)
        err = str(e)
        try:
            _clips_set(pid, status="error", message=None, error=err)
        except Exception:
            pass
    finally:
        _longform_release(pid, "clips")


def _longform_start_clip_search(pid: str, links: List[str]) -> None:
    _longform_claim(pid, "clips")
    _clips_set(pid, status="searching", message="Starting...", error=None)
    threading.Thread(target=_longform_clip_search, args=(pid, links), daemon=True).start()


class ClipSearchRequest(BaseModel):
    links: List[str] = []


@protected.post("/api/longform/projects/{pid}/clip-search")
def longform_clip_search(pid: str, req: ClipSearchRequest) -> dict:
    project = _longform_project(pid)
    if not _is_explainer(project):
        raise HTTPException(409, "streamer clip search is for Caught On Code episodes")
    if not os.environ.get("TWITCH_CLIENT_ID"):
        raise HTTPException(409, "Twitch isn't connected (TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET aren't set on Railway).")
    _longform_start_clip_search(pid, [str(x) for x in req.links if str(x).strip()][:20])
    return _longform_store.load(pid)


class ClipUseRequest(BaseModel):
    use: bool


@protected.delete("/api/longform/projects/{pid}/library/{clip_id}")
def longform_clip_remove(pid: str, clip_id: str) -> dict:
    """Throw away a streamer clip that's no good (not one the script uses)."""
    project = _longform_project(pid)
    if not _is_explainer(project):
        raise HTTPException(409, "a documentary's clips come with its research")
    if any(s.get("clip") == clip_id for s in project.get("scenes") or []):
        raise HTTPException(409, "the script uses this clip -- remove its moment from the script first")

    def apply(pr: dict) -> None:
        lib = pr.get("library") or []
        if not any(c.get("id") == clip_id for c in lib):
            raise HTTPException(404, "no such clip")
        pr["library"] = [c for c in lib if c.get("id") != clip_id]
    out = _longform_store.update(pid, apply)
    if re.fullmatch(r"C\d{1,3}", clip_id):
        shutil.rmtree(_longform_store.path(pid) / "clips" / clip_id, ignore_errors=True)
    return out


@protected.put("/api/longform/projects/{pid}/library/{clip_id}")
def longform_clip_use(pid: str, clip_id: str, req: ClipUseRequest) -> dict:
    """Tick or untick a streamer clip for the script."""
    def apply(pr: dict) -> None:
        for c in pr.get("library") or []:
            if c.get("id") == clip_id:
                c["use"] = req.use
                return
        raise HTTPException(404, "no such clip")
    _longform_project(pid)
    return _longform_store.update(pid, apply)


class ExplainerCreateRequest(BaseModel):
    topic: str
    notes: str = ""
    links: List[str] = []


@protected.get("/api/explainers")
def explainers_meta() -> dict:
    """What the Caught On Code page needs besides its episodes: topic ideas
    (from the niche research) and whether its YouTube account is connected."""
    return {"channel": explainer.CHANNEL, "topics": explainer.TOPIC_IDEAS,
            "oauth_configured": youtube_oauth.is_configured(),
            "youtube_connected": _youtube_token_stores[explainer.PROFILE].is_connected()}


@protected.post("/api/explainers")
def explainer_create(req: ExplainerCreateRequest) -> dict:
    topic = " ".join(req.topic.split())[:160]
    if len(topic) < 4:
        raise HTTPException(400, "type the topic first, e.g. how kernel anti-cheat works")
    project = _longform_store.create({"kind": explainer.KIND, "topic": topic, "title": topic,
                                      "notes": req.notes.strip()[:6000], "links": _clean_links(req.links)})
    _longform_start(project["id"], "research", _longform_research, "researching", "Starting research...")
    return {"id": project["id"]}


@protected.get("/api/longform/projects/{pid}")
def longform_get(pid: str) -> dict:
    return _longform_project(pid)


@protected.delete("/api/longform/projects/{pid}")
def longform_delete(pid: str) -> dict:
    _longform_project(pid)
    _longform_store.delete(pid)
    return {"ok": True}


@protected.post("/api/longform/projects/{pid}/research")
def longform_rerun_research(pid: str, req: LongformResearchRequest) -> dict:
    project = _longform_project(pid)
    if project.get("kind") not in ("documentary", explainer.KIND):
        raise HTTPException(409, "this is an old aviation test video -- delete it and start a new episode")
    updates = {}
    if req.notes is not None:
        updates["notes"] = req.notes.strip()[:6000]
    if req.links is not None:
        updates["links"] = _clean_links(req.links)
    if updates:
        _longform_store.update(pid, lambda pr: pr.update(**updates))
    (_longform_store.path(pid) / "dossier.json").unlink(missing_ok=True)
    _longform_start(pid, "research", _longform_research, "researching", "Starting research...")
    return {"ok": True}


@protected.post("/api/longform/projects/{pid}/write")
def longform_write_story(pid: str) -> dict:
    project = _longform_project(pid)
    if (not project.get("library") and not _is_explainer(project)) or project.get("status") not in ("research_ready", "script_ready"):
        raise HTTPException(409, "run the research first")
    _longform_start(pid, "write", _longform_write, "writing", "Claude is writing the story (1-2 minutes)...")
    return {"ok": True}


@protected.put("/api/longform/projects/{pid}/scenes")
def longform_save_scenes(pid: str, req: LongformScenesRequest) -> dict:
    """Save story edits (reordering, rewording, a different clip or cut).
    A narrated scene keeps its recorded take only if its words didn't change."""
    project = _longform_project(pid)
    if project.get("status") != "script_ready":
        raise HTTPException(409, "write the story first")
    if _is_explainer(project):
        try:
            dossier = json.loads((_longform_store.path(pid) / "dossier.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dossier = {"topic": project.get("topic") or ""}
        new = explainer.normalize_scenes(req.scenes, dossier, project.get("library") or [])
    else:
        new = documentary.normalize_scenes(req.scenes, project.get("library") or [])
    if not any(s["kind"] == "narrate" for s in new):
        raise HTTPException(400, "the story needs at least one narrated scene")

    def apply(pr: dict) -> None:
        takes = {s.get("narration"): s.get("take") for s in pr.get("scenes") or [] if s.get("kind") == "narrate"}
        for sc in new:
            if sc["kind"] == "narrate":
                sc["take"] = takes.get(sc["narration"])
        pr["scenes"] = new
    return _longform_store.update(pid, apply)


@protected.get("/api/longform/projects/{pid}/scenes/{index}/preview")
async def longform_scene_preview(pid: str, index: int) -> FileResponse:
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not 0 <= index < len(scenes):
        raise HTTPException(404, "scene not found")
    sc = scenes[index]
    key = hashlib.sha1(json.dumps({k: sc.get(k) for k in ("kind", "clip", "start", "end", "caption", "title")}, sort_keys=True).encode()).hexdigest()[:12]
    d = _longform_store.path(pid)
    out = d / "render" / "previews" / f"{key}.jpg"
    if not out.exists():
        try:
            await run_in_threadpool(longform_video.preview_still, d, sc, _longform_library(project), _longform_brand(), out)
        except Exception as e:
            raise HTTPException(500, f"Couldn't draw this scene: {e}")
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@protected.get("/api/longform/projects/{pid}/clips/{clip_id}")
def longform_clip_file(pid: str, clip_id: str) -> FileResponse:
    project = _longform_project(pid)
    clip = _longform_library(project).get(clip_id)
    path = _longform_store.path(pid) / clip["file"] if clip else None
    if not path or not path.is_file():
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="video/mp4")


@protected.get("/api/longform/projects/{pid}/clips/{clip_id}/words")
def longform_clip_words(pid: str, clip_id: str) -> dict:
    _longform_project(pid)
    return {"words": documentary.clip_words(_longform_store.path(pid), clip_id)}


_TAKE_EXTS = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
              "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/aac": ".aac", "video/webm": ".webm"}


@protected.post("/api/longform/projects/{pid}/scenes/{index}/take")
async def longform_take(pid: str, index: int, request: Request) -> dict:
    """One recorded take of one narrated scene: convert, transcribe, compare
    with the script. A take that matches is accepted; a flagged one is kept
    aside until it's re-read (or kept anyway)."""
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not 0 <= index < len(scenes) or not longform.needs_take(scenes[index]):
        raise HTTPException(404, "that scene isn't narrated")
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
        return {"duration": round(duration, 2), "heard": " ".join(heard).strip(), **longform.check_take(script, heard)}

    try:
        result = await run_in_threadpool(work)
    except Exception as e:
        wav.unlink(missing_ok=True)
        print(f"[longform] take check failed for {pid} scene {index}: {e}", flush=True)
        raise HTTPException(500, f"Couldn't check that take: {e}")

    take = {"file": wav.name, "recorded_at": time.time(), "kept": False, **result}
    _longform_set_take(pid, index, script, take, wav)
    return {"take": take}


def _longform_set_take(pid: str, index: int, script: str, take: dict, wav: Path) -> None:
    """Make `take` the scene's take and delete the one it replaces -- unless
    the scene's words changed meanwhile, then the new take is dropped."""
    takes_dir = wav.parent
    replaced = []

    def apply(pr: dict) -> None:
        scs = pr.get("scenes") or []
        sc = scs[index] if index < len(scs) else None
        if sc is None or sc.get("narration") != script:
            raise HTTPException(409, "the script changed while this take was being checked -- record it again")
        old = sc.get("take") or {}
        if old.get("file") and old["file"] != wav.name:
            replaced.append(old["file"])
        sc["take"] = take
    try:
        _longform_store.update(pid, apply)
    except HTTPException:
        wav.unlink(missing_ok=True)
        raise
    for name in replaced:
        (takes_dir / name).unlink(missing_ok=True)


@protected.post("/api/longform/projects/{pid}/scenes/{index}/keep")
def longform_keep_take(pid: str, index: int) -> dict:
    """Accept a flagged take as it is (e.g. Whisper misheard, not you)."""
    def apply(pr: dict) -> None:
        scs = pr.get("scenes") or []
        if not 0 <= index < len(scs) or not (scs[index].get("take") or {}).get("file"):
            raise HTTPException(404, "no take recorded for that scene")
        scs[index]["take"]["kept"] = True
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


# ---- Dean's AI voice (clipper/voice_clone.py): one sample, shared by all
# projects, used to voice a narrated scene he'd rather not re-read.
_voice_dir = BASE_DIR / "_longform" / "_voice"
_voice_busy = threading.Lock()  # one generation at a time; the model is single-threaded anyway


@protected.get("/api/longform/voice")
def longform_voice_status() -> dict:
    return voice_clone.status(_voice_dir)


@protected.post("/api/longform/voice/sample")
async def longform_voice_sample(request: Request) -> dict:
    """Dean reading voice_clone.SAMPLE_TEXT once. Checked the same way as a
    take, loosely: it only has to be him, clearly, reading most of it."""
    data = await request.body()
    if len(data) < 1000:
        raise HTTPException(400, "that recording is empty -- check your mic")
    if len(data) > longform.MAX_TAKE_BYTES:
        raise HTTPException(413, "that recording is too long -- about 30 seconds is plenty")
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    _voice_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    raw = _voice_dir / f"upload_{stamp}{_TAKE_EXTS.get(ctype, '.webm')}"
    wav = _voice_dir / f"upload_{stamp}.wav"

    def work() -> dict:
        raw.write_bytes(data)
        try:
            longform.to_wav(raw, wav)
        finally:
            raw.unlink(missing_ok=True)
        duration = longform.audio_duration(wav)
        if duration < voice_clone.MIN_SAMPLE_SECONDS:
            raise ValueError(f"That was only {duration:.0f} seconds -- read the whole text (about 30 seconds).")
        if duration > voice_clone.MAX_SAMPLE_SECONDS:
            raise ValueError("That was over a minute -- just read the text once.")
        heard = longform.transcribe_take(wav)
        check = longform.check_take(voice_clone.SAMPLE_TEXT, heard)
        if check["coverage"] < 0.6:
            raise ValueError("Couldn't hear you reading the text clearly -- find a quiet spot and try again.")
        return {"duration": duration, "coverage": check["coverage"], "heard": " ".join(heard).strip()}

    try:
        result = await run_in_threadpool(work)
    except ValueError as e:
        wav.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        wav.unlink(missing_ok=True)
        print(f"[longform] voice sample check failed: {e}", flush=True)
        raise HTTPException(500, f"Couldn't check that recording: {e}")
    wav.replace(voice_clone.sample_path(_voice_dir))
    voice_clone.save_sample_meta(_voice_dir, result["duration"], result)
    return voice_clone.status(_voice_dir)


@protected.get("/api/longform/voice/sample")
def longform_voice_sample_audio() -> FileResponse:
    path = voice_clone.sample_path(_voice_dir)
    if not path.is_file():
        raise HTTPException(404, "no voice sample recorded")
    return FileResponse(path, media_type="audio/wav")


@protected.delete("/api/longform/voice/sample")
def longform_voice_sample_delete() -> dict:
    voice_clone.sample_path(_voice_dir).unlink(missing_ok=True)
    (_voice_dir / "sample.json").unlink(missing_ok=True)
    return voice_clone.status(_voice_dir)


# ---- Other people's AI voices: a friend who agreed to narrate. One folder
# each under _voices/<id>/ with its own sample and voice.json (name, who
# agreed, an optional private link). The friend lives far away, so the
# sample can be an uploaded voice note, or recorded by the friend on a
# private link that needs no app password (/voice-sample/<token>). Videos
# that use one are marked "altered or synthetic content" on YouTube.
_voices_dir = BASE_DIR / "_longform" / "_voices"
_VOICE_ID_RE = re.compile(r"^[0-9a-f]{8}$")
VOICE_LINK_DAYS = 7
_voice_meta_lock = threading.Lock()


def _voice_path(vid: Optional[str]) -> Path:
    """The folder of a voice: "me" (or nothing) is Dean's own."""
    if not vid or vid == "me":
        return _voice_dir
    if not _VOICE_ID_RE.match(vid) or not (_voices_dir / vid / "voice.json").is_file():
        raise HTTPException(404, "that voice doesn't exist (any more)")
    return _voices_dir / vid


def _voice_ids() -> List[str]:
    if not _voices_dir.is_dir():
        return []
    found = [d for d in _voices_dir.iterdir() if _VOICE_ID_RE.match(d.name) and (d / "voice.json").is_file()]
    return [d.name for d in sorted(found, key=lambda d: voice_clone.read_meta(d).get("created_at") or 0)]


def _voice_name(vid: Optional[str]) -> str:
    if not vid or vid == "me":
        return "You"
    try:
        return voice_clone.read_meta(_voice_path(vid)).get("name") or "Friend"
    except HTTPException:
        return "a deleted voice"


def _voice_row(vid: str) -> dict:
    d = _voice_path(vid)
    st = voice_clone.status(d)
    row = {"id": vid, "name": _voice_name(vid), "mine": vid == "me", "sample": st["sample"], "ready": st["ready"]}
    if vid != "me":
        meta = voice_clone.read_meta(d)
        link = meta.get("link") or {}
        row["consent"] = meta.get("consent")
        row["link"] = ({"url": f"/voice-sample/{link['token']}", "expires_at": link["expires_at"]}
                       if link.get("token") and link.get("expires_at", 0) > time.time() else None)
    return row


@protected.get("/api/voices")
def list_voices() -> dict:
    st = voice_clone.status(_voice_dir)
    return {"installed": st["installed"], "configured": st["configured"], "setup_hint": st["setup_hint"],
            "error": st["error"], "sample_text": voice_clone.SAMPLE_TEXT,
            "voices": [_voice_row(v) for v in ["me", *_voice_ids()]]}


class VoiceCreateRequest(BaseModel):
    name: str
    consent: bool = False


@protected.post("/api/voices")
def create_voice(req: VoiceCreateRequest) -> dict:
    name = " ".join(req.name.split())[:40]
    if not name:
        raise HTTPException(400, "Give the voice a name.")
    if not req.consent:
        raise HTTPException(400, "Only copy someone's voice after they've said yes. Tick the box once they have.")
    vid = uuid.uuid4().hex[:8]
    voice_clone.write_meta(_voices_dir / vid, {
        "name": name, "created_at": time.time(),
        "consent": {"confirmed_by": "Dean", "confirmed_at": time.time()},
    })
    return _voice_row(vid)


@protected.delete("/api/voices/{vid}")
def delete_voice(vid: str) -> dict:
    """Removes the sample and the private link. Takes already read in this
    voice keep their audio."""
    if vid == "me":
        raise HTTPException(400, "Your own voice sample is deleted from the long-form page.")
    d = _voice_path(vid)
    shutil.rmtree(d, ignore_errors=True)
    return list_voices()


def _store_voice_sample(d: Path, data: bytes, ctype: str, filename: str = "") -> dict:
    """Check an uploaded or recorded sample and make it the voice's sample.
    Anything ffmpeg can read works (a WhatsApp voice note, an mp3, a phone
    recording); the first voice_clone.KEEP_SECONDS of talking are kept.
    It only has to be clear speech -- reading SAMPLE_TEXT is best but a
    voice note saying anything works too."""
    if len(data) < 1000:
        raise HTTPException(400, "that recording is empty -- check the mic")
    if len(data) > longform.MAX_TAKE_BYTES:
        raise HTTPException(413, "that file is too big -- a minute of talking is plenty")
    d.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    ext = Path(filename).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{2,5}", ext):
        ext = _TAKE_EXTS.get(ctype, ".bin")
    raw = d / f"upload_{stamp}{ext}"
    wav = d / f"upload_{stamp}.wav"

    def work() -> dict:
        raw.write_bytes(data)
        try:
            try:
                full = longform.audio_duration(raw)
            except Exception:
                raise ValueError("That file isn't audio the app can read -- try an mp3, m4a or a voice note.")
            if full > voice_clone.UPLOAD_MAX_SECONDS:
                raise ValueError("That's over 10 minutes -- send a shorter clip of the voice, about 30 seconds of talking.")
            try:
                voice_clone.trim_sample(raw, wav)
            except subprocess.CalledProcessError:
                raise ValueError("That file isn't audio the app can read -- try an mp3, m4a or a voice note.")
        finally:
            raw.unlink(missing_ok=True)
        duration = longform.audio_duration(wav)
        if duration < voice_clone.MIN_SAMPLE_SECONDS:
            raise ValueError(f"Only {duration:.0f} seconds of talking -- it needs at least 10 (30 is best).")
        heard = longform.transcribe_take(wav)
        words = " ".join(heard).split()
        if len(words) < voice_clone.MIN_SAMPLE_WORDS:
            raise ValueError("Couldn't hear clear talking in that -- a quiet room and one voice works best.")
        check = longform.check_take(voice_clone.SAMPLE_TEXT, heard)
        return {"duration": duration, "coverage": check["coverage"], "heard": " ".join(words)}

    try:
        result = work()
    except ValueError as e:
        wav.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        wav.unlink(missing_ok=True)
        print(f"[voices] sample check failed for {d.name}: {e}", flush=True)
        raise HTTPException(500, f"Couldn't check that recording: {e}")
    wav.replace(voice_clone.sample_path(d))
    voice_clone.save_sample_meta(d, result["duration"], result)
    return result


@protected.post("/api/voices/{vid}/sample")
async def upload_voice_sample(vid: str, request: Request) -> dict:
    if vid == "me":
        raise HTTPException(400, "Record your own sample on the long-form page.")
    d = _voice_path(vid)
    data = await request.body()
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    await run_in_threadpool(_store_voice_sample, d, data, ctype, request.headers.get("x-filename") or "")
    return _voice_row(vid)


@protected.get("/api/voices/{vid}/sample")
def voice_sample_audio(vid: str) -> FileResponse:
    path = voice_clone.sample_path(_voice_path(vid))
    if not path.is_file():
        raise HTTPException(404, "no voice sample yet")
    return FileResponse(path, media_type="audio/wav")


@protected.post("/api/voices/{vid}/link")
def create_voice_link(vid: str) -> dict:
    """A private link the person opens on their phone to record their own
    sample: no app password, works for VOICE_LINK_DAYS, one per voice (a
    new one replaces the old)."""
    if vid == "me":
        raise HTTPException(400, "That's your own voice -- record it on the long-form page.")
    d = _voice_path(vid)
    with _voice_meta_lock:
        meta = voice_clone.read_meta(d)
        meta["link"] = {"token": secrets.token_urlsafe(18), "expires_at": time.time() + VOICE_LINK_DAYS * 86400}
        voice_clone.write_meta(d, meta)
    return _voice_row(vid)


@protected.delete("/api/voices/{vid}/link")
def delete_voice_link(vid: str) -> dict:
    d = _voice_path(vid)
    with _voice_meta_lock:
        meta = voice_clone.read_meta(d)
        meta.pop("link", None)
        voice_clone.write_meta(d, meta)
    return _voice_row(vid)


class VoicePreviewRequest(BaseModel):
    text: str


@protected.post("/api/voices/{vid}/preview")
async def preview_voice(vid: str, req: VoicePreviewRequest) -> FileResponse:
    """Say one line in this voice, to hear how it came out."""
    text = " ".join(req.text.split())[:300]
    if not text:
        raise HTTPException(400, "Type something for the voice to say.")
    d = _voice_path(vid)
    _voice_check(vid)
    if not _voice_busy.acquire(blocking=False):
        raise HTTPException(409, "The AI voice is busy reading something else -- try again in a few seconds.")
    out = d / "preview.wav"
    try:
        await run_in_threadpool(voice_clone.synthesize, d, text, out)
    except Exception as e:
        print(f"[voices] preview failed for {vid}: {e}", flush=True)
        raise HTTPException(500, f"The voice couldn't read that: {e}")
    finally:
        _voice_busy.release()
    return FileResponse(out, media_type="audio/wav")


def _voice_by_token(token: str) -> Path:
    if _voices_dir.is_dir() and re.fullmatch(r"[A-Za-z0-9_-]{20,40}", token or ""):
        for vid in _voice_ids():
            link = voice_clone.read_meta(_voices_dir / vid).get("link") or {}
            if link.get("token") and secrets.compare_digest(link["token"], token) and link.get("expires_at", 0) > time.time():
                return _voices_dir / vid
    raise HTTPException(404, "This link has expired or was switched off. Ask for a new one.")


# Public on purpose (no app password): the friend who agreed to narrate
# records a sample here from far away. The unguessable token is the key;
# it only lets them set this one voice's sample, nothing else.
@app.get("/voice-sample/{token}", response_class=HTMLResponse)
def voice_link_page(token: str) -> str:
    try:
        d = _voice_by_token(token)
    except HTTPException as e:
        return VOICE_LINK_HTML.replace("__STATE__", json.dumps({"error": e.detail}))
    meta = voice_clone.read_meta(d)
    return VOICE_LINK_HTML.replace("__STATE__", json.dumps({
        "name": meta.get("name") or "", "sample_text": voice_clone.SAMPLE_TEXT,
        "has_sample": voice_clone.sample_path(d).is_file()}))


@app.post("/voice-sample/{token}")
async def voice_link_upload(token: str, request: Request) -> dict:
    d = _voice_by_token(token)
    if request.headers.get("x-consent") != "yes":
        raise HTTPException(400, "Tick the box to say it's OK first.")
    data = await request.body()
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    result = await run_in_threadpool(_store_voice_sample, d, data, ctype, request.headers.get("x-filename") or "")
    with _voice_meta_lock:
        meta = voice_clone.read_meta(d)
        meta.setdefault("consent", {})["self_confirmed_at"] = time.time()
        voice_clone.write_meta(d, meta)
    return {"ok": True, "duration": round(result["duration"], 1)}


def _voice_check(vid: Optional[str] = None) -> None:
    st = voice_clone.status(_voice_path(vid))
    if not st["installed"] or not st["configured"]:
        raise HTTPException(409, voice_clone.SETUP_HINT)
    if not st["sample"]:
        if vid and vid != "me":
            raise HTTPException(409, f"{_voice_name(vid)}'s voice has no sample yet -- add one on the Voices page.")
        raise HTTPException(409, "Record your voice sample first (the “Your AI voice” box at the top).")


def _clone_take(pid: str, index: int, script: str, vid: str = "me") -> dict:
    """Read one narrated scene in an AI voice (Dean's own unless the episode
    picked another), check it like a recorded take and make it the scene's
    take. The caller holds _voice_busy."""
    takes_dir = _longform_store.path(pid) / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    raw = takes_dir / f"scene{index:02d}_{stamp}_ai24k.wav"
    wav = takes_dir / f"scene{index:02d}_{stamp}_ai.wav"
    started = time.time()
    try:
        try:
            voice_clone.synthesize(_voice_path(vid), script, raw)
            longform.to_wav(raw, wav)
        finally:
            raw.unlink(missing_ok=True)
        duration = longform.audio_duration(wav)
        heard = longform.transcribe_take(wav)
    except Exception:
        wav.unlink(missing_ok=True)
        raise
    result = {"duration": round(duration, 2), "heard": " ".join(heard).strip(),
              "took_seconds": round(time.time() - started, 1), **longform.check_take(script, heard)}
    take = {"file": wav.name, "recorded_at": time.time(), "kept": False, "voice": "ai", **result}
    if vid != "me":
        # Someone else's voice: the episode gets YouTube's "altered or
        # synthetic content" label when it's posted.
        take["voice_id"] = vid
    _longform_set_take(pid, index, script, take, wav)
    return take


@protected.post("/api/longform/projects/{pid}/scenes/{index}/clone")
async def longform_clone_take(pid: str, index: int) -> dict:
    """Voice one narrated scene with Dean's AI voice instead of recording it.
    The result is checked like a recorded take, so a garbled or skipped
    phrase gets flagged; generating again gives a slightly different read."""
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if not 0 <= index < len(scenes) or not longform.needs_take(scenes[index]):
        raise HTTPException(404, "that scene isn't narrated")
    vid = project.get("ai_voice") or "me"
    _voice_check(vid)
    if not _voice_busy.acquire(blocking=False):
        raise HTTPException(409, "Your AI voice is already reading another scene -- give it a few seconds.")
    script = scenes[index]["narration"]
    try:
        take = await run_in_threadpool(_clone_take, pid, index, script, vid)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[longform] AI voice failed for {pid} scene {index}: {e}", flush=True)
        raise HTTPException(500, f"Your AI voice couldn't read this scene: {e}")
    finally:
        _voice_busy.release()
    return {"take": take}


# ---- "Let my AI voice read everything": every narrated scene without a
# usable take, one after another in the background. Scenes Dean already
# recorded (or that pass the check) are left alone. A take the misread check
# flags is tried once more; one still flagged stays for him to play and keep
# or redo. Progress lives in project["ai_all"].

def _ai_all_set(pid: str, **changes) -> dict:
    return _longform_store.update(pid, lambda pr: pr.update(ai_all={**(pr.get("ai_all") or {}), **changes}))


def _longform_ai_all(pid: str, todo: List[int], vid: str = "me") -> None:
    done, flagged, failed = 0, [], []
    try:
        for index in todo:
            project = _longform_store.load(pid)
            if (project.get("ai_all") or {}).get("stop"):
                _ai_all_set(pid, status="stopped", current=None)
                return
            scenes = project.get("scenes") or []
            if index >= len(scenes) or not longform.needs_take(scenes[index]) or longform.scene_ready(scenes[index]):
                done += 1
                continue
            _ai_all_set(pid, current=index)
            script = scenes[index]["narration"]
            if not _voice_busy.acquire(timeout=300):
                failed.append(index + 1)
                continue
            try:
                take = None
                for _attempt in range(2):
                    take = _clone_take(pid, index, script, vid)
                    if take.get("ok"):
                        break
                if take and not take.get("ok"):
                    flagged.append(index + 1)
            except HTTPException:  # the script changed meanwhile
                failed.append(index + 1)
            except Exception as e:
                print(f"[longform] AI voice failed for {pid} scene {index}: {e}", flush=True)
                failed.append(index + 1)
            finally:
                _voice_busy.release()
            done += 1
            _ai_all_set(pid, done=done, flagged=flagged, failed=failed)
        _ai_all_set(pid, status="done", current=None, done=done, flagged=flagged, failed=failed, finished_at=time.time())
    except Exception as e:
        print(f"[longform] AI voice (all scenes) for {pid} stopped: {e}", flush=True)
        try:
            _ai_all_set(pid, status="error", current=None, error=str(e))
        except Exception:
            pass
    finally:
        _longform_release(pid, "ai_all")


@protected.post("/api/longform/projects/{pid}/clone-all")
def longform_clone_all(pid: str) -> dict:
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if project.get("status") != "script_ready" or not scenes:
        raise HTTPException(409, "write the script first")
    vid = project.get("ai_voice") or "me"
    _voice_check(vid)
    todo = [i for i, sc in enumerate(scenes) if longform.needs_take(sc) and not longform.scene_ready(sc)]
    if not todo:
        raise HTTPException(409, "Every narrated scene already has a take.")
    _longform_claim(pid, "ai_all")
    _longform_store.update(pid, lambda pr: pr.update(ai_all={
        "status": "running", "total": len(todo), "done": 0, "current": None, "flagged": [], "failed": [],
        "started_at": time.time()}))
    threading.Thread(target=_longform_ai_all, args=(pid, todo, vid), daemon=True).start()
    return _longform_store.load(pid)


@protected.delete("/api/longform/projects/{pid}/clone-all")
def longform_stop_clone_all(pid: str) -> dict:
    """Stop after the scene being read now; what's read so far is kept."""
    _longform_project(pid)
    if not _longform_is_busy(pid, "ai_all"):
        raise HTTPException(409, "your AI voice isn't reading anything")
    return _ai_all_set(pid, stop=True)


class AiVoiceRequest(BaseModel):
    voice: str = "me"


@protected.put("/api/longform/projects/{pid}/ai-voice")
def longform_set_ai_voice(pid: str, req: AiVoiceRequest) -> dict:
    """Which AI voice reads this episode's lines: Dean's own ("me") or a
    friend's from the Voices page. Takes already read stay as they are."""
    _longform_project(pid)
    vid = req.voice or "me"
    _voice_path(vid)
    if _longform_is_busy(pid, "ai_all"):
        raise HTTPException(409, "wait for the AI voice to finish (or stop it) first")
    return _longform_store.update(pid, lambda pr: pr.update(ai_voice=vid))


_MUSIC_EXTS = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mp4": ".m4a",
               "audio/x-m4a": ".m4a", "audio/aac": ".aac", "audio/ogg": ".ogg", "audio/flac": ".flac"}


@protected.post("/api/longform/projects/{pid}/music")
async def longform_upload_music(pid: str, request: Request, name: str = "") -> dict:
    """Optional background music (e.g. from the YouTube Audio Library), mixed
    quietly under the whole video and ducked under the voice and clips."""
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


class LongformMusicLevelRequest(BaseModel):
    level: str


@protected.put("/api/longform/projects/{pid}/music-level")
def longform_music_level(pid: str, req: LongformMusicLevelRequest) -> dict:
    """Quieter / normal / louder background music. Only the final mix
    changes, so the next render reuses every scene and takes a minute."""
    if req.level not in longform_video.MUSIC_LEVELS:
        raise HTTPException(400, "level must be quiet, normal or loud")
    _longform_project(pid)
    return _longform_store.update(pid, lambda pr: pr.update(
        music_level=req.level, music_db=longform_video.music_db(req.level)))


class LongformMixRequest(BaseModel):
    voice_db: Optional[float] = None
    music_db: Optional[float] = None


@protected.put("/api/longform/projects/{pid}/mix")
def longform_mix(pid: str, req: LongformMixRequest) -> dict:
    """The voice and music sliders, in dB. Your voice is first levelled take
    by take (longform_video.VOICE_TARGET); voice_db moves it from there.
    Only the final mix changes, so "Update sound only" takes a minute."""
    _longform_project(pid)
    changes = {}
    if req.voice_db is not None:
        lo, hi = longform_video.VOICE_DB_RANGE
        changes["voice_db"] = round(max(lo, min(hi, float(req.voice_db))), 1)
    if req.music_db is not None:
        changes["music_db"] = round(longform_video.music_db(db=req.music_db), 1)
    return _longform_store.update(pid, lambda pr: pr.update(changes))


@protected.delete("/api/longform/projects/{pid}/music")
def longform_remove_music(pid: str) -> dict:
    _longform_project(pid)
    for old in _longform_store.path(pid).glob("music.*"):
        old.unlink(missing_ok=True)
    return _longform_store.update(pid, lambda pr: pr.update(music=None))


@protected.post("/api/longform/projects/{pid}/visuals")
def longform_plan_visuals(pid: str) -> dict:
    project = _longform_project(pid)
    if project.get("status") != "script_ready":
        raise HTTPException(409, "write the story first")
    if _is_explainer(project):
        raise HTTPException(409, "an explainer's diagrams come with its script")
    _longform_claim(pid, "visuals")
    _longform_store.update(pid, lambda pr: pr.update(visuals={"status": "planning", "message": "Starting..."}))
    threading.Thread(target=_longform_visuals, args=(pid, True), daemon=True).start()
    return {"ok": True}


_VISUAL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_VISUAL_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".mp4": "video/mp4"}


@protected.get("/api/longform/projects/{pid}/visuals/{name}")
def longform_visual_file(pid: str, name: str) -> FileResponse:
    _longform_project(pid)
    path = _longform_store.path(pid) / "visuals" / name
    if not _VISUAL_NAME.match(name) or name.startswith(".") or path.suffix.lower() not in _VISUAL_TYPES or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type=_VISUAL_TYPES[path.suffix.lower()], headers={"Cache-Control": "max-age=86400"})


@protected.get("/api/longform/emoji/{code}")
def longform_emoji(code: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{2,6}(-[0-9a-f]{2,6}){0,4}", code):
        raise HTTPException(404, "not found")
    # The page drops U+FE0F from the code; some bundled files keep it (🎙 is
    # 1f399-fe0f.webp), so look the emoji up the way the renderer does.
    try:
        path = visual_sources.emoji_file("".join(chr(int(h, 16)) for h in code.split("-")))
    except ValueError:  # not a real code point
        path = None
    if path is None:
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "max-age=604800"})


@protected.get("/api/longform/visual-sources")
def longform_visual_sources() -> dict:
    return visual_sources.stock_available()


def _longform_takes(scenes: List[dict]) -> List[Optional[float]]:
    return [float((s.get("take") or {}).get("duration") or 0) if longform.needs_take(s) else None for s in scenes]


def _longform_render(pid: str, music_only: bool = False) -> None:
    started = time.time()

    def progress(p: float, msg: str) -> None:
        _longform_store.update(pid, lambda pr: pr.update(render={**(pr.get("render") or {}), "status": "rendering",
                                                                 "progress": round(p, 3), "message": msg}))
    try:
        project = _longform_store.load(pid)
        d = _longform_store.path(pid)
        scenes = project["scenes"]
        music = (project.get("music") or {}).get("file")
        final, starts, total = longform_video.render_documentary(
            d, scenes, _longform_library(project),
            _longform_takes(scenes), _longform_brand(project), d / music if music else None, on_progress=progress,
            music_level=project.get("music_level") or "normal", music_only=music_only,
            voice_db=float(project.get("voice_db") or 0.0), music_db_value=project.get("music_db"),
            end_line=explainer.END_LINE if _is_explainer(project) else None,
        )
        _longform_store.update(pid, lambda pr: pr.update(render={
            "status": "done", "progress": 1.0, "message": None, "error": None, "built_at": time.time(),
            "duration": total, "starts": starts, "took_seconds": round(time.time() - started),
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
def longform_start_render(pid: str, music_only: bool = False) -> dict:
    """Render the episode. music_only: only redo the final mix (after a
    music change or a voice / music volume change) -- about a minute, never the scenes; it
    refuses if anything else changed, so it can't turn into a long render."""
    project = _longform_project(pid)
    scenes = project.get("scenes") or []
    if project.get("status") != "script_ready" or not scenes:
        raise HTTPException(409, "write the story first")
    missing = [i + 1 for i, s in enumerate(scenes) if not longform.scene_ready(s)]
    if missing:
        raise HTTPException(409, "record these scenes first: " + ", ".join(map(str, missing)))
    if _longform_is_busy(pid, "visuals"):
        raise HTTPException(409, "the visuals are still being planned -- give it a minute")
    if music_only:
        if (project.get("render") or {}).get("status") != "done":
            raise HTTPException(409, "render the video once first")
        changed = longform_video.changed_scenes(_longform_store.path(pid), scenes, _longform_library(project),
                                                _longform_takes(scenes), _longform_brand(project))
        if changed:
            raise HTTPException(409, "Scenes " + ", ".join(map(str, changed)) + " changed since the last render, so this "
                                     "needs a full “Render again” (only those scenes are redone).")
    _longform_claim(pid, "render")
    _longform_store.update(pid, lambda pr: pr.update(render={
        "status": "rendering", "progress": 0.0, "message": "Updating the sound..." if music_only else "Starting..."}))
    threading.Thread(target=_longform_render, args=(pid, music_only), daemon=True).start()
    return {"ok": True}


@protected.get("/api/longform/projects/{pid}/video")
def longform_video_file(pid: str) -> FileResponse:
    project = _longform_project(pid)
    path = _longform_store.path(pid) / "final.mp4"
    if (project.get("render") or {}).get("status") != "done" or not path.is_file():
        raise HTTPException(404, "render the video first")
    safe = "".join(ch if ch.isalnum() or ch in " -_" else "" for ch in (project.get("title") or "video")).strip()[:60] or "video"
    return FileResponse(path, media_type="video/mp4", filename=f"{safe}.mp4")


def _longform_sources(project: dict) -> List[str]:
    src = project.get("sources") or {}
    out = []
    if src.get("wikipedia"):
        out.append(f"Wikipedia: {src['wikipedia']}")
    out += [f"{a['title']}: {a['url']}" for a in src.get("articles") or []]
    return out


@protected.post("/api/longform/projects/{pid}/publish-text")
async def longform_publish_text(pid: str) -> dict:
    """Three title options and a description with chapters and credits."""
    project = _longform_project(pid)
    render = project.get("render") or {}
    if render.get("status") != "done":
        raise HTTPException(409, "render the video first")
    if _is_explainer(project):
        try:
            dossier = json.loads((_longform_store.path(pid) / "dossier.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dossier = {"topic": project.get("topic") or ""}
        try:
            text = await run_in_threadpool(explainer.write_publish_text, project, dossier, project["scenes"], render.get("starts") or [])
        except Exception as e:
            raise HTTPException(500, f"Couldn't write the title and description: {e}")
        return _longform_store.update(pid, lambda pr: pr.update(publish=text))
    try:
        text = await run_in_threadpool(longform_video.write_publish_text, project.get("streamer") or {"login": project.get("login")},
                                       _longform_brand(), project["scenes"], render.get("starts") or [],
                                       _longform_sources(project) + documentary.visual_credits(project["scenes"]))
    except Exception as e:
        raise HTTPException(500, f"Couldn't write the title and description: {e}")
    return _longform_store.update(pid, lambda pr: pr.update(publish=text))


def _longform_other_voice(project: dict) -> bool:
    """True when a narrated scene was read by someone else's AI voice (a
    friend's, from the Voices page) -- YouTube wants that disclosed as
    altered or synthetic content. Dean's own cloned voice doesn't need it."""
    return any((sc.get("take") or {}).get("voice_id") for sc in project.get("scenes") or [])


class LongformUploadRequest(BaseModel):
    title: str
    description: str = ""
    privacy_status: str = "private"


@protected.post("/api/longform/projects/{pid}/upload")
def longform_upload(pid: str, req: LongformUploadRequest) -> dict:
    """Upload the finished episode to the Caught On Stream channel as a
    regular (long-form) video -- private by default, so it can be checked
    and scheduled in YouTube Studio before it goes live."""
    project = _longform_project(pid)
    path = _longform_store.path(pid) / "final.mp4"
    if (project.get("render") or {}).get("status") != "done" or not path.is_file():
        raise HTTPException(409, "render the video first")
    title = " ".join(req.title.split())
    if not title:
        raise HTTPException(400, "Title cannot be empty.")
    access_token = _longform_access_token(project)
    try:
        video_id = youtube_upload.upload_video(access_token, path, title=title, description=req.description,
                                               privacy_status=req.privacy_status, is_short=False,
                                               synthetic_media=_longform_other_voice(project))
    except (youtube_upload.UploadError, ValueError) as e:
        raise HTTPException(502, str(e)) from e
    url = f"https://youtu.be/{video_id}"
    project = _longform_store.update(pid, lambda pr: pr.update(youtube_video_id=video_id, youtube_url=url,
                                                               uploaded_privacy=req.privacy_status, uploaded_at=time.time()))
    if (project.get("thumbnails") or {}).get("items"):
        project = _longform_apply_thumbnail(pid, project, access_token)
    if _is_explainer(project):
        return project  # no clip moments to cut cliffhanger Shorts from
    # Straight away, cut 2 promo Shorts from the episode (unless they're
    # already being made) and tie them to it, so each Short's upload carries
    # the link to the full video.
    try:
        job_id = project.get("promo_job_id") if project.get("promo_job_id") in jobs else None
        if not job_id:
            job_id = _longform_start_promo(project, path)
        _attach_promo(job_id, url, title)
        project = _longform_store.update(pid, lambda pr: pr.update(promo_job_id=job_id))
    except Exception as e:  # the episode is up either way
        print(f"[longform] promo Shorts for {pid} didn't start: {e}", flush=True)
    return project


def _longform_chosen_thumb(pid: str, project: dict) -> Path:
    rec = project.get("thumbnails") or {}
    items = rec.get("items") or []
    if not items:
        raise HTTPException(409, "make the thumbnails first")
    item = items[min(int(rec.get("chosen") or 0), len(items) - 1)]
    return _longform_store.path(pid) / "thumbs" / item["file"]


def _longform_apply_thumbnail(pid: str, project: dict, access_token: str) -> dict:
    """Set the chosen thumbnail on the uploaded video; the outcome is kept
    on the project (the upload itself never fails because of it)."""
    try:
        youtube_upload.set_thumbnail(access_token, project["youtube_video_id"], _longform_chosen_thumb(pid, project))
        status = {"ok": True, "message": "Thumbnail set on YouTube.", "at": time.time()}
    except (youtube_upload.UploadError, HTTPException, OSError) as e:
        status = {"ok": False, "message": getattr(e, "detail", None) or str(e), "at": time.time()}
    return _longform_store.update(pid, lambda pr: pr.update(thumbnail_status=status))


@protected.post("/api/longform/projects/{pid}/thumbnails")
async def longform_make_thumbnails(pid: str) -> dict:
    """Three thumbnail options (longform_thumbnail.py): real frames from the
    streamer's clips, faces first, with a short hook line."""
    project = _longform_project(pid)
    if _is_explainer(project) and project.get("status") == "script_ready":
        try:
            try:
                dossier = json.loads((_longform_store.path(pid) / "dossier.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                dossier = None
            rec = await run_in_threadpool(explainer.make_thumbnails, _longform_store.path(pid), project, dossier)
        except Exception as e:
            raise HTTPException(500, f"Couldn't make thumbnails: {e}")
        return _longform_store.update(pid, lambda pr: pr.update(thumbnails=rec, thumbnail_status=None))
    if project.get("status") != "script_ready" or not project.get("library"):
        raise HTTPException(409, "write the story first")
    d = _longform_store.path(pid)
    try:
        dossier = json.loads((d / "dossier.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        dossier = {"profile": project.get("streamer") or {}}
    try:
        rec = await run_in_threadpool(longform_thumbnail.make_thumbnails, d, project, dossier, project["library"])
    except Exception as e:
        raise HTTPException(500, f"Couldn't make thumbnails: {e}")
    return _longform_store.update(pid, lambda pr: pr.update(thumbnails=rec, thumbnail_status=None))


class LongformThumbTextRequest(BaseModel):
    hook: str


@protected.put("/api/longform/projects/{pid}/thumbnails/{index}")
async def longform_redraw_thumbnail(pid: str, index: int, req: LongformThumbTextRequest) -> dict:
    project = _longform_project(pid)
    rec = project.get("thumbnails") or {}
    if not 0 <= index < len(rec.get("items") or []):
        raise HTTPException(404, "no such thumbnail")
    hook = " ".join(req.hook.split())[:40]
    if not hook:
        raise HTTPException(400, "type the text first")
    try:
        redraw = explainer.redraw if _is_explainer(project) else longform_thumbnail.redraw
        rec = await run_in_threadpool(redraw, _longform_store.path(pid), rec, index, hook)
    except Exception as e:
        raise HTTPException(500, f"Couldn't redraw it: {e}")
    return _longform_store.update(pid, lambda pr: pr.update(thumbnails=rec))


@protected.post("/api/longform/projects/{pid}/thumbnails/{index}/choose")
def longform_choose_thumbnail(pid: str, index: int) -> dict:
    project = _longform_project(pid)
    if not 0 <= index < len((project.get("thumbnails") or {}).get("items") or []):
        raise HTTPException(404, "no such thumbnail")
    return _longform_store.update(pid, lambda pr: pr["thumbnails"].update(chosen=index))


@protected.get("/api/longform/projects/{pid}/thumbnails/{index}")
def longform_thumbnail_file(pid: str, index: int, download: bool = False) -> FileResponse:
    project = _longform_project(pid)
    items = (project.get("thumbnails") or {}).get("items") or []
    path = _longform_store.path(pid) / "thumbs" / items[index]["file"] if 0 <= index < len(items) else None
    if path is None or not path.is_file():
        raise HTTPException(404, "not found")
    if download:
        return FileResponse(path, media_type="image/jpeg", filename=f"thumbnail_{index + 1}.jpg")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@protected.post("/api/longform/projects/{pid}/thumbnails/apply")
def longform_apply_thumbnail(pid: str) -> dict:
    """Set (or change) the thumbnail on the already-uploaded video."""
    project = _longform_project(pid)
    if not project.get("youtube_video_id"):
        raise HTTPException(409, "upload the video first -- the chosen thumbnail is set then")
    _longform_chosen_thumb(pid, project)
    return _longform_apply_thumbnail(pid, project, _longform_access_token(project))


PROMO_SHORTS = 2


def _longform_start_promo(project: dict, path: Path) -> str:
    """Two cliffhanger Shorts cut from the episode (longform_promo.py): each
    stops right before a payoff, so viewers go to the full video to find
    out. They land on Home as a finished job, ready to post like any clip."""
    name = (project.get("streamer") or {}).get("display_name") or project.get("login") or "the streamer"
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id, "source_url": f"Cliffhanger Shorts from {project.get('title') or 'the episode'}",
            "created_at": time.time(), "state": "running", "message": "Cutting 2 cliffhanger Shorts from the episode...",
            "progress": 0.1, "estimate_minutes": 2, "clips": [], "error": None, "saved": False,
            "request": JobRequest(source=str(path), num_clips=PROMO_SHORTS, focus=f"cliffhangers from the story of {name}"),
            "channel_profile": DEFAULT_CHANNEL_PROFILE, "promo_episode": project["id"],
        }
        cancel_events[job_id] = threading.Event()
    _persist(job_id)
    threading.Thread(target=_longform_build_promo, args=(job_id, project, path), daemon=True).start()
    return job_id


def _longform_build_promo(job_id: str, project: dict, path: Path) -> None:
    out_dir = BASE_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cuts = longform_promo.plan(project, _longform_store.path(project["id"]))
        if not cuts:
            raise RuntimeError("the story has no setup-then-clip moments to cut a cliffhanger from")
        clips = []
        stop = cancel_events.get(job_id)
        for k, cut in enumerate(cuts[:PROMO_SHORTS], start=1):
            if stop is not None and stop.is_set():
                break
            name = f"cliffhanger_{k}.mp4"
            dur = longform_promo.render_short(path, cut, _longform_brand(), out_dir / name)
            clips.append({
                "file": name, "start": cut["start"], "end": cut["end"], "duration": dur,
                "title": cut["title"], "upload_title": cut["title"], "description": cut["description"],
                "hook_caption": cut["hook"], "hook_text": cut["hook"], "reason": f"Cliffhanger ({cut['kind']}) from the episode",
                # Made from Dean's own documentary, not picked by the clip
                # pipeline: kept out of the clip registry so they don't skew
                # what the Shorts learn from YouTube stats.
                "promo": True,
                **({"synthetic_voice": True} if _longform_other_voice(project) else {}),
            })
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None:
                if stop is not None and stop.is_set():
                    job.update(state="cancelled", message="Stopped", clips=clips if job.get("saved") else [])
                else:
                    job.update(state="done", message="Done", progress=1.0, clips=clips)
    except Exception as e:
        print(f"[longform] cliffhanger Shorts for {project.get('id')} failed: {e}", flush=True)
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None:
                job.update(state="error", message="Failed", error=f"Couldn't cut the cliffhanger Shorts: {e}")
    _persist(job_id)


def _attach_promo(job_id: str, url: str, title: str) -> None:
    """Mark a clip job as promo Shorts for a long-form video: the Home
    page then adds "Full story: <link>" to each Short's description and,
    once a Short is posted, points to where YouTube Studio's "Related
    video" is set (YouTube's API can't set that link itself)."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        job["promo_for"] = {"url": url, "title": title}
    _persist(job_id)


@protected.post("/api/longform/projects/{pid}/promo-shorts")
def longform_promo_shorts(pid: str) -> dict:
    """Send the finished episode through the normal clip pipeline to cut
    promo Shorts from it; they show up in Home's jobs list as usual."""
    project = _longform_project(pid)
    if _is_explainer(project):
        raise HTTPException(409, "cliffhanger Shorts are cut from documentary clip moments; explainers don't have any")
    path = _longform_store.path(pid) / "final.mp4"
    if (project.get("render") or {}).get("status") != "done" or not path.is_file():
        raise HTTPException(409, "render the video first")
    with jobs_lock:
        running = (jobs.get(project.get("promo_job_id") or "") or {}).get("state") not in (None, *TERMINAL_STATES)
    if running:
        # Already being cut (e.g. the button pressed twice): one set at a time.
        return project
    job_id = _longform_start_promo(project, path)
    if project.get("youtube_url"):
        _attach_promo(job_id, project["youtube_url"], project.get("title") or "")
    return _longform_store.update(pid, lambda pr: pr.update(promo_job_id=job_id))


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
<a class="longform-link" href="/long-form"><span>🎬 Go to long-form videos</span><span>→</span></a>
<a class="longform-link" href="/caught-on-code"><span>💻 Go to Caught On Code</span><span>→</span></a>

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
// Jobs cut from a long-form episode as its promo Shorts: {url, title} of
// the episode, so each Short's description links to it.
const jobPromoFor = {};
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
  const promo = jobPromoFor[jobId];
  if (promo && promo.url && !youtubeUploadDescriptionInput.value.includes(promo.url)) {
    youtubeUploadDescriptionInput.value = (youtubeUploadDescriptionInput.value.trim() + '\\n\\nFull story: ' + promo.url).trim();
  }
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
    jobPromoFor[jobId] = job.promo_for || null;
    if (job.promo_for && job.promo_for.url) {
      const note = document.createElement('div');
      note.className = 'hint';
      note.style.marginTop = '6px';
      if (c.youtube_video_id) {
        // YouTube's API can't set a Short's "Related video"; this opens the
        // Short in Studio, where it's one dropdown.
        note.appendChild(document.createTextNode('🔗 Last step: '));
        const studio = document.createElement('a');
        studio.href = `https://studio.youtube.com/video/${encodeURIComponent(c.youtube_video_id)}/edit`;
        studio.target = '_blank';
        studio.rel = 'noopener';
        studio.textContent = 'open it in YouTube Studio';
        note.appendChild(studio);
        note.appendChild(document.createTextNode(` and set “Related video” to “${job.promo_for.title || 'the full episode'}”, so viewers can tap through to it.`));
      } else {
        note.textContent = `🎬 Promo Short for “${job.promo_for.title || 'your episode'}”. Its description will link to the full video.`;
      }
      div.appendChild(note);
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

      if (!c.is_recap && !c.promo) {
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
  .learning { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); }
  .learning h3 { margin: 0 0 4px; font-size: 1rem; }
  .learn-sub { margin-top: 12px; font-weight: 600; font-size: 0.85rem; }
  .learn-list { margin: 6px 0 0; padding-left: 18px; font-size: 0.85rem; }
  .learn-list li { margin: 3px 0; overflow-wrap: anywhere; }
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
  .lf-episode { border: 1px solid var(--border); border-radius: 12px; padding: 14px; margin-top: 12px; }
  .lf-episode h3 { margin: 0; font-size: 1rem; }
  .lf-episode h3 a { color: var(--text); }
  .lf-chart { position: relative; margin-top: 8px; }
  .lf-chart svg { width: 100%; height: auto; display: block; touch-action: pan-y; }
  .lf-tip { position: absolute; pointer-events: none; background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 8px; font-size: 0.75rem; box-shadow: 0 4px 14px rgba(0,0,0,0.12); white-space: nowrap; display: none; }
  .lf-chapters { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.74rem; color: var(--muted); margin-top: 4px; }
  .lf-drop { font-size: 0.82rem; margin-top: 6px; }
  .lf-drop b { font-variant-numeric: tabular-nums; }
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
  <label style="margin-top:0">🎬 Long-form episodes</label>
  <p class="hint">How each &ldquo;Story Of&rdquo; episode is doing on YouTube. The retention line is laid over the
    episode&rsquo;s chapters, so a drop shows which part people left in. YouTube&rsquo;s numbers run about two days behind.</p>
  <div id="longform-body"><div class="hint">Loading...</div></div>
  <button id="longform-refresh-btn" type="button">↻ Refresh from YouTube</button>
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
  ['', 'Shorts', 'Stayed', 'At 3s', 'Watched', 'Median views', 'Subs / 1k views'].forEach(h => headRow.appendChild(el('th', { text: h })));
  head.appendChild(headRow);
  table.appendChild(head);
  const body = el('tbody');
  groups.forEach(g => {
    const nameRow = el('tr', { className: 'group-name' });
    const nameCell = el('th', { text: g.name + (g.made_here_only ? ' (clips made here only)' : '') });
    nameCell.colSpan = 7;
    nameRow.appendChild(nameCell);
    body.appendChild(nameRow);
    g.buckets.forEach(b => {
      const row = el('tr', { className: b.enough ? '' : 'thin' });
      row.appendChild(el('td', { text: b.label + (b.enough ? '' : ' *') }));
      row.appendChild(el('td', { text: String(b.n) }));
      row.appendChild(el('td', { text: fmtFraction(b.stayed) }));
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

// What the clip picker learns from (clip_performance.build_learning): the
// last two weeks vs the two before, best and worst recent Shorts, streamers.
function buildLearningBlock(L) {
  const box = el('div', { className: 'learning' });
  box.appendChild(el('h3', { text: 'What the clip picker learns from' }));
  box.appendChild(el('div', { className: 'hint', text: 'Every new batch of clips is picked with these in front of Claude, alongside the comparisons below.' }));
  const a = L.last_14 || {}, b = L.prev_14 || {};
  if (a.n && b.n) {
    const tiles = el('div', { className: 'stat-row' });
    const change = (x, y) => (x != null && y) ? `${x >= y ? '+' : ''}${Math.round((x - y) / y * 100)}% vs the 2 weeks before` : '';
    tiles.appendChild(statTile('Median views, last 2 weeks', fmtCount(a.median_views), change(a.median_views, b.median_views) || `${a.n} Shorts`));
    if (a.stayed != null) tiles.appendChild(statTile('Stayed to watch, last 2 weeks', fmtFraction(a.stayed), b.stayed != null ? `was ${fmtFraction(b.stayed)}` : ''));
    box.appendChild(tiles);
  }
  const list = (title, items) => {
    if (!items || !items.length) return;
    box.appendChild(el('div', { className: 'learn-sub', text: title }));
    const ul = el('ul', { className: 'learn-list' });
    items.forEach(e => {
      const li = el('li');
      const link = el('a', { text: e.title, href: `https://youtube.com/shorts/${encodeURIComponent(e.id)}` });
      link.target = '_blank'; link.rel = 'noopener';
      li.appendChild(link);
      const bits = [fmtCount(e.views) + ' views', e.stayed != null ? fmtFraction(e.stayed) + ' stayed' : null,
        e.duration != null ? Math.round(e.duration) + 's' : null, e.moment_type, e.streamer].filter(Boolean);
      li.appendChild(el('span', { className: 'hint', text: ' · ' + bits.join(' · ') }));
      ul.appendChild(li);
    });
    box.appendChild(ul);
  };
  list('Best recent Shorts', L.best);
  list('Worst recent Shorts', L.worst);
  if (L.streamers && L.streamers.length) {
    box.appendChild(el('div', { className: 'learn-sub', text: 'By streamer (last 60 days)' }));
    const wrap = el('div', { className: 'table-scroll' });
    const t = el('table', { className: 'perf-table' });
    const hr = el('tr');
    ['Streamer', 'Shorts', 'Median views', 'Stayed', 'Last 3 weeks', 'Before'].forEach(h => hr.appendChild(el('th', { text: h })));
    const th = el('thead'); th.appendChild(hr); t.appendChild(th);
    const tb = el('tbody');
    L.streamers.forEach(x => {
      const r = el('tr');
      const down = x.recent_median != null && x.earlier_median != null && x.recent_median < x.earlier_median * 0.7;
      const up = x.recent_median != null && x.earlier_median != null && x.recent_median > x.earlier_median * 1.3;
      r.appendChild(el('td', { text: x.name + (down ? ' ↓' : up ? ' ↑' : '') }));
      r.appendChild(el('td', { text: String(x.n) }));
      r.appendChild(el('td', { text: fmtCount(x.median_views) }));
      r.appendChild(el('td', { text: fmtFraction(x.stayed) }));
      r.appendChild(el('td', { text: x.recent_median == null ? '–' : `${fmtCount(x.recent_median)} (${x.recent_n})` }));
      r.appendChild(el('td', { text: x.earlier_median == null ? '–' : `${fmtCount(x.earlier_median)} (${x.earlier_n})` }));
      tb.appendChild(r);
    });
    t.appendChild(tb); wrap.appendChild(t); box.appendChild(wrap);
  }
  return box;
}

function buildVideoTable(videos) {
  const details = el('details', { className: 'perf-videos' });
  details.appendChild(el('summary', { text: `Every Short in this analysis (${videos.length})` }));
  const wrap = el('div', { className: 'table-scroll' });
  const table = el('table', { className: 'perf-table' });
  const head = el('thead');
  const headRow = el('tr');
  ['Title', 'Stayed', 'At 3s', 'Watched', 'Views', 'Shares / 1k', 'Subs', 'Length', 'Posted', 'Made here'].forEach(h => headRow.appendChild(el('th', { text: h })));
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
    row.appendChild(el('td', { text: fmtFraction(v.stayed) }));
    row.appendChild(el('td', { text: fmtFraction(v.retention ? v.retention.watch_3s : null) }));
    row.appendChild(el('td', { text: fmtPercent(v.avg_view_pct) }));
    row.appendChild(el('td', { text: fmtCount(v.views) }));
    row.appendChild(el('td', { text: v.shares_per_1k == null ? '–' : v.shares_per_1k.toFixed(1) }));
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
  if (s.stayed != null) tiles.appendChild(statTile('Stayed to watch', fmtFraction(s.stayed), 'plays not swiped away · 70%+ is viral, ~50% average'));
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
  if (data.learning) body.appendChild(buildLearningBlock(data.learning));
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

// ---------- long-form episodes ----------
const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
  return n;
}
function fmtHours(minutes) {
  if (minutes == null) return '–';
  return minutes >= 600 ? `${Math.round(minutes / 60)} h` : minutes >= 60 ? `${(minutes / 60).toFixed(1)} h` : `${Math.round(minutes)} min`;
}

// One series (share of viewers still watching), so one hue and no legend;
// chapters are numbered dashed markers listed under the chart; hovering
// (or dragging a finger) shows the exact point and the chapter it's in.
function buildRetentionChart(ep, W) {
  const wrap = el('div', { className: 'lf-chart' });
  const H = W < 500 ? 200 : 220, L = 38, R = 10, T = 18, B = 26;
  const iw = W - L - R, ih = H - T - B;
  const maxY = Math.max(1, ...ep.curve.map(p => p[1]));
  const x = f => L + f * iw, y = v => T + ih - (v / maxY) * ih;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': `Share of viewers still watching across ${ep.title}` });
  [0, 0.25, 0.5, 0.75, 1].forEach(v => {
    svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: 'var(--border)', 'stroke-width': 1 }));
    const t = svgEl('text', { x: L - 6, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)' });
    t.textContent = `${Math.round(v * 100)}%`;
    svg.appendChild(t);
  });
  (W < 500 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).forEach(f => {
    const t = svgEl('text', { x: x(f), y: H - 6, 'text-anchor': f === 0 ? 'start' : f === 1 ? 'end' : 'middle', 'font-size': 11, fill: 'var(--muted)' });
    t.textContent = formatSeconds(f * ep.duration);
    svg.appendChild(t);
  });
  const chapters = (ep.chapters || []).filter(c => ep.duration && c.t / ep.duration < 0.99);
  chapters.forEach((c, i) => {
    const cx = x(c.t / ep.duration);
    svg.appendChild(svgEl('line', { x1: cx, x2: cx, y1: T, y2: T + ih, stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0.6 }));
    const n = svgEl('text', { x: cx + 3, y: T - 5, 'font-size': 11, fill: 'var(--muted)' });
    n.textContent = String(i + 1);
    svg.appendChild(n);
  });
  const d = ep.curve.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ');
  svg.appendChild(svgEl('path', { d, fill: 'none', stroke: 'var(--chart-you)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  const guide = svgEl('line', { y1: T, y2: T + ih, stroke: 'var(--text)', 'stroke-width': 1, opacity: 0 });
  const dot = svgEl('circle', { r: 5, fill: 'var(--chart-you)', stroke: 'var(--card)', 'stroke-width': 2, opacity: 0 });
  svg.appendChild(guide); svg.appendChild(dot);
  const hit = svgEl('rect', { x: L, y: 0, width: iw, height: H, fill: 'transparent' });
  svg.appendChild(hit);
  const tip = el('div', { className: 'lf-tip' });
  function show(evt) {
    const box = svg.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, ((evt.clientX - box.left) / box.width * W - L) / iw));
    const p = ep.curve.reduce((a, b) => Math.abs(b[0] - f) < Math.abs(a[0] - f) ? b : a);
    const sec = p[0] * ep.duration;
    let ch = null;
    chapters.forEach((c, i) => { if (c.t <= sec) ch = `${i + 1}. ${c.title}`; });
    guide.setAttribute('x1', x(p[0])); guide.setAttribute('x2', x(p[0])); guide.setAttribute('opacity', 0.35);
    dot.setAttribute('cx', x(p[0])); dot.setAttribute('cy', y(p[1])); dot.setAttribute('opacity', 1);
    tip.textContent = `${formatSeconds(sec)} · ${Math.round(p[1] * 100)}% still watching` + (ch ? ` · ${ch}` : '');
    tip.style.display = 'block';
    const px = (x(p[0]) / W) * box.width;
    tip.style.left = Math.min(box.width - tip.offsetWidth, Math.max(0, px - tip.offsetWidth / 2)) + 'px';
    tip.style.top = Math.max(0, (y(p[1]) / H) * box.height - 44) + 'px';
  }
  function hide() { tip.style.display = 'none'; guide.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); }
  hit.addEventListener('pointermove', show);
  hit.addEventListener('pointerdown', show);
  hit.addEventListener('pointerleave', hide);
  wrap.appendChild(svg);
  wrap.appendChild(tip);
  if (chapters.length) {
    const list = el('div', { className: 'lf-chapters' });
    chapters.forEach((c, i) => list.appendChild(el('span', { text: `${i + 1}. ${c.title} (${formatSeconds(c.t)})` })));
    wrap.appendChild(list);
  }
  return wrap;
}

// Views per day since upload: one series, bars with the value on hover.
function buildDailyChart(daily, W) {
  const wrap = el('div', { className: 'lf-chart' });
  const H = 120, L = 6, R = 6, T = 8, B = 20;
  const iw = W - L - R, ih = H - T - B, n = daily.length;
  const maxV = Math.max(1, ...daily.map(d => d.views));
  const bw = Math.max(2, iw / n - 2);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Views per day since upload' });
  svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: T + ih, y2: T + ih, stroke: 'var(--border)', 'stroke-width': 1 }));
  daily.forEach((d, i) => {
    const h = Math.max(d.views ? 2 : 0, (d.views / maxV) * ih);
    const bx = L + i * (iw / n) + 1;
    const r = svgEl('rect', { x: bx, y: T + ih - h, width: bw, height: h, rx: Math.min(4, bw / 2), fill: 'var(--chart-you)' });
    const ttl = svgEl('title'); ttl.textContent = `${d.date}: ${Number(d.views).toLocaleString()} views`;
    r.appendChild(ttl);
    svg.appendChild(r);
    const hitR = svgEl('rect', { x: bx - 1, y: T, width: iw / n, height: ih, fill: 'transparent' });
    hitR.appendChild(ttl.cloneNode(true));
    svg.appendChild(hitR);
  });
  [[0, 'start'], [n - 1, 'end']].forEach(([i, anchor]) => {
    if (i < 0 || (i === n - 1 && n < 2)) return;
    const t = svgEl('text', { x: i === 0 ? L : W - R, y: H - 5, 'text-anchor': anchor, 'font-size': 11, fill: 'var(--muted)' });
    t.textContent = daily[i].date.slice(5);
    svg.appendChild(t);
  });
  wrap.appendChild(svg);
  return wrap;
}

function buildEpisode(ep, W) {
  const box = el('div', { className: 'lf-episode' });
  const h = el('h3');
  const a = el('a', { text: ep.title, href: ep.url || '#' }); a.target = '_blank';
  h.appendChild(a);
  box.appendChild(h);
  if (ep.error) { box.appendChild(el('div', { className: 'hint', text: ep.error })); return box; }
  const st = ep.stats || {};
  box.appendChild(el('div', { className: 'hint', text: `${formatSeconds(ep.duration)} long` + (ep.privacy ? ` · uploaded ${ep.privacy}` : '') }));
  const tiles = el('div', { className: 'stat-row' });
  tiles.appendChild(statTile('Views', fmtCount(st.views)));
  tiles.appendChild(statTile('Watch time', fmtHours(st.estimatedMinutesWatched)));
  tiles.appendChild(statTile('Average view', st.averageViewDuration != null ? formatSeconds(st.averageViewDuration) : '–',
    st.averageViewPercentage != null ? `${fmtPercent(st.averageViewPercentage)} of the video` : null));
  if (ep.watch_30s != null) tiles.appendChild(statTile('Still watching at 0:30', fmtFraction(ep.watch_30s), 'the opening'));
  tiles.appendChild(statTile('Subscribers gained', fmtCount(st.subscribersGained)));
  if (st.click_rate != null) tiles.appendChild(statTile('Thumbnail click rate', `${(st.click_rate * (st.click_rate <= 1 ? 100 : 1)).toFixed(1)}%`, `${fmtCount(st.impressions)} impressions`));
  if (ep.vs_similar != null) tiles.appendChild(statTile('Retention vs similar videos', ep.vs_similar.toFixed(2), '0.5 = typical for the length'));
  box.appendChild(tiles);
  if (!st.views) {
    box.appendChild(el('div', { className: 'hint', text: ep.privacy === 'private'
      ? 'No views yet -- it is still private. Numbers appear here once it is public (and about two days behind).'
      : 'No views counted yet -- YouTube runs about two days behind.' }));
  }
  if (ep.curve && ep.curve.length) {
    const t = el('div', { className: 'chart-title' }); t.style.flexWrap = 'wrap';
    t.appendChild(el('span', { text: 'Who is still watching' }));
    t.appendChild(el('span', { className: 'hint', text: 'share of viewers at each point; numbered lines are chapters' }));
    t.style.marginTop = '14px';
    box.appendChild(t);
    box.appendChild(buildRetentionChart(ep, W));
    (ep.drops || []).forEach(dp => {
      const row = el('div', { className: 'lf-drop' });
      row.appendChild(el('b', { text: `${formatSeconds(dp.t)} ` }));
      row.appendChild(document.createTextNode(`lost ${Math.round(dp.fall * 100)}% of viewers, during scene ${dp.scene}: ${dp.what}`));
      box.appendChild(row);
    });
  } else if (st.views) {
    box.appendChild(el('div', { className: 'hint', text: 'YouTube shows the retention curve once the video has enough views.' }));
  }
  if (ep.daily && ep.daily.length > 1) {
    const t = el('div', { className: 'chart-title' }); t.style.marginTop = '14px';
    t.appendChild(el('span', { text: 'Views per day' }));
    box.appendChild(t);
    box.appendChild(buildDailyChart(ep.daily, W));
  }
  if (ep.traffic && ep.traffic.length) {
    const block = el('div', { className: 'chart-block' });
    const t = el('div', { className: 'chart-title' });
    t.appendChild(el('span', { text: 'Where the views came from' }));
    block.appendChild(t);
    ep.traffic.forEach(s => {
      const row = el('div', { className: 'bar-row' });
      row.title = `${fmtCount(s.views)} views from ${s.source}`;
      const lc = el('div', { className: 'bar-label' }); lc.appendChild(el('div', { className: 'bar-title', text: s.source }));
      row.appendChild(lc);
      const track = el('div', { className: 'bar-track' }); const fill = el('div', { className: 'bar-fill' });
      fill.style.width = Math.max(2, Math.round(s.share * 100)) + '%'; fill.style.background = 'var(--chart-you)';
      track.appendChild(fill); row.appendChild(track);
      row.appendChild(el('div', { className: 'bar-value', text: fmtFraction(s.share) }));
      block.appendChild(row);
    });
    box.appendChild(block);
  }
  if (ep.shorts && ep.shorts.length) {
    const t = el('div', { className: 'chart-title' }); t.style.marginTop = '14px';
    t.appendChild(el('span', { text: 'Its cliffhanger Shorts' }));
    box.appendChild(t);
    const tbl = el('table', { className: 'perf-table' });
    const head = el('tr');
    ['Short', 'Views', 'Watched', 'Subs'].forEach(x => head.appendChild(el('th', { text: x })));
    const thead = el('thead'); thead.appendChild(head); tbl.appendChild(thead);
    const tb = el('tbody');
    ep.shorts.forEach(s => {
      const tr = el('tr');
      const td = el('td'); const sa = el('a', { text: s.title || s.video_id, href: s.url }); sa.target = '_blank'; td.appendChild(sa);
      tr.appendChild(td);
      tr.appendChild(el('td', { text: fmtCount(s.views) }));
      tr.appendChild(el('td', { text: fmtPercent(s.averageViewPercentage) }));
      tr.appendChild(el('td', { text: fmtCount(s.subscribersGained) }));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    box.appendChild(tbl);
  }
  return box;
}

async function loadLongform(refresh) {
  const body = document.getElementById('longform-body');
  const btn = document.getElementById('longform-refresh-btn');
  btn.disabled = true; body.style.opacity = '0.5';
  let data;
  try {
    data = await (await fetch('/api/longform-analytics' + (refresh ? '?refresh=true' : ''))).json();
  } catch (e) {
    data = { available: false, reason: 'Could not load the long-form numbers.' };
  }
  body.style.opacity = ''; btn.disabled = false;
  body.innerHTML = '';
  if (!data.available) { body.appendChild(el('div', { className: 'hint', text: data.reason || 'Not available.' })); return; }
  // Charts are drawn at the width they're shown at, so their labels stay
  // readable on a phone instead of a desktop drawing scaled down.
  const W = Math.max(280, Math.min(900, body.clientWidth - 30));
  data.episodes.forEach(ep => body.appendChild(buildEpisode(ep, W)));
}
loadLongform(false);
document.getElementById('longform-refresh-btn').addEventListener('click', () => loadLongform(true));

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


# The /long-form page -- "The Story Of" streamer documentaries; see clipper/documentary.py
# and the /api/longform routes.
# ---- Voices page and the friend's private recording page ----------------------
_SIMPLE_PAGE_CSS = """
  :root {
    color-scheme: light dark;
    --bg: #f2f3f7; --card: #ffffff; --text: #1a1b1f; --muted: #6b7280; --border: #e5e7eb;
    --accent: #6d28d9; --accent2: #ec4899; --accent-text: #ffffff; --danger: #dc2626;
    --ok: #15803d; --warn: #b45309;
    --shadow: 0 1px 2px rgba(16,24,40,0.04), 0 8px 24px rgba(16,24,40,0.06);
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0f1115; --card: #1a1c23; --text: #f2f3f7; --muted: #9aa0ac; --border: #2b2e37;
      --ok: #4ade80; --warn: #fbbf24; --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.4); }
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; background: var(--bg);
    color: var(--text); margin: 0; padding: 40px 16px; }
  .page { max-width: 760px; margin: 0 auto; }
  .topnav { display: flex; flex-wrap: wrap; gap: 4px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px; margin-bottom: 16px; box-shadow: var(--shadow); }
  .topnav a { flex: 1 1 auto; text-align: center; padding: 9px 10px; border-radius: 9px; font-size: 0.84rem;
    font-weight: 600; color: var(--muted); text-decoration: none; }
  .topnav a:hover { color: var(--text); }
  .topnav a.active { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: var(--accent-text); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--shadow);
    padding: 28px 28px 32px; }
  .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
  .brand .logo { font-size: 1.3rem; line-height: 1; display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 10px; background: linear-gradient(135deg, var(--accent), var(--accent2)); }
  h1 { font-size: 1.3rem; margin: 0; letter-spacing: -0.01em; }
  h3 { font-size: 1.02rem; margin: 0 0 4px; }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin: 4px 0 20px; }
  label { display: block; margin-top: 14px; font-size: 0.82rem; font-weight: 600; color: var(--muted); }
  input, select, textarea { width: 100%; padding: 10px 12px; margin-top: 6px; font-size: 0.95rem; background: var(--bg);
    color: var(--text); font-family: inherit; border: 1px solid var(--border); border-radius: 10px; }
  input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
  input[type=checkbox] { width: auto; margin: 0; }
  button { margin-top: 14px; padding: 10px 18px; font-size: 0.92rem; font-weight: 600; cursor: pointer; border: none;
    border-radius: 10px; background: linear-gradient(135deg, var(--accent), var(--accent2)); color: var(--accent-text); }
  button.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
  button.danger-link { background: none; border: none; color: var(--danger); padding: 0; font-weight: 600; font-size: 0.85rem; }
  button:disabled { opacity: 0.45; cursor: default; }
  button.recording { background: var(--danger); }
  .hint { font-size: 0.8rem; color: var(--muted); margin-top: 6px; line-height: 1.45; }
  .section { margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--border); }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .actions button { margin-top: 10px; }
  .status { border-radius: 10px; padding: 10px 12px; font-size: 0.86rem; margin-top: 10px; background: var(--bg);
    border: 1px solid var(--border); }
  .status.ok { border-color: color-mix(in srgb, var(--ok) 45%, var(--border)); }
  .status.bad { border-color: color-mix(in srgb, var(--danger) 45%, var(--border)); }
  .tick { display: flex; gap: 10px; align-items: flex-start; margin-top: 14px; font-size: 0.88rem; font-weight: 500;
    color: var(--text); }
  .tick input { margin-top: 3px; }
  a { color: var(--accent); }
"""

VOICES_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper — Voices</title>
<style>
__CSS__
  .voice { border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-top: 12px; }
  .voice .top { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .voice .name { font-weight: 700; font-size: 1rem; }
  .badge { font-size: 0.75rem; font-weight: 700; border-radius: 999px; padding: 3px 9px; background: var(--bg);
    border: 1px solid var(--border); white-space: nowrap; }
  .badge.ok { color: var(--ok); }
  .badge.wait { color: var(--warn); }
  .link-box { display: flex; gap: 8px; margin-top: 10px; }
  .link-box input { margin-top: 0; font-size: 0.82rem; }
  .link-box button { margin-top: 0; white-space: nowrap; }
  .try { display: flex; gap: 8px; margin-top: 10px; }
  .try input { margin-top: 0; }
  .try button { margin-top: 0; white-space: nowrap; }
  details summary { cursor: pointer; font-weight: 600; font-size: 0.88rem; margin-top: 10px; color: var(--accent); }
  @media (max-width: 520px) { .card { padding: 20px 16px 24px; } .link-box, .try { flex-direction: column; } }
</style>
</head>
<body>
<div class="page">
__NAV_LINKS__
<div class="card">
  <div class="brand"><span class="logo">🗣</span><h1>Voices</h1></div>
  <p class="subtitle">AI copies of voices that can read your lines: yours, and friends who said yes. Pick one when the AI reads a long-form episode or a ranking Short.</p>
  <div id="setup" class="status" style="display:none"></div>
  <div id="voices"></div>

  <div class="section">
    <h3>➕ Add a friend's voice</h3>
    <div class="hint">Only copy a voice after the person has said yes. Videos that use their voice get YouTube's "altered or synthetic content" label automatically, so viewers know it's AI.</div>
    <label for="new-name">Their name</label>
    <input id="new-name" maxlength="40" placeholder="e.g. Sam">
    <label class="tick"><input type="checkbox" id="new-consent"> <span>They agreed to an AI copy of their voice narrating my videos.</span></label>
    <button id="add-btn" type="button">Add voice</button>
    <div class="hint" id="add-msg"></div>
  </div>
</div>
</div>
<audio id="player" style="display:none"></audio>
<script>
const $ = (id) => document.getElementById(id);
let data = null;
async function api(path, opts) {
  const r = await fetch(path, opts);
  const out = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(out.detail || `Request failed (${r.status})`);
  return out;
}
const jsonOpts = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function day(ts) { return new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }); }
function play(url) { const a = $('player'); a.src = url + (url.includes('?') ? '&' : '?') + 't=' + Date.now(); a.play(); }

async function load() {
  data = await api('/api/voices');
  render();
}
function render() {
  const st = $('setup');
  if (!data.installed) { st.style.display = 'block'; st.textContent = 'The AI voice isn’t installed on the server yet.'; }
  else if (data.setup_hint) { st.style.display = 'block'; st.textContent = '⚙️ ' + data.setup_hint; }
  else if (data.error) { st.style.display = 'block'; st.textContent = '⚠️ ' + data.error; }
  else st.style.display = 'none';
  const box = $('voices'); box.innerHTML = '';
  for (const v of data.voices) box.appendChild(voiceCard(v));
}
function voiceCard(v) {
  const c = el('div', 'voice');
  const top = el('div', 'top');
  top.appendChild(el('span', 'name', v.mine ? '🎙 You' : '👤 ' + v.name));
  top.appendChild(el('span', 'badge ' + (v.sample ? 'ok' : 'wait'),
    v.sample ? `✅ Sample: ${Math.round(v.sample.duration)} s` : '⏳ No sample yet'));
  c.appendChild(top);
  if (v.mine) {
    c.appendChild(el('div', 'hint', v.sample ? `Recorded ${day(v.sample.recorded_at)}. Re-record it on the long-form page under “Your AI voice”.`
      : 'Record your sample on the long-form page, under “Your AI voice”.'));
  } else {
    const cons = v.consent || {};
    let line = cons.confirmed_at ? `You confirmed they agreed on ${day(cons.confirmed_at)}.` : '';
    if (cons.self_confirmed_at) line += ` They agreed on the recording link on ${day(cons.self_confirmed_at)}.`;
    if (line) c.appendChild(el('div', 'hint', line.trim()));
  }
  const acts = el('div', 'actions');
  if (v.sample) {
    const p = el('button', 'secondary', '▶ Play sample');
    p.type = 'button'; p.onclick = () => play(v.mine ? '/api/longform/voice/sample' : `/api/voices/${v.id}/sample`);
    acts.appendChild(p);
  }
  if (!v.mine) {
    const lk = el('button', 'secondary', v.link ? '🔗 New recording link' : '🔗 Get a recording link');
    lk.type = 'button';
    lk.onclick = async () => { try { await api(`/api/voices/${v.id}/link`, { method: 'POST' }); await load(); } catch (e) { alert(e.message); } };
    acts.appendChild(lk);
    const up = el('button', 'secondary', '⬆ Upload a voice note'); up.type = 'button';
    const file = el('input'); file.type = 'file'; file.accept = 'audio/*,video/*,.opus,.m4a,.ogg,.mp3,.wav,.aac'; file.style.display = 'none';
    up.onclick = () => file.click();
    file.onchange = async () => {
      const f = file.files[0]; if (!f) return;
      up.disabled = true; up.textContent = 'Checking…';
      try {
        await api(`/api/voices/${v.id}/sample`, { method: 'POST', headers: { 'Content-Type': f.type || 'application/octet-stream', 'X-Filename': f.name }, body: f });
        await load();
      } catch (e) { alert(e.message); up.disabled = false; up.textContent = '⬆ Upload a voice note'; }
    };
    acts.appendChild(up); acts.appendChild(file);
    const del = el('button', 'danger-link', 'Delete voice'); del.type = 'button'; del.style.marginLeft = 'auto';
    del.onclick = async () => {
      if (!confirm(`Delete ${v.name}'s voice? Lines already read in it keep their audio.`)) return;
      try { data = await api(`/api/voices/${v.id}`, { method: 'DELETE' }); render(); } catch (e) { alert(e.message); }
    };
    acts.appendChild(del);
  }
  c.appendChild(acts);
  if (!v.mine && v.link) {
    const url = location.origin + v.link.url;
    const lb = el('div', 'link-box');
    const inp = el('input'); inp.readOnly = true; inp.value = url; inp.onclick = () => inp.select();
    const cp = el('button', '', navigator.share ? '📤 Send' : '📋 Copy'); cp.type = 'button';
    cp.onclick = async () => {
      const text = `Could you record a 30-second voice sample for my videos? Open this on your phone: ${url}`;
      try {
        if (navigator.share) await navigator.share({ text });
        else { await navigator.clipboard.writeText(url); cp.textContent = '✅ Copied'; }
      } catch (e) { /* closed the share sheet */ }
    };
    lb.appendChild(inp); lb.appendChild(cp);
    c.appendChild(lb);
    const off = el('button', 'danger-link', 'Switch the link off'); off.type = 'button'; off.style.marginTop = '8px';
    off.onclick = async () => { try { await api(`/api/voices/${v.id}/link`, { method: 'DELETE' }); await load(); } catch (e) { alert(e.message); } };
    c.appendChild(el('div', 'hint', `They open it on their phone, read a short text and tap Stop, no password needed. Works until ${day(v.link.expires_at)}.`));
    c.appendChild(off);
  } else if (!v.mine && !v.sample) {
    c.appendChild(el('div', 'hint', 'Get a link they open on their phone to record the sample themselves (best), or upload a voice note they sent you: 30 seconds of them talking clearly, in a quiet room, nobody else talking.'));
  }
  if (v.sample && v.ready) {
    const d = el('details'); d.appendChild(el('summary', '', '🗣 Hear it say something'));
    const t = el('div', 'try');
    const inp = el('input'); inp.placeholder = 'Number two. Chat saw it coming five seconds before he did.';
    const go = el('button', '', 'Say it'); go.type = 'button';
    go.onclick = async () => {
      go.disabled = true; go.textContent = 'Reading…';
      try {
        const r = await fetch(`/api/voices/${v.id}/preview`, jsonOpts('POST', { text: inp.value || inp.placeholder }));
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'It couldn’t read that');
        const a = $('player'); a.src = URL.createObjectURL(await r.blob()); a.play();
      } catch (e) { alert(e.message); }
      go.disabled = false; go.textContent = 'Say it';
    };
    t.appendChild(inp); t.appendChild(go); d.appendChild(t);
    d.appendChild(el('div', 'hint', 'The first one after a restart takes up to a minute while the model loads.'));
    c.appendChild(d);
  }
  return c;
}
$('add-btn').onclick = async () => {
  const name = $('new-name').value.trim();
  $('add-msg').textContent = '';
  try {
    await api('/api/voices', jsonOpts('POST', { name, consent: $('new-consent').checked }));
    $('new-name').value = ''; $('new-consent').checked = false;
    $('add-msg').textContent = `✅ Added. Now get a recording link for ${name}, or upload a voice note.`;
    await load();
  } catch (e) { $('add-msg').textContent = '⚠️ ' + e.message; }
};
load().catch(e => { $('voices').textContent = '⚠️ ' + e.message; });
</script>
</body>
</html>
""".replace("__CSS__", _SIMPLE_PAGE_CSS)


VOICE_LINK_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Record a voice sample</title>
<style>
__CSS__
  body { padding: 24px 16px; }
  .page { max-width: 560px; }
  .prompter { font-size: 1.15rem; line-height: 1.6; background: var(--bg); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px; margin-top: 14px; }
  .big { font-size: 1.05rem; padding: 14px 22px; width: 100%; }
  .timer { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.9rem; margin-left: 8px; }
</style>
</head>
<body>
<div class="page"><div class="card" id="card">
  <div class="brand"><span class="logo">🎙</span><h1 id="hello">Record a voice sample</h1></div>
  <p class="subtitle">You've been asked to lend your voice to some YouTube videos. Read the text below out loud once (about 30 seconds). The app makes an AI copy of your voice from it, which then reads short lines in its videos, marked on YouTube as AI-made.</p>
  <label class="tick"><input type="checkbox" id="consent"> <span>I'm OK with an AI copy of my voice narrating these videos.</span></label>
  <div class="hint">Changed your mind later? Tell the person who sent you this link and they'll delete it.</div>
  <div class="prompter" id="text"></div>
  <div class="hint">A quiet room, phone about a hand's width from your mouth, your normal talking voice. Nobody else talking.</div>
  <button id="rec" class="big" type="button">🎙 Start recording</button><span id="timer" class="timer"></span>
  <div id="result" class="status" style="display:none"></div>
  <div class="hint" style="margin-top:16px">Can't record here? <a href="#" id="pick">Upload a recording instead</a>.</div>
  <input type="file" id="file" accept="audio/*,video/*,.opus,.m4a,.ogg,.mp3,.wav,.aac" style="display:none">
</div></div>
<script>
const STATE = __STATE__;
const $ = (id) => document.getElementById(id);
if (STATE.error) {
  $('card').innerHTML = '<div class="brand"><span class="logo">🎙</span><h1>Link not working</h1></div><p class="subtitle"></p>';
  $('card').querySelector('.subtitle').textContent = STATE.error;
} else {
  if (STATE.name) $('hello').textContent = `Hi ${STATE.name}! Record a voice sample`;
  $('text').textContent = STATE.sample_text;
  if (STATE.has_sample) show('ok', '✅ We already have a sample from you. Recording again replaces it.');
}
let rec = null, chunks = [], stream = null, started = 0, tick = null;
function show(kind, text) { const r = $('result'); r.style.display = 'block'; r.className = 'status ' + kind; r.textContent = text; }
function needConsent() { if ($('consent').checked) return false; show('bad', 'Tick the box first to say it\\'s OK.'); return true; }
async function send(blob, type, name) {
  show('', '👂 Checking your recording… (about 20 seconds)');
  $('rec').disabled = true;
  try {
    const r = await fetch(location.pathname, { method: 'POST', body: blob,
      headers: { 'Content-Type': (type || 'application/octet-stream').split(';')[0], 'X-Consent': 'yes', 'X-Filename': name || '' } });
    const out = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(out.detail || 'That didn\\'t work, try again.');
    show('ok', `✅ Got it, ${Math.round(out.duration)} seconds. Thank you! You can close this page.`);
  } catch (e) { show('bad', '⚠️ ' + e.message); }
  $('rec').disabled = false;
}
if (!STATE.error) $('rec').onclick = async () => {
  if (rec) { clearInterval(tick); rec.stop(); stream.getTracks().forEach(t => t.stop()); return; }
  if (needConsent()) return;
  if (!navigator.mediaDevices || !window.MediaRecorder) { show('bad', 'This browser can\\'t record. Use the upload link below.'); return; }
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
  catch (e) { show('bad', 'Microphone access was blocked. Allow it and try again.'); return; }
  const opts = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
  const mime = opts.find(t => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || '';
  rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined); chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = () => { const type = rec.mimeType || 'audio/webm'; const b = new Blob(chunks, { type }); rec = null;
    $('rec').className = 'big'; $('rec').textContent = '🎙 Record again'; $('timer').textContent = ''; send(b, type, ''); };
  rec.start(); started = Date.now();
  $('rec').className = 'big recording'; $('rec').textContent = '⏹ Stop';
  $('result').style.display = 'none';
  tick = setInterval(() => { const s = Math.round((Date.now() - started) / 1000); $('timer').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }, 500);
};
if (!STATE.error) $('pick').onclick = (e) => { e.preventDefault(); if (!needConsent()) $('file').click(); };
if (!STATE.error) $('file').onchange = () => { const f = $('file').files[0]; if (f) send(f, f.type, f.name); };
</script>
</body>
</html>
""".replace("__CSS__", _SIMPLE_PAGE_CSS)


_LONGFORM_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper — __PAGE_TITLE__</title>
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
  .page { max-width: 760px; margin: 0 auto; }
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
  h3 { font-size: 1.02rem; margin: 0 0 4px; display: flex; align-items: center; gap: 8px; }
  h3 .n { display: inline-flex; width: 24px; height: 24px; border-radius: 50%; align-items: center; justify-content: center;
    font-size: 0.78rem; background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; }
  button.linkish { background: none; border: none; color: var(--accent); padding: 0; margin: 0; font-weight: 700; font-size: 0.88rem; }
  button.danger-link { background: none; border: 1px solid var(--border); color: var(--danger); margin-top: 0; font-size: 0.85rem; padding: 8px 14px; }
  .sched-head { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
  .sched-head input { width: auto; margin-top: 0; }
  .slot { display: grid; grid-template-columns: 110px 1fr auto; gap: 10px; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border); }
  .slot .date { font-weight: 800; font-size: 0.88rem; }
  .slot .date span { display: block; font-weight: 500; color: var(--muted); font-size: 0.74rem; }
  .slot input { margin-top: 0; }
  .slot button, .slot a { margin-top: 0; padding: 8px 12px; font-size: 0.8rem; white-space: nowrap; }
  .slot a { color: var(--accent); font-weight: 700; text-decoration: none; }
  .project { display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 12px 14px; margin-top: 10px; border: 1px solid var(--border);
    border-radius: 12px; background: var(--bg); cursor: pointer; width: 100%; text-align: left; color: var(--text); font-weight: 400; }
  .project:hover { border-color: var(--accent); opacity: 1; }
  .project .t { font-weight: 700; font-size: 0.95rem; }
  .project .m { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
  .streamer { display: flex; gap: 14px; align-items: center; margin-top: 12px; }
  .streamer img { width: 64px; height: 64px; border-radius: 50%; background: var(--bg); }
  .streamer .t { font-weight: 800; font-size: 1.15rem; }
  .streamer .m { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
  .steps { display: flex; gap: 6px; margin: 14px 0 4px; flex-wrap: wrap; }
  .steps span { font-size: 0.74rem; font-weight: 700; padding: 5px 10px; border-radius: 999px; background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
  .steps span.done { color: var(--accent); border-color: var(--accent); }
  .steps span.on { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; border-color: transparent; }
  .status-box { margin-top: 16px; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border); background: var(--bg); font-size: 0.9rem; }
  .status-box.err { border-color: var(--danger); color: var(--danger); }
  .facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px; margin-top: 10px; }
  .fact { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font-size: 0.84rem; }
  .fact b { display: block; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 3px; }
  .fact a { color: var(--accent); }
  .meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted); margin: 8px 0 4px; }
  .meta b { color: var(--text); }
  details { margin-top: 12px; }
  summary { cursor: pointer; font-weight: 700; font-size: 0.9rem; color: var(--accent); }
  .scene { display: grid; grid-template-columns: 170px 1fr; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--border); }
  .scene img { width: 170px; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 8px; background: #0b0c10; display: block; }
  .scene .kind { font-size: 0.74rem; font-weight: 800; margin-bottom: 4px; display: flex; gap: 6px; align-items: center; }
  .scene .kind .num { color: var(--accent); }
  .scene .kind .tools { margin-left: auto; display: flex; gap: 4px; }
  .scene .kind .tools button { margin: 0; padding: 3px 8px; font-size: 0.74rem; background: transparent; color: var(--muted); border: 1px solid var(--border); }
  .scene textarea, .scene input, .scene select { margin-top: 5px; padding: 7px 9px; font-size: 0.85rem; }
  .scene textarea { min-height: 80px; resize: vertical; line-height: 1.45; }
  .scene .row { display: flex; gap: 6px; align-items: center; }
  .scene .row input { width: 90px; }
  .scene .excerpt { font-size: 0.78rem; color: var(--muted); margin-top: 5px; font-style: italic; }
  .cues { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .lessons { margin: 10px 0 0; padding: 0; list-style: none; }
  .lessons li { display: flex; gap: 8px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.88rem; }
  .lessons li .src { display: block; font-size: 0.74rem; color: var(--muted); margin-top: 2px; }
  .lessons li button { margin: 0 0 0 auto; padding: 2px 8px; font-size: 0.74rem; background: transparent; color: var(--muted); border: 1px solid var(--border); }
  .cue { display: inline-flex; align-items: center; gap: 6px; padding: 4px 6px 4px 8px; border-radius: 9px; font-size: 0.76rem;
    background: color-mix(in srgb, var(--accent) 10%, var(--bg)); border: 1px solid var(--border); max-width: 100%; }
  .cue .at { font-weight: 700; }
  .cue .what { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 260px; min-width: 0; }
  .scene > div { min-width: 0; }
  .cue img, .cue video { width: 34px; height: 22px; object-fit: cover; border-radius: 4px; background: #0b0c10; }
  .cue img.emo { width: 22px; height: 22px; object-fit: contain; background: none; }
  .cue button { margin: 0; padding: 0 5px; font-size: 0.72rem; background: transparent; color: var(--muted); border: none; }
  .visuals-bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 10px; font-size: 0.85rem; }
  .visuals-bar button { margin-top: 0; }
  .scene.narrate { }
  .scene.moment .kind { color: #0891b2; }
  .scene.title .kind { color: #b45309; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .actions button { margin-top: 12px; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
  .chips button { margin-top: 0; padding: 6px 0; width: 36px; font-size: 0.8rem; background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
  .chips button.ready { background: color-mix(in srgb, #059669 18%, var(--bg)); color: #059669; border-color: #059669; }
  .chips button.flag { background: color-mix(in srgb, #f59e0b 18%, var(--bg)); color: #b45309; border-color: #f59e0b; }
  .chips button.cur { outline: 3px solid var(--accent); outline-offset: 1px; }
  .chips button.ai { border-style: dashed; }
  #voice-text { font-size: 1.1rem; }
  .prompter { background: #0b0c10; color: #f3f4f6; border-radius: 14px; padding: 22px; margin-top: 12px; font-size: 1.35rem; line-height: 1.65; }
  .prompter .miss { background: rgba(239, 68, 68, 0.4); color: #fff; border-radius: 5px; padding: 0 3px; }
  .prompter .label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 700; }
  .rec-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .rec-row button { white-space: nowrap; }
  #rec-btn { font-size: 1.05rem; padding: 14px 22px; }
  #rec-btn.recording { background: var(--danger); }
  .rec-timer { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--danger); margin-top: 18px; }
  .take-result { margin-top: 12px; padding: 12px 14px; border-radius: 12px; font-size: 0.92rem; font-weight: 600; }
  .take-result.ok { background: color-mix(in srgb, #059669 14%, var(--bg)); color: #047857; }
  .take-result.bad { background: color-mix(in srgb, #f59e0b 16%, var(--bg)); color: #92400e; }
  .take-result.wait { background: var(--bg); color: var(--muted); }
  .clips-box { margin-top: 14px; }
  .clip-row { display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-top: 1px solid var(--border); }
  .clip-row img { width: 128px; height: 72px; object-fit: cover; border-radius: 8px; flex: none; background: var(--bg); }
  .clip-row .ct { font-weight: 600; font-size: 0.92rem; }
  .clip-row .cm, .clip-row .cw { font-size: 0.82rem; color: var(--muted); margin-top: 2px; }
  .clip-row .use { display: flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 0.85rem;
    text-transform: none; letter-spacing: normal; font-weight: 500; color: var(--text); }
  .clip-row a.ct { color: var(--text); text-decoration: none; }
  .clip-row a.ct:hover { text-decoration: underline; }
  .clip-row .use input { width: auto; margin: 0; }
  .clip-row button { margin-top: 6px; padding: 5px 12px; font-size: 0.8rem; }
  .clip-row video { width: 100%; max-width: 420px; margin-top: 8px; border-radius: 8px; }
  @media (max-width: 520px) { .clip-row img { width: 96px; height: 54px; } }
  .ai-all { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }
  .ai-voice-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 10px; font-size: 0.85rem; color: var(--muted); }
  .ai-voice-row select { width: auto; margin-top: 0; padding: 6px 10px; font-size: 0.88rem; }
  .ai-voice-row a { color: var(--accent); font-weight: 600; text-decoration: none; }
  .ai-all button { margin-top: 0; }
  #ai-all-progress .bar { margin-top: 10px; }
  #ai-all-bar { height: 100%; width: 0; background: var(--accent); transition: width 0.4s; }
  .take-result .heard { display: block; font-weight: 400; font-size: 0.8rem; margin-top: 6px; color: var(--muted); }
  .music-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .music-row input { flex: 1; min-width: 200px; }
  .music-row button { margin-top: 6px; }
  .mix { margin-top: 14px; }
  .mix-row { display: flex; gap: 10px; align-items: center; margin-top: 6px; }
  .mix-row .mix-label { width: 110px; flex: none; font-size: 0.9rem; white-space: nowrap; }
  .mix-row input[type=range] { flex: 1; min-width: 0; margin: 0; accent-color: var(--accent); }
  .mix-row .mix-val { width: 56px; flex: none; text-align: right; font-size: 0.85rem; color: var(--muted); font-variant-numeric: tabular-nums; }
  .mix button { margin-top: 10px; padding: 7px 14px; font-size: 0.85rem; }
  .bar { height: 10px; border-radius: 99px; background: var(--track); margin-top: 14px; overflow: hidden; }
  .bar div { height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.4s; }
  #final-video { width: 100%; border-radius: 12px; margin-top: 12px; background: #000; }
  .title-opt { display: flex; gap: 8px; align-items: center; margin-top: 8px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg); font-size: 0.9rem; cursor: pointer; }
  .title-opt:hover { border-color: var(--accent); }
  a.dl-link { display: inline-flex; margin-top: 12px; padding: 10px 16px; border-radius: 10px; border: 1px solid var(--border);
    color: var(--text); text-decoration: none; font-weight: 600; font-size: 0.9rem; }
  .clip-player { width: 100%; border-radius: 8px; margin-top: 6px; background: #000; }
  .thumbs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
  .thumbs figure { margin: 0; cursor: pointer; border-radius: 10px; padding: 3px; border: 3px solid transparent; }
  .thumbs figure.on { border-color: var(--accent); }
  .thumbs img { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 7px; display: block; background: #0b0c10; }
  .thumbs figcaption { font-size: 0.74rem; color: var(--muted); margin-top: 4px; text-align: center; }
  @media (max-width: 600px) { .thumbs { grid-template-columns: minmax(0, 1fr); } }
  @media (max-width: 600px) {
    .card { padding: 20px 16px 24px; }
    .prompter { font-size: 1.2rem; padding: 18px; }
    .scene { grid-template-columns: minmax(0, 1fr); }
    .cue { flex-wrap: wrap; }
    .cue .what { white-space: normal; max-width: 100%; }
    .scene img { width: 100%; }
    .slot { grid-template-columns: 90px 1fr; }
    .slot > :last-child { grid-column: 2; justify-self: start; }
  }
  body[data-series="explainer"] .doc-only { display: none !important; }
  body:not([data-series="explainer"]) .exp-only { display: none !important; }
  .topics { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .topics button { font-size: 0.85rem; padding: 8px 12px; text-align: left; }
  .diagrams { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
  .diagram { display: flex; align-items: center; gap: 8px; min-width: 0; padding: 6px 10px; border: 1px solid var(--border); border-radius: 10px; font-size: 0.82rem; }
  .diagram .what { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .diagram .at { color: var(--muted); white-space: nowrap; }
  .diagram button { margin: 0; padding: 2px 8px; }
  .quiz-item { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 8px; font-size: 0.88rem; }
  .quiz-item b { color: var(--accent); }
</style>
</head>
<body data-series="__SERIES_KIND__">
<div class="page">
<a class="back" href="/">← Back to clips</a>
<div class="card">
  <div class="brand"><span class="logo">__LOGO__</span><h1>__H1__</h1></div>
  <p class="subtitle">__SUBTITLE__</p>

  <div id="list-view">
    <div class="section exp-only" style="margin-top:0;padding-top:0;border-top:none">
      <h3>📺 The channel</h3>
      <div id="exp-channel"></div>
      <h3 style="margin-top:18px">💡 Topic ideas</h3>
      <div class="hint">From the niche research: tech questions gamers search for that no big channel owns yet. Tap one to fill it in below.</div>
      <div class="topics" id="exp-topics"></div>
    </div>
    <div class="section doc-only" style="margin-top:0;padding-top:0;border-top:none">
      <h3>📅 Bi-weekly schedule</h3>
      <div class="hint">One episode every two weeks. Put a streamer on each date, then start the episode when you're ready to make it. Mix big names (search traffic) with the streamers your Shorts already cover (your audience).</div>
      <div class="sched-head">
        <label style="margin:0">First episode date</label>
        <input id="series-start" type="date">
      </div>
      <div id="slots"></div>
      <datalist id="streamer-suggestions"></datalist>
      <button id="save-series" type="button" class="secondary">💾 Save schedule</button>
    </div>

    <div class="section" id="projects-wrap" style="display:none">
      <h3>Your episodes</h3>
      <div id="projects"></div>
    </div>

    <div class="section">
      <h3>Start an episode now</h3>
      <label class="doc-only">Streamer's Twitch login</label>
      <input id="new-login" class="doc-only" placeholder="e.g. stableronaldo" list="streamer-suggestions">
      <label class="exp-only">Topic</label>
      <input id="new-topic" class="exp-only" maxlength="160" placeholder="e.g. How kernel anti-cheat works">
      <label>Your notes (optional)</label>
      <textarea id="new-notes" rows="3" placeholder="Big moments, rivalries, how they blew up, running jokes... Claude uses this alongside the research."></textarea>
      <label>Article links (optional, one per line)</label>
      <textarea id="new-links" rows="2" placeholder="News articles, interviews, a fan wiki page..."></textarea>
      <button id="start-new" type="button">🔎 Research and start</button>
      <div class="hint" id="new-status"></div>
    </div>

    <div class="section doc-only" id="lessons-section">
      <h3>🧠 What the series has learned</h3>
      <div class="hint">Claude follows these every time it writes a story, picks the clip moments and plans the visuals. Remove any that don’t fit (✕).</div>
      <ul class="lessons" id="lessons"></ul>
      <details>
        <summary>Teach it something new</summary>
        <div class="hint">Paste YouTube’s review of an episode (YouTube Studio → the video → Editing feedback), comments you agree with, or your own notes. Claude turns it into lessons and merges them with the ones above.</div>
        <input id="learn-source" placeholder="Where it's from, e.g. YouTube review of The Story of Jynxzi">
        <textarea id="learn-text" rows="5" placeholder="Paste the feedback here..."></textarea>
        <button id="learn-btn" type="button" class="secondary">🧠 Learn from this</button>
        <div class="hint" id="learn-status"></div>
      </details>
    </div>

    <div class="section" id="voice-section">
      <h3>🗣 Your AI voice</h3>
      <div class="hint">A copy of your voice for the odd line you keep fluffing, or one you add after recording. Your real voice is what keeps people watching, so use it to patch lines, not to read whole episodes. It’s free and runs on your own server. A friend’s voice can be added on the <a href="/voices">Voices page</a>.</div>
      <div id="voice-status" class="status-box" style="display:none"></div>
      <details id="voice-record">
        <summary id="voice-summary">Record your voice sample</summary>
        <div class="hint">Read this once in your normal narrating voice, in a quiet room (about 30 seconds). Your AI voice copies the recording, background noise included, so a clean take matters more than a long one.</div>
        <div class="prompter" id="voice-text"></div>
        <div class="rec-row">
          <button id="voice-rec-btn" type="button">🎙 Record sample</button>
          <span id="voice-timer" class="rec-timer"></span>
        </div>
        <div id="voice-result" class="take-result" style="display:none"></div>
        <div class="actions">
          <button id="voice-play" type="button" class="secondary" style="display:none">▶ Play my sample</button>
          <button id="voice-delete" type="button" class="danger-link" style="display:none;margin-top:18px">Delete sample</button>
        </div>
        <audio id="voice-audio" style="display:none"></audio>
      </details>
    </div>
  </div>

  <div id="project-view" style="display:none">
    <button id="to-list" type="button" class="linkish">← All episodes</button>
    <div class="streamer" id="streamer-card"></div>
    <div class="steps" id="steps"></div>
    <div id="p-status" class="status-box" style="display:none"></div>
    <div id="legacy" class="status-box" style="display:none">This is the old aviation test video. That format was dropped, so it can't be edited any more. Delete it below.</div>

    <div class="section" id="research-section">
      <h3><span class="n">1</span>Research</h3>
      <div id="research-summary"></div>
      <details id="research-edit">
        <summary>Add notes or links and research again</summary>
        <label>Your notes</label>
        <textarea id="p-notes" rows="3"></textarea>
        <label>Article links (one per line)</label>
        <textarea id="p-links" rows="2"></textarea>
        <button id="rerun-research" type="button" class="secondary">🔎 Run research again</button>
      </details>
      <div class="exp-only clips-box" id="clips-box">
        <label>🎬 Streamer clips</label>
        <div class="hint">Real moments of what this episode explains, from Twitch and YouTube. Only clips that clearly show it get ticked. Ticked clips go to Claude when it writes the script: it plays 1–4 of them, and the streamer or channel is credited on screen and in the description. Change the ticks, then write the script (again).</div>
        <div class="hint" id="clips-status"></div>
        <div id="clips-list"></div>
        <details id="clips-more">
          <summary>Paste clip links (Twitch or YouTube) or search again</summary>
          <textarea id="clip-links" rows="2" placeholder="Twitch clip or YouTube links, one per line"></textarea>
          <button id="clip-search-btn" type="button" class="secondary">🔎 Find clips</button>
        </details>
      </div>
      <button id="write-story" type="button">✍️ Write the story</button>
    </div>

    <div class="section" id="story-section" style="display:none">
      <h3><span class="n">2</span><span class="doc-only">Story</span><span class="exp-only">Script</span></h3>
      <div class="hint doc-only">🎙 Narrated scenes are what you read. 🎬 Moments are clips that play with their own sound. 📖 Chapter cards become YouTube chapters. Edit anything, then save.</div>
      <div class="hint exp-only">🎙 Narrated scenes are what you read; under each are the animated diagrams that play while you say it (each appears on the words in quotes). 🎮 Game simulations show the idea as a slowed-down top-down match. ❓ Quiz diagrams are the “pause and guess” moments. 📖 Chapter cards become YouTube chapters. Edit the words or remove a diagram, then save.</div>
      <div class="meta" id="story-meta"></div>
      <div class="visuals-bar doc-only">
        <span id="visuals-status"></span>
        <button id="plan-visuals" type="button" class="secondary">🎨 Plan visuals again</button>
      </div>
      <div class="hint doc-only" id="visuals-hint">The picture changes on key words: big text, 3D emoji, stat cards, posts and headlines you pasted, timelines, and free stock footage. Remove any you don’t want (✕) and save.</div>
      <div class="hint doc-only" id="stock-hint"></div>
      <details id="story-details">
        <summary id="story-summary"><span class="doc-only">Show and edit the story</span><span class="exp-only">Show and edit the script</span></summary>
        <div id="scenes"></div>
        <div class="actions">
          <button type="button" class="secondary add-scene" data-kind="narrate">+ Narration</button>
          <button type="button" class="secondary add-scene" data-kind="moment" id="add-moment">+ Moment</button>
          <button type="button" class="secondary add-scene" data-kind="title">+ Chapter</button>
        </div>
        <div class="actions">
          <button id="save-story" type="button" disabled>💾 Save changes</button>
          <button id="rewrite-story" type="button" class="secondary">🔄 Write it again</button>
        </div>
      </details>
    </div>

    <div class="section" id="record-section" style="display:none">
      <h3><span class="n">3</span>Record</h3>
      <div class="hint">Read each narrated scene, then tap Stop. The app listens back, and if you skipped or fluffed something it asks you to read that scene again.</div>
      <div class="chips" id="chips"></div>
      <div class="prompter" id="prompter"></div>
      <div class="rec-row">
        <button id="rec-btn" type="button">🎙 Record</button>
        <button id="ai-btn" type="button" class="secondary">🤖 Use my AI voice</button>
        <span id="rec-timer" class="rec-timer"></span>
      </div>
      <div id="take-result" class="take-result" style="display:none"></div>
      <div class="actions">
        <button id="play-take" type="button" class="secondary" style="display:none">▶ Play my take</button>
        <button id="keep-take" type="button" class="secondary" style="display:none">Keep anyway</button>
        <button id="prev-scene" type="button" class="secondary">← Previous</button>
        <button id="next-scene" type="button" class="secondary">Next →</button>
      </div>
      <div class="hint" id="ai-hint"></div>
      <div class="ai-voice-row" id="ai-voice-row" style="display:none">
        <span>AI voice for this episode:</span>
        <select id="ai-voice"></select>
        <a href="/voices">Voices</a>
      </div>
      <div class="ai-all" id="ai-all" style="display:none">
        <button id="ai-all-btn" type="button" class="secondary">🤖 Let my AI voice read everything</button>
        <div id="ai-all-progress" style="display:none"><div class="bar"><div id="ai-all-bar"></div></div></div>
        <div class="hint" id="ai-all-msg"></div>
      </div>
      <audio id="take-audio" style="display:none"></audio>
    </div>

    <div class="section" id="render-section" style="display:none">
      <h3><span class="n">4</span>Render &amp; post</h3>
      <label style="margin-top:4px">Background music (optional)</label>
      <div class="hint">A track from the YouTube Audio Library (free on monetised videos). It plays quietly and dips under your voice and the clips.</div>
      <div id="music-current" class="hint" style="display:none"></div>
      <div class="music-row">
        <input id="music-file" type="file" accept="audio/*">
        <button id="music-upload" type="button" class="secondary">Upload music</button>
        <button id="music-remove" type="button" class="secondary" style="display:none">Remove</button>
      </div>
      <div class="mix" id="mix">
        <label style="margin-top:0">Sound</label>
        <div class="hint">Every take of your voice (the AI voice too) is levelled to the same loudness. Move it up or down from there.</div>
        <div class="mix-row"><span class="mix-label">🎙 Your voice</span><input type="range" id="voice-db" min="-8" max="8" step="1" value="0"><span class="mix-val" id="voice-db-val">0 dB</span></div>
        <div class="mix-row" id="music-db-row" style="display:none"><span class="mix-label">🎵 Music</span><input type="range" id="music-db" min="-9" max="9" step="1" value="0"><span class="mix-val" id="music-db-val">0 dB</span></div>
        <button type="button" id="music-only" style="display:none">🔊 Update sound only (about a minute)</button>
      </div>
      <div class="hint" id="music-level-hint"></div>
      <div class="hint" id="render-hint" style="margin-top:14px"></div>
      <button id="render-btn" type="button">🎬 Render video (1080p)</button>
      <div id="render-progress" style="display:none">
        <div class="bar"><div id="render-bar"></div></div>
        <div class="hint" id="render-msg"></div>
      </div>
      <div id="render-error" class="status-box err" style="display:none"></div>
      <div id="render-out" style="display:none">
        <video id="final-video" controls preload="metadata"></video>
        <div class="actions">
          <a id="video-dl" class="dl-link" href="#">⬇ Download</a>
          <button id="publish-btn" type="button" class="secondary">✍️ Write title &amp; description</button>
        </div>
        <div class="hint" id="render-info"></div>
        <label>Thumbnail</label>
        <div class="hint doc-only">Three options made from real frames of the clips: the streamer&rsquo;s face big and cut out from a darkened background, with a short line. The one you pick is set on YouTube when you upload.</div>
        <div class="hint exp-only">Three options made from the episode&rsquo;s own diagrams, with a short line. The one you pick is set on YouTube when you upload.</div>
        <button id="thumb-make" type="button" class="secondary">🖼 Make thumbnails</button>
        <div class="thumbs" id="thumbs"></div>
        <div id="thumb-edit" style="display:none">
          <div class="music-row">
            <input id="thumb-hook" maxlength="40" placeholder="Text on the thumbnail">
            <button id="thumb-redraw" type="button" class="secondary">Redraw</button>
            <a id="thumb-dl" class="dl-link" href="#" style="margin-top:6px">⬇ Download</a>
            <button id="thumb-apply" type="button" class="secondary" style="display:none">Set on YouTube</button>
          </div>
        </div>
        <div class="hint" id="thumb-status"></div>
        <div id="publish-out" style="display:none">
          <label>Title options (tap one to use it)</label>
          <div id="title-list"></div>
          <label>Title to post</label>
          <input id="post-title" maxlength="100">
          <label>Description (with chapters and credits)</label>
          <textarea id="desc-text" rows="10"></textarea>
          <label>Post as</label>
          <select id="post-privacy">
            <option value="private">Private: check and schedule it in YouTube Studio</option>
            <option value="unlisted">Unlisted</option>
            <option value="public">Public now</option>
          </select>
          <div class="hint doc-only">Uploading also cuts 2 cliffhanger Shorts from the episode (on the Home page): each stops right before a payoff, with the link to the full video in its description.</div>
          <div class="exp-only" id="quiz-wrap">
            <label>Quizzes to add in YouTube Studio</label>
            <div class="hint">After uploading: YouTube Studio &gt; this video &gt; Video elements &gt; <i>Add a quiz</i>. Paste each question at its time (YouTube has no way to add them automatically). A saved quiz can’t be edited, only deleted and redone.</div>
            <div id="quiz-list"></div>
          </div>
          <div class="actions">
            <button id="upload-btn" type="button">⬆ Upload to Caught On Stream</button>
            <button id="promo-btn" type="button" class="secondary doc-only">✂️ Make 2 cliffhanger Shorts</button>
          </div>
          <div class="hint" id="post-status"></div>
        </div>
      </div>
    </div>

    <div class="section">
      <button id="delete-project" type="button" class="danger-link">🗑 Delete this episode</button>
    </div>
  </div>
</div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const NL = String.fromCharCode(10);
const KIND_LABEL = { narrate: '🎙 Narration', moment: '🎬 Moment', title: '📖 Chapter' };
const SERIES = __SERIES_JSON__;
const EXP = SERIES.kind === 'explainer';
let project = null;
let pollTimer = null;
let storyDirty = false;
let recIndex = 0;
let recorder = null, recStream = null, recChunks = [], recStarted = 0, recTick = null;
let recTarget = 'scene';  // 'scene' (a narrated scene) or 'sample' (the AI voice sample)
let lastResult = null;
let voice = null;  // /api/longform/voice
let voices = null;  // /api/voices: Dean's own + friends' AI voices
let aiBusy = false;

function fmtTime(sec) { sec = Math.max(0, Math.round(sec || 0)); return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`; }
function fmtDate(iso) { if (!iso) return ''; const d = new Date(iso + 'T12:00:00'); return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }); }
function needsTake(s) { return s.kind === 'narrate'; }
function sceneReady(s) { if (!needsTake(s)) return true; const t = s.take || {}; return !!t.file && !!(t.ok || t.kept); }
function sceneFlagged(s) { const t = s.take || {}; return needsTake(s) && !!t.file && !t.ok && !t.kept; }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
  return data;
}
const jsonOpts = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// ---------- list view ----------
let series = null;

async function showList() {
  stopPolling();
  project = null;
  history.replaceState(null, '', SERIES.path);
  $('project-view').style.display = 'none';
  $('list-view').style.display = 'block';
  const [{ projects }, s] = await Promise.all([api('/api/longform/projects?series=' + SERIES.kind),
                                              api(EXP ? '/api/explainers' : '/api/longform/series')]);
  if (EXP) renderExplainerHome(s);
  else { series = s; renderSeries(); loadLessons(); }
  loadVoice();
  $('projects-wrap').style.display = projects.length ? 'block' : 'none';
  const pl = $('projects');
  pl.innerHTML = '';
  projects.forEach(p => {
    const b = el('button', 'project'); b.type = 'button';
    const info = el('div');
    info.appendChild(el('div', 't', ['documentary', 'explainer'].includes(p.kind) ? p.title : `${p.title} (old aviation test)`));
    const state = p.youtube_video_id ? '✅ Posted' : p.rendered ? '🎬 Rendered'
      : p.status === 'researching' ? '🔎 Researching…' : p.status === 'writing' ? '✍️ Writing…'
      : p.status === 'script_ready' ? `🎙 ${p.recorded}/${p.scenes} scenes recorded`
      : p.status === 'research_ready' ? '✍️ Ready to write' : p.status === 'error' ? '⚠️ Needs attention' : '';
    info.appendChild(el('div', 'm', `${state}${p.minutes ? ` · ~${p.minutes} min` : ''}`));
    b.appendChild(info);
    b.appendChild(el('span', 'm', '→'));
    b.addEventListener('click', () => openProject(p.id));
    pl.appendChild(b);
  });
}

function renderSeries() {
  $('series-start').value = series.start;
  const dl = $('streamer-suggestions');
  dl.innerHTML = '';
  (series.suggested || []).forEach(l => { const o = document.createElement('option'); o.value = l; dl.appendChild(o); });
  const wrap = $('slots');
  wrap.innerHTML = '';
  series.slots.forEach((slot, i) => {
    const row = el('div', 'slot');
    const date = el('div', 'date', fmtDate(slot.date));
    date.appendChild(el('span', '', i === 0 ? 'next up' : `+${i * series.every_days} days`));
    const input = el('input'); input.value = slot.login; input.placeholder = 'streamer login'; input.dataset.date = slot.date;
    input.setAttribute('list', 'streamer-suggestions');
    row.appendChild(date); row.appendChild(input);
    if (slot.project) {
      const a = el('a', '', slot.project.youtube_video_id ? '✅ Posted →' : 'Open episode →'); a.href = '#';
      a.addEventListener('click', (e) => { e.preventDefault(); openProject(slot.project.id); });
      row.appendChild(a);
    } else {
      const b = el('button', 'secondary', 'Start'); b.type = 'button';
      b.addEventListener('click', async () => {
        if (!input.value.trim()) { alert('Put a streamer on this date first.'); return; }
        b.disabled = true;
        try {
          const { id } = await api('/api/longform/projects', jsonOpts('POST', { login: input.value, slot: slot.date }));
          openProject(id);
        } catch (e) { alert(e.message); b.disabled = false; }
      });
      row.appendChild(b);
    }
    wrap.appendChild(row);
  });
}

$('save-series').addEventListener('click', async () => {
  const plan = {};
  document.querySelectorAll('#slots input').forEach(i => { plan[i.dataset.date] = i.value; });
  try { series = await api('/api/longform/series', jsonOpts('PUT', { start: $('series-start').value, plan })); renderSeries(); }
  catch (e) { alert(e.message); }
});

// Caught On Code: the channel's YouTube account and the topic ideas
function renderExplainerHome(meta) {
  const box = $('exp-channel');
  box.innerHTML = '';
  const line = el('div', 'hint');
  if (!meta.oauth_configured) line.textContent = 'YouTube uploads need the Google OAuth keys set on Railway (the same ones the main channel uses).';
  else if (meta.youtube_connected) line.textContent = '✅ ' + meta.channel + '’s YouTube account is connected: finished episodes upload there.';
  else line.textContent = 'Connect the ' + meta.channel + ' YouTube account once, so finished episodes upload there (not to Caught On Stream).';
  box.appendChild(line);
  if (meta.oauth_configured) {
    if (meta.youtube_connected) {
      const b = el('button', 'secondary', 'Disconnect'); b.type = 'button';
      b.addEventListener('click', async () => {
        if (!confirm('Disconnect the ' + meta.channel + ' YouTube account?')) return;
        await api('/api/youtube/disconnect?profile=code', { method: 'POST' }); showList();
      });
      box.appendChild(b);
    } else {
      const a = el('a', 'dl-link', '🔗 Connect ' + meta.channel + ' on YouTube'); a.href = '/auth/youtube/login?profile=code';
      box.appendChild(a);
    }
  }
  const tw = $('exp-topics');
  tw.innerHTML = '';
  (meta.topics || []).forEach(topic => {
    const b = el('button', 'secondary', topic); b.type = 'button';
    b.addEventListener('click', () => { $('new-topic').value = topic; $('new-topic').focus(); });
    tw.appendChild(b);
  });
  $('new-notes').placeholder = 'What you already know, angles you want, games to use as examples... Claude uses this alongside the research.';
  $('new-links').placeholder = 'Articles, docs, dev blogs, a Wikipedia page...';
}

$('start-new').addEventListener('click', async () => {
  const login = $('new-login').value.trim();
  const topic = $('new-topic').value.trim();
  if (EXP ? !topic : !login) { $('new-status').textContent = EXP ? 'Type the topic first.' : 'Enter the streamer’s Twitch login first.'; return; }
  $('start-new').disabled = true;
  try {
    const links = $('new-links').value.split(NL).map(s => s.trim()).filter(Boolean);
    const { id } = EXP
      ? await api('/api/explainers', jsonOpts('POST', { topic, notes: $('new-notes').value, links }))
      : await api('/api/longform/projects', jsonOpts('POST', { login, notes: $('new-notes').value, links }));
    $('new-status').textContent = '';
    openProject(id);
  } catch (e) { $('new-status').textContent = e.message; }
  finally { $('start-new').disabled = false; }
});

// ---------- project view ----------
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function openProject(id) {
  history.replaceState(null, '', `${SERIES.path}?v=${encodeURIComponent(id)}`);
  $('list-view').style.display = 'none';
  $('project-view').style.display = 'block';
  storyDirty = false;
  lastResult = null;
  if (!voice) loadVoice();
  await loadProject(id, true);
  if (project) $('story-details').open = !(project.scenes || []).some(s => s.take);
}

async function loadProject(id, resetIndex) {
  try { project = await api(`/api/longform/projects/${encodeURIComponent(id)}`); }
  catch (e) { alert(e.message); showList(); return; }
  if (resetIndex) {
    const todo = (project.scenes || []).findIndex(s => needsTake(s) && !sceneReady(s));
    recIndex = todo === -1 ? (project.scenes || []).findIndex(needsTake) : todo;
    if (recIndex < 0) recIndex = 0;
  }
  render();
  stopPolling();
  const busy = ['researching', 'writing'].includes(project.status) || (project.render || {}).status === 'rendering'
    || (project.visuals || {}).status === 'planning' || (project.ai_all || {}).status === 'running'
    || (project.clip_search || {}).status === 'searching';
  if (busy) pollTimer = setInterval(() => loadProject(project.id, project.status !== 'script_ready'), 3000);
}

function render() {
  const p = project;
  const legacy = !['documentary', 'explainer'].includes(p.kind);
  $('legacy').style.display = legacy ? 'block' : 'none';
  ['research-section', 'story-section', 'record-section', 'render-section'].forEach(s => { if (legacy) $(s).style.display = 'none'; });
  renderStreamer();
  if (legacy) return;
  const scenes = p.scenes || [];
  const hasStory = p.status === 'script_ready' && scenes.length > 0;
  const narr = scenes.filter(needsTake);
  const allRec = hasStory && narr.every(sceneReady);
  const r = p.render || {};
  const researched = EXP ? ['research_ready', 'writing', 'script_ready'].includes(p.status) : !!p.library;
  const steps = [
    ['1 Research', researched ? 'done' : 'on'],
    [EXP ? '2 Script' : '2 Story', hasStory ? 'done' : (researched ? 'on' : '')],
    ['3 Record', allRec ? 'done' : (hasStory ? 'on' : '')],
    ['4 Render & post', p.youtube_video_id ? 'done' : (allRec ? 'on' : '')],
  ];
  $('steps').innerHTML = '';
  steps.forEach(([t, c]) => { const s = el('span', c, t + (c === 'done' ? ' ✓' : '')); $('steps').appendChild(s); });
  const st = $('p-status');
  if (['researching', 'writing'].includes(p.status)) { st.style.display = 'block'; st.className = 'status-box'; st.textContent = '⏳ ' + (p.message || 'Working…'); }
  else if (p.error) { st.style.display = 'block'; st.className = 'status-box err'; st.textContent = '⚠️ ' + p.error; }
  else st.style.display = 'none';
  $('research-section').style.display = 'block';
  $('story-section').style.display = hasStory ? 'block' : 'none';
  $('record-section').style.display = hasStory ? 'block' : 'none';
  $('render-section').style.display = hasStory ? 'block' : 'none';
  renderResearch();
  if (hasStory) {
    if (!storyDirty) renderStory();
    renderRecorder();
    renderRender();
  }
}

function renderStreamer() {
  const p = project, s = p.streamer || {};
  const c = $('streamer-card');
  c.innerHTML = '';
  if (EXP) {
    const info = el('div');
    info.appendChild(el('div', 't', p.title || p.topic || 'Untitled'));
    info.appendChild(el('div', 'm', SERIES.channel + ' explainer' + (p.topic && p.topic !== p.title ? ' · ' + p.topic : '')));
    c.appendChild(info);
    return;
  }
  if (s.profile_image_url) { const img = el('img'); img.src = s.profile_image_url; img.alt = ''; c.appendChild(img); }
  const info = el('div');
  info.appendChild(el('div', 't', p.title || 'Untitled'));
  const bits = [];
  if (s.login) bits.push(`twitch.tv/${s.login}`);
  if (s.created_at) bits.push(`on Twitch since ${fmtDate(s.created_at.slice(0, 10))}`);
  if (s.current_game) bits.push(`streams ${s.current_game}`);
  info.appendChild(el('div', 'm', bits.join(' · ')));
  c.appendChild(info);
}

function renderResearch() {
  const p = project;
  const box = $('research-summary');
  box.innerHTML = '';
  const lib = p.library || [];
  const src = p.sources || {};
  if (EXP) {
    if (p.status === 'researching') box.appendChild(el('div', 'hint', 'Reading Wikipedia and your links (a minute or so).'));
    const pages = src.pages || [];
    if (pages.length || (src.failed_links || []).length) {
      const facts = el('div', 'facts');
      pages.forEach(pg => { const f = el('div', 'fact'); f.appendChild(el('b', '', 'Source')); const a = el('a', '', pg.title); a.href = pg.url; a.target = '_blank'; a.rel = 'noopener'; f.appendChild(a); facts.appendChild(f); });
      if ((src.failed_links || []).length) { const f = el('div', 'fact'); f.appendChild(el('b', '', 'Couldn’t open')); f.appendChild(document.createTextNode(src.failed_links.join(', '))); facts.appendChild(f); }
      box.appendChild(facts);
    }
  }
  if (!EXP && p.status === 'researching' && !lib.length) { box.appendChild(el('div', 'hint', 'Gathering their Twitch history, top clips, Wikipedia and your links. Downloading and transcribing the clips takes a few minutes.')); }
  if (!EXP && (lib.length || src.wikipedia !== undefined)) {
    const facts = el('div', 'facts');
    const add = (label, text, href) => { const f = el('div', 'fact'); f.appendChild(el('b', '', label)); if (href) { const a = el('a', '', text); a.href = href; a.target = '_blank'; a.rel = 'noopener'; f.appendChild(a); } else f.appendChild(document.createTextNode(text)); facts.appendChild(f); };
    if (lib.length) {
      const years = lib.map(c => c.date.slice(0, 4)).sort();
      add('Clips', `${lib.length} downloaded (${years[0]}–${years[years.length - 1]})`);
      const top = [...lib].sort((a, b) => b.views - a.views)[0];
      add('Most-viewed clip', `${top.title} · ${top.views.toLocaleString()} views`, top.url);
    }
    add('Wikipedia', src.wikipedia ? 'Article found' : 'No article', src.wikipedia || null);
    add('Your links', `${(src.articles || []).length} read` + ((src.failed_links || []).length ? `, ${src.failed_links.length} couldn’t be opened` : ''));
    box.appendChild(facts);
  }
  if (EXP) renderClips();
  $('p-notes').value = p.notes || '';
  $('p-links').value = (p.links || []).join(NL);
  const busy = ['researching', 'writing'].includes(p.status);
  $('rerun-research').disabled = busy;
  const wb = $('write-story');
  wb.style.display = (EXP ? p.status === 'research_ready' : p.library && p.status !== 'script_ready') ? 'inline-block' : 'none';
  wb.textContent = EXP ? '✍️ Write the script' : '✍️ Write the story';
  wb.disabled = busy;
}

function renderClips() {
  const p = project, cs = p.clip_search || {}, lib = p.library || [];
  const games = (cs.games || []).join(', ');
  const where = [games ? `Twitch (${games})` : '', cs.youtube ? 'YouTube' : ''].filter(Boolean).join(' and ');
  const unres = cs.unresolved ? ` ${cs.unresolved} link${cs.unresolved === 1 ? '' : 's'} couldn’t be used (not a Twitch clip or YouTube video, or over 15 minutes long).` : '';
  $('clips-status').textContent = cs.status === 'searching'
    ? '⏳ ' + (cs.message || 'Searching…') + ' You can write the script now, or wait for the clips (a few minutes).'
    : cs.status === 'error' ? '⚠️ ' + (cs.error || 'The clip search failed.')
    : cs.status === 'done' ? (cs.added ? `Found ${cs.added} clip${cs.added === 1 ? '' : 's'}` + (where ? ` on ${where} (looked at ${cs.scanned}).` : '.')
      : where ? `Nothing on ${where} clearly shows it (looked at ${cs.scanned}). Paste a link below if you know one.`
      : 'This topic doesn’t really happen on stream, so there was nothing to search. Paste a link below if you know a clip.') + unres
    : 'Not searched yet.';
  const list = $('clips-list');
  list.innerHTML = '';
  lib.forEach(c => {
    const row = el('div', 'clip-row');
    if (c.thumbnail) { const im = el('img'); im.loading = 'lazy'; im.alt = ''; im.src = c.thumbnail; row.appendChild(im); }
    const body = el('div');
    const t = el('a', 'ct', c.title || c.id); t.href = c.url; t.target = '_blank'; t.rel = 'noopener'; body.appendChild(t);
    body.appendChild(el('div', 'cm', [c.source === 'youtube' ? 'YouTube' : 'Twitch', c.streamer, c.game, (c.views || 0).toLocaleString() + ' views', Math.round(c.duration || 0) + 's'].filter(Boolean).join(' · ')));
    if (c.what) body.appendChild(el('div', 'cw', (c.pasted ? '📎 Your link · ' : c.fits === false ? '✗ Claude: doesn’t seem to show it · ' : c.fits ? '✓ Claude: shows it · ' : '') + c.what));
    const use = el('label', 'use');
    const cb = el('input'); cb.type = 'checkbox'; cb.checked = !!c.use;
    cb.addEventListener('change', async () => {
      try { project = await api(`/api/longform/projects/${p.id}/library/${c.id}`, jsonOpts('PUT', { use: cb.checked })); renderClips(); }
      catch (e) { alert(e.message); cb.checked = !cb.checked; }
    });
    use.appendChild(cb); use.appendChild(document.createTextNode('Use in the script'));
    body.appendChild(use);
    const play = el('button', 'secondary', '▶ Play'); play.type = 'button';
    play.addEventListener('click', () => {
      let v = body.querySelector('video');
      if (!v) { v = el('video'); v.controls = true; body.appendChild(v); }
      v.src = `/api/longform/projects/${p.id}/clips/${c.id}`; v.play();
    });
    body.appendChild(play);
    const rm = el('button', 'secondary', '🗑 Remove'); rm.type = 'button'; rm.style.marginLeft = '6px';
    rm.addEventListener('click', async () => {
      if (!confirm('Remove this clip from the episode?')) return;
      try { project = await api(`/api/longform/projects/${p.id}/library/${c.id}`, { method: 'DELETE' }); renderClips(); }
      catch (e) { alert(e.message); }
    });
    body.appendChild(rm);
    row.appendChild(body);
    list.appendChild(row);
  });
  $('clip-search-btn').disabled = cs.status === 'searching';
}
$('clip-search-btn').addEventListener('click', async () => {
  try {
    project = await api(`/api/longform/projects/${project.id}/clip-search`, jsonOpts('POST', { links: $('clip-links').value.split(NL).map(s => s.trim()).filter(Boolean) }));
    $('clip-links').value = '';
    loadProject(project.id, false);
  } catch (e) { alert(e.message); }
});

$('rerun-research').addEventListener('click', async () => {
  if ((project.scenes || []).length && !confirm('Research again? You can then rewrite the story with the new research.')) return;
  try {
    await api(`/api/longform/projects/${project.id}/research`, jsonOpts('POST', { notes: $('p-notes').value, links: $('p-links').value.split(NL).map(s => s.trim()).filter(Boolean) }));
    loadProject(project.id, true);
  } catch (e) { alert(e.message); }
});
$('write-story').addEventListener('click', async () => {
  try { await api(`/api/longform/projects/${project.id}/write`, { method: 'POST' }); loadProject(project.id, true); }
  catch (e) { alert(e.message); }
});

// ---------- story editor ----------
function clipLabel(c) { return `${c.id} · ${c.date} · ${c.title}`.slice(0, 80); }
const wordsCache = {};
async function clipWords(id) {
  if (!wordsCache[id]) wordsCache[id] = api(`/api/longform/projects/${project.id}/clips/${id}/words`).then(r => r.words).catch(() => []);
  return wordsCache[id];
}

function markDirty() { storyDirty = true; $('save-story').disabled = false; }

function renderStory() {
  const scenes = project.scenes || [];
  const narr = scenes.filter(needsTake);
  const words = narr.reduce((n, s) => n + s.narration.split(' ').length, 0);
  const momentSecs = scenes.filter(s => s.kind === 'moment').reduce((n, s) => n + (s.end - s.start), 0);
  const chapters = scenes.filter(s => s.kind === 'title').length;
  $('story-meta').innerHTML = '';
  const quizzes = scenes.reduce((n, s) => n + (s.visuals || []).filter(v => v.type === 'quiz').length, 0);
  const diagrams = scenes.reduce((n, s) => n + (s.visuals || []).length, 0);
  [[`~${Math.round(words / 150 + momentSecs / 60 + chapters * 3 / 60)} min`, ' long'], [String(words), ' words to read'],
   EXP ? [String(diagrams), ' diagrams'] : [String(scenes.filter(s => s.kind === 'moment').length), ' clip moments'],
   EXP ? [String(quizzes), quizzes === 1 ? ' quiz' : ' quizzes'] : [String(chapters), ' chapters']]
   .concat(EXP && momentSecs ? [[String(scenes.filter(s => s.kind === 'moment').length), ' streamer clips']] : []).forEach(([b, t]) => {
    const s = el('span'); s.appendChild(el('b', '', b)); s.appendChild(document.createTextNode(t)); $('story-meta').appendChild(s);
  });
  $('add-moment').style.display = !EXP || (project.library || []).some(c => c.use) ? '' : 'none';
  const wrap = $('scenes');
  wrap.innerHTML = '';
  scenes.forEach((s, i) => wrap.appendChild(sceneRow(s, i)));
  $('save-story').disabled = true;
  renderVisualsBar();
}

let stockKeys = null;
function renderVisualsBar() {
  if (EXP) return;
  if (stockKeys === null) { stockKeys = {}; api('/api/longform/visual-sources').then(r => { stockKeys = r; renderVisualsBar(); }).catch(() => {}); }
  const sh = $('stock-hint');
  sh.innerHTML = '';
  if (stockKeys.pexels === false && stockKeys.pixabay === false) {
    sh.textContent = 'Stock video is off until you add a free Pexels key (PEXELS_API_KEY on Railway, from pexels.com/api). Photos still come from Openverse’s copyright-free collection.';
  } else if (stockKeys.pexels || stockKeys.pixabay) {
    sh.appendChild(document.createTextNode('Free stock photos and videos provided by '));
    [['Pexels', 'https://www.pexels.com'], ['Pixabay', 'https://pixabay.com'], ['Openverse', 'https://openverse.org']].forEach(([n, u], k) => {
      const a = el('a', '', n); a.href = u; a.target = '_blank'; a.rel = 'noopener'; sh.appendChild(a);
      sh.appendChild(document.createTextNode(k < 2 ? (k === 1 ? ' and ' : ', ') : '.'));
    });
  }
  const v = project.visuals || {};
  const n = (project.scenes || []).reduce((k, s) => k + (s.cues || []).length, 0);
  const planning = v.status === 'planning';
  $('visuals-status').textContent = planning ? '⏳ ' + (v.message || 'Planning the visuals…')
    : v.status === 'error' ? '⚠️ Visuals: ' + (v.error || 'failed')
    : n ? `🎨 ${n} key-word visuals planned` : '🎨 No key-word visuals yet';
  $('plan-visuals').disabled = planning;
  $('plan-visuals').textContent = n ? '🎨 Plan visuals again' : '🎨 Plan visuals';
}
$('plan-visuals').addEventListener('click', async () => {
  if (storyDirty && !confirm('Save your story changes first? Planning again replaces the visuals of every scene.')) return;
  if (!storyDirty && (project.scenes || []).some(s => (s.cues || []).length) && !confirm('Plan all the visuals again? Your current ones are replaced.')) return;
  try {
    if (storyDirty) { project = await api(`/api/longform/projects/${project.id}/scenes`, jsonOpts('PUT', { scenes: collectScenes() })); storyDirty = false; }
    await api(`/api/longform/projects/${project.id}/visuals`, { method: 'POST' });
    loadProject(project.id, false);
  } catch (e) { alert(e.message); }
});

function clipSelect(value, allowNone) {
  const sel = el('select');
  if (allowNone) { const o = el('option', '', '(no clip: title card)'); o.value = ''; sel.appendChild(o); }
  (project.library || []).forEach(c => { const o = el('option', '', clipLabel(c)); o.value = c.id; if (c.id === value) o.selected = true; sel.appendChild(o); });
  return sel;
}

function sceneRow(s, i) {
  const row = el('div', 'scene ' + s.kind);
  row.dataset.kind = s.kind;
  const img = el('img'); img.loading = 'lazy'; img.alt = '';
  img.src = `/api/longform/projects/${project.id}/scenes/${i}/preview?k=${encodeURIComponent([s.clip, s.start, s.end, s.caption, s.title].join('|') + (s.visuals ? keyOf(JSON.stringify(s.visuals)) : ''))}`;
  row.appendChild(img);
  const body = el('div');
  const kind = el('div', 'kind');
  kind.appendChild(el('span', 'num', String(i + 1)));
  kind.appendChild(el('span', '', KIND_LABEL[s.kind]));
  if (needsTake(s)) kind.appendChild(el('span', '', sceneReady(s) ? '· ✅ recorded' : sceneFlagged(s) ? '· ⚠️ retake' : ''));
  const tools = el('span', 'tools');
  [['↑', -1], ['↓', 1]].forEach(([t, d]) => { const b = el('button', '', t); b.type = 'button'; b.title = 'Move'; b.addEventListener('click', () => moveScene(i, d)); tools.appendChild(b); });
  const del = el('button', '', '✕'); del.type = 'button'; del.title = 'Remove'; del.addEventListener('click', () => removeScene(i)); tools.appendChild(del);
  kind.appendChild(tools);
  body.appendChild(kind);
  if (s.kind === 'title') {
    const t = el('input'); t.value = s.title; t.dataset.f = 'title'; t.addEventListener('input', markDirty); body.appendChild(t);
  } else {
    if (s.kind === 'narrate') {
      const ta = el('textarea'); ta.value = s.narration; ta.dataset.f = 'narration'; ta.addEventListener('input', markDirty); body.appendChild(ta);
      if (EXP) {
        const dw = el('div', 'diagrams'); dw.dataset.visuals = JSON.stringify(s.visuals || []);
        renderDiagrams(dw);
        body.appendChild(dw);
        row.appendChild(body);
        return row;
      }
      const cw = el('div', 'cues'); cw.dataset.cues = JSON.stringify(s.cues || []);
      renderCues(cw);
      body.appendChild(cw);
    }
    const sel = clipSelect(s.clip, s.kind === 'narrate'); sel.dataset.f = 'clip'; sel.addEventListener('change', markDirty); body.appendChild(sel);
    if (s.kind === 'moment') {
      const r = el('div', 'row');
      const a = el('input'); a.type = 'number'; a.step = '0.5'; a.min = '0'; a.value = s.start; a.dataset.f = 'start';
      const b = el('input'); b.type = 'number'; b.step = '0.5'; b.min = '0'; b.value = s.end; b.dataset.f = 'end';
      [a, b].forEach(x => x.addEventListener('input', markDirty));
      const play = el('button', 'secondary', '▶ Play'); play.type = 'button'; play.style.marginTop = '5px';
      r.appendChild(el('span', 'hint', 'from')); r.appendChild(a); r.appendChild(el('span', 'hint', 'to')); r.appendChild(b); r.appendChild(el('span', 'hint', 's')); r.appendChild(play);
      body.appendChild(r);
      const ex = el('div', 'excerpt', '');
      body.appendChild(ex);
      clipWords(s.clip).then(ws => { ex.textContent = '“' + ws.filter(w => w.e > s.start && w.s < s.end).map(w => w.w).join(' ') + '”'; });
      play.addEventListener('click', () => {
        let v = body.querySelector('video');
        if (!v) { v = el('video', 'clip-player'); v.controls = true; body.appendChild(v); }
        v.src = `/api/longform/projects/${project.id}/clips/${sel.value}#t=${a.value},${b.value}`;
        v.play();
      });
    }
    if (EXP) { body.appendChild(el('div', 'hint', 'Credited on screen: ' + (s.caption || 'the streamer'))); row.appendChild(body); return row; }
    const cap = el('input'); cap.placeholder = 'Corner caption, e.g. March 2023 · 1.2M views'; cap.value = s.caption || ''; cap.dataset.f = 'caption';
    cap.addEventListener('input', markDirty); body.appendChild(cap);
  }
  row.appendChild(body);
  return row;
}

const CUE_ICON = { words: '🔠', emoji: '', stat: '📊', stock: '🎞', photo: '🖼', clip: '🎬', post: '💬', headline: '📰', timeline: '📅' };
function emojiCode(e) { return [...e].map(c => c.codePointAt(0).toString(16)).filter(h => h !== 'fe0f').join('-'); }
function cueWhat(c) {
  if (c.type === 'words') return `big words “${c.text || c.at}”`;
  if (c.type === 'emoji') return c.text ? `“${c.text}”` : 'emoji';
  if (c.type === 'stat') return `${c.label || 'stat'}: ${c.value}`;
  if (c.type === 'stock' || c.type === 'photo') return `${c.type === 'stock' ? 'footage' : 'photo'}: ${c.query}` + (c.credit && c.credit.source ? ` (${c.credit.source})` : '');
  if (c.type === 'clip') return `clip ${c.clip} at ${Math.round(c.start || 0)}s`;
  if (c.type === 'post') return 'your pasted post' + (c.highlight ? `: “${c.highlight}”` : '');
  if (c.type === 'headline') return 'article headline' + (c.highlight ? `: “${c.highlight}”` : '');
  if (c.type === 'timeline') return 'timeline ' + (c.points || []).map(p => p[0]).join(' → ');
  return c.type;
}
function renderCues(cw) {
  const cues = JSON.parse(cw.dataset.cues || '[]');
  cw.innerHTML = '';
  cues.forEach((c, k) => {
    const chip = el('span', 'cue');
    if (c.type === 'emoji' && c.emoji) { const im = el('img', 'emo'); im.src = `/api/longform/emoji/${emojiCode(c.emoji)}`; im.alt = c.emoji; chip.appendChild(im); }
    else if (c.type === 'photo' && c.asset) { const im = el('img'); im.loading = 'lazy'; im.src = `/api/longform/projects/${project.id}/visuals/${c.asset}`; chip.appendChild(im); }
    else if (c.type === 'stock' && c.asset) {
      const src = `/api/longform/projects/${project.id}/visuals/${c.asset}`;
      if (c.asset.endsWith('.mp4')) { const v = el('video'); v.muted = true; v.preload = 'metadata'; v.src = src + '#t=1'; chip.appendChild(v); }
      else { const im = el('img'); im.loading = 'lazy'; im.src = src; chip.appendChild(im); }
    } else chip.appendChild(el('span', '', CUE_ICON[c.type] || '•'));
    chip.appendChild(el('span', 'at', `“${c.at}”`));
    chip.appendChild(el('span', 'what', '→ ' + cueWhat(c)));
    const x = el('button', '', '✕'); x.type = 'button'; x.title = 'Remove this visual';
    x.addEventListener('click', () => { cues.splice(k, 1); cw.dataset.cues = JSON.stringify(cues); renderCues(cw); markDirty(); });
    chip.appendChild(x);
    cw.appendChild(chip);
  });
  if (!cues.length) cw.appendChild(el('span', 'hint', 'No key-word visuals yet: clips with slow zooms and captions play under this one.'));
}

// ---------- explainer diagrams (Caught On Code) ----------
const DIAGRAM_ICON = { arena: '🎮', flow: '➡️', network: '🌐', race: '⏱', bars: '📊', bignum: '🔢', grid: '▦', layers: '🧱', neural: '🧠', compare: '⚖️', quiz: '❓', words: '🔠' };
function keyOf(s) { let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return String(h); }
function diagramWhat(v) {
  const t = v.title ? v.title + ': ' : '';
  if (v.type === 'flow') return t + (v.nodes || []).map(n => n.label).join(' → ');
  if (v.type === 'network') return t + ((v.center || {}).label || 'server') + ' with ' + (v.clients || []).map(n => n.label).join(', ') + (v.tick_rate ? ` · ${v.tick_rate} ticks/s` : '');
  if (v.type === 'race') return t + (v.events || []).map(e => `${e.label} (${e.ms} ${v.unit || 'ms'})`).join(' → ') + (v.example ? ' · example timings' : '');
  if (v.type === 'bars') return t + (v.items || []).map(i => `${i.label} ${i.value}${v.unit ? ' ' + v.unit : ''}`).join(', ');
  if (v.type === 'bignum') return `${v.value} ${v.label}`;
  if (v.type === 'grid') return `${v.count} dots` + (v.highlight ? `, ${v.highlight} lit` : '') + (v.label ? `: ${v.label}` : '');
  if (v.type === 'layers') return t + (v.items || []).map(i => i.label).join(' / ');
  if (v.type === 'neural') return 'neural network' + (v.label ? ': ' + v.label : '');
  if (v.type === 'compare') return `${(v.left || {}).title} vs ${(v.right || {}).title}`;
  if (v.type === 'quiz') return `${v.question} (answer: ${(v.options || [])[v.answer] || ''})`;
  if (v.type === 'words') return `“${v.text}”`;
  if (v.type === 'arena') return 'game simulation: ' + (v.mode === 'ticks' ? `what the server knows at ${(v.tick_rates || []).join(' vs ')} tick`
    : v.mode === 'rewind' ? `server rewinds ${v.delay_ms} ms, hit behind the wall` : `peek, ${v.delay_ms} ms head start`)
    + (v.example ? ' · example numbers' : '');
  return v.type;
}
function renderDiagrams(dw) {
  const vis = JSON.parse(dw.dataset.visuals || '[]');
  dw.innerHTML = '';
  vis.forEach((v, k) => {
    const chip = el('div', 'diagram');
    chip.appendChild(el('span', '', DIAGRAM_ICON[v.type] || '•'));
    chip.appendChild(el('span', 'what', diagramWhat(v)));
    chip.appendChild(el('span', 'at', k === 0 ? 'from the start' : `on “${v.at || ''}”`));
    const x = el('button', 'secondary', '✕'); x.type = 'button'; x.title = 'Remove this diagram';
    x.addEventListener('click', () => { vis.splice(k, 1); dw.dataset.visuals = JSON.stringify(vis); renderDiagrams(dw); markDirty(); });
    chip.appendChild(x);
    dw.appendChild(chip);
  });
  if (!vis.length) dw.appendChild(el('span', 'hint', 'No diagram: the key words show in big type.'));
}

function collectScenes() {
  return [...$('scenes').querySelectorAll('.scene')].map(row => {
    const out = { kind: row.dataset.kind };
    row.querySelectorAll('[data-f]').forEach(f => { out[f.dataset.f] = f.value; });
    const cw = row.querySelector('.cues');
    if (cw) out.cues = JSON.parse(cw.dataset.cues || '[]');
    const dw = row.querySelector('.diagrams');
    if (dw) out.visuals = JSON.parse(dw.dataset.visuals || '[]');
    if (out.start !== undefined) out.start = parseFloat(out.start);
    if (out.end !== undefined) out.end = parseFloat(out.end);
    return out;
  });
}
function applyLocal(list) { project.scenes = list.map(s => ({ ...s })); renderStory(); markDirty(); }
function moveScene(i, d) {
  const list = collectScenes(); const j = i + d;
  if (j < 0 || j >= list.length) return;
  [list[i], list[j]] = [list[j], list[i]];
  list.forEach(s => { if (s.kind === 'narrate') s.take = (project.scenes.find(o => o.narration === s.narration) || {}).take; });
  applyLocal(list);
}
function removeScene(i) {
  const list = collectScenes(); list.splice(i, 1);
  list.forEach(s => { if (s.kind === 'narrate') s.take = (project.scenes.find(o => o.narration === s.narration) || {}).take; });
  applyLocal(list);
}
document.querySelectorAll('.add-scene').forEach(b => b.addEventListener('click', () => {
  const list = collectScenes();
  list.forEach(s => { if (s.kind === 'narrate') s.take = (project.scenes.find(o => o.narration === s.narration) || {}).take; });
  const first = (EXP && (project.library || []).find(c => c.use)) || (project.library || [])[0] || {};
  const k = b.dataset.kind;
  list.push(k === 'title' ? { kind: 'title', title: 'New chapter' }
    : k === 'moment' ? { kind: 'moment', clip: first.id, start: 0, end: Math.min(10, first.duration || 10), caption: '' }
    : EXP ? { kind: 'narrate', narration: 'Write what you want to say here.', visuals: [] }
    : { kind: 'narrate', narration: 'Write what you want to say here.', clip: first.id, caption: '' });
  applyLocal(list);
}));

$('save-story').addEventListener('click', async () => {
  $('save-story').disabled = true;
  try {
    project = await api(`/api/longform/projects/${project.id}/scenes`, jsonOpts('PUT', { scenes: collectScenes() }));
    storyDirty = false;
    render();
  } catch (e) { alert(e.message); $('save-story').disabled = false; }
});
$('rewrite-story').addEventListener('click', async () => {
  if (!confirm('Have Claude write the whole story again? Your edits and recorded scenes will be replaced.')) return;
  try { await api(`/api/longform/projects/${project.id}/write`, { method: 'POST' }); storyDirty = false; loadProject(project.id, true); }
  catch (e) { alert(e.message); }
});

// ---------- recorder (narrated scenes only) ----------
function narrIndices() { return (project.scenes || []).map((s, i) => needsTake(s) ? i : -1).filter(i => i >= 0); }

function renderRecorder() {
  const scenes = project.scenes;
  const idx = narrIndices();
  if (!idx.includes(recIndex)) recIndex = idx[0] || 0;
  const chips = $('chips');
  chips.innerHTML = '';
  idx.forEach((i, k) => {
    const s = scenes[i];
    const b = el('button', (sceneReady(s) ? 'ready' : sceneFlagged(s) ? 'flag' : '') + ((s.take || {}).voice === 'ai' ? ' ai' : '') + (i === recIndex ? ' cur' : ''), String(k + 1));
    b.type = 'button';
    b.addEventListener('click', () => { if (recorder || aiBusy) return; recIndex = i; lastResult = null; renderRecorder(); });
    chips.appendChild(b);
  });
  const s = scenes[recIndex];
  const take = s.take || {};
  const missed = new Set(sceneFlagged(s) ? (take.missed || []) : []);
  const pr = $('prompter');
  pr.innerHTML = '';
  pr.appendChild(el('div', 'label', `Narration ${idx.indexOf(recIndex) + 1} of ${idx.length} · scene ${recIndex + 1}`));
  s.narration.split(' ').forEach((w, i) => { pr.appendChild(el('span', missed.has(i) ? 'miss' : '', w + ' ')); });
  const rb = $('rec-btn');
  if (!recorder) { rb.textContent = sceneFlagged(s) ? '🎙 Read it again' : sceneReady(s) ? '🎙 Re-record' : '🎙 Record'; rb.className = ''; }
  const res = $('take-result');
  const show = lastResult || (take.file ? take : null);
  if (show && !recorder && !aiBusy) {
    res.style.display = 'block';
    const good = show.ok || show.kept;
    const who = show.voice === 'ai' ? '🤖 AI voice · ' : '';
    res.className = 'take-result ' + (good ? 'ok' : 'bad');
    res.textContent = good ? (show.kept && !show.ok ? '✅ ' + who + 'Kept as it is.' : '✅ ' + who + (show.message || 'Sounds right.'))
      : '⚠️ ' + who + (show.voice === 'ai' ? 'Your AI voice garbled part of this. Try it again, or record it yourself.' : show.message);
    if (show.heard) res.appendChild(el('span', 'heard', 'Heard: ' + show.heard));
  } else if (!recorder && !aiBusy) res.style.display = 'none';
  $('play-take').style.display = take.file ? 'inline-block' : 'none';
  $('play-take').textContent = take.voice === 'ai' ? '▶ Play AI take' : '▶ Play my take';
  const ab = $('ai-btn');
  const av = aiVoice(), who = aiWho(av);
  const vReady = !!(av && av.ready);
  renderVoicePick();
  const aa = project.ai_all || {}, aiAll = aa.status === 'running';
  ab.disabled = !!recorder || aiBusy || !vReady || aiAll;
  if (!recorder) $('rec-btn').disabled = aiBusy || aiAll;
  renderAiAll(vReady, aa, idx.filter(i => !sceneReady(scenes[i])).length);
  ab.textContent = aiBusy ? '🤖 Reading…' : take.voice === 'ai' ? '🤖 AI voice again' : `🤖 Use ${who} AI voice`;
  const narr = idx.map(i => scenes[i]);
  const aiCount = narr.filter(x => (x.take || {}).voice === 'ai').length;
  $('ai-hint').textContent = !vReady ? (av && !av.mine ? `${av.name}’s voice has no sample yet: add one on the Voices page.` : 'Want a line read in your AI voice instead? Set it up under “Your AI voice” on the episodes list.')
    : aiCount ? `AI voice on ${aiCount} of ${narr.length} narrated scenes.` + (av && av.mine && aiCount * 3 > narr.length ? ' Keep most of it in your real voice; that’s what viewers stay for.' : '')
      + (av && !av.mine ? ` The video gets YouTube’s “altered or synthetic content” label because it uses ${av.name}’s AI voice.` : '')
    : `Stuck on a line? “Use ${who} AI voice” reads this scene for you in 10–30 seconds.`;
  $('keep-take').style.display = sceneFlagged(s) ? 'inline-block' : 'none';
  const pos = idx.indexOf(recIndex);
  $('prev-scene').disabled = pos <= 0 || !!recorder || aiBusy;
  $('next-scene').disabled = pos >= idx.length - 1 || !!recorder || aiBusy;
}

function pickMime() {
  const opts = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
  return opts.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || '';
}
function recEls() {
  return recTarget === 'sample' ? { btn: $('voice-rec-btn'), timer: $('voice-timer'), res: $('voice-result') }
    : { btn: $('rec-btn'), timer: $('rec-timer'), res: $('take-result') };
}
async function startRecording(target) {
  if (!navigator.mediaDevices || !window.MediaRecorder) { alert('This browser can’t record audio. Try Chrome or Safari.'); return; }
  try { recStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
  catch (e) { alert('Microphone access was blocked. Allow it for this site and try again.'); return; }
  const mime = pickMime();
  recorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined);
  recChunks = [];
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
  recTarget = target;
  recorder.onstop = target === 'sample' ? uploadSample : uploadTake;
  recorder.start();
  recStarted = Date.now();
  lastResult = null;
  const { btn, timer, res } = recEls();
  res.style.display = 'none';
  btn.textContent = '⏹ Stop and check'; btn.className = 'recording';
  timer.textContent = '0:00';
  recTick = setInterval(() => { timer.textContent = fmtTime((Date.now() - recStarted) / 1000); }, 500);
  if (target === 'scene') renderRecorder();
}
function stopRecording() { if (!recorder) return; clearInterval(recTick); recorder.stop(); recStream.getTracks().forEach(t => t.stop()); }
async function uploadTake() {
  const idx = recIndex;
  const type = (recorder && recorder.mimeType) || (recChunks[0] && recChunks[0].type) || 'audio/webm';
  const blob = new Blob(recChunks, { type });
  recorder = null;
  $('rec-timer').textContent = '';
  const rb = $('rec-btn'); rb.disabled = true; rb.className = ''; rb.textContent = 'Checking…';
  const res = $('take-result'); res.style.display = 'block'; res.className = 'take-result wait'; res.textContent = '👂 Listening back to your take…';
  try {
    const { take } = await api(`/api/longform/projects/${project.id}/scenes/${idx}/take`, { method: 'POST', headers: { 'Content-Type': type.split(';')[0] }, body: blob });
    lastResult = take;
    project.scenes[idx].take = take;
    rb.disabled = false;
    render();
    if (take.ok) {
      const next = narrIndices().find(i => i > idx && !sceneReady(project.scenes[i]));
      if (next !== undefined) setTimeout(() => { if (!recorder && recIndex === idx) { recIndex = next; lastResult = null; renderRecorder(); } }, 1500);
    }
  } catch (e) {
    rb.disabled = false; lastResult = null; renderRecorder();
    res.style.display = 'block'; res.className = 'take-result bad'; res.textContent = '⚠️ ' + e.message;
  }
}
$('rec-btn').addEventListener('click', () => { if (recorder) { if (recTarget === 'scene') stopRecording(); } else startRecording('scene'); });
$('ai-btn').addEventListener('click', async () => {
  if (recorder || aiBusy) return;
  const idx = recIndex;
  aiBusy = true; lastResult = null;
  $('rec-btn').disabled = true;
  renderRecorder();
  const res = $('take-result');
  res.style.display = 'block'; res.className = 'take-result wait';
  res.textContent = '🤖 Your AI voice is reading this scene, then the app listens back to check it (10–30 seconds, a minute the first time)…';
  try {
    const { take } = await api(`/api/longform/projects/${project.id}/scenes/${idx}/clone`, { method: 'POST' });
    aiBusy = false; $('rec-btn').disabled = false;
    lastResult = take;
    project.scenes[idx].take = take;
    render();
  } catch (e) {
    aiBusy = false; $('rec-btn').disabled = false; lastResult = null; renderRecorder();
    res.style.display = 'block'; res.className = 'take-result bad'; res.textContent = '⚠️ ' + e.message;
  }
});
function renderAiAll(vReady, aa, left) {
  const running = aa.status === 'running';
  $('ai-all').style.display = vReady && (left || aa.status) ? 'block' : 'none';
  const b = $('ai-all-btn');
  b.textContent = running ? (aa.stop ? '⏳ Stopping after this scene…' : '⏹ Stop') : `🤖 Let ${aiWho(aiVoice())} AI voice read everything`;
  b.disabled = (running && !!aa.stop) || (!running && (!left || !!recorder || aiBusy));
  $('ai-all-progress').style.display = running ? 'block' : 'none';
  const total = aa.total || 0, done = aa.done || 0;
  if (running) $('ai-all-bar').style.width = Math.round(100 * done / Math.max(total, 1)) + '%';
  // The job counts scenes (chapter cards included); the chips count narrations.
  const no = n => narrIndices().indexOf(n - 1) + 1;
  const flagged = (aa.flagged || []).map(no).filter(Boolean), failed = (aa.failed || []).map(no).filter(Boolean);
  const extra = (flagged.length ? ` Narration${flagged.length > 1 ? 's' : ''} ${flagged.join(', ')} came out garbled twice: play ${flagged.length > 1 ? 'them' : 'it'}, then “Keep anyway”, try the AI voice again or record ${flagged.length > 1 ? 'them' : 'it'} yourself.` : '')
    + (failed.length ? ` Couldn’t read narration${failed.length > 1 ? 's' : ''} ${failed.join(', ')}; try ${failed.length > 1 ? 'them' : 'it'} one by one.` : '');
  $('ai-all-msg').textContent = running
    ? `🤖 Reading narration ${aa.current != null ? no(aa.current + 1) : '…'} · ${done} of ${total} done. Each one takes 10–30 seconds and is checked like a recording. You can leave this page.`
    : aa.status === 'done' ? `✅ Your AI voice read ${done} scene${done === 1 ? '' : 's'}.` + extra
    : aa.status === 'stopped' ? `Stopped after ${done} of ${total}. What it read is kept.` + extra
    : aa.status === 'error' ? '⚠️ ' + (aa.error || 'It stopped.') + extra
    : `Reads the ${left} narrated scene${left === 1 ? '' : 's'} that ${left === 1 ? 'has' : 'have'} no take yet, one after another (about ${Math.max(1, Math.round(left * 25 / 60))} min). Scenes you recorded yourself are kept.`;
}
$('ai-all-btn').addEventListener('click', async () => {
  const running = (project.ai_all || {}).status === 'running';
  try {
    if (running) project = await api(`/api/longform/projects/${project.id}/clone-all`, { method: 'DELETE' });
    else {
      const left = (project.scenes || []).filter(s => needsTake(s) && !sceneReady(s)).length;
      const av = aiVoice();
      if (!confirm(`${av && !av.mine ? av.name + '’s' : 'Your'} AI voice will read the ${left} narrated scene${left === 1 ? '' : 's'} without a take. Scenes you recorded yourself stay as they are. Start?`)) return;
      project = await api(`/api/longform/projects/${project.id}/clone-all`, { method: 'POST' });
    }
    loadProject(project.id, false);
  } catch (e) { alert(e.message); }
});
$('prev-scene').addEventListener('click', () => { const idx = narrIndices(); const p = idx.indexOf(recIndex); if (p > 0) { recIndex = idx[p - 1]; lastResult = null; renderRecorder(); } });
$('next-scene').addEventListener('click', () => { const idx = narrIndices(); const p = idx.indexOf(recIndex); if (p < idx.length - 1) { recIndex = idx[p + 1]; lastResult = null; renderRecorder(); } });
$('play-take').addEventListener('click', () => { const a = $('take-audio'); a.src = `/api/longform/projects/${project.id}/scenes/${recIndex}/take?t=${Date.now()}`; a.play(); });
$('keep-take').addEventListener('click', async () => {
  try { project = await api(`/api/longform/projects/${project.id}/scenes/${recIndex}/keep`, { method: 'POST' }); lastResult = null; render(); } catch (e) { alert(e.message); }
});

// ---------- lessons ----------
function renderLessons(list) {
  const ul = $('lessons');
  ul.innerHTML = '';
  list.forEach((l, i) => {
    const li = el('li');
    const body = el('div');
    body.appendChild(document.createTextNode(l.text));
    body.appendChild(el('span', 'src', l.source || ''));
    li.appendChild(body);
    const x = el('button', '', '✕'); x.type = 'button'; x.title = 'Remove this lesson';
    x.addEventListener('click', async () => {
      if (!confirm('Remove this lesson? Claude stops following it from the next story.')) return;
      try { renderLessons((await api(`/api/longform/lessons/${i}`, { method: 'DELETE' })).lessons); } catch (e) { alert(e.message); }
    });
    li.appendChild(x);
    ul.appendChild(li);
  });
  if (!list.length) ul.appendChild(el('li', 'hint', 'No lessons yet.'));
}
async function loadLessons() {
  try { renderLessons((await api('/api/longform/lessons')).lessons); } catch (e) { /* page still works */ }
}
$('learn-btn').addEventListener('click', async () => {
  const text = $('learn-text').value.trim();
  if (text.length < 20) { alert('Paste the feedback first.'); return; }
  const b = $('learn-btn'); b.disabled = true; $('learn-status').textContent = '🧠 Claude is reading it (about 20 seconds)…';
  try {
    const r = await api('/api/longform/lessons', jsonOpts('POST', { text, source: $('learn-source').value }));
    renderLessons(r.lessons);
    $('learn-text').value = ''; $('learn-source').value = '';
    $('learn-status').textContent = '✅ Learned. The next story you write uses these.';
  } catch (e) { $('learn-status').textContent = '⚠️ ' + e.message; }
  finally { b.disabled = false; }
});

// ---------- your AI voice ----------
async function loadVoice() {
  try { voice = await api('/api/longform/voice'); } catch (e) { voice = null; }
  try { voices = (await api('/api/voices')).voices; } catch (e) { voices = null; }
  renderVoice();
}
// The AI voice this episode reads with: Dean's own unless he picked a
// friend's on the Voices page.
function aiVoice() {
  const id = (project && project.ai_voice) || 'me';
  return (voices || []).find(v => v.id === id) || (id === 'me' ? { id: 'me', name: 'You', mine: true, ready: !!(voice && voice.ready) } : null);
}
function aiWho(v) { return !v || v.mine ? 'my' : `${v.name}’s`; }
function renderVoicePick() {
  const row = $('ai-voice-row'), sel = $('ai-voice');
  const list = voices || [];
  row.style.display = project && list.length > 1 ? 'flex' : 'none';
  if (!project || list.length < 2) return;
  const cur = project.ai_voice || 'me';
  sel.innerHTML = '';
  for (const v of list) {
    const o = document.createElement('option');
    o.value = v.id; o.textContent = (v.mine ? 'You' : v.name) + (v.sample ? '' : ' (no sample yet)');
    o.selected = v.id === cur; sel.appendChild(o);
  }
  if (!list.some(v => v.id === cur)) {
    const o = document.createElement('option'); o.value = cur; o.textContent = 'A deleted voice'; o.selected = true; sel.appendChild(o);
  }
  sel.disabled = !!recorder || aiBusy || (project.ai_all || {}).status === 'running';
}
$('ai-voice').addEventListener('change', async () => {
  try { project = await api(`/api/longform/projects/${project.id}/ai-voice`, jsonOpts('PUT', { voice: $('ai-voice').value })); lastResult = null; renderRecorder(); }
  catch (e) { alert(e.message); renderVoicePick(); }
});
function renderVoice() {
  const v = voice || {};
  const st = $('voice-status');
  const lines = [];
  if (v.sample) lines.push(`✅ Voice sample saved ${new Date(v.sample.recorded_at * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} (${Math.round(v.sample.duration)} seconds).`);
  if (!v.installed) lines.push('The AI voice isn’t installed on the server yet (it comes with the next deploy).');
  else if (v.setup_hint) lines.push('⚙️ ' + v.setup_hint);
  else if (v.error) lines.push('⚠️ ' + v.error);
  else if (v.sample) lines.push('Ready: tap “🤖 Use my AI voice” on any narrated scene.');
  else lines.push('Record your sample below to switch it on.');
  st.style.display = 'block';
  st.textContent = lines.join(' ');
  $('voice-text').textContent = v.sample_text || '';
  $('voice-summary').textContent = v.sample ? 'Record your voice sample again' : 'Record your voice sample';
  $('voice-play').style.display = $('voice-delete').style.display = v.sample ? 'inline-block' : 'none';
  if (project) renderRecorder();
}
async function uploadSample() {
  const type = (recorder && recorder.mimeType) || (recChunks[0] && recChunks[0].type) || 'audio/webm';
  const blob = new Blob(recChunks, { type });
  recorder = null;
  recTarget = 'scene';
  $('voice-timer').textContent = '';
  const rb = $('voice-rec-btn'); rb.disabled = true; rb.className = ''; rb.textContent = 'Checking…';
  const res = $('voice-result'); res.style.display = 'block'; res.className = 'take-result wait'; res.textContent = '👂 Listening back to your sample…';
  try {
    voice = await api('/api/longform/voice/sample', { method: 'POST', headers: { 'Content-Type': type.split(';')[0] }, body: blob });
    res.className = 'take-result ok'; res.textContent = '✅ Sample saved. Play it back: if it sounds clean, you’re set.';
  } catch (e) {
    res.className = 'take-result bad'; res.textContent = '⚠️ ' + e.message;
  }
  rb.disabled = false; rb.textContent = '🎙 Record sample';
  renderVoice();
}
$('voice-rec-btn').addEventListener('click', () => { if (recorder) { if (recTarget === 'sample') stopRecording(); } else startRecording('sample'); });
$('voice-play').addEventListener('click', () => { const a = $('voice-audio'); a.src = `/api/longform/voice/sample?t=${Date.now()}`; a.play(); });
$('voice-delete').addEventListener('click', async () => {
  if (!confirm('Delete your voice sample? Scenes already read in your AI voice keep their audio.')) return;
  try { voice = await api('/api/longform/voice/sample', { method: 'DELETE' }); $('voice-result').style.display = 'none'; renderVoice(); } catch (e) { alert(e.message); }
});

// ---------- thumbnails ----------
const THUMB_LABEL = { face: 'Face + text', full: 'Big face', split: 'Then vs now', left: 'Diagram + text', center: 'Big text', bottom: 'Text below' };
function renderThumbs() {
  const p = project, rec = p.thumbnails || {}, items = rec.items || [];
  const box = $('thumbs'); box.innerHTML = '';
  items.forEach((it, i) => {
    const f = el('figure', i === (rec.chosen || 0) ? 'on' : '');
    const im = el('img'); im.alt = it.hook; im.src = `/api/longform/projects/${p.id}/thumbnails/${i}?f=${encodeURIComponent(it.file)}`;
    f.appendChild(im);
    f.appendChild(el('figcaption', '', (i === (rec.chosen || 0) ? '✓ ' : '') + (THUMB_LABEL[it.layout] || it.layout)));
    f.addEventListener('click', async () => {
      try { project = await api(`/api/longform/projects/${p.id}/thumbnails/${i}/choose`, { method: 'POST' }); renderThumbs(); } catch (e) { alert(e.message); }
    });
    box.appendChild(f);
  });
  $('thumb-make').textContent = items.length ? '🖼 Make new ones' : '🖼 Make thumbnails';
  $('thumb-edit').style.display = items.length ? 'block' : 'none';
  if (items.length) {
    const i = Math.min(rec.chosen || 0, items.length - 1);
    if (document.activeElement !== $('thumb-hook')) $('thumb-hook').value = items[i].hook;
    $('thumb-dl').href = `/api/longform/projects/${p.id}/thumbnails/${i}?download=1`;
    $('thumb-apply').style.display = p.youtube_video_id ? 'inline-block' : 'none';
  }
  const st = p.thumbnail_status;
  $('thumb-status').textContent = st ? (st.ok ? '✅ ' : '⚠️ ') + st.message : '';
}
$('thumb-make').addEventListener('click', async () => {
  const b = $('thumb-make'); b.disabled = true; b.textContent = 'Making thumbnails… (about a minute)';
  try { project = await api(`/api/longform/projects/${project.id}/thumbnails`, { method: 'POST' }); }
  catch (e) { alert(e.message); }
  finally { b.disabled = false; renderThumbs(); }
});
$('thumb-redraw').addEventListener('click', async () => {
  const rec = project.thumbnails || {}; const i = rec.chosen || 0;
  const b = $('thumb-redraw'); b.disabled = true;
  try { project = await api(`/api/longform/projects/${project.id}/thumbnails/${i}`, jsonOpts('PUT', { hook: $('thumb-hook').value })); $('thumb-hook').blur(); renderThumbs(); }
  catch (e) { alert(e.message); } finally { b.disabled = false; }
});
$('thumb-apply').addEventListener('click', async () => {
  const b = $('thumb-apply'); b.disabled = true;
  try { project = await api(`/api/longform/projects/${project.id}/thumbnails/apply`, { method: 'POST' }); renderThumbs(); }
  catch (e) { alert(e.message); } finally { b.disabled = false; }
});

// ---------- render & post ----------
function renderRender() {
  const p = project, r = p.render || {};
  const rendering = r.status === 'rendering';
  const narr = (p.scenes || []).filter(needsTake);
  const left = narr.filter(s => !sceneReady(s)).length;
  $('music-current').style.display = p.music ? 'block' : 'none';
  $('music-current').textContent = p.music ? `🎵 ${p.music.name}` : '';
  $('music-remove').style.display = p.music ? 'inline-block' : 'none';
  $('music-db-row').style.display = p.music ? 'flex' : 'none';
  [['voice-db', p.voice_db || 0], ['music-db', musicDb(p)]].forEach(([id, v]) => {
    if (document.activeElement !== $(id)) $(id).value = v;
    $(id + '-val').textContent = fmtDb($(id).value);
    $(id).disabled = rendering;
  });
  $('music-only').style.display = r.status === 'done' ? 'inline-block' : 'none';
  $('music-only').disabled = rendering;
  $('music-upload').disabled = $('music-remove').disabled = rendering;
  const rb = $('render-btn');
  rb.disabled = rendering || left > 0;
  rb.textContent = r.status === 'done' ? '🎬 Render again' : '🎬 Render video (1080p)';
  $('render-hint').textContent = left ? `Record the last ${left} narrated scene${left === 1 ? '' : 's'} first.` : rendering ? '' : 'Rendering takes roughly as long as the video. You can leave this page while it runs.';
  $('render-progress').style.display = rendering ? 'block' : 'none';
  if (rendering) { $('render-bar').style.width = Math.round((r.progress || 0) * 100) + '%'; $('render-msg').textContent = r.message || 'Rendering…'; }
  $('render-error').style.display = r.status === 'error' ? 'block' : 'none';
  $('render-error').textContent = r.status === 'error' ? '⚠️ ' + (r.error || 'Render failed.') : '';
  const out = $('render-out');
  if (r.status !== 'done') { out.style.display = 'none'; return; }
  out.style.display = 'block';
  const src = `/api/longform/projects/${p.id}/video?t=${Math.round(r.built_at || 0)}`;
  if ($('final-video').getAttribute('src') !== src) $('final-video').setAttribute('src', src);
  $('video-dl').href = src;
  $('render-info').textContent = `${fmtTime(r.duration)} long · rendered in ${Math.max(1, Math.round((r.took_seconds || 0) / 60))} min.`;
  const pub = p.publish;
  $('publish-out').style.display = pub ? 'block' : 'none';
  if (!$('upload-btn').disabled) $('upload-btn').textContent = '⬆ Upload to ' + SERIES.channel;
  if (EXP) {
    const ql = $('quiz-list'); ql.innerHTML = '';
    const quiz = (pub && pub.quiz) || [];
    $('quiz-wrap').style.display = quiz.length ? 'block' : 'none';
    quiz.forEach(q => {
      const d = el('div', 'quiz-item');
      d.appendChild(el('b', '', q.time + '  '));
      d.appendChild(document.createTextNode(q.question));
      const ul = el('div', 'hint');
      ul.textContent = q.options.map((o, i) => `${'ABCD'[i]}. ${o}${i === q.answer ? ' ✓' : ''}`).join('   ');
      d.appendChild(ul);
      ql.appendChild(d);
    });
  }
  if (pub) {
    const tl = $('title-list'); tl.innerHTML = '';
    (pub.titles || []).forEach(t => { const o = el('div', 'title-opt', t); o.addEventListener('click', () => { $('post-title').value = t; }); tl.appendChild(o); });
    if (!$('post-title').value) $('post-title').value = (pub.titles || [])[0] || p.title;
    if (document.activeElement !== $('desc-text') && !$('desc-text').dataset.edited) $('desc-text').value = pub.description || '';
  }
  renderThumbs();
  const ps = $('post-status'); ps.innerHTML = '';
  if (p.youtube_url) { ps.appendChild(document.createTextNode(`✅ Uploaded (${p.uploaded_privacy}): `)); const a = el('a', '', p.youtube_url); a.href = p.youtube_url; a.target = '_blank'; ps.appendChild(a); }
  if (p.promo_job_id) {
    if (ps.childNodes.length) ps.appendChild(el('br'));
    ps.appendChild(document.createTextNode(p.youtube_url
      ? '✂️ 2 cliffhanger Shorts are being cut from it: find them on the Home page. Their descriptions already link to this video; after posting one, a link takes you straight to where YouTube Studio lets you set it as the Short’s “Related video”.'
      : '✂️ 2 cliffhanger Shorts are being cut: find them on the Home page. Once you upload this video, they’ll carry the link to it.'));
  }
}
$('desc-text').addEventListener('input', () => { $('desc-text').dataset.edited = '1'; });
$('render-btn').addEventListener('click', async () => {
  try { await api(`/api/longform/projects/${project.id}/render`, { method: 'POST' }); loadProject(project.id, false); } catch (e) { alert(e.message); }
});
$('music-upload').addEventListener('click', async () => {
  const f = $('music-file').files[0];
  if (!f) { alert('Choose a music file first.'); return; }
  $('music-upload').disabled = true;
  try { project = await api(`/api/longform/projects/${project.id}/music?name=${encodeURIComponent(f.name)}`, { method: 'POST', headers: { 'Content-Type': f.type || 'audio/mpeg' }, body: f }); $('music-file').value = ''; render(); }
  catch (e) { alert(e.message); } finally { $('music-upload').disabled = false; }
});
function musicDb(p) { return p.music_db != null ? p.music_db : ({ quiet: -3, loud: 3 }[p.music_level] || 0); }
function fmtDb(v) { v = Number(v); return (v > 0 ? '+' : '') + v + ' dB'; }
['voice-db', 'music-db'].forEach(id => {
  $(id).addEventListener('input', () => { $(id + '-val').textContent = fmtDb($(id).value); });
  $(id).addEventListener('change', async () => {
    const body = id === 'voice-db' ? { voice_db: Number($(id).value) } : { music_db: Number($(id).value) };
    try {
      project = await api(`/api/longform/projects/${project.id}/mix`, jsonOpts('PUT', body));
      $(id).blur();
      render();
      $('music-level-hint').textContent = (project.render || {}).status === 'done'
        ? 'Press “🔊 Update sound only” to hear it. Your scenes are kept; only the sound is mixed again.' : 'Used when you render.';
    } catch (e) { alert(e.message); }
  });
});
$('music-only').addEventListener('click', async () => {
  try {
    await api(`/api/longform/projects/${project.id}/render?music_only=true`, { method: 'POST' });
    $('music-level-hint').textContent = '';
    loadProject(project.id, false);
  } catch (e) { alert(e.message); }
});
$('music-remove').addEventListener('click', async () => { try { project = await api(`/api/longform/projects/${project.id}/music`, { method: 'DELETE' }); render(); } catch (e) { alert(e.message); } });
$('publish-btn').addEventListener('click', async () => {
  const b = $('publish-btn'); b.disabled = true; b.textContent = 'Writing…';
  try { project = await api(`/api/longform/projects/${project.id}/publish-text`, { method: 'POST' }); delete $('desc-text').dataset.edited; $('post-title').value = ''; render(); }
  catch (e) { alert(e.message); } finally { b.disabled = false; b.textContent = '✍️ Write title & description'; }
});
$('upload-btn').addEventListener('click', async () => {
  const privacy = $('post-privacy').value;
  if (privacy === 'public' && !confirm('Post it publicly right now?')) return;
  const b = $('upload-btn'); b.disabled = true; b.textContent = 'Uploading… (a few minutes)';
  try { project = await api(`/api/longform/projects/${project.id}/upload`, jsonOpts('POST', { title: $('post-title').value, description: $('desc-text').value, privacy_status: privacy })); render(); }
  catch (e) { alert(e.message); } finally { b.disabled = false; b.textContent = '⬆ Upload to ' + SERIES.channel; }
});
$('promo-btn').addEventListener('click', async () => {
  const b = $('promo-btn'); b.disabled = true;
  try { project = await api(`/api/longform/projects/${project.id}/promo-shorts`, { method: 'POST' }); render(); } catch (e) { alert(e.message); }
  finally { b.disabled = false; }
});

$('delete-project').addEventListener('click', async () => {
  if (!confirm('Delete this episode, its research, clips and recordings? This can’t be undone.')) return;
  try { await api(`/api/longform/projects/${project.id}`, { method: 'DELETE' }); showList(); } catch (e) { alert(e.message); }
});
$('to-list').addEventListener('click', () => { if (recorder) stopRecording(); showList(); });

const startId = new URLSearchParams(location.search).get('v');
if (startId) openProject(startId); else showList();
</script>
</body>
</html>
"""


_LONGFORM_SERIES = {
    "documentary": {
        "path": "/long-form", "channel": "Caught On Stream", "title": "The Story Of", "logo": "🎬",
        "subtitle": "A bi-weekly documentary series for Caught On Stream: a different streamer each episode. The app researches them and pulls their best clips, Claude drafts the story, you narrate it, and the app edits it together.",
    },
    "explainer": {
        "path": "/caught-on-code", "channel": "Caught On Code", "title": "Caught On Code", "logo": "💻",
        "subtitle": "The tech behind gaming, streaming and the internet, explained: netcode, anti-cheat, matchmaking, how a stream reaches you. The app researches the topic, Claude writes the script with animated diagrams and pause-and-guess quizzes, you narrate it, and the app draws and edits it together.",
    },
}


def _longform_page(kind: str) -> str:
    """The long-form page for one series: the documentaries (/long-form) or
    the Caught On Code explainers (/caught-on-code). Same template; the
    series decides the header, the list view and a few steps."""
    cfg = _LONGFORM_SERIES[kind]
    return (_LONGFORM_TEMPLATE
            .replace("__SERIES_KIND__", kind)
            .replace("__SERIES_JSON__", json.dumps({"kind": kind, "path": cfg["path"], "channel": cfg["channel"]}))
            .replace("__PAGE_TITLE__", cfg["title"]).replace("__LOGO__", cfg["logo"]).replace("__H1__", cfg["title"])
            .replace("__SUBTITLE__", cfg["subtitle"]))


LONGFORM_HTML = _longform_page("documentary")
EXPLAINER_HTML = _longform_page("explainer")

