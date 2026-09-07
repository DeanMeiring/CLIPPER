"""Web wrapper around the clipper CLI pipeline: submit a video, poll status,
download the rendered clips. One job runs at a time on a background worker
thread so a small Railway instance doesn't try to transcode multiple videos
at once.
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from clipper.captions import build_ass
from clipper.download import _ffprobe_duration, download_video, is_url, probe_video
from clipper.long_vod import gather_candidates, is_long_vod, select_and_map
from clipper.reframe import compute_layout
from clipper.render import render_clip
from clipper.select_moments import select_clips
from clipper.transcribe import Word, get_transcript
from clipper.trending import get_trending_sections, search_creator
from clipper.notify import send_telegram

BASE_DIR = Path(os.environ.get("CLIPPER_JOBS_DIR", "/tmp/clipper_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

JOB_META_NAME = "job.json"
TERMINAL_STATES = ("done", "error", "cancelled")


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


def _job_public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("request", "pending_regenerate")}


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
        (out_dir / JOB_META_NAME).write_text(json.dumps(data), encoding="utf-8")
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


def _run_job(job_id: str) -> None:
    with jobs_lock:
        pending_regenerate = jobs[job_id].pop("pending_regenerate", None)
    if pending_regenerate is not None:
        _run_regenerate(job_id, pending_regenerate)
        return

    req: JobRequest = jobs[job_id]["request"]
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
        _set(job_id, state="selecting", message=f"Asking Claude to pick up to {req.num_clips} moments...")
        _progress(job_id, 0.55)
        picks = select_clips(
            words, dl.duration,
            n_clips=req.num_clips, min_len=req.min_len, max_len=req.max_len,
            focus=req.focus, source_title=dl.title,
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
        render_clip(video_path, pick.start, pick.end, layout, ass_path, out_path)
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
    max_len = float(req.get("max_len") or 90.0)

    with jobs_lock:
        job = jobs[job_id]
        pipeline = job.get("pipeline")
        source_title = job.get("source_title") or ""
        existing_clips = list(job.get("clips") or [])
        used_ranges = [tuple(r) for r in (job.get("used_ranges") or [])]
        used_window_indices = set(job.get("used_window_indices") or [])
    if not pipeline:
        pipeline = "long_vod" if list(raw_dir.glob("cand_*.mp4")) else "short"

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
        )
        cand_words = {c["index"]: c["words"] for c in candidates}
        render_items = [(video_path, cand_words[pick.window_index], pick) for video_path, pick in mapped]
        used_window_indices |= {pick.window_index for _, pick in mapped}
        _set(job_id, used_window_indices=list(used_window_indices))
    else:
        cached = _load_transcript_for_regenerate(raw_dir)
        if not cached:
            _set(job_id, state="error", error="The downloaded source video is gone -- resubmit the source URL instead.")
            return
        cancel()
        picks = select_clips(
            cached["words"], cached["duration"],
            n_clips=num_clips + len(used_ranges), min_len=min_len, max_len=max_len,
            focus=focus, source_title=source_title,
        )

        def _overlaps_used(p) -> bool:
            return any(not (p.end <= u[0] or p.start >= u[1]) for u in used_ranges)

        picks = [p for p in picks if not _overlaps_used(p)][:num_clips]
        render_items = [(cached["video_path"], cached["words"], pick) for pick in picks]
        used_ranges = used_ranges + [(pick.start, pick.end) for pick in picks]
        _set(job_id, used_ranges=[list(r) for r in used_ranges])

    if not render_items:
        _set(job_id, state="error", error="Claude didn't return any new, non-overlapping moments this time -- try a different focus.")
        return

    clips_meta = _render_all(job_id, out_dir, render_items, 0.15, existing_clips)
    _progress(job_id, 1.0)
    _set(job_id, state="done", message=f"Done. {len(clips_meta)} clip(s) total.")


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
        n = len(job.get("clips") or [])
        text = f'✅ Clipper done: "{label}" -- {n} clip(s) ready.'
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
def get_clip(job_id: str, filename: str) -> FileResponse:
    path = BASE_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


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
  #stop-modal-overlay, #mood-modal-overlay, #regen-modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    align-items: center; justify-content: center; z-index: 100; padding: 16px;
  }
  #stop-modal-overlay.open, #mood-modal-overlay.open, #regen-modal-overlay.open { display: flex; }
  .modal { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px; max-width: 380px; box-shadow: var(--shadow); }
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
  .actions { display: flex; align-items: center; gap: 0; }
  #search-row { display: flex; gap: 8px; margin-top: 0; }
  #search-row input { flex: 1; margin-top: 0; }
  #search-row button { margin-top: 0; padding: 0 16px; white-space: nowrap; }
  #search-results-section { margin-bottom: 16px; display: none; }
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
    <input id="max_len" type="number" value="90">
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
    <label>Focus (optional)</label>
    <input id="regen-focus" placeholder="e.g. funniest moments">
    <label>How many more clips?</label>
    <input id="regen-num-clips" type="number" value="3" min="1" max="10">
    <div class="modal-actions" style="margin-top:16px">
      <button id="regen-go-btn" type="button">Generate</button>
      <button id="regen-cancel-btn" type="button" class="ghost">Cancel</button>
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
      meta.textContent = job.state === 'done'
        ? `${(job.clips || []).length} clip(s)`
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
      if (!running) {
        const regenBtn = document.createElement('button');
        regenBtn.type = 'button';
        regenBtn.textContent = 'Generate more clips';
        regenBtn.addEventListener('click', () => openRegenModal(job.id));
        row.appendChild(regenBtn);

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
const regenGoBtn = document.getElementById('regen-go-btn');
let regenJobId = null;

function openRegenModal(jobId) {
  regenJobId = jobId;
  regenFocusInput.value = '';
  regenNumClipsInput.value = '3';
  regenModal.classList.add('open');
}

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
    link.href = `/api/jobs/${jobId}/clips/${c.file}`;
    link.setAttribute('download', '');
    link.textContent = `Download ${c.file}`;
    div.appendChild(link);

    clipsEl.appendChild(div);
  });

  if (job.state === 'done' || job.state === 'error' || job.state === 'cancelled') {
    if (timer) clearInterval(timer);
    setRunning(false);
    cancelBtn.disabled = false;
    cancelBtn.textContent = 'Emergency stop';
    deleteBtn.style.display = (job.clips || []).length ? 'block' : 'none';
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
