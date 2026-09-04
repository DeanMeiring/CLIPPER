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
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from clipper.captions import build_ass
from clipper.download import download_video, is_url, probe_video
from clipper.long_vod import gather_candidates, is_long_vod, select_and_map
from clipper.reframe import compute_layout
from clipper.render import render_clip
from clipper.select_moments import select_clips
from clipper.transcribe import get_transcript
from clipper.trending import get_trending_sections

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


def _job_public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "request"}


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

    if not render_items:
        _set(job_id, state="error", error="Model returned no usable picks.")
        return

    clips_meta = []
    render_span = 1.0 - render_base
    for i, (video_path, words, pick) in enumerate(render_items, start=1):
        cancel()
        _set(job_id, state="rendering", message=f'Rendering clip {i}/{len(render_items)}: "{pick.title}"')
        _progress(job_id, render_base + render_span * ((i - 1) / len(render_items)))
        clip_words = [w for w in words if w.start >= pick.start and w.end <= pick.end]
        layout = compute_layout(video_path, pick.start, pick.end, target_w=1080, target_h=1920)
        out_path = out_dir / f"clip_{i:02d}.mp4"
        ass_path = out_dir / f"_clip_{i:02d}.ass"
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
            "reason": pick.reason,
        })
        _set(job_id, clips=list(clips_meta))
        _progress(job_id, render_base + render_span * (i / len(render_items)))

    _progress(job_id, 1.0)
    _set(job_id, state="done", message=f"Done. {len(clips_meta)} clip(s).")


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
            job_queue.task_done()


_load_persisted_jobs()
threading.Thread(target=_worker, daemon=True).start()


@protected.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "message": "Queued",
            "progress": 0.0,
            "estimate_minutes": None,
            "clips": [],
            "error": None,
            "request": req,
        }
        cancel_events[job_id] = threading.Event()
    _persist(job_id)
    job_queue.put(job_id)
    return {"job_id": job_id}


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


@protected.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return {k: v for k, v in job.items() if k != "request"}


@protected.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        event = cancel_events.get(job_id)
        if event is None or job["state"] in ("done", "error", "cancelled"):
            return {"ok": True, "state": job["state"]}
        event.set()
        job["message"] = "Stopping..."
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
    """Three rows for the UI: configured creators' latest YouTube upload,
    configured creators' latest Twitch VOD, and Twitch's biggest live
    streams globally (see clipper/trending.py). Cached briefly so
    refreshing the page doesn't re-hit the Twitch/YouTube APIs (and
    YouTube's daily quota) every time."""
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


@app.get("/healthz")
def healthz() -> dict:
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
  .title-row button { margin-top: 0; padding: 8px 12px; font-size: 0.82rem; }
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
</style>
</head>
<body>
<div class="page">
<div class="card">

<div class="brand"><span class="logo">🎬</span><h1>clipper</h1></div>
<p class="subtitle">Paste a YouTube or Twitch link, get back short vertical highlight clips picked by Claude.</p>

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
const TRENDING_SECTIONS = ['youtube_channels', 'twitch_vods', 'trending_live'];
let timer = null;
let currentJobId = null;
let jobStartedAt = null;

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

function setRunning(running) {
  submitBtn.disabled = running;
  cancelBtn.style.display = running ? 'inline-block' : 'none';
  progressWrap.style.display = running ? 'block' : 'none';
}

document.getElementById('submit').addEventListener('click', async () => {
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
  currentJobId = job_id;
  jobStartedAt = Date.now();
  if (timer) clearInterval(timer);
  timer = setInterval(() => poll(job_id), 2000);
  poll(job_id);
});

cancelBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  cancelBtn.disabled = true;
  cancelBtn.textContent = 'Stopping...';
  await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
});

async function poll(jobId) {
  const resp = await fetch(`/api/jobs/${jobId}`);
  if (!resp.ok) return;
  const job = await resp.json();
  statusEl.textContent = `[${job.state}] ${job.message || ''}`;
  if (job.error) statusEl.textContent += `\\nError: ${job.error}`;

  const pct = Math.round((job.progress || 0) * 100);
  progressBar.style.width = pct + '%';
  progressPct.textContent = pct + '%';
  if (job.estimate_minutes) {
    const elapsedMin = (Date.now() - jobStartedAt) / 60000;
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
  } else {
    deleteBtn.textContent = "🗑 I've downloaded these — delete from server";
  }
  deleteBtn.disabled = false;
});
</script>
</body>
</html>
"""
