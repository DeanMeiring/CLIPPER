"""Web wrapper around the clipper CLI pipeline: submit a video, poll status,
download the rendered clips. One job runs at a time on a background worker
thread so a small Railway instance doesn't try to transcode multiple videos
at once.
"""
from __future__ import annotations

import os
import queue
import secrets
import threading
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

BASE_DIR = Path(os.environ.get("CLIPPER_JOBS_DIR", "/tmp/clipper_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)


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


def _set(job_id: str, **kwargs) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            job.update(kwargs)


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


def _worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            _run_job(job_id)
        except JobCancelled:
            _set(job_id, state="cancelled", message="Cancelled by user.")
        except Exception as e:  # noqa: BLE001 - surface any pipeline failure to the client
            _set(job_id, state="error", error=str(e))
        finally:
            cancel_events.pop(job_id, None)
            job_queue.task_done()


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
    job_queue.put(job_id)
    return {"job_id": job_id}


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
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 1.4rem; }
  label { display: block; margin-top: 14px; font-size: 0.9rem; font-weight: 600; }
  input, textarea { width: 100%; padding: 8px; margin-top: 4px; font-size: 1rem; box-sizing: border-box; }
  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }
  button { margin-top: 18px; padding: 10px 18px; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  #status { margin-top: 24px; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.85rem; }
  #progress-wrap { display: none; margin-top: 12px; }
  #progress-meta { display: flex; justify-content: space-between; font-size: 0.8rem; color: #888; margin-bottom: 4px; }
  #progress-track { background: #8883; border-radius: 8px; height: 14px; overflow: hidden; }
  #progress-bar { background: #2e8b57; height: 100%; width: 0%; transition: width 0.6s ease; }
  #cancel-btn { display: none; margin-top: 10px; background: #b33; color: #fff; border: none; border-radius: 6px; }
  #cancel-btn:hover { background: #a22; }
  .clip { margin-top: 10px; padding: 10px; border: 1px solid #8888; border-radius: 8px; }
  .clip a { display: inline-block; margin-top: 6px; }
  .title-row { display: flex; gap: 6px; margin-top: 6px; align-items: center; }
  .title-row input { flex: 1; margin-top: 0; font-weight: 600; }
  .title-row button { margin-top: 0; padding: 6px 10px; font-size: 0.85rem; }
  .checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 14px; }
  .checkbox-row input { width: auto; margin-top: 0; }
  .checkbox-row label { margin-top: 0; }
  .hint { font-size: 0.8rem; color: #888; font-weight: 400; margin-top: 2px; }
</style>
</head>
<body>
<h1>clipper</h1>
<p>Paste a YouTube URL, get back short vertical highlight clips picked by Claude.</p>

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

<button id="submit">Generate clips</button>
<button id="cancel-btn" type="button">Emergency stop</button>

<div id="status"></div>
<div id="progress-wrap">
  <div id="progress-meta">
    <span id="progress-pct">0%</span>
    <span id="progress-eta"></span>
  </div>
  <div id="progress-track"><div id="progress-bar"></div></div>
</div>
<div id="clips"></div>

<script>
const statusEl = document.getElementById('status');
const clipsEl = document.getElementById('clips');
const submitBtn = document.getElementById('submit');
const cancelBtn = document.getElementById('cancel-btn');
const progressWrap = document.getElementById('progress-wrap');
const progressBar = document.getElementById('progress-bar');
const progressPct = document.getElementById('progress-pct');
const progressEta = document.getElementById('progress-eta');
let timer = null;
let currentJobId = null;
let jobStartedAt = null;

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
  }
}
</script>
</body>
</html>
"""
