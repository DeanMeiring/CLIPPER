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
from clipper.download import download_video
from clipper.reframe import compute_layout
from clipper.render import render_clip
from clipper.select_moments import select_clips
from clipper.transcribe import get_transcript

BASE_DIR = Path(os.environ.get("CLIPPER_JOBS_DIR", "/tmp/clipper_jobs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

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
        jobs[job_id].update(kwargs)


def _run_job(job_id: str) -> None:
    req: JobRequest = jobs[job_id]["request"]
    out_dir = BASE_DIR / job_id
    raw_dir = out_dir / "_source"

    _set(job_id, state="downloading", message=f"Fetching source: {req.source}")
    dl = download_video(req.source, raw_dir)
    _set(job_id, source_title=dl.title, duration=dl.duration)

    _set(job_id, state="transcribing", message="Getting transcript...")
    words = get_transcript(dl.video_path, dl.captions_path, prefer_whisper=req.whisper)
    if not words:
        _set(job_id, state="error", error="No speech/captions found -- nothing to clip.")
        return

    _set(job_id, state="selecting", message=f"Asking Claude to pick up to {req.num_clips} moments...")
    picks = select_clips(
        words, dl.duration,
        n_clips=req.num_clips, min_len=req.min_len, max_len=req.max_len,
        focus=req.focus, source_title=dl.title,
    )
    if not picks:
        _set(job_id, state="error", error="Model returned no usable picks.")
        return

    clips_meta = []
    for i, pick in enumerate(picks, start=1):
        _set(job_id, state="rendering", message=f'Rendering clip {i}/{len(picks)}: "{pick.title}"')
        clip_words = [w for w in words if w.start >= pick.start and w.end <= pick.end]
        layout = compute_layout(dl.video_path, pick.start, pick.end, target_w=1080, target_h=1920)
        out_path = out_dir / f"clip_{i:02d}.mp4"
        ass_path = out_dir / f"_clip_{i:02d}.ass"
        build_ass(clip_words, pick.start, ass_path)
        render_clip(dl.video_path, pick.start, pick.end, layout, ass_path, out_path)
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

    _set(job_id, state="done", message=f"Done. {len(clips_meta)} clip(s).")


def _worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            _run_job(job_id)
        except Exception as e:  # noqa: BLE001 - surface any pipeline failure to the client
            _set(job_id, state="error", error=str(e))
        finally:
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
            "clips": [],
            "error": None,
            "request": req,
        }
    job_queue.put(job_id)
    return {"job_id": job_id}


@protected.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return {k: v for k, v in job.items() if k != "request"}


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
  #status { margin-top: 24px; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.85rem; }
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

<div id="status"></div>
<div id="clips"></div>

<script>
const statusEl = document.getElementById('status');
const clipsEl = document.getElementById('clips');
let timer = null;

document.getElementById('submit').addEventListener('click', async () => {
  clipsEl.innerHTML = '';
  statusEl.textContent = 'Submitting...';
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
    return;
  }
  const { job_id } = await resp.json();
  if (timer) clearInterval(timer);
  timer = setInterval(() => poll(job_id), 2000);
  poll(job_id);
});

async function poll(jobId) {
  const resp = await fetch(`/api/jobs/${jobId}`);
  if (!resp.ok) return;
  const job = await resp.json();
  statusEl.textContent = `[${job.state}] ${job.message || ''}`;
  if (job.error) statusEl.textContent += `\\nError: ${job.error}`;

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

  if (job.state === 'done' || job.state === 'error') {
    if (timer) clearInterval(timer);
  }
}
</script>
</body>
</html>
"""
