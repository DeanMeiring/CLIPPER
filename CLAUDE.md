# Working with Dean on CLIPPER

## Merging pull requests

Default: **create the PR and stop there.** Dean merges it himself on GitHub.
Do not merge PRs automatically unless he explicitly says so for that specific
PR. This applies across sessions unless he states otherwise in the moment.

## Watching PRs

When asked to watch/babysit a PR (or by default after opening one), subscribe
to PR activity and drive it to green per the standing PR rules — but never
merge it yourself; merging stays Dean's call per the rule above.

- **No hourly "still no change" check-in loops.** Dean told a session to stop
  them. This repo has no CI checks, so a scheduled poll on an open PR finds
  nothing to do; rely on the PR-activity subscription instead, and re-check
  the PR only when an event arrives or he asks.
- Don't commit a new change onto a branch that already backs an open PR —
  it silently joins that PR. Cut a fresh branch from `origin/main`.

## Branches

- Everything uses **flat** paths: `clipper/*.py`, `webapp/main.py`.
- Ship each change on its own branch cut from `origin/main`
  (`git checkout -b claude/<topic> origin/main`), push, open a PR. When a
  change depends on a PR that hasn't merged yet, branch from that PR's
  branch instead and say "merge #N first" in the PR body.
- The old session branch `claude/clipper-github-setup-pyyn2i` on GitHub
  still holds a stale nested-path (`clipper/clipper/*.py`) history whose
  content is all in `main` already. Don't build on it or try to push it:
  it has diverged, and the force-push needed is blocked in auto mode.

## What the app is now

- Dean cut the app back to the core clip pipeline (Sept 2026) to build on
  from there. The Spanish channel (`/espanol`), Weekly Recap and Game
  Recap pages were removed, including their code, and the recap-only
  helpers in `clipper/trending.py`. Pages left: Home (`/`), Analytics,
  Hook Line.
- `CHANNEL_PROFILES` in `webapp/main.py` holds just `main`. The profile
  plumbing (per-channel Twitch logins env var, brand name, mascot accent,
  YouTube token file, `output_language`) was kept on purpose so a new
  channel can be added later as another entry.
- Old Spanish or recap jobs may still sit on the Railway volume. They
  show in the jobs list so they can be downloaded or deleted. Uploading
  a clip from a removed profile, and regenerating a recap job, both
  return 409, so a Spanish clip can never go to the main channel.
- Dean likes how clips are picked and rendered. Changes to shared
  pipeline code must keep that byte-for-byte unless he asks otherwise.
  For the clip-picking prompts in `clipper/select_moments.py` this was
  checked by capturing old vs new prompts and comparing byte for byte.
- The "learns from YouTube stats" notes (`_load_strategy_notes` /
  `_load_performance_notes`) come from the main channel's account. Dean
  especially values this feature — don't break it.

## Long-form: "The Story Of" streamer documentaries

- History: an aviation-incident format (NTSB reports, maps, charts) was
  built, rendered once and judged "quite bad" by Dean, then replaced. The
  research said why: channels in that niche win on 3D recreations and real
  ATC audio, and on a narrator's voice/take, not on generated visuals.
- Now: a bi-weekly documentary series on the **main Caught On Stream
  channel** (same audience as the Shorts, so Shorts can link to it), a
  different streamer each episode, SunnyV2-style ("The Story of X").
- `/long-form` (Home link "Go to long-form videos"), its own page. Code:
  `clipper/documentary.py` (research + story), `clipper/longform.py`
  (projects, scene-by-scene recording, the misread check),
  `clipper/longform_video.py` (render), `/api/longform/*` routes and
  `LONGFORM_HTML` in `webapp/main.py`. Projects: `BASE_DIR/_longform/<id>/`
  (`project.json`, `dossier.json`, `clips/<Cnn>/`, `takes/`, `render/`,
  `final.mp4`); the schedule is `BASE_DIR/_longform/_series.json`.
- Flow: schedule slot (every 14 days) → research (Twitch profile, the
  streamer's most-viewed clips all-time + per year, Wikipedia, pasted
  article links, Dean's notes; clips downloaded with yt-dlp and transcribed
  with word timings) → Claude writes scenes of three kinds: `narrate`
  (Dean reads it, over a clip), `moment` (a clip cut that plays at full
  volume with subtitles), `title` (chapter card → YouTube chapter) →
  editable story → record narrated scenes only (same misread check) →
  render → title/description with chapters + credits → upload to Caught On
  Stream as a regular (not Short) video, private by default → optional
  promo Shorts via a normal clip job on `final.mp4`.
- Script rules that matter: facts only from the research, quotes only from
  clip transcripts, no accusations/private-life speculation, no hype words.
  Commentary over clips is what keeps it on the right side of fair use and
  YouTube's July 2025 "inauthentic content" rule — Dean's own narration is
  the point, never AI voice.
- **Keyword visuals (Sept 2026):** Dean said the voice-over was fine but
  the visuals were the problem: one dimmed clip looping under a 20-40 s
  scene. Now `clipper/longform_beats.py` cuts each narrated scene into
  beats that change on the words he says (every ~3-6 s), timed by
  Whisper word timings of his take (cached as `takes/<file>.words.json`)
  aligned to the script. Cue types: words, emoji, stat, stock, photo,
  clip, post, headline, timeline; between cues his clips play with slow
  zooms and quick cuts; word-by-word captions throughout; animated
  chapter cards. `documentary.plan_visuals` (Claude) picks the cues right
  after the story is written; `normalize_cues` drops anything untrue: the
  phrase must be in the narration, a stat's number and a timeline's years
  must be in the research. Story editor shows them as removable chips.
- Free-only visual sources (Dean: "free, not free to an extent"):
  bundled Fluent Emoji 3D (MIT, 326 picked, `clipper/assets/emoji`),
  bundled fonts Inter/Anton/Source Serif 4 (OFL, `clipper/assets/fonts`,
  opened by path, never registered with fontconfig so Shorts captions are
  unchanged), Pexels + Pixabay (free keys `PEXELS_API_KEY`,
  `PIXABAY_API_KEY`; without them stock video is skipped), Openverse
  limited to CC0/public domain (no key), X posts via the public embed
  endpoint `cdn.syndication.twimg.com` (no key). Credits go in the
  YouTube description automatically. All of these hosts are blocked in
  the sandbox: test with mocks and the scikit-video sample clips
  (`pip download scikit-video`, `skvideo/datasets/data/*.mp4`).
- Rendering is Pillow frames piped into ffmpeg, about 2x real time on
  CPU (a 12-minute episode renders in ~25 min).
- **AI voice (Sept 2026):** Dean narrates with his own voice (never his
  face). `clipper/voice_clone.py` clones it with Kyutai Pocket TTS (100M
  params, CPU, ~0.6 GB RAM, CC-BY-4.0 weights) to patch lines he'd rather
  not re-read. It is NOT meant to replace his reading of whole episodes;
  the page nudges him when over a third of the scenes use it. One sample
  (him reading `SAMPLE_TEXT`) lives in `BASE_DIR/_longform/_voice/`; the
  "🤖 Use my AI voice" button on a narrated scene makes a take with
  `"voice": "ai"`, checked by the same misread check. The cloning weights
  are gated on Hugging Face (free, auto-approved), so Railway needs
  `HF_TOKEN` from an account that accepted the terms at
  huggingface.co/kyutai/pocket-tts. Chatterbox was rejected: ~7.5 GB RAM
  on CPU, and the box has 8 GB that already peaks near full. The
  Dockerfile installs CPU-only torch first. From PyPI it would pull GBs of
  CUDA. Adding pocket-tts changed no existing package version (checked
  with `pip install --dry-run --report`). YouTube doesn't require the
  altered-content label for cloning your own voice for voiceover. The
  model can't be downloaded in the sandbox (HF blocked), so test with a
  fake `pocket_tts` module.
- Quiz format: prototyped (higher-or-lower on clip views, who-said-it,
  guess-the-year) and shelved once Dean said he'll use his voice. The
  documentary is the one long-form format for now.
- Old aviation projects on the volume show as "old aviation test" and can
  only be deleted. pypdf/cryptography and the map/chart/Pexels code were
  removed with that format.
- The Docker image isn't changed for fonts on purpose: ffmpeg's
  dependencies already bring DejaVu, and adding fonts could change the
  clip captions' font fallback. `longform_video.font()` falls back to
  `fc-match`, then Pillow's built-in font.
- Twitch, Wikipedia and ntsb-style external hosts are blocked from the
  Claude Code sandbox; test with mocks. A standalone ffmpeg for local
  render tests: `pip install imageio-ffmpeg` and point `CLIPPER_FFMPEG` at
  `imageio_ffmpeg.get_ffmpeg_exe()`.

## Clip layout

- Clips render in the IRL layout by default (`reframe.LetterboxLayout`,
  the whole scene letterboxed; `JobRequest.irl_layout=True`). Each clip's
  "Switch to facecam" button opens the manual box-picker, and "apply to
  the other clips" converts a whole job. Unticking "IRL layout" brings back
  automatic facecam detection (`compute_layout`). Dean chose this default.

## Gotchas

- ASS subtitle colours are `&HBBGGRR` (blue-green-red), not RGB: `00CCFF`
  is yellow. Render a frame and look before claiming a colour.
- Sandbox: commands that `pkill`/background a server often exit 144
  without having failed — re-check state instead of retrying blindly.
  `rm -rf "$VAR"/…` is refused by a safety check; use literal paths.
- After checking out another branch, a stray `webapp/__pycache__` can be
  left behind; it's untracked and safe to delete.
- `_persist(job_id)` takes `jobs_lock` itself, and `jobs_lock` is a plain
  (non-reentrant) `threading.Lock`. Call it *after* leaving a
  `with jobs_lock:` block, never inside one — inside, the request deadlocks.
- Words in the clip-picking prompt leak into generated titles/descriptions.
  That includes the mood buttons' `data-mood` text, which is sent verbatim
  as the `focus`. "banter", "insane", "chaos" etc. in prompt text made the
  output sound AI-written; keep hype words out of anything the model reads.

## Testing without ffmpeg or API keys

- `import webapp.main` is fast (it starts daemon worker threads). Set
  `CLIPPER_JOBS_DIR` to a temp dir, leave `APP_PASSWORD` unset (auth is
  skipped), then use FastAPI's `TestClient` and patch the ffmpeg/Claude
  helpers as `webapp.main.<name>` (e.g. `trim_clip`, `_render_atomic`).
- A hung request: run under `timeout` with
  `faulthandler.dump_traceback_later(N, exit=True)` to see where it's stuck.
- Page JS lives in Python triple-quoted strings (`_CHANNEL_HOME_TEMPLATE`,
  `*_HTML`). To syntax-check it, evaluate the string with
  `ast.literal_eval` first, then `node --check` the `<script>` body. A raw
  regex extract keeps `\\'` escapes unprocessed and reports false errors.

## Railway deploys

- Project `12720d82-23db-498d-b4f6-2a69cf5fff42`, service
  `978f7dca-a5bf-4663-8446-ed3a6625841f`, environment
  `a30ad832-9974-4645-bed0-37002a3af844` (production).
- Merges to `main` auto-deploy. After a merge, check `get-logs` /
  `environment-status` to confirm the deploy went out clean before telling
  Dean it's live.

## Other standing preferences

- No paid signups/subscriptions for libraries or tools — check for a free
  option first (e.g. plain CSS charts instead of a charting library).
- Dean posts ~2 clips/day, chosen manually by eye — factor this into any
  quota/cost estimates (YouTube API, Claude API, Railway usage, etc.).
- He prefers a separate page per feature/channel (linked from the top nav)
  over controls bolted onto an existing page.
- For UI changes he likes a mockup first when the change is big, then a
  live-browser check (Playwright screenshots) before the PR.
