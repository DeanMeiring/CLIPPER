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

## Long-form videos (second channel)

- Dean is starting a second YouTube channel for 10–12 minute aviation
  incident stories (suggested name "Seconds to Decide"), promoted with
  Shorts cut from each long video. The Stable Ronaldo / Jynxzi Shorts
  channel carries on unchanged alongside it.
- `/long-form` (linked from Home as "Go to long-form videos") is its own
  page, no clip controls on it. Code: `clipper/longform.py` plus the
  `/api/longform/*` routes and `LONGFORM_HTML` in `webapp/main.py`.
  Projects live in `BASE_DIR/_longform/<id>/` (`project.json`,
  `report.txt`, `takes/`, `narration.wav|m4a`).
- Flow: pick an incident (curated NTSB reports in `longform.INCIDENTS`, or
  paste a report PDF link / upload the PDF) → Claude writes a scene-by-
  scene script from the report (facts from the report only, no invented
  dialogue, no victim names) → Dean records each scene in the browser →
  Whisper transcribes the take and `check_take` compares it with the
  script: a run of 3+ skipped/misread words, 3+ added/repeated words, or
  low overall match flags the take and he reads that scene again (or
  "Keep anyway"). Numbers, filler words and a single misheard word are
  ignored on purpose. Accepted takes are joined into one narration track
  with known scene start times.
- Still to build: visuals per scene (route maps, cockpit-transcript cards,
  charts, free Pexels/Pixabay stock, NTSB report images) timed to the
  narration, 1080p render, upload to the second channel's YouTube account,
  and promo Shorts. No AI images of real crashes and no AI voice.
- `ntsb.gov` is blocked from the Claude Code sandbox but reachable from
  Railway; test with mocks (see the testing section).

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
