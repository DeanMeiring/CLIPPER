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
  Hook Line; since added: Rankings (`/rankings`) and Voices (`/voices`).
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
- **Ranking Shorts (Oct 2026):** Dean liked the "Ranking <streamer>'s Funny
  Moments" format (a 1-5 list on screen, filled in out of order, #1 last)
  and asked for it after views declined. Researched before building: the
  format keeps viewers to the end and gets comments, but a list with
  labels only is the "clips edited together with little or no narrative"
  YouTube's reused-content rules and the 1 Oct 2026 change name. So each
  moment ends with a short spoken line (Dean, or a voice from `/voices`)
  placing it: that's the commentary. `/rankings` (in the top nav):
  pick 3-5 finished clips whose downloaded source is still in
  `<job>/_source/` → `clipper/ranking.py` `plan()` (Claude: rank, 3-9 s
  cut, 2-4 word label, emoji from the bundled Fluent set, the narrator's
  line; `normalize_plan` keeps it inside each clip) → edit / reorder /
  record or AI-voice each line (same misread check) → `render()`: one
  segment per moment in the reveal order (shuffled, #1 last), the
  narrator's line over a freeze of the moment's last frame, the label pops
  in as he says it, captions in the Shorts' ASS style; wide streams sit
  under the list, vertical ones fill the screen with the list over them.
  Fits in 58 s by trimming moments from their start. The video lands on
  Home as its own done job (`job["ranking_id"]`, clip `ranking`/`promo`,
  `synthetic_voice` when a friend's voice read a line); a re-render
  replaces it until posted. Projects: `BASE_DIR/_rankings/<id>/`. Posted
  rankings are kept out of `clip_performance.build_stats` (the picker's
  learning) and shown on their own (`data["rankings"]`,
  `GET /api/rankings-compare`: views + stayed-to-watch vs the normal
  Shorts' middle). Chat-reaction numbers were pitched but not built: chat
  replay fetching fails often in production logs.
- **Learning upgrade (Oct 2026):** Dean saw Shorts views declining and
  asked the picker to learn more from past videos. `clip_performance`
  now also gets `youtube_analytics.get_video_engagement` (engagedViews,
  likes, comments, shares per video; falls back without engagedViews):
  "stayed to watch" = engagedViews / views, the API's nearest thing to
  Studio's "viewed vs swiped away" (since March 2025 a Shorts view counts
  every start or replay). `build_learning` adds the last 2 weeks vs the 2
  before, the 6 best and 6 worst Shorts of the last 60 days (with moment
  type, hook text, length, Claude's own score) and per-streamer momentum
  (streamer = first `TRENDING_TWITCH_LOGINS` name in the title or source
  title; recent = last 3 weeks). All of it goes into the performance notes
  the picker and the AI overview read (`render_prompt_text`), and into a
  "What the clip picker learns from" block on Analytics. The picker
  prompt's own wording is unchanged.
- **Short clips (Oct 2026):** Dean noticed shorter clips win. Checked on
  the last 47 Shorts (vidIQ): <= 20 s median ~3.8k views, 6 of 7 over 2k;
  21-60 s median ~1.3-1.5k. Clip length now defaults to 10-20 s
  (`JobRequest` / `RegenerateRequest` / Generate more / the Home boxes);
  the picker prompt only changes in those two numbers.
- YouTube announced on 1 Oct 2026 that the Shorts feed will cut reach for
  channels that mainly re-upload others' clips without significant changes
  (voice-over describing the clip, minor edits and templates don't count;
  own commentary, analysis, storytelling do; permission doesn't help).
  This is the biggest outside risk to the clip channel; Dean was told.

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
  Stream as a regular (not Short) video, private by default → 2
  **cliffhanger** promo Shorts start automatically on upload
  (`clipper/longform_promo.py`, Dean asked for clips that leave viewers
  wanting the full video): cut from the episode's own structure -- the
  cold open + the question it raises, and a narrated setup + 1.4 s of the
  clip it leads to (Claude picks the strongest turning point and writes
  the hook/title) -- vertical, question on top, "What happened next? Full
  story on the channel" end card, under 60 s. They land on Home as a
  finished job whose clips have `"promo": True`, which keeps them out of
  the clip registry so they don't skew the Shorts' "learns from YouTube
  stats" data. The job has `job["promo_for"] = {url, title}`: the Home upload
  window adds "Full story: <url>" to their descriptions, and a posted one
  links to its YouTube Studio edit page to set "Related video". YouTube's
  API can't set a Short's Related video, so that one click stays manual.
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
- **Lessons (Sept 2026):** `clipper/longform_lessons.py` keeps editing
  lessons in `BASE_DIR/_longform/_lessons.json`, seeded from YouTube's
  editing review of the first episode (StableRonaldo): long opening
  laughter, a moment opening on a static desktop, a loud clip cut
  straight into serious news, an 11 s still end card. They go into the
  story and visuals prompts (`prompt_block`); Dean pastes new reviews or
  notes on /long-form and Claude merges them (max 15, removable). Clip
  transcripts given to Claude mark "(silence Ns)", and
  `documentary.tighten_moments` trims dead air off Claude's moment picks
  (≤0.8 s before the first word, ≤1.5 s after the last, cold open ≤ ~10 s)
  when a story is written -- never Dean's own edits. Dean rated that
  first rendered episode "really good".
- **Thumbnails (Sept 2026):** `clipper/longform_thumbnail.py` makes 3
  options on the Render & post step: `face` (face big on the right, hook
  left), `full` (face big in the middle, hook along the bottom) and `split`
  (then vs now: oldest vs newest face frame, two years). Dean found the
  first version "not that good": it shrank frames to 720p before looking
  for faces, so it picked small faces with the chat overlay and a
  gameplay HUD with no face at all. Now frames are grabbed at the clip's
  own resolution (max 1920 wide), and only frames with a real face are
  used (OpenCV Haar, then checked by MediaPipe BlazeFace: score >= 1.4,
  which rejects game characters and emotes). The crop is tight on the
  face (zoom capped at 2.5x), which also cuts out the chat and timers.
  The person is cut out with MediaPipe selfie segmentation over a blurred,
  darkened background, with a white outline. Both models are tiny
  `.tflite` files (Apache 2.0) bundled in `clipper/assets/models`, copied
  from the mediapipe wheel and run with `cv2.dnn.readNetFromTFLite`, so
  there are no new packages. GitHub's copies are Git LFS, which the
  sandbox can't fetch. Test faces: `skimage/data/astronaut.png` from the
  scikit-image wheel. Claude writes 1-3 word hooks (max 22 characters, no
  in-jokes or clip quotes). A number in a hook must appear in the research
  or the story, or the hook is dropped. Dean can edit the text and redraw,
  pick one, or download it. The chosen one is set with `thumbnails.set`
  right after the upload (`youtube_upload.set_thumbnail`). A failure never
  fails the upload: it lands in `project["thumbnail_status"]`, and "Set on
  YouTube" retries. Custom thumbnails need a phone-verified channel;
  YouTube answers 403 without it.
- **Analytics (Oct 2026):** the Analytics page has a "Long-form episodes"
  panel (`GET /api/longform-analytics`, cached 10 min like the Shorts
  panel; `clipper/longform_analytics.py` builds it from
  `youtube_analytics.get_video_stats / get_retention_curve /
  get_video_daily / get_video_traffic`). For each uploaded episode it
  shows views, watch time, average view, the share still watching at
  0:30, subscribers, the thumbnail click rate (when YouTube reports it),
  and retention vs similar videos. The retention line carries numbered
  chapter markers taken from `render.starts`, and the biggest drops after
  0:30 are named with the scene playing then. It also shows views per
  day, traffic sources and how the episode's cliffhanger Shorts did. The
  upload now stores `uploaded_at`, which sets the start of the reporting
  window. Charts are hand-made SVG (no chart library), drawn at their real
  width so the labels stay readable on a phone.
- Rendering is Pillow frames piped into ffmpeg, about 2x real time on
  CPU (a 12-minute episode renders in ~25 min). Scenes are cached by a
  hash of their inputs, so a change to the final mix only (music, music
  volume) re-renders in a minute or two.
- Background music: levelled with loudnorm, then set to
  `longform_video.MUSIC_LEVELS` (quiet -37 / normal -34 / loud -31 LUFS,
  ducked under voice and clips). Dean found the first flat 0.16 gain "a
  bit too loud"; the page has Quieter/Normal/Louder buttons
  (`project["music_level"]`) and "🎵 Update music only"
  (`POST .../render?music_only=true`), which reuses every finished scene
  and refuses with the changed scene numbers (`changed_scenes`) rather
  than turning into a long render. Dean didn't want a music change to
  re-render the whole video.
- **Sound sliders (Oct 2026):** Dean asked to adjust his voice and the
  music loudness. Narrated takes used to go in raw (whatever the mic gave),
  so an AI-voice take and his own could differ a lot. Now the final mix
  levels each narrated scene's take to `VOICE_TARGET` (-16 LUFS, what the
  clip moments are loudnormed to) with a plain gain over that scene's time
  range (measured once with ebur128, cached as `takes/<file>.loudness.json`;
  capped so peaks stay under -1 dBFS). The page has two sliders: voice
  `project["voice_db"]` (-8..+8 dB from levelled) and music
  `project["music_db"]` (-9..+9 dB around the old "normal" -34 LUFS; the old
  `music_level` maps to -3/0/+3). `PUT .../mix`; "🔊 Update sound only"
  is the old music-only re-mix, so no scene renders again.
- **AI voice (Sept 2026):** Dean narrates with his own voice (never his
  face). `clipper/voice_clone.py` clones it with Kyutai Pocket TTS (100M
  params, CPU, ~0.6 GB RAM, CC-BY-4.0 weights) to patch lines he'd rather
  not re-read. It is NOT meant to replace his reading of whole episodes;
  the page nudges him when over a third of the scenes use it. One sample
  (him reading `SAMPLE_TEXT`) lives in `BASE_DIR/_longform/_voice/`; the
  "🤖 Use my AI voice" button on a narrated scene makes a take with
  `"voice": "ai"`, checked by the same misread check. Oct 2026: Dean asked
  for "🤖 Let my AI voice read everything" (Record step, both series):
  `POST/DELETE .../clone-all` runs every narrated scene without a usable
  take through `_clone_take` in a background thread (one at a time under
  `_voice_busy`), keeps scenes he recorded, retries a flagged take once,
  and reports progress/flagged/failed in `project["ai_all"]` (scene
  numbers; the page shows them as narration numbers). Stop finishes the
  current scene. The cloning weights
  are gated on Hugging Face (free, auto-approved), so Railway needs
  `HF_TOKEN` from an account that accepted the terms at
  huggingface.co/kyutai/pocket-tts. Chatterbox was rejected: ~7.5 GB RAM
  on CPU, and the box has 8 GB that already peaks near full. The
  Dockerfile installs CPU-only torch first. From PyPI it would pull GBs of
  CUDA. Adding pocket-tts changed no existing package version (checked
  with `pip install --dry-run --report`). YouTube doesn't require the
  altered-content label for cloning your own voice for voiceover. The
  model can't be downloaded in the sandbox (HF blocked), so test with a
  fake `pocket_tts` module. Dean: it sounds right but "butchered numbers",
  so `voice_clone.spoken_text` spells numbers out before TTS ("2019" ->
  "twenty nineteen", "4.3M" -> "four point three million", "March 14" ->
  "March fourteenth"); the script and captions keep digits. The Railway
  variable was once saved as " HF_TOKEN" with a leading space: if the page
  says it isn't switched on, check the name (Railway MCP `list-variables`
  shows names only).
- **Friends' voices (Oct 2026):** Dean's friend has a voice he wanted
  for narration, agreed to it, and lives far away. `/voices` (linked from
  long-form's "Your AI voice" box) lists Dean's own voice plus other
  people's, each in `BASE_DIR/_longform/_voices/<8 hex>/` (`sample.wav`,
  `sample.json`, `voice.json` with name + consent). Adding one needs the
  "they agreed" box ticked. The sample is an uploaded voice note (any
  format ffmpeg reads; silence at the start is cut and the first 45 s
  kept; it only has to be clear talking, >= 15 words) or recorded by the
  friend on a private link `/voice-sample/<token>`: public on purpose (no
  app password), the token is the key, works 7 days, only sets that one
  voice's sample, and needs their own "I'm OK with it" tick (stored as
  `consent.self_confirmed_at`). A long-form episode picks its AI voice
  (`project["ai_voice"]`, "me" by default; `PUT .../ai-voice`); takes read
  by someone else's voice carry `voice_id`. Any upload using another
  person's cloned voice sets YouTube's `status.containsSyntheticMedia`
  (`youtube_upload.upload_video(synthetic_media=True)`; the long-form
  upload, its promo Shorts via `clip["synthetic_voice"]`). Dean's own voice
  doesn't need the label, and other uploads send exactly what they did.
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

## Caught On Code: tech explainers (Oct 2026)

- A second long-form series and YouTube channel: the tech behind gaming,
  streaming and the internet (netcode, anti-cheat, matchmaking, stream
  latency, AI voice cloning...). Dean narrates, no face. The niche came
  from research: 3Blue1Brown, Kurzgesagt, Branch Education, Real
  Engineering and The B1M already own the big science topics, but
  gaming-tech questions players search for have no channel that owns them.
  It's the same audience as Caught On Stream.
- Page `/caught-on-code` (linked from Home). The page is the same
  template as `/long-form` (`_LONGFORM_TEMPLATE`, built per series by
  `_longform_page`). JS branches on `SERIES` / `EXP`, and elements marked
  `doc-only` / `exp-only` are hidden by CSS. Projects are in the same
  long-form store with `kind: "explainer"`
  (`/api/longform/projects?series=`). Recording, the misread check,
  render, music, thumbnails, upload and analytics routes are shared; they
  branch with `_is_explainer`.
- `clipper/explainer.py`:
  - Research: Wikipedia top articles, pasted links, notes.
  - The script prompt: narrate scenes, each with 1-4 `visuals`, plus 2-3
    "pause and guess" quiz scenes. Dean found the first episodes had too
    few scenes (one diagram held 20-40 s), so scenes are now 20-60 words
    and the diagram changes about every 8-10 s (the prompt asks for
    ~words/40 scenes and ~words/22 diagrams).
  - `normalize_scenes` / `normalize_visual`: a number on screen must be in
    the research, or be simple maths (`"math": "1000 / 128"`) on numbers
    that are, plus unit constants (60, 1000...). Unsourced race timings
    get an "EXAMPLE TIMINGS" tag. Thumbnail hooks get the same number
    check.
  - `quiz_suggestions`: YouTube Studio in-video quizzes are added by
    hand; there's no API.
  - Publish text and diagram-based thumbnails (layouts left / center /
    bottom).
- `clipper/explainer_visuals.py`:
  - Templates: flow, network, race, bars, bignum, grid, layers, neural,
    compare, quiz, words, arena.
  - `arena` (Oct 2026, Dean wanted more than diagrams and liked a sample):
    a top-down game simulation, slowed down (`_nice_slow`) and always
    tagged "SIMPLIFIED · SLOWED DOWN N×". Modes: `peek` (peeker's
    advantage; `delay_ms` head start, timeline), `ticks` (server
    snapshots every 1000/rate ms; 1-2 `tick_rates`), `rewind` (lag
    compensation: you shoot his lagging image, he's behind a wall on his
    screen, server rewinds `delay_ms`, hit). `normalize_visual` checks
    `delay_ms` / `tick_rates` against the research (unsourced: "EXAMPLE
    NUMBERS" tag) and drops a `note` with an unsourced number. Line of
    sight is computed (`_blocked`), so the moment he becomes visible or
    hidden is real geometry, not hand-timed.
  - Pillow drawn at 2x and reduced, about 1.8x real time on CPU.
  - Each element appears on its `at` phrase, matched against the take's
    word timings (`longform_beats.script_timings`). Captions and
    watermark are the documentaries'.
  - `longform_video.render_scene` dispatches on `scene["visuals"]`.
    `_scene_key` only adds `visuals` when present, so documentary
    scene caches are unchanged (checked).
- **Streamer clips (Oct 2026):** `clipper/clip_search.py` finds real
  Twitch moments of what an episode explains, to play as `moment` scenes
  (credited `twitch.tv/<name>` on screen and under "Clips" in the
  description). Twitch can't search clips by keyword, so: Claude names the
  games + title words, `game_clips` reads each game's top English clips
  (all time + six 2-month windows), `shortlist` keeps matching titles,
  Claude picks up to 8, `documentary.build_library` downloads and
  transcribes them, and `check` (Claude, on the transcripts) sets
  `fits` / `what`; `use` starts as `fits`. Pasted Twitch clip links are
  always `use`d. It runs on its own after an episode's first research
  (needs `TWITCH_CLIENT_ID`), or from "🔎 Find clips" (`POST
  .../clip-search`); Dean ticks clips (`PUT .../library/{id}`), and only
  ticked ones reach the script prompt (`explainer._clips_block`; with no
  clips the prompt is byte-identical to before). Explainer chapter cards
  stay plain (`_scene_plan`), and topics that never happen on stream (DLSS)
  just find nothing.
  - First real run: Dean found the picks "unrelated". Now (a) Claude's
    title words must be situation-specific, not generic ("server", "lag",
    "100"), (b) pick and check both drop anything unsure -- an empty list
    is fine, (c) YouTube is searched too (`youtube_search`: Claude writes
    3-4 queries, `videoDuration=short`, kept if <= 4 min; other channels'
    explainers/news/reactions are skipped in the pick prompt) with the
    episode's channel OAuth token (~400 of the 10,000 daily units), and
    YouTube links can be pasted (<= 15 min). Credit reads
    "YouTube · <channel>". YouTube clips carry more Content ID risk than
    Twitch clips; Dean was told. Clips can be removed (`DELETE
    .../library/{id}`, refused if the script uses it).
- Its own YouTube account, `EXTRA_YOUTUBE_ACCOUNTS["code"]` (token file
  `_youtube_oauth_token_code.json`). It's deliberately not in
  `CHANNEL_PROFILES`, which would add it to Home's channel switcher and
  the Shorts pipeline. Connect with `/auth/youtube/login?profile=code`;
  the callback returns to `/caught-on-code`. Upload, thumbnail and
  analytics pick the account per episode (`_longform_account`). There
  are no cliffhanger Shorts for explainers (no clip moments); the route
  returns 409.
- Manim was used for the first pilot clip (sandbox only: needs Pango dev
  libs and a venv). The app uses the Pillow templates instead, so there's
  no new dependency or Docker change.

## The Deniboy: Dean's own Rocket League channel (Oct 2026)

- Dean plays Rocket League (SA Grand Champ, channel "The Deniboy") and has a
  backlog of his own clips to post as one Short a day. Home has "🚗 Go to my
  Rocket League channel" → `/rocket-league` (`RL_HTML`), code in
  `clipper/rocket_league.py`, routes `/api/rl/*`, data in `BASE_DIR/_rl/`
  (`clips/<id>/` with `source.*`, `clip.json`, `short.mp4`; `music/` +
  `music.json`; `settings.json`). Nothing shared with the clip pipeline.
- Its own YouTube account `EXTRA_YOUTUBE_ACCOUNTS["rl"]` (token
  `_youtube_oauth_token_rl.json`; connect with
  `/auth/youtube/login?profile=rl`, the callback returns to the page).
- Flow: upload many clips (streamed to disk, ≤ 2 GB each) → `find_goal`
  (loudest burst in the game audio: explosion + horn) sets the goal second
  and a ≤ 59 s window around it → Dean types the text (also the YouTube
  title) → `render`: full screen (middle of the frame; default) or whole
  frame over a blur, the text on top, his name at the bottom, a white flash
  + quick zoom on the goal, the song's drop (`find_drop`: biggest jump in
  bass energy, editable per song) lined up with the goal, game sound
  loudnormed under the music (quiet/medium/loud) → "📅 Schedule" uploads
  it private with `publishAt` on the next free day at his time (SA time,
  default 17:00; `youtube_upload.upload_video(publish_at=...)`), so YouTube
  posts it even while the Railway app sleeps. Max 4 uploads a day
  (`RL_UPLOADS_PER_DAY`): the 10,000-unit quota is shared with Caught On
  Stream. After upload the source is deleted; the Short is deleted 3 days
  after it goes public (`RL_KEEP_DAYS`) -- the volume is near the 5 GB cap.
- Music is his own uploads ("Auto" rotates the least-used song). Free
  tracks (NCS, free phonk) keep a Short earning; the song's credit is
  added to the description. A label's song gets a Content ID claim: under
  60 s the Short stays up but the label takes its ad money. Dean was told.
- Trimming is done on the original (Dean asked): a timeline under the
  video (kept part, goal in orange, playhead; tap to seek) with Start here /
  End here / Goal here / Play the cut; a made Short can be flipped back to
  the original. The Music box links to NCS, Pixabay free phonk and the
  YouTube Audio Library. Uploads warn before leaving the page and "Connect"
  opens in a new tab: on the first real try, tapping Connect mid-upload cut
  the upload off (ClientDisconnect, now a quiet 400). The sandbox Chromium
  can't play H.264, so browser tests of the player need a WebM clip.
- His branding (pfp from his Steam avatar or a "D" logo, banner with an
  original GC badge, not Psyonix's rank icon) was made in the sandbox, not
  in the app.

## Ball Evolution: physics Shorts (Oct 2026)

- Dean asked for a "satisfying" format: a ball falls slowly through a small
  opening, multiplies when it hits certain points, and keeps going until
  the jar is full, with a sound on every hit. Researched first: a real
  niche (marble/ball simulation channels), but the "millions of views"
  claims come mostly from people selling templates, and YouTube's July
  2026 "generic or repetitive content" wording won't monetize
  template-looking videos with little variation, so each one must differ.
  It doesn't fit Caught On Stream's audience; a separate channel if posted.
- He then asked for Cow Evolution-style merging (same things combine into
  the next, bigger one), with audio, keeping the first design. Both
  prototypes were rendered in the sandbox; he said "I like it, it's good".
- `clipper/ball_evolution.py` (CLI: `python -m clipper.ball_evolution
  out.mp4 [--seed N]`): bees drip from a hole, gold pegs send a copy back
  out of the hole, and in the jar matching animals merge: bee > mouse >
  frog > chicken > cat > dog > panda > lion > unicorn (bundled Fluent
  Emoji, not Cow Evolution's art). A ladder at the top reveals each animal;
  it ends on "EVOLVED!" at the unicorn. All audio is generated in numpy
  (see "Noise audio" below), so nothing can be claimed.
- Physics is pymunk (added to requirements; `pip --dry-run --report`
  showed no other package version change, only cffi/pycparser added).
  Fixes from the prototype: a wider neck plus "the hole waits while 90
  are still falling" stopped bees piling up over the pegs; frogs and up
  pull toward their nearest match in the jar (force applied every
  substep: pymunk clears forces after each step), which stopped runs
  stalling with two lions apart. Runs still vary, so `pick_seed`
  simulates seeds without drawing (~1 s each, in parallel) and keeps one
  with the unicorn at 40-62 s and no wait over 16 s between reveals
  (about 1 in 5 seeds). A 55 s video renders in ~3 min on 4 CPUs.
- **Variety (Oct 2026):** Dean will post almost daily on a new channel and
  asked how to make each video unique, plus sound for the balls. Every
  video is now a *recipe* (`make_recipe` / `pick_recipe`): theme (8
  evolution chains from the bundled emoji: animals, sports, food, space,
  money, laughs, vehicles, weather -- each with its own colours, hooks and
  starter item), course (`pegs`, `triangle`, `spinners` = rotating
  kinematic bars, `ramps` = short steep deflectors over pegs, `bumpers`),
  jar (`box`, `bowl`, `flask`), ending text, and music key/scale/tempo.
  `--history file.json` keeps the theme off the last 3 videos and never
  repeats a theme+course+jar combo. Ball sound: every peg hit makes a
  sound, walls/ramps/spinners/bumpers knock, a first landing in the jar
  thuds. Rate-limited per peg (0.1 s) and per sound type in the mix.
- **Noise audio (Oct 2026):** Dean found the musical audio (marimba notes
  per peg, chimes, fanfare, a four-chord music loop) "horrible" and asked
  for "more white noise". Now every sound is band-limited noise
  (`_noise`/`_burst`, FFT-shaped, numpy only): marble-like taps per peg
  (a bit brighter to the right, 3 takes each), knocks, a soft puff per
  merge that deepens with size (+ a low thump from the 4th item up), a
  whoosh landing on a thump for each reveal, a long whoosh + deep boom for
  the last item, all over a quiet noise bed (rain / air / hush, picked per
  video). No music, no melodies. The recipe's key/scale only seed the bed
  choice and a small per-video pitch shift now. Don't bring tonal music
  back without asking him. Then he asked for an option to drop the noise
  bed and keep only the balls' sounds: the page's "Background sound"
  setting (`settings["bed"]`: auto / rain / air / hush / off, default
  auto) goes into each recipe as `recipe["bed"]`; "off" makes the bed
  silent.
- **More variety (Oct 2026):** Dean will run the channel fully automated
  (2 videos a day) and asked for "at least 50 different" themes plus
  different ball sounds. `clipper/ball_themes.py` holds 57 chains
  (8-9 emoji each, own hooks + counter word), built from the Fluent Emoji
  3D set: 265 more emoji were copied in from the same npm package
  (`@lobehub/fluent-emoji-3d`, MIT, ~6 KB each; `code_for` finds a file
  with or without fe0f). Themes without `bg` get one from their last
  item's colour (`_auto_bg`). Each recipe also picks `skin` (bubble /
  glass / plain / neon), `backdrop` (gradient / glow / stars / grid /
  dots) and `sound` (marble / glass / wood / plastic / rubber / water /
  metal / pop: short percussive hits from `_pack_hit`, no melody).
  `pick_recipe` scores candidates: theme not in the last ~20, new
  theme+course+jar combo, and course/skin/backdrop/sound different from
  the previous video. 30 picks in a row gave 30 different themes.
- **Circle formats (Oct 2026):** Dean asked for two more video types,
  rotated by day with the evolve one. `clipper/ball_circles.py`
  (`recipe["format"]`: `escape` / `touch`; `ball_evolution._parts`
  dispatches simulation + painter, so picking, rendering and audio are
  shared). `escape`: a spinning ring (2.5-3.2 rad/s) with a gap (0.75-0.9
  rad), gravity 700, elastic balls; each ball that gets out spawns two in
  the middle; full = 0.42 x (ring/ball radius)^2 = 145 balls; at full the
  ring stops with its gap at the top and the count holds. `touch`: closed
  ring, zero gravity, speeds held at 280-650 px/s with a tiny random turn
  each step (without it two balls can loop forever and never meet); two
  touching balls spawn one between them, then rest 1.2 s; full = 173.
  Theme chain = milestones on a log scale up to full (new balls come out
  as the next item; ladder + "NEW:" banner). Tuning lessons: escape was
  ~65 s with a slow spin / small gap; now 35-45 s; touch 26-38 s.
  `good_end` for both is 25-60 s.
- **Gates, wheel, Halloween (Oct 2026):** Dean asked for multiplier and
  spinning-wheel versions of Evolve, and Halloween themes. Two more
  courses in `COURSES`: `gates` (three rows of sensor segments,
  collision_type 4, labelled ×2 / +1 / +2 and shuffled per seed; a tier-0
  item passing one spawns that many copies just under it, each gate once
  per item, copies inherit `hit`; no gold pegs: the hole drips one item
  every `GATE_DRIP` 1.1 s instead; counter reads "made"; a "×2" label pops
  where items pass, at most one per gate per 0.35 s, faded in `Sim.step`)
  and `wheel` (one kinematic body: three bars through a solid 70 px hub =
  six spokes, slick spokes; with four bars and grippy spokes items rode
  the pockets by the hub forever and runs stalled; gold pegs above and in
  columns beside it). Both reach the end in the good window on about half
  of seeds, which `pick_seed` handles. Halloween pack in `ball_themes`
  (haunted, witch, graveyard, monsters, trick_or_treat, plus the old
  spooky; 14 more emoji copied in from the same Fluent package);
  `ball_themes.SEASONS` (October = Halloween, December = winter) gives
  those themes +3 in `pick_recipe`, about 40% of picks in October. Long
  hooks now shrink to fit the width.
- **Gumball (Oct 2026):** Dean found the box course + narrow neck + jar
  "weird and difficult to follow" and picked the gumball from five drawn
  shapes (tube, gumball, hourglass, round flask, vase). `JARS` is now just
  `gumball` (old box/bowl/flask in `OLD_JARS` still render old recipes
  exactly as before): one outline, a 260 px neck from the hole down into a
  globe (`GUMBALL` cx 540, cy 1100, r 520). `_build`'s `Y()` maps every
  course's 400..1030 band into 665..1085 and `inside()` keeps pegs,
  spinners, bumpers, ramps, gates (as wide as the globe at that height) and
  the wheel inside it. Lessons: pegs right under the neck made wedges
  (band moved down); squeezed staggered rows got closer than an item and
  formed a mesh (gumball pegs keep >= 2r+40 apart); spinners flicked items
  out of the open neck, and a lid made items roll off its outside, so an
  item flung back up the neck simply vanishes "into the hole"; fewer pegs
  fit, so more are gold (x1.3, max 0.62); gates' drip starts at 1.5 s and
  speeds up (a steady drip left a long wait for the last item). `Sim.geo`
  (`_geo`) holds the merge / landed / pull / wait-count lines and the
  counter position per machine. 30-80% of seeds per course are good.
- Course tuning lessons: long full-width ramps were too slow (~16 s to
  roll down three); balls rest forever in any gap narrower than a ball
  (peg-wall pockets, peg pairs, ramp ends at a wall), so `peg()` skips
  pegs within a ball's width of a wall. 8-item chains finish faster, so
  `good_end` is 28-52 s for them, 38-62 s for 9. Most seeds that fail do
  so on a long wait before the last reveal (`MAX_UNLOCK_GAP` = 18 s).
- Dean suggested famous actors or streamers as the evolving items. Told
  him: actors no (photo copyright + right of publicity + impersonation
  rules); streamers only with their permission (their pfps/emotes are
  their art). Emoji themes instead.
- **Page (Oct 2026):** `/ball-evolution` (`BALLS_HTML`, Home link "🫧 Go
  to Ball Evolution"), routes `/api/balls/*`, store
  `clipper/ball_channel.py` in `BASE_DIR/_balls/` (`videos/<id>/video.json`
  + `video.mp4`, `settings.json`). Its own YouTube account
  `EXTRA_YOUTUBE_ACCOUNTS["balls"]` (token `_youtube_oauth_token_balls.json`,
  connect with `/auth/youtube/login?profile=balls`). "Make" queues 1/3/7
  videos (theme/course/jar or Auto); one background thread makes them in
  turn by running `python -m clipper.ball_evolution --recipe <json>
  --progress` as a subprocess (keeps the 3-5 min render off the web
  server's GIL; PROGRESS/RESULT lines drive the page's progress bar), and
  `pick_seed` uses "spawn" workers (max 4) because forking a threaded
  server can deadlock. The history for `pick_recipe` is the recipes in
  every video.json, kept after the mp4 is gone. A make interrupted by a
  restart is queued again from scratch on the next page load; Cancel kills
  the subprocess. Upload/schedule mirrors Rocket League (`publishAt` on
  the next free posting slot, `_balls_next_free_slot`), max 2
  uploads/day here (`BALLS_UPLOADS_PER_DAY`; the 10,000-unit quota is
  shared by every channel), category Entertainment (`upload_video
  (category_id="24")`; every other upload still sends Gaming), mp4 deleted
  3 days after it's public. Title = the hook + the first item's emoji;
  description never names the last item (no spoiler). "Name on the
  videos" (watermark) is drawn bottom-right on videos made after it's set.
  Uploads are `selfDeclaredMadeForKids: False` like every upload here;
  Dean was told to decide whether this channel is "made for kids".
- First real upload (Oct 2026) failed with "connection expired" while the
  page said connected: `upload_video` turned every 401 into that message.
  Now `_unauthorized_message` reads YouTube's reason (`youtubeSignupRequired`
  = the Google account has no channel, e.g. the channel is a brand account
  and the personal account got picked). The YouTube login now uses
  `prompt=consent select_account` (always shows the account picker), the
  callback stores the connected channel (`token["channel"]`, `{"none":
  true}` without one), and /ball-evolution shows its name with Reconnect /
  Disconnect links.
- **Autopilot (Oct 2026):** Dean wants the channel fully automated: 2
  videos a day at 08:00 and 15:00 SA time, YouTube + Instagram, the kind
  rotating by day (`BALLS_FORMAT_ROTATION` evolve -> escape -> touch,
  `date.toordinal() % 3`, both slots that day). Settings `autopilot`,
  `slots` (1-4 HH:MM), `auto_instagram`, `ahead_days` (2).
  `_balls_autopilot_tick` (non-blocking lock): (1) plans one video per
  coming slot within `ahead_days` (`options = {format, slot, auto}`; a
  slot is taken by a plan or an upload's publish time, `_slot_key`
  "YYYY-MM-DDTHH:MM" SA time); (2) uploads ready ones with
  `publishAt` = their slot (now, if the slot passed < 3 h ago); (3) posts
  to Instagram the videos that went public in the last 12 h. It runs
  after every make, every 10 min while the app is awake, on saving the
  settings, and from the public `GET /autopilot/tick?key=$AUTOPILOT_KEY`
  (404 without the variable), which a Railway cron service calls to wake
  the sleeping app (a few minutes after each slot, so the Instagram post
  follows). `_balls_work` keeps the app awake (`_keepalive_loop`) while
  it renders. Upload count "today" is YouTube's quota day (midnight
  Pacific = 09:00/10:00 SA), not the last 24 h: the two slots fall in
  different Pacific days, so a rolling window would sometimes block the
  second upload. Hitting the 2/day cap is normal (`quota_wait`), not an
  error. Deleting an uploaded video only hides it (`hidden`), so its slot
  stays taken and the autopilot doesn't post a second one there. "🚀 Post
  now to YouTube + Instagram" (`POST .../post-both`) shows when Instagram
  is connected.
  First real morning (9 Oct): the 08:00 video missed its slot. A manual
  post the day before used one of the 2 uploads of that YouTube day, and
  the 08:00 upload has to happen before 08:00, still in that same day
  (it resets 09:00/10:00 SA). Fixes: the waker also runs at 09:05 and
  10:05 SA (cron `5 6,7,8,13 * * *` UTC) so a missed slot posts late
  within the 3 h window, and `BALLS_UPLOADS_PER_DAY` is 3 so the backlog
  can catch up (with 2 it would stay a day behind for good).
  9 Oct: the first ring video (escape, 1.1k views in a day) beat the
  evolve one (~250), and Dean asked for more ring videos: the kind now
  rotates per slot, not per day, through `BALLS_FORMAT_CYCLE` (escape,
  touch, escape, evolve, touch, escape, touch, evolve: 3 in 4 rings),
  `_balls_format_for(dt)`; a manual Make defaults to the next slot's kind.
  Restarts (a deploy, Railway waking the app) used to leave the autopilot
  idle until the page was opened: now a startup thread restarts the loop
  when autopilot is on, and every tick runs `_balls_resume` (requeues a
  video a restart cut off mid-make, starts the worker for queued ones).

## Instagram posting (Oct 2026)

- Dean pasted Meta's Content Publishing docs and asked to set it up, for
  the Ball Evolution channel ("Dropvolve", the name/pfp/banner/bio were
  made for him in the sandbox) first. `clipper/instagram.py` uses the
  Instagram API with Instagram Login (no Facebook Page): authorize at
  instagram.com/oauth/authorize (scopes `instagram_business_basic,
  instagram_business_content_publish`, `force_reauth`), code -> short
  token (api.instagram.com/oauth/access_token, response wrapped in
  `data[]`) -> 60-day token (`ig_exchange_token`) + `/me` user_id/username,
  stored in `BASE_DIR/_instagram_token_<account>.json`, refreshed
  (`ig_refresh_token`) once over a day old with < 20 days left.
- Posting a Reel: Instagram downloads the video itself, so the app serves
  it at a public one-off link `/ig-media/<token>.mp4` (no app password,
  random token, 1 hour, in memory). Container (`media_type=REELS`,
  `share_to_feed`) -> poll `status_code` every 10 s until FINISHED (10 min
  cap) -> `media_publish` -> permalink. Runs in a background thread; the
  video's json gets `instagram: {status: posting|posted|error, ...}`.
  Caption = YouTube title + description, "#Shorts" removed.
- `INSTAGRAM_ACCOUNTS` in `webapp/main.py` holds just "balls"; routes
  `/auth/instagram/login?account=`, `/auth/instagram/callback` (redirect
  URL to register in the Meta app: `https://<RAILWAY_PUBLIC_DOMAIN>/auth/
  instagram/callback`), `POST /api/instagram/disconnect`, `POST
  /api/balls/videos/{id}/instagram`. Needs `INSTAGRAM_APP_ID` /
  `INSTAGRAM_APP_SECRET`. Own accounts need no App Review: the Instagram
  account is added as an Instagram Tester (accepted in the Instagram app).
- **Facebook Login route (Oct 2026):** on both of Dean's Meta apps the
  Instagram use case only offered "API setup with Facebook login" (no
  Instagram-login setup, nothing under "Add more to this use case"), so
  `INSTAGRAM_LOGIN` defaults to `facebook`: facebook.com dialog (scopes
  `instagram_basic, instagram_content_publish, pages_show_list,
  pages_read_engagement, business_management`, or `config_id` from
  `INSTAGRAM_FB_CONFIG_ID`) -> user token -> `fb_exchange_token` long-lived
  -> `/me/accounts` with `instagram_business_account` -> the first Page
  with a linked Instagram account; its Page token (doesn't expire) is
  stored with `graph: graph.facebook.com` and used for /media and
  media_publish. @dropvolve must be linked to a Facebook Page. The
  redirect URL goes in Facebook Login for Business -> Settings -> Valid
  OAuth Redirect URIs; App ID/secret from App settings -> Basic. Dean is
  the app admin, so no tester invite. `INSTAGRAM_LOGIN=instagram` keeps
  the Instagram Login route.
- Instagram's API can't schedule a post, and the Railway app sleeps, so
  posting is "📸 Post to Instagram" now, not on the YouTube schedule.
  Meta's hosts are blocked in the sandbox: tested with a fake `requests`.

## Clip layout

- Clips render in the IRL layout by default (`reframe.LetterboxLayout`,
  the whole scene letterboxed; `JobRequest.irl_layout=True`). Each clip's
  "Switch to facecam" button opens the manual box-picker, and "apply to
  the other clips" converts a whole job. Unticking "IRL layout" brings back
  automatic facecam detection (`compute_layout`). Dean chose this default.
- **Card style (Oct 2026, default):** Dean liked the "Cheerful Videos"
  look and asked for it on every clip, switchable by hand. A whole-scene
  clip renders as `reframe.CardLayout(video_y, video_h)`: white 1080x1920,
  the mascot as the profile picture, channel name + handle (profile
  `handle`, `CLIPPER_BRAND_HANDLE`, default "@caughtonstream24"), the hook
  as the post text for the whole clip (Inter, wrapped by measuring with
  Pillow, shrinks past 3 lines), then the whole frame at full width, the
  spoken captions over its bottom edge. No blue verified tick (claiming
  verification you don't have reads as misleading); Dean was told.
  `captions.build_card_ass` makes `_clip_NN.card.ass` FROM the clip's
  normal `_clip_NN.ass` (its caption lines keep their timing), so the
  normal file is untouched and facecam re-renders keep using it; any clip,
  old ones too, can switch. Render: `scale` + `pad` (keeps the source fps),
  `render.ass_filter` adds the bundled fonts dir only for `.card.ass`
  files, so normal clips burn exactly as before. `JobRequest.card_style`
  (default on; old jobs' new clips get it too) applies wherever the
  letterbox would be used, vertical sources stay letterboxed. Per clip:
  "🪪 Switch to old look / card style" = `POST .../mark-irl?card=`
  (no flag follows the job). Registry layout name "card". "✏️ Edit card
  text" (Dean asked to change the post text) = `POST .../card-text`: sets
  `clip["hook_text"]` (emoji stripped, the card font has none; empty = no
  post text), swaps the HookText line in the normal .ass too
  (`captions.set_hook_text`) and re-renders the card via mark-irl.

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

- Uvicorn runs with `--log-config webapp/log_config.json`: its INFO lines go
  to stdout, warnings/tracebacks to stderr. Railway shows every stderr line
  as a red "error", so with uvicorn's default config a healthy startup
  looked like errors to Dean. A red line in the logs should now mean a real
  problem.
- Project `12720d82-23db-498d-b4f6-2a69cf5fff42`, service
  `978f7dca-a5bf-4663-8446-ed3a6625841f`, environment
  `a30ad832-9974-4645-bed0-37002a3af844` (production).
- Merges to `main` auto-deploy. After a merge, check `get-logs` /
  `environment-status` to confirm the deploy went out clean before telling
  Dean it's live.

## Whose videos get clipped

- The README deliberately says nothing about whose videos to clip. Keep it
  out of there; this is the place for it.
- The tool works the same on Dean's own uploads and on other people's
  public videos. The difference is legal/platform, not technical:
  re-publishing someone else's content can run into copyright and
  platform ToS issues depending on how much is used, how (commentary /
  criticism / fair use vs. straight re-upload) and where it's posted.
  That's a per-video gut-check before publishing, not before
  experimenting locally.

## Other standing preferences

- No paid signups/subscriptions for libraries or tools — check for a free
  option first (e.g. plain CSS charts instead of a charting library).
- Dean posts ~2 clips/day, chosen manually by eye — factor this into any
  quota/cost estimates (YouTube API, Claude API, Railway usage, etc.).
- He prefers a separate page per feature/channel (linked from the top nav)
  over controls bolted onto an existing page.
- For UI changes he likes a mockup first when the change is big, then a
  live-browser check (Playwright screenshots) before the PR.
