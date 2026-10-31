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
- **Your take at the end (Oct 2026, standard):** after the 1 Oct change Dean
  wanted his own commentary on every clip without showing his face. Each
  clip ends with Pixel -- a retro-TV mascot in the channel's colours
  (yellow case, black screen with the viewfinder corners, white pixel face,
  red mouth; chosen from 4 sandbox designs) -- popping in small at the
  bottom left over the clip's held last frame and saying one short line in
  Dean's cloned voice, 1-4 words of it in a bubble. The clip itself is
  rendered exactly as before; `clipper/mascot_take.py` appends the tail
  (`apply`: clip + tail re-encoded into the same file, the take-free clip
  kept as `<job>/_takes/<clip>.notake.mp4` so takes never stack and
  `remove` restores it; voice in `_takes/<clip>.wav`). Claude suggests the
  line (`suggest`: `say` <= 10 words, his opinion or a question to viewers,
  never a description of the clip; `text` <= 4 words; a separate prompt,
  the picker's prompts are untouched; no stock fallback line -- the same
  line on every clip is the template YouTube doesn't count).
  `JobRequest.mascot_take` (default on, Home checkbox "📺 Your take at the
  end"); `_queue_takes` runs after a job's / Generate more's new clips are
  done, one at a time under `_voice_busy`, and after a facecam / card-style
  re-render (`redo=True`: same words and voice put back, nothing new
  generated). `clip["take"]` = {status queued/working/done/error/off/
  removed, say, text, tail, base_duration}; `clip["duration"]` includes the
  tail. Home clip cards: ✏️ Change take (his own words), 🔄 New suggestion,
  🗑 Remove take; nothing changes once posted. Without a voice sample the
  take is skipped ("off"). Mascot sizes: 380 px box, bottom at y 1650.
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
- **Neon ring look (Oct 2026):** the first ring video on black beat the
  rest, and Dean found the backgrounds not eye-catching. Researched (the
  2023 "will the ball escape" Shorts: bright balls on dark, glow and trails
  are the genre's tools; no real study compares backgrounds), then 8
  mockups; he picked trails + rainbow ring and four backgrounds. Ring
  recipes now draw `backdrop` from `be.RING_BACKDROPS` (black / space /
  synthwave / blurred; black weighted 3, space 2, synthwave 2, blurred 1
  in `_RING_BACKDROP_DRAW`, then `pick_recipe` avoids repeating the last
  one, so black is ~40%). With one of them `CirclePainter` adds light
  trails (full-res layer from just above the ring down, faded x0.82 a
  frame, lines from the last position so fast balls streak), a rainbow ring
  with glow (drawn once; the gap is cut out per frame with a pieslice mask,
  so the colours stay put while the gap turns) and a glow on each ball
  sprite. Synthwave is the "calm" version Dean preferred over a busy one:
  dark sky, faint grid below the count (`SYNTH_HORIZON` 1640), its lines
  scroll toward the viewer. ~31 ms a frame vs ~19 before (per-frame
  rotation, blur and big alpha arcs were 80-90 ms and were dropped). Older
  ring recipes (backdrop gradient/glow/...) draw byte for byte as before
  (checked), and evolve videos are unchanged.
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
  within the 3 h window, and `BALLS_UPLOADS_PER_DAY` is 3 (4 since 12 Oct,
  Dean's call after a Remake day used them up; the API budget is ~6
  uploads/day across all channels, so it's tight on busy days) so the backlog
  can catch up (with 2 it would stay a day behind for good).
  9 Oct: the first ring video (escape, 1.1k views in a day) beat the
  evolve one (~250), and Dean asked for more ring videos: the kind now
  rotates per slot, not per day, through `BALLS_FORMAT_CYCLE` (escape,
  touch, escape, evolve, touch, escape, touch, evolve: 3 in 4 rings),
  `_balls_format_for(dt)`; a manual Make defaults to the next slot's kind.
  (Since Break in: a 10-slot cycle, 8 in 10 rings, Evolve every 5th.)
  Restarts (a deploy, Railway waking the app) used to leave the autopilot
  idle until the page was opened: now a startup thread restarts the loop
  when autopilot is on, and every tick runs `_balls_resume` (requeues a
  video a restart cut off mid-make, starts the worker for queued ones).
  "♻️ Remake upcoming videos with the newest look" (Posting section,
  `POST /api/balls/remake-upcoming`; Dean asked after a design fix left
  the next days' videos in the old look): every video that isn't public
  yet and was already made (or is being made: its process is killed) is
  queued again from scratch -- new recipe, title, description, same slot;
  not-yet-made ones are left alone. Ones already scheduled on YouTube
  can't be swapped: the OAuth scopes allow upload but not delete (adding
  delete would mean a broader scope + reconnecting), so they're listed
  with Studio links; after Dean deletes one there, pressing again sees it
  gone (`videos.list` with his token, 1 unit), hides the record with
  `youtube.removed` (still counts toward the day's uploads, no longer
  holds its slot or gets an Instagram post) and plans a new video for
  that time. This includes videos hidden with the app's own 🗑 (first real
  use: Dean hid Saturday's two that way, the button skipped hidden ones,
  and Saturday stayed blocked); a hidden one still on YouTube is listed as
  "removed from this list, still on YouTube" because it will still post.
  12 Oct: the 08:00 video missed its slot again. The logs showed no
  upload attempt after yesterday's three Remakes (13:19, 14:12, 17:55
  UTC): the same five videos were made three times and none went up, so
  YouTube's day (09:00 SA 11 Oct to 09:00 SA 12 Oct) had already used its
  `BALLS_UPLOADS_PER_DAY` (videos deleted in Studio still count: YouTube
  charged for them), and an 08:00 video must be uploaded before 08:00,
  inside that same day. It goes up late via the 09:05 waker. The tick
  used to be silent about this; it now logs every upload ("[autopilot]
  uploaded ...") and a wait for a slot under 12 h away ("... waits"),
  so the next miss can be read straight from the logs. Same day:
  `_balls_resume` (run on every /api/balls poll) requeued a video that
  had just finished between reading the list and writing it, so videos
  were made twice; it now re-checks inside `_balls.update`, and
  `_balls_making` covers the minute of picking the pictures before the
  render process starts.
  Then Dean asked for the autopilot to post directly, like Instagram, so
  changes and Remakes stop costing uploads: the tick no longer uploads
  ahead with `publishAt`; a ready video is uploaded public once its slot
  has come (up to 3 h late), so it costs one upload only when it posts and
  Remake can swap anything not yet posted. It relies on the Railway waker
  (`ball-autopilot-waker`, cron `5 6,7,8,13 * * *` UTC = 08:05, 09:05,
  10:05, 15:05 SA): posts go out ~5 min after 08:00 / 15:00. Changing the
  slot times on the page needs that cron changed too. The manual
  "📅 Schedule" button still schedules ahead with `publishAt`.

- **Break in (Oct 2026):** Dean's idea, from two sandbox prototypes. The
  first (ring on top, corridor packed with tiles, doubling on each return,
  box below) broke through "way too quick"; he asked for it upside down
  with a plain corridor. `clipper/ball_breakin.py` (`format: "breakin"`,
  dispatched by `_parts`): one ball in a rainbow ring at the bottom (no
  gravity, steady 520 px/s, tiny random turns, balls pass through each
  other), a 92 px gap at the top into a plain corridor, a room of 11x11
  tiles on top. Every broken tile spawns a ball there. With 1-hit tiles the
  room went in ~1 s, so hits needed rise from 1 at the entrance row to 6
  at the far row (numbers on the tiles): the room clears ~10-15 s after
  the first ball gets in. The ladder runs on ball count (log scale); the
  last item appears when the room is cleared. `good_end` 18-42 s, which
  with `MAX_UNLOCK_GAP` keeps the wait to get in at ~6-18 s (about 1 in 4
  seeds pass). Its own hooks ("Can one cookie clear the room?",
  `BREAKIN_HOOKS`), endings (`CLEARED!`...) and description, drawn from a
  separate rng, so the other formats' recipes are unchanged (checked on
  800 recipes). Neon look like the rings, backgrounds black / space /
  blurred (`_BREAKIN_BACKDROP_DRAW`; the synthwave grid would sit behind
  the ring). (Since 11 Oct: 420 bricks, see "Rotation + Break in".)

- **Merge to final form + picture packs (Oct 2026):** Dean sent a
  music.bouncingball reel (308K likes): cut-out meme pictures drop down a
  chute into a rainbow ring, two of the same merge into the next, ending on
  a final form; every bounce plays a note. He asked for meme / "cursed"
  pictures instead of emoji ("the emoji doesn't grab attention") and a song
  from the bounces. `clipper/ball_merge.py` (`format: "merge"`): capped
  chute (items flew back out without the cap), gravity 900, bounce 0.92,
  drops every 0.3 s, bigger tiers join the drops once revealed (a 9-step
  chain needs 256 of the smallest otherwise), tier >= 3 drift to their
  match. Every ring hit plays the next note of a public-domain tune
  (`TUNES`: Mountain King, Ode to Joy, Fur Elise, Turkish March, Twinkle;
  Jingle Bells in December) on a numpy piano, max ~7 notes/s
  (`NOTE_GAP`); `be.mix_audio` adds `sim.music_track(n)` when a sim has
  one (others unchanged), no noise bed. Dean once hated random marimba
  notes; this is a recognisable tune, which he asked for.
  `clipper/ball_packs.py`: packs of 6-9 pictures in order (last = final
  form), `BASE_DIR/_balls/packs/<id>/` (pack.json + 01.png..). Uploaded
  on the page (shrunk to 512 px in the browser, sent as base64 JSON), or
  automatic: Claude plans an 8-step normal -> cursed chain of GIPHY
  *sticker* searches (stickers are already cut out), the app searches
  (`GIPHY_API_KEY`, free; rating pg), then ONE Claude vision call sees all
  candidates (up to 6 per step) and picks one per step, rejecting nudity,
  gore, real people, offensive text; any empty step fails the whole pack.
  Real people / celebrity faces are excluded on purpose (likeness), and
  Dean was told meme pictures carry copyright risk; he chose to go ahead.
  A merge make (`_balls_pick_pack`) uses the chosen pack, else the oldest
  unused upload, else a new GIPHY pack, else the theme's emoji; the pack id
  is saved in the video's options so a retry/remake keeps the pictures.
  Plain photos are drawn in a circle with a white rim, stickers as
  cut-outs (`is_cutout`: >30% see-through edge). The sandbox can't reach
  GIPHY, Reddit, Imgur, Wikimedia etc. (all blocked); tested with a fake
  `requests` and a fake `_claude_json`. `BALLS_FORMAT_CYCLE` is 12 slots:
  merge x3, escape x3, touch x2, breakin x2, evolve x2.
  First real runs (11 Oct): the drop pace is per chain length
  (`ball_merge.PACE`, 6-9 pictures all ~26-44 s; one pace for all made a
  7-picture pack finish in ~15 s and every seed failed); GIPHY packs log
  each step, try 3 searches and every image version, leave out a step with
  nothing usable (>= 6 kept, final form required), and check pictures step
  by step when the one big check fails. The model behind `CLIPPER_MODEL`
  thinks before answering: a 400-token limit returned only a thinking
  block, so the pack calls allow 4000-8000 tokens.
  **Claude's last look (Oct 2026, Dean asked "have Claude look at the
  ending, that it looks good"):** after a merge video renders,
  `clipper/ball_review.py` sends 4 frames (start, 40%, 70%, the ending
  1.2 s after the final form) + the pictures' names to Claude: does the
  ending work, does anything look broken, are the pictures fine to post.
  The prompt says the ending text over the pile is normal (it always is).
  `_balls_redo`: a "looks" no gets one new run with the same pictures; a
  "pictures" no on a GIPHY pack flags it (`pack["flagged"]`) and swaps in
  a new pack, then the theme's emoji; Dean's own uploaded packs are never
  swapped. At most 2 remakes, then it goes out with the note on its card
  (`video["review"]`, past no's in `reviews`). A failed check (no key, API
  error) lets the video through. Flag and logo packs add a note on what their
  pictures look like (`ball_review._style_note`): on 31 Oct Claude called
  Japan's countryball "a generic red smiley" and remade a good video.
- **Pictures in every kind + starter packs (Oct 2026):** Dean found the
  emoji "too generic" and made 4 GIPHY packs; every kind now uses a picture
  pack. `be.recipe_chain` / `be.load_pack_images` / `be.apply_pack` (hook:
  the pack's own, Break in's "Can one {first} clear the room?", else
  `PACK_HOOKS`); `be.Painter` draws pack pictures for all kinds (merge's
  own copy is gone; no-pack videos draw byte for byte as before, checked).
  `_balls_pick_pack` for every make: chosen > oldest unused pack (uploads
  first, then GIPHY / starter; `next_unused`) > new GIPHY pack > the pack
  used longest ago (`least_used`) > emoji. Claude's last look now checks
  every video with a pack (`ball_review.KINDS`).
- **Starter + cursed packs (Oct 2026):** Dean asked for 10 cursed / chad /
  sigma / country templates "from a source you're allowed to use or made
  yourself". All 10 ship in `clipper/assets/packs/` and are seeded into the
  store as source "starter" (`ball_starters.seed_bundled`): 3 countryball
  flag packs (flag-icons SVGs, MIT, rendered with cairosvg in the sandbox
  only, crown on the final one) and 7 packs from `clipper/ball_cursed.py`:
  one Fluent emoji face (21 `BASES`, eye positions measured) through our
  own edits: plain > Chubby/Long > Cool (🕶) > Buff (💪 x2) > Sigma (B&W,
  🕶, 👔) > Cursed (👄, extra 👁, swirl) > Deep Fried (saturated, JPEG
  crunch, red laser eyes) > God (👑, 🪽, gold glow). When every pack is
  used the app makes a new cursed pack itself, so GIPHY is no longer used
  (stickers belong to their makers; `ball_packs.auto_pack` is kept but
  nothing calls it now). Dean asked about scraping Google Images for
  chad-cat style photos; told him why not (others' copyright, real
  people's faces, Google's terms/CAPTCHA, reused-content demonetization)
  and he chose free real photos + our edits: `clipper/ball_photos.py`
  searches Openverse (CC0 / public domain only) for one of 20 animal
  faces, Claude picks a photo with one animal facing the camera (no
  people, text, watermarks) and gives the face box + eyes as fractions,
  `cut_face` cuts a circle centred on the eyes, and `ball_cursed.make`
  runs `PHOTO_STEPS` on it (adds "Chad": a smooth widen + stretch from the
  eyes down). Source "photo", credit "Photo by X, CC0 via Openverse" in
  the description. Order when nothing is unused: photo pack (needs the
  Claude key) > emoji cursed pack (`ball_cursed.new_pack`). "📷 Make a
  photo pack now" (`POST /api/balls/packs/auto`). Openverse is blocked in
  the sandbox: tested with a fake `requests`, a fake Claude and
  skimage's `chelsea.png` cat.
- **Rotation + Break in (11 Oct):** Dean's numbers: the gap ring and Break
  in do best, the peg/gumball Evolve ~200 views. `BALLS_FORMAT_CYCLE` (12
  slots): escape x4, breakin x4, merge x2, touch x2; Evolve only by hand.
  An autopilot video (`options.auto` + `slot`) takes its kind from the
  rotation at make time, not from its plan: the first Remake after this
  change still made Evolve videos for slots planned while Evolve was in
  the cycle, and Dean ("the maze path in the circle is too confusing")
  doesn't want them.
  **Four more kinds (11 Oct):** Dean sent music.bouncingball's grid and
  asked for all four, rotating (`clipper/ball_arenas.py`, `be._parts` ->
  `ball_arenas.PARTS`; neon dark look; a tune from the bounces via
  `music_track` like Merge; own hooks/endings `be.ARENA_HOOKS` /
  `ARENA_ENDINGS`, skin "plain"): `center` Hold the center (every pack
  picture bounces around a black hole that grows; falling in = out,
  crossed out in the ladder; last one wins; `simulate` now returns `ok`
  and `is_good` needs it: the seed must make the pack's final picture win,
  ~1 in 8, and the first one must fall in by 8 s (`C_FIRST_OUT`: no dead
  start), so `pick_seed` tries 240 seeds for center/spikes; a pack whose
  title is a question, "Which country is the best?", is the hook),
  `spikes` Spikes will grow (one ball, steady speed, no gravity, in an
  octagon; a solid spike grows where it hits a wall once it has moved off;
  only a near head-on hit on a tip pops it -- with gravity, whole-spike
  pops or spikes grown under the ball it popped after 2-5 spikes; next
  picture every ~28 bounces / pictures; seeds where it survives to the final form),
  `shrink` The ball grows, the circle shrinks (own physics, ~68 bounces,
  ~25 s), `jaws` 4 seconds to escape (a turning ring of 3 arcs; each ball
  has 4 s or pops, an escape makes two; rotating teeth were tried and swept
  every resting ball). Rotation (16 slots): escape x3, breakin x3, center
  x2, merge x2, spikes x2, shrink x2, jaws x1, touch x1. Tuned with the
  countryball pack: good seeds per 12 center ~2, spikes ~3, shrink 11,
  jaws 5.
  **Four more (11 Oct)**, from reels Dean sent with big numbers, same
  module: `spiral` "Will the ball multiply?" (809K likes: balls break an
  Archimedean spiral wall, pieces take 2 hits in the middle up to 18 on the
  outer turn, every broken piece makes a ball, counter "x28"; pictures
  follow the share of wall gone -- by ball count they all came in the last
  second), `rays` (570K: rounds of rainbow rays fall into a parabola and
  bounce through the focus, 1..150 rays, the round's picture at the focus;
  analytic paths, no physics, every seed works; our own tune, the reel used
  Interstellar's), `box` "+3 after each escape" (360K: a box with a hole in
  the floor, each ball out adds 3, counter to 2,500, max 520 simulated at
  once, the rest wait), `zen` "Reminds me of something" (106K: a spiked ring;
  a ball that touches the spikes turns grey and stays; each picture gets
  `Z_TRIES` 14 / pictures tries; the jaw only opens when the final picture
  drops in -- open from the start or on a timer, earlier balls got out
  first). `trail` "With every bounce, the ball evolves" (one ball in a
  ring under gravity, every bounce stamps a coloured circle on a layer that
  stays and the ball grows a little; the picture changes every ~N bounces;
  ends when its radius is 46% of the ring's). Rotation (22 slots,
  `BALLS_FORMAT_CYCLE`): escape x3, spiral, trail, center, breakin, rays,
  box, zen x2 each, merge, spikes, shrink, jaws, touch x1.
  **Bounce sounds (11 Oct):** merge and every arena kind pick
  `recipe["bounce_sound"]` (own rng `bounce-sound-<seed>`, so the other
  recipe fields are unchanged): `tune` 2 : `meow` 1 : `phonk` 1.
  `clipper/ball_sounds.py`, numpy only: Dean asked for "a cursed meow,
  heavily deep, deeper the more it bounces" (a synthesised meow, slowed a
  bit more each bounce up to 3.5x) and "the song people post mog videos
  to" -- those are labels' phonk tracks that Content ID would claim on
  every video, so it's an original 808 cowbell riff + 808 bass in the same
  style; Dean was told. The card chip shows "🐱 cursed meow" / "🥁 phonk".
  **Songs for the bounces (11 Oct):** Dean found the piano tunes "just
  standard piano" and asked for non-copyright songs. /ball-evolution has a
  "🎵 Songs for the bounces" box (store `BASE_DIR/_balls/music/` +
  `music.json`, routes `/api/balls/music*`, same pattern as Rocket League's
  music; "🚗 Add my Rocket League songs" copies those with their credits).
  A video whose recipe drew `bounce_sound` "tune" plays a song instead
  when there are any (`_balls_pick_song` in `_balls_make`: least-used,
  `recipe["song"] = {path, start, name, credit}`). 12 Oct: Dean wants his
  (phonk) songs on every one, in rotation, so any `bounce_sound` (tune,
  meow, phonk riff) becomes a song when songs exist; the others are only
  the fallback without songs. The Make box has a "Song" menu (Dean wanted
  to test one song): Auto (rotation), any one song, or "No song"; it goes
  in the video's `options["song"]` (so a retry keeps it) and is greyed
  out for kinds without bounce music (evolve, escape, touch, breakin).
  14 Oct: a song video did very well and Dean asked for music on every
  video, so those four get songs too (`_balls_pick_song` no longer needs
  a `bounce_sound`; the menu is enabled for every kind). `be.mix_audio`:
  a sim without `music_track` whose recipe has a song plays it gated by
  its ball hits (tick / knock / thud events, `ball_sounds.song_track`),
  over the ball sounds at half level and with no noise bed. Recipes
  without a song mix exactly as before. `ball_sounds.song_track`: each bounce lets the song
  play on for `SONG_OPEN` 0.42 s (it only moves while the ball bounces,
  like the reels), starting at its drop (`rocket_league.find_drop`,
  editable). The credit goes under "Music:" in the description. Links on
  the page: YouTube Audio Library (never claimed), NCS (credit), Pixabay
  (keep the license certificate). Labels' songs aren't offered: claimed.
  **Rainbow trails (12 Oct):** Dean asked for the balls to leave a
  colourful trail. Every kind but Evolve now gets `recipe["trail"] =
  "rainbow"` (`make_recipe`), and the trail loops in the arena, merge,
  circle and Break in painters draw through `be.draw_trail` /
  `trail_segments`: the colour moves through the rainbow with distance
  (a full rainbow every ~320 px, each ball starting at its own hue), in
  ~7 px steps so it blends instead of striping (3 steps max when > 40
  balls), with a round joint at each bounce; arena trails last longer
  (`RAINBOW_TRAIL_FADE` 0.92, was 0.82; Box keeps 0.6). Recipes without
  the field (every video made before) draw byte for byte as before
  (checked on 240 frames of 7 kinds). Then (same day) Dean asked for
  the shrinking-circle trail to stay for the whole video:
  `ShrinkPainter.KEEP_TRAIL` (no fade at all; paint outside the shrunk
  ring stays too), and on the zen spike ring "just a normal line" that
  goes once the ball stops: `ZenPainter` draws a 7 px line that never
  fades while the ball moves, and when it turns grey that line fades out
  (x0.8 a frame) while the next ball starts a new one. Spikes will grow
  keeps the normal fading trail. Then Dean asked for the circle to stop
  shrinking: `shrink` ("🔴 Ball grows every bounce") keeps the ring at
  `G_R0` (`G_RING_STEP` 0) and the ball grows 6 px a bounce (was 2.3 with
  the ring -4), so it fills the ring in about the same ~70 bounces (good
  seeds 12/12 for 6, 8 and 9 pictures, 25-35 s); new hooks ("The ball
  grows with every bounce"...) and description.
  Break in has 420 bricks (44x22, was 132 at 80x40), up to 8 hits deep,
  `MAX_BALLS` 600: the room clears ~11-25 s after the ball gets in, ~5 in
  12 seeds pass `pick_seed`, ~5 min to make a 34 s video.

- **Duel, race, known pictures, cursed merges (19 Oct):** Dean sent reels
  (a "Bouncing Ball Duel" with 1M likes, "First to 300 wins" with 149K) and
  his weakest Shorts (emoji packs from before songs were on every kind).
  `clipper/ball_duel.py` (`NO_PACK` kinds: no picture pack, the balls are
  the contestants; `be._parts` dispatches; own draws in `duel_recipe` /
  `race_recipe`, so other kinds' recipes are unchanged): `duel` -- two balls
  with faces ("Cherry Red VS Ocean Blue", `COLOURS`), 10 hearts, weapons
  appear in grey rings; Dean wanted mostly guns that fire once: pistol,
  shotgun (7 pellets, max 2 hearts), minigun (1.2 s burst, max 3), rocket
  launcher (homing, 3 hearts close / 2 at the edge), drawn in `_gun` (the
  emoji set only has a water pistol); a knife (2 hits) and a shield (blocks
  2, hurts on contact) now and then. Faces react, balls crack, the loser
  shatters, "X WINS!". `sim.ok` = a comeback (winner 2+ hearts behind at
  some point), winner on <= 3 hearts, both had 2+ weapons; `sim.max_gap`
  (longest stretch without a hit) replaces the unlock gap in `simulate`;
  ~1 seed in 7, `pick_seed` tries 480. `race` -- four plain colour balls
  (Dean: not phone brands there, and the first try was too fast to
  follow: 360 -> 640 px/s now); every wall hit ties a line, running into
  another's lines steals them (a stolen line rests 1.5 s); first to 40
  (`RACE_TARGET`) wins; ok = 3+ lead changes and second place within 30%.
  Both play Dean's song straight through (not gated by bounces) under
  their own numpy sounds (`_sfx_mix`), bed off. Rotation (25 slots) adds
  duel x2, race x1. **New kinds first:** `BALLS_DEBUT` (duel 3, race 2,
  merge 2 videos made since `BALLS_DEBUT_SINCE`) -- an autopilot video
  takes the debut kind furthest from its count before the rotation; after
  a deploy, "♻️ Remake upcoming" turns the queued videos into them. The
  plan list shows the kind a made video actually is.
  **Known pictures:** Dean wants merges (and every kind) with things people
  know, not emoji. 5 logo packs (`logos_phones/apps/cars/gaming/sports`,
  Simple Icons, CC0 data; logos are trademarks -- Dean's call after being
  told the risk; the credit line says not made or endorsed by them) and 2
  more flag packs (Africa, Asia). `_balls_pick_pack`: chosen > unused
  upload > `next_known` (unused logo/flag pack, else the one used longest
  ago) > the old order (cursed/photo packs only if no known pack exists).
  **Cursed merge noises:** `clipper/ball_cursed_sounds.py` (numpy voices:
  chirp-down meow, dying meow, kitten moan, demon meow, moan, reversed wail,
  deep laugh, out-of-tune choir, demon growl; small merges short and spaced
  >= 0.3 s, big ones deep and long). New merge recipes carry
  `merge_sound: "cursed"`; `mix_audio` then drops the merge puffs and lays
  `merge_track` over the music. Older recipes sound as before.

- **16-ball tournament (25 Oct):** a viewer commented "Pls make a tournament
  16 balls" under a duel, and Dean asked for it. `format: "tourney"` in
  `clipper/ball_duel.py` (`TourneySim` / `TourneyPainter`, `NO_PACK`): a
  bracket on top (16 colour dots, losers crossed out, winners' lines in their
  colour, 🏆 then the champion with a crown), all fights of a round at once
  below it (8 arenas, then 4, then 2, then the final full width with names),
  each ball's hearts in a column beside its arena. Every fight is a `DuelSim`
  in the duel's own coordinates; the painter only draws it smaller
  (`DuelPainter.place`; plain mode draws the duel byte for byte as before,
  checked on two whole duels, sim + frames + audio). Hearts by round 2/2/3/5
  (`T_HEARTS`). Tuning lessons: rounds ran 20-40 s because balls kept
  swapping weapons before firing (now `keep`: a gun-holder walks past drops),
  circling balls outran the aim (fires near enough after 1.2 s), and shots
  kept missing; so `T_SUDDEN` 6/7/8 s: sudden death = one heart each + a
  rocket each, and the ring closes in (`DuelSim.ring`, red on screen,
  "SUDDEN DEATH") until nothing misses; rounds now ~9-15 s, the whole video
  ~45-55 s (`good_end` 36-58), ~40% of seeds have a close final (comeback or
  winner on <= 2 hearts). 16 colours: `COLOURS` + `EXTRA_COLOURS`, every
  first fight one of each. Early rounds drop wall/bump/pickup sounds (eight
  fights at once was a wall of noise). In the rotation once (26 slots) and
  `BALLS_DEBUT` 1, so the next autopilot video is a tournament.
  ~2 min to make a 51 s video.
- **Numbers + more fights (25 Oct):** Dean asked to "pull analytics for the
  balls and see what does well". vidIQ's public numbers for the 30 Shorts
  (8-23 Oct; Dropvolve isn't connected to vidIQ, so no retention/swipe
  data -- Dean was told to connect it): almost all sit at 1-2k views (the
  test audience), the Royal Purple VS Cherry Red duel went to 12k and still
  climbing, 3.5 likes per 100 views; the other duel 2.2; everything else
  ~1; the growing ball 3k; cursed packs weakest. Dean said yes to more
  fights + varied duels:
  - `BALLS_FORMAT_CYCLE` 18 slots, a third fights: duel x4, tourney x2,
    shrink x2, escape x2, then one each of center, zen, spiral, race, merge,
    jaws, box, spikes. Rays, break in, trail, touch out of the autopilot.
  - Duel looks (`duel_recipe`, own rng `duel-look-<seed>`, so the colour
    duel's draws are unchanged): 45% **flags** (`COUNTRIES`, 45 plain
    flag-icons discs in `clipper/assets/flags`, the duel's own face drawn
    on them; 70% a football-style `RIVALRIES` pair; countries in an armed
    conflict with each other left out on purpose; a too-close second colour
    takes `ALT_COLOURS`; "Flags: flag-icons (MIT)" in the description),
    25% **2v2** (`TeamDuelSim` / `TeamDuelPainter`: a warm and a cool pair,
    team-coloured rings, 6 hearts each, aim at the nearest enemy, no
    friendly fire, a team is out when both are; ok = a 3+ heart comeback
    and the winners on <= 5 hearts, ~1 seed in 10), 30% the old colours.
    The tournament is half flags ("16 countries, 1 winner"). The duel's
    team-ready hooks (`_hp_row`, `_ko`, `_movers`, `_pairs`, `_collide`,
    `_bullet_hit`, `_boom_targets`) keep the 1v1 duel byte for byte as
    before (checked on two whole duels from their saved recipes).
- **Battle royale (26 Oct):** Dean asked for "16 balls, one circle,
  1v1v1". `format: "royale"` (`RoyaleSim` / `RoyalePainter` in
  `ball_duel.py`, `NO_PACK`, half 16 countries / half 16 colours): every
  ball for itself (each its own team in the 2v2 machinery: aims at the
  nearest ball, bullets hit whoever they reach, rockets hurt everyone near),
  4 hearts each (pips above each ball), up to 6 drops at once. It runs in
  a bigger ring (`R_START` 800 in the duel's units) drawn scaled into the
  duel's ring, so 16 balls have room; from `R_ZONE_AT` 8 s the ring closes
  in like a battle-royale zone (red, 14/s down to 250), and once it is
  smaller than the duel ring guns fire without waiting to aim. HUD: the
  hook, all 16 in two rows (crossed out when out), "N LEFT". ok = winner
  on <= 2 hearts and the first knockout by 6 s; `max_gap` is the longest
  wait between knockouts; `good_end` 24-45 (median ~29 s, ~1 seed in 3).
  Early tests with 3 hearts ended in ~24 s. Rotation 20 slots: royale x2
  added; `BALLS_DEBUT` royale 1 (the next autopilot video). White winners
  (England, Snow White) get a dark text outline (`_stroke`). ~1.75 min to
  make a 35 s video.
- **Finish the job (26 Oct):** Dean: some videos stop at the final picture
  with the job half done (ring not full, box not full, spikes not
  covering the walls). New recipes of `be.FILL_KINDS` carry
  `recipe["fill"]`; the final picture still comes when it did
  (`ArenaSim.final()`), then the video goes on until it's visibly done:
  escape keeps adding balls to `FILL_SHARE` 0.74 of the ring's area (it
  fills to the top; at 145 gravity left the top half empty), spiral until
  every wall piece is gone (a ball per piece), box until `B_FULL` 1150
  balls are in it (full to the top; the live cap rises after the final),
  spikes grow by themselves every `S_AUTO` 0.12 s in free spots (not on
  the ball, which can't pop after the final form) until the walls are
  covered, trail until the ball's radius is `T_FILL_END` 0.9 of the ring.
  Adds ~1-8 s. Recipes without the field (every older video) simulate
  exactly as before (checked on 12 recipes of 6 kinds). Shrink, touch,
  zen, merge, center, jaws, break in already ended on a finished job.
- **Learns from its numbers (29 Oct):** Dean: views aren't climbing, "it
  sees which video gets a lot of views ... and then it decides which videos
  to make". `clipper/ball_learn.py`: once a day (`_balls_autopilot_tick`,
  when `learn.json` is over 20 h old; or "🧠 Learn now", `POST
  /api/balls/learn`) it reads the channel's own YouTube Analytics per video
  (views, engagedViews = kept watching, likes, shares, average % watched;
  report sorted by views, retried without engagedViews on a 400), matches
  them to each video's recipe (fights split by look: `duel:flags`...),
  counts only videos at least 48 h old, and Claude sets the mix
  (`weights` per kind, `looks` per fight, 2-4 plain `notes`). Guardrails
  in code (`clamp` / `_spread`): every `BALLS_FORMAT_CYCLE` kind >= 2%,
  none over 34%, each fight look >= 10%, Evolve never. No key or no answer:
  the same table by a formula (`formula_mix`). `_balls_format_for` then
  draws each slot's kind from the mix (`_balls_learned_kind`: seeded by
  slot + learn time, never the kind the slot before got), `_balls_make`
  forces the fight's look (`duel_recipe(recipe, force=...)` etc.; with no
  `force` the recipes are byte for byte as before, checked on 1,200), and
  `BALLS_DEBUT` still goes first. Setting `learn` (default on); off = the
  fixed cycle. Page: "🧠 Learning from your numbers" (notes, per-kind
  table, share of the next videos). Store: `BASE_DIR/_balls/learn.json`.
  Dean was told it shifts the mix toward kinds that keep viewers, but
  can't promise a breakout: most Shorts sit at the 1-2k test audience.
  31 Oct: a run at 05:07 UTC got no rows from YouTube (its endDate was
  still tomorrow on Pacific time) and reset every kind to an even 7%.
  `fetch_stats` now ends yesterday (UTC), and a run that matches 0
  videos while the last one had some raises instead of saving (the mix
  stays; it retries in 3 h).
- **The daily report, for Claude Code (29 Oct):** Dean: "let it pull a
  report ... so that you can read it. Then you can edit it as you see
  fit." Each learning run also writes `BASE_DIR/_balls/report.json` and
  prints it to the app's logs as `[balls-report] {json}` lines, all with
  the same `run` time: `summary` (channel middle, Claude's notes), `mix`,
  one `kind` line per kind, one `video` line per posted video (kind/look,
  title, hook, pictures, sound, length, views, stayed %, avg % watched,
  likes, comments, shares, age) and `curve` lines (share still watching
  at 5%, 10% ... 100%) for the 3 best and 3 worst videos over 48 h old,
  then `end`. **To read it:** Railway `get-logs` (project/service above)
  with `filter: "balls-report"`, `limit: 500`, and keep the latest `run`.
  Use it to change the videos themselves (pacing, hooks, endings, sounds),
  not just the mix; tell Dean what the numbers showed. Dean can download
  it from the page (`GET /api/balls/report`).
- **First report's fixes (29 Oct):** the watch curves showed viewers
  leaving the moment the result shows (best duel: 80% still watching at
  90%, 26% at the end), so `END_HOLD` is 1.8 s in every kind (was 3.2-4;
  every end animation is done within ~1 s) and `render` fades the last
  `END_FADE` 0.35 s of audio so the loop back isn't a cut-off sound. The
  flag royale lost half its viewers 5-10 s in, before any knockout: now
  `R_HEARTS` 3, `R_SPAWN` 420, zone from 3 s at 16/s, first knockout by
  `R_FIRST_KO` 3 s (median 3.2 s, was 6.2), ~22 s long, `good_end`
  18-38 (12 of 40 seeds pass). Claude's mix is limited to the
  `BALLS_FORMAT_CYCLE` kinds (the first run brought back rays/touch/break
  in/trail). The report's `pictures` for fights names the contestants.
- **Survivor, new weapons, clean look (29 Oct):** Dean: "change the
  visuals a bit... new weapons or new battle royale or whatever", after the
  research. (1) `format: "survivor"` (`clipper/ball_survivor.py`, NO_PACK,
  `be._parts` dispatches; label "🌍 Countries, 1 ring, 1 survivor"): 24
  flag balls (`S_N`) bounce at a steady speed in a ring whose gap turns
  (`S_SPIN`) and widens as they go (`S_GAP0` x (N/left)^`S_GAP_GROW`); a
  ball that finds the gap falls out and is crossed out in the grid on top
  ("N LEFT", "X IS OUT"), the last one wins ("X SURVIVES!"). Red caps mark
  the gap's ends. ok = first out by 2.5 s, no wait over 4.5 s; `good_end`
  20-36, ~1 seed in 2, ~23 s long. Description opens "Comment your country
  before you watch 👇" (the reel it's from asked that). In the cycle x2
  (22 slots) and `BALLS_DEBUT` 2. (2) `recipe["arsenal"] = 2` (every new
  duel / 2v2 / tournament / royale) adds `WEAPONS2`: laser (aims with a
  red sight line `LASER_CHARGE` 0.5 s, then a beam to the ring's edge, 2
  hearts to everyone on it), saw (spinning blade, 3 hits), freeze gun (1
  heart + frozen `FREEZE_S` 1.5 s: can't move or use its weapon), bomb (put
  down near the enemy, `BOMB_FUSE` 1.4 s, the blast zone pulses red; uses
  `_boom`). Weapons drop every 1.5-2.5 s with it (was 2-3.2): duels median
  ~34 s (was ~36). Recipes without `arsenal` simulate exactly as before
  (hash-checked on 12 duels/tournaments). (3) `recipe["style"] = "clean"`
  (new duels and 2v2s only; royale/tournament keep their own HUDs): the
  arena drawn 1.08x and higher (`C_CY` 1050, via `place`), fighting-game
  health bars (a segment per heart, the ones just lost turn red and drain),
  a VS badge, the hook under them, colour trails, a white flash on the
  knockout, and 2v2 rings in the team's own colours (they used to follow
  the team's place, so a blue team could have orange rings).
- **Idea board: weapons + arenas (29 Oct):** Dean was shown a board of 9
  arena and 16 weapon mockups (sandbox only). He dropped lava ring, the
  cage, black hole and comic; kept the rest and all the weapons.
  (1) `WEAPONS2` gained katana (dash, 2 hearts), hammer (shockwave), boomerang
  (curves back, can hit twice), axe (thrown, 2), bow (draws, arrow goes
  through everyone, 2), lightning (chains to 3 enemies), flamethrower (cone,
  `FLAME_S`, max 3 hearts), magnet (pulls in, contact = 1), banana (peel
  trap: 1 + spin-out, reuses `frozen` + `spin`), glove (fling; a wall right
  after = 1 more, `_wall_slam`), pan (bats bullets back, bonks on contact),
  tornado / black hole / Moai boulder (`self.things`, `_things`), wand
  (`small` for `SMALL_S`: +1 per hit, drawn 0.65x), potion (heal 2 / fast /
  giant). New duels pick `recipe["weapons"]` (4 `CLASSICS` + 6 `NEWER`,
  own rng `weapons-<seed>`); tournaments and royales draw from all of
  `WEAPONS2`. Status timers live in `_status`, speeds in `_spd` (always
  `D_SPEED` for older fights). Pictures: Fluent Emoji 3D as white-outlined
  stickers (`_sticker`, `EMOJI_ARMS`, `HELD_ANGLE` turns each to aim; 7 more
  emoji copied in from `@lobehub/fluent-emoji-3d`); sounds are numpy
  (`_sound`: slash, slam, twang, thunder, fire, magnet, slip, punch, bonk,
  wind, hole, rumble, sparkle, gulp). Duels median ~32 s, every weapon
  gets used (80-seed run).
  (2) `recipe["arena"]` for new duels / 2v2s (`ARENAS`, own rng): neon (the
  clean look), notebook (lined paper, pen-drawn ring, ink outlines, pencil
  trails, Permanent Marker titles), chalk (chalkboard, chalk ring, Gochi Hand
  titles), ice (rounded rink, skate-mark trails) and boxing (square ring,
  ropes, crowd). Ice and boxing have box walls (`ARENA_SHAPES`, `_box_wall`
  / `_box_bounce` / `_outside`; spawns, bullets, rockets and the laser use
  them). Fonts: `PermanentMarker-Regular.woff` (Apache 2.0) and
  `GochiHand-Regular.woff` (OFL) from @fontsource, opened by path like the
  others.
  (3) Falling floor (`recipe["floor"]`, half of new royales, own rng
  `floor-<seed>`, `FLOOR_HOOKS`): hex tiles (`F_TILE`) crack `F_CRACK` s
  before they drop, outside first from `F_START` at `F_SPEED`, the middle
  `F_KEEP` stays; a ball over a hole falls (shrinks away) and is out; balls
  turn back from a hole `F_DODGE` of the time. ~17 s, ~6 falls, `good_end`
  15-34 (1 seed in 3). Recipes without these fields simulate and draw
  exactly as before (hash + pixel checks on duels, tournaments, royales).
- **Ball League (31 Oct):** Dean asked for live numbers + research + one
  good idea to grow the channel (one video in a few takes off, the rest sit
  at the 1-2k test audience). Numbers: the colour duels were the only
  breakout (Purple VS Red 14.6k, 68% kept watching); colour tournament 136,
  flag royale 877. Research (subagents; vidIQ ran out of credits, YouTube /
  Social Blade blocked in the sandbox): growing ball-sim channels post daily
  without gaps and own one promise; the biggest ball-fight channel
  (Earclacks' Weapon Balls, ~209K) grew on a fixed cast with records and
  leagues; YouTube pushes a Short past the test audience on "chose to
  watch" (70-90% in the winners). So every colour 1v1 duel is now a league
  fight (`clipper/ball_league.py`): the champion (left, 👑, "CHAMPION 5-1")
  VS a challenger (record or "NEW"), a pill "REMATCH · FIGHT #7", league
  hooks ("Can anyone stop X? 3 wins in a row"), and under "X WINS!" "NEW
  CHAMPION!" / "KEEPS THE BELT · N IN A ROW". Title "<hook> ⚔️ Ball League
  #N"; the description asks "Who should fight the champion next? Comment a
  colour 👇". Nothing is stored apart from the videos: each league video's
  recipe has `league` (no, names, records, tag, streak, champ, and after
  render `winner` / `winner_hp` via `render`'s info), and the standings
  before a fight are the league videos posted or slotted earlier
  (`_balls_league_history`, `_balls_when`: YouTube time > slot > made),
  minus ones removed from YouTube or hidden before posting. Matchmaking
  (`next_match`): fight #1 is the Purple VS Red rematch; Dean's pick
  (`settings["league_next"]`, used once, the page's "Next challenger"
  menu) > a rematch after a close fight (winner on <= 2 hearts; never a
  rematch of a rematch) > a colour that hasn't fought (seeded by the
  fight number, so the page's "Next fight" matches) > the best record;
  nobody from the last two fights, no two of a colour (Lime / Mint Green).
  Roster: the 8 duel colours + Mint Green, Deep Maroon, Navy Blue, Choco
  Brown, Shiny Gold (Coal Black, Snow White, Steel Grey left out: they
  vanish on the dark arenas / the notebook's paper). `settings["league"]`
  (default on; off = duels as before) and `league_slot` (was "15:00", now
  "" -- see Comment fights): that slot is always a league fight (`_balls_league_slot`, before the
  learned mix and BALLS_DEBUT; the learned chain treats it as a duel so
  the next slot isn't one too). Flag and 2v2 duels only happen with the
  league off. Page: "🏆 Ball League" (standings, next fight, last 5).
  Duels without `league` draw byte for byte as before (checked).
  Not built, told Dean: a first comment and playlists need a reconnect with
  the `youtube.force-ssl` scope; pinning a comment has no API.
- **Comment fights (31 Oct):** Dean asked for numbers + research + one idea
  to grow the channel (one video in a few takes off). The colour duel was
  the only breakout (Purple VS Red 14.6k); a "Ball League" of colours was
  built (PR #142) but Dean said nobody relates to "Royal Purple"; #142 was
  merged anyway, and since #143 the league only covers the colour 1v1s the
  08:00 mix draws (its `league_slot` defaults to "" and never takes the
  comment fight's slot: `_balls_league_slot` checks `_balls_crowd_slot`). Pitched three relatable ideas (viewers' names as the balls,
  "which one are you" groups, everyday debates); he picked the first, asked
  for "better images", the 15:00 video always a comment fight, the 08:00
  one still changing, and no need to check on it. Research: Marble Drop
  Daily runs the same idea with marbles on Shorts (no growth numbers found);
  YouTube's fake-engagement policy bans content whose *only* purpose is
  rewarding comments, so: no prizes, never "like/subscribe to enter",
  "remove me" takes a person out. `clipper/ball_crowd.py` (`format:
  "crowd"`, NO_PACK; label "💬 Comment fight"): `fetch` reads top-level
  comments (commentThreads.list, the channel's own token -- its
  `youtube.readonly` scope already covers it, 1 unit per video) on the
  channel's videos from the last `LOOKBACK_DAYS` 21, never its own;
  `update_people` keeps one entry per channel (newest comment);
  `moderate` (Claude, batches of 40, once per comment) sets `name_ok`,
  `quote` (<= 48 chars of their own words, family-friendly, or null) and
  `opt_out`; the word list `_BAD` / `_OPT_OUT` also has a say, and is all
  there is without a key (no quotes then). `cast`: 3+ people waiting
  (`ROYALE_MIN`) -> a 16-ball royale (champion, the waiting ones oldest
  first, then people who were in before, then "Bot N" balls); fewer -> a
  1v1 duel, champion VS the next one, each with their comment as a card
  (character, name, line) for the first `CARD_S` 4 s. Every person gets a
  fixed character (`CHARACTERS`, 55 bundled Fluent 3D emoji, by channel id)
  and colour. A bot never wins while real people are in it (`_real_wins`
  makes that seed not ok). The royale shows everyone's character + name in
  the top grid (crown on the champ), names under the balls, "@x IS OUT";
  the end says "@x WINS!", "Keeps / Takes / Wins the crown. Defends it next
  fight." and a "Comment to get in the next one" button. Names keep their
  case (`Handle`, `DuelPainter.UPPER_NAMES`). The sims are RoyaleSim /
  DuelSim (draws from `royale_recipe` / `duel_recipe` with colours, then
  `apply`); a ball with `pic` (5th player field) draws the character
  instead of the face -- duels/royales without it draw byte for byte as
  before (checked). After render, `info["crowd"]["winner"]` (cast index).
  App: `settings["crowd"]` (on), `crowd_slot` "15:00", `crowd_blocked`.
  `_balls_format_for` gives the crowd slot "crowd" (before the learned mix
  and BALLS_DEBUT; the learned chain skips it), the planner only plans it
  `BALLS_CROWD_AHEAD_H` 8 h ahead (made at the 08:05 waker, so the newest
  comments get in), and `_balls_next_free_slot` keeps it free. At make:
  `_balls_crowd_cast` -> `_balls_crowd_refresh` (skipped if read < 20 min
  ago) -> `cast` with who's been in (`_balls_crowd_used`) and the champion
  (`_balls_crowd_champ`: winner of the latest earlier comment fight,
  `_balls_when` order), both from the videos' recipes; the store is only
  `BASE_DIR/_balls/crowd.json` (people + checks). Page "💬 Comment fights":
  on/off, last read, the crown, who's waiting (with "Take out"), who's left
  out (and "Let back in"), "Read the comments now"
  (`POST /api/balls/crowd/refresh`, `POST /api/balls/crowd/block`). The
  report names the commenters (`ball_learn._pictures`).
- **Art looks (31 Oct):** Dean asked for new designs / 2D art on the 08:00
  videos, rotating, after research. Research: no study says one art style
  wins for ball videos (sources are mostly tool sellers); what holds is a
  bold, high-contrast first frame and YouTube's July 2026 rule against
  channels whose videos look like one template. A board of 8 mockups (neon,
  notebook, chalk, pixel, paper cut-out, blueprint, sticker pop, arcade);
  Dean said "do it" to the suggested set. `clipper/ball_looks.py`:
  `ARTS` neon / pixel / sticker / notebook / arcade, `recipe["art"]`,
  applied over each finished frame in `be.render` (`Look.apply`), so every
  kind gets it without its painter changing; recipes without `art` (and
  "neon") render exactly as before. pixel = 1/4 size, 5 levels per
  channel, NEAREST back up; arcade = brightness to green phosphor + glow +
  scanlines + vignette; sticker / notebook swap the painter's `bg` (the
  backdrop it draws on) for yellow stripes / lined paper and outline
  everything drawn on it (pixels > `FG_DIFF` 40 from `bg`, worked out at
  half size; cv2 dilate) in thick black / thin ink. Turning white to ink
  on the paper made white pictures black, so it's outlines only. Kinds:
  `KINDS` (every non-fight kind but Evolve; fights keep their arenas, 15:00
  is the comment fight); Hold the center skips sticker / notebook (its
  black hole is the backdrop's black). `_balls_make` picks one
  (`ball_looks.pick`: never the last made video's look,
  `_balls_last_art`) and sets `backdrop` "black" for pixel / sticker /
  notebook (a space nebula pixelates into blobs). ~40 ms a frame extra.
  Claude's last look is told the look is on purpose (`ball_review._ART_NOTES`),
  the report has an `art` field per video so the numbers can show which
  look keeps viewers, and cards show "🎨 <look>".
- **Music only on the growing ball (21 Oct):** Dean: his songs only on
  `shrink` ("🔴 Ball grows every bounce"); every other kind plays just ball
  sounds and cursed sounds. `_balls_sound` in `_balls_make`: a song for
  `BALLS_SONG_KINDS` (or any kind when a Make picks one song by hand);
  otherwise `BALLS_NO_SONG_SOUND` sets the bounce sound -- the cursed meow
  (deeper every bounce) for the ring/arena kinds, ball taps
  (`ball_sounds.knock`, bounce_sound "knock") for Merge, whose merges make
  the cursed noises, and "none" for duel/race (their own hit/shot/steal
  sounds only). Escape, touch, break in and evolve have no bounce music:
  their ball sounds (+ the noise bed setting). The piano tune and phonk
  riff are no longer used by the autopilot.

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
