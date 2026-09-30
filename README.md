# clipper

A personal project: an AI-powered video clipping tool that turns a long
video (YouTube link, Twitch VOD, or a local file) into a batch of short,
vertical (9:16), captioned highlight clips — the kind of tool Opus Clip /
Vizard / Klap sell as a subscription, except this one runs on your own
machine (or your own Railway deployment), for the cost of a few cents of
Claude API usage per video. Built to run my own clipping workflow for the
**Caught On Stream** channel — I post a couple of clips a day, picked and
rendered through this instead of a paid SaaS.

Usable two ways: as a **command-line tool** (`clipper/`) for one-off local
runs, or as a **web app** (`webapp/`, see [Web app](#web-app)) that wraps
the whole pipeline behind a browser UI you can deploy to Railway and use
from your phone. The web app adds finding what to clip, uploading straight
to YouTube, channel analytics that feed back into how clips are picked, and
a long-form documentary series tool.

## Key features

- **AI clip selection** — Claude reads the full transcript and picks the
  most "clippable" moments (funny lines, hot takes, payoffs, hooks),
  steerable with a focus prompt, and explains *why* it picked each one,
  with a score and a breakdown (hook, controversy, reaction, payoff, makes
  sense alone). Cuts land exactly on word boundaries.
- **It learns from your channel** — the web app reads your channel's real
  YouTube retention curves and view counts, compares what your posted
  Shorts had in common, and feeds those findings (plus a saved AI strategy
  overview) back into how the next clips are picked and titled.
- **IRL layout by default** — every clip shows the whole scene,
  letterboxed into 9:16. Any clip can be switched to a facecam-over-gameplay
  split with a box-picker (once per job is enough: "apply to the other
  clips"), or you can untick "IRL layout" to have facecams auto-detected
  (OpenCV face detection with a Claude-vision check).
- **Karaoke-style burned-in captions** — word-by-word highlighted captions
  (ffmpeg/libass), plus a hook line at the top for the first 3 seconds, the
  channel mascot and name, and quiet pauses cut out.
- **Long Twitch VODs** — finds candidate highlight windows from chat
  spikes and existing viewer clips *before* downloading anything, so an
  hours-long stream doesn't need to be fully fetched and transcribed.
- **Finding what to clip** — tracked streamers' latest VODs and uploads,
  yesterday's VODs, what's trending live on Twitch, suggested creators, a
  creator search, and an AI "which VOD to clip today" recommendation.
- **Posting** — one-click upload to YouTube as a Short (with a trim step),
  generated titles/descriptions, downloadable thumbnails (a real frame plus
  one bold line of text) and a separate spoiler-style "hook line" pass.
- **Long-form documentaries** — "The Story Of <streamer>", a bi-weekly
  series: research, a Claude-written script, you record the narration
  scene by scene, keyword visuals, render, upload, and two promo Shorts.
- **Telegram notifications** when a job finishes, plus a Sunday reminder
  to post.

## What it does (the clip pipeline)

1. Downloads the source video (`yt-dlp`), or points at a local file, and
   grabs YouTube's auto-captions if available. For a long Twitch VOD it
   first scans chat replay and existing viewer clips for highlight
   windows, then downloads only those.
2. Gets a transcript with word-level timing (local Whisper by default in
   the web app; YouTube's captions are the faster option).
3. Scans the audio for loud moments (shouting, big reactions) as an extra
   signal.
4. Sends the transcript to Claude, which picks the N most clippable
   moments and writes a title, a hook caption, an upload title and a
   description for each. Its rough times are snapped to exact word
   boundaries.
5. For each moment: cuts quiet pauses (optional), picks the layout (IRL by
   default), builds the burned-in captions, and renders the clip with
   ffmpeg.
6. Writes the clips plus their metadata (titles, timestamps, hook text, why
   Claude picked each one). The CLI writes a `clips.json`; the web app
   keeps it with the job.

## Tech stack

- **Python 3.10+**: `clipper/` (CLI + pipeline) and `webapp/` (FastAPI),
  one codebase for both.
- **[Claude](https://www.anthropic.com/claude)** (Anthropic API): moment
  selection, facecam checks, strategy overviews, hook lines, and the
  long-form research, scripts and visual plans.
- **yt-dlp** (with Deno for YouTube's signature challenge): video and
  caption downloads.
- **faster-whisper**: local speech-to-text with word-level timing.
- **OpenCV**: face detection for automatic facecam layouts.
- **ffmpeg** (via subprocess): cutting, cropping, burning in `.ass`
  captions.
- **Pillow**: long-form cards, captions and frames.
- **chat-downloader**: Twitch VOD chat replay for highlight detection.
- **pocket-tts** (Kyutai Pocket TTS, CPU PyTorch): the long-form "AI voice",
  a clone of your own voice for patching narration lines.
- **FastAPI + Uvicorn**: the web app, deployed as a **Docker** container on
  **Railway**.

## Setup (Windows)

You'll need three things installed once:

1. **Python 3.10+**: https://www.python.org/downloads/ (tick "Add
   python.exe to PATH" during install).
2. **ffmpeg**: https://www.gyan.dev/ffmpeg/builds/ (grab the "essentials"
   build, unzip it, add the `bin` folder to your PATH). Confirm with
   `ffmpeg -version` in a new terminal.
3. **This tool's Python dependencies**, from the repo folder:

   ```
   pip install -r requirements.txt
   ```

   (or `pip install -e .` if you want the `clipper` command available
   globally instead of running it as `python -m clipper.cli`).
   `requirements.txt` includes the long-form AI voice (`pocket-tts`, which
   pulls in PyTorch). For CLI-only use, installing the other lines is
   enough.

### Anthropic API key

Clip selection is done by Claude, so you need an API key:

1. Get one at https://console.anthropic.com/ (pay-as-you-go; the clip
   prompts are text-only and cheap, typically a few cents per video).
2. Set it as an environment variable before running:

   ```
   setx ANTHROPIC_API_KEY "sk-ant-your-key-here"
   ```
   (close and reopen your terminal after `setx`, or set it for the current
   session only with `set ANTHROPIC_API_KEY=sk-ant-...`)

   Or copy `.env.example` to `.env`, fill in your key, and load it however
   you prefer.

## Usage (command line)

Basic: 5 clips from a YouTube video, using its own captions:

```
python -m clipper.cli "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Clips land in `./clips/` by default: `clip_01.mp4`, `clip_02.mp4`, ...,
plus `clips.json`.

More options:

```
python -m clipper.cli SOURCE [options]

  SOURCE                  YouTube/Twitch URL or path to a local video file

  -o, --out PATH          Output folder (default: ./clips)
  -n, --clips N           Number of clips to produce (default: 5)
  --min-len SECONDS       Minimum clip length (default: 20)
  --max-len SECONDS       Maximum clip length (default: 90)
  --focus "TEXT"          Steer what Claude looks for,
                          e.g. --focus "funniest moments"
  --whisper               Force local Whisper transcription instead of
                          YouTube's captions (slower, but works on videos
                          with no captions, and is more accurate)
  --whisper-model SIZE    tiny / base / small (default) / medium / large-v3
                          -- bigger = more accurate, slower, more RAM
  --api-key KEY           Anthropic API key (defaults to ANTHROPIC_API_KEY)
  --width / --height      Output resolution (default: 1080x1920, i.e. 9:16)
  --no-captions           Skip burning in captions
```

Examples:

```
# Local file, 3 short clips, no captions
python -m clipper.cli "C:\Videos\podcast_ep12.mp4" -n 3 --max-len 45 --no-captions

# YouTube video with no captions -- use Whisper, look for funny bits
python -m clipper.cli "https://youtu.be/XXXXXXXXXXX" --whisper --focus "funniest moments"
```

The CLI uses automatic layouts (face detection) and plain captions. The IRL
default, pacing edits, branding, uploads and analytics are web-app features.

## A note on whose videos to clip

This tool works the same way on your own uploads and on other people's
public videos — it's just software, it doesn't know the difference. The
difference that matters is legal/platform, not technical: clipping your
own content is obviously fine; clipping someone else's and re-publishing
it can run into copyright and platform ToS issues depending on how much
you use, how you use it (commentary/criticism/fair use vs. straight
re-upload), and where you post it. Worth a quick gut-check per video
before you publish, not before you experiment locally.

## How it's organized

```
clipper/                    # pipeline + CLI, importable on its own
  cli.py                     # command-line entry point
  download.py                # yt-dlp wrapper + captions (incl. ranged downloads)
  transcribe.py              # captions / Whisper -> word-level timing
  select_moments.py          # Claude: which parts to clip, and why
  cut_points.py              # snap rough clip times to exact word boundaries
  loud_moments.py            # loudness spikes, an extra selection signal
  highlights.py              # chat-spike + viewer-clip signals for long Twitch VODs
  long_vod.py                # long-VOD pipeline: candidate windows -> download -> select
  edit_plan.py               # pacing: cut dead air, optional payoff teaser
  reframe.py                 # layouts: IRL letterbox, face-detected crop, facecam split
  facecam_vision.py          # Claude-vision check of facecam placement
  captions.py                # burned-in .ass captions, hook text, mascot + name stamp
  render.py                  # ffmpeg cut/crop/caption render, trim, hook-line overlay
  hook_line.py               # spoiler-style flash hook line for a finished clip
  thumbnail.py               # real frame + one bold line -> downloadable thumbnail
  trending.py                # tracked streamers' latest content, trending live, VOD candidates
  clip_registry.py           # every rendered clip, kept after its job is deleted
  clip_features.py           # measurable traits of a clip (time to speech, pacing...)
  clip_performance.py        # posted Shorts' real retention vs what the clips had in common
  channel_insights.py        # no-auth channel snapshot + best-day-to-post heuristic
  youtube_analytics.py       # real Analytics for the OAuth-connected channel
  youtube_oauth.py           # Google OAuth flow for the connected channel
  youtube_upload.py          # resumable upload of a clip (or long-form video) to YouTube
  channel_strategy.py        # Claude-written strategy overview + "which VOD today"
  competitor_discovery.py    # find channels clipping a given streamer
  competitor_content.py      # what's in a competitor's top Shorts (transcript + loudness)
  notify.py                  # Telegram notifications
  documentary.py             # long-form: research + story for "The Story Of"
  longform.py                # long-form: projects, scene-by-scene recording, misread check
  longform_beats.py          # long-form: keyword visuals timed to the narration
  longform_lessons.py        # long-form: editing lessons fed back into the prompts
  longform_video.py          # long-form: rendering (Pillow frames into ffmpeg)
  visual_sources.py          # long-form: free visuals (emoji, stock, CC0 photos, X posts)
  voice_clone.py             # long-form: your cloned voice (Pocket TTS)
  assets/                    # bundled emoji (Fluent Emoji 3D, MIT) and fonts (OFL)
webapp/
  main.py                    # FastAPI app: job queue, API routes, HTML pages
Dockerfile
pyproject.toml
requirements.txt
.env.example
```

## Web app

`webapp/main.py` is a FastAPI app around the same pipeline: paste a video
URL (or pick one it suggests), it downloads, transcribes, picks and renders
in the background, and you review, download or upload the clips from the
page. One job runs at a time; jobs are saved to disk, so they survive a
restart.

Deploy: push this repo to Railway (the `Dockerfile` installs ffmpeg, Deno,
CPU-only PyTorch and the Python deps), set `ANTHROPIC_API_KEY`, mount a
volume and point `CLIPPER_JOBS_DIR` at it, and generate a domain. "Sleep on
idle" is safe to enable: while a job runs, the app pings its own `/healthz`
so the container isn't put to sleep mid-render.

**Password-protect it** (recommended for a public domain) by setting
`APP_PASSWORD`. Every route except `/healthz` then requires HTTP Basic
auth (any username, that password). Leave it unset for local dev.

**YouTube downloads from a cloud IP**: datacenter IPs regularly hit
YouTube's "sign in to confirm you're not a bot" wall, which only real
session cookies get past. Export cookies from a logged-in browser (e.g. the
"Get cookies.txt LOCALLY" extension) and paste the file's contents into the
`YTDLP_COOKIES` variable, or point `YTDLP_COOKIES_FILE` at a file already
on disk.

Runs locally too:

```
pip install -r requirements.txt
uvicorn webapp.main:app --reload
```

### Pages

**Home (`/`)**: making clips.

- *What to clip*: yesterday's VODs from your tracked streamers, the latest
  VODs and uploads, what's trending live on Twitch (100k+ viewers),
  suggested creators not on your watchlist, a creator search, and
  "Recommend one" (Claude ranks your tracked streamers' recent VODs and
  checks the top picks are actually downloadable).
- *Options per job*: number and length of clips, a focus or mood,
  Whisper captions, tighten pacing, payoff teaser, mascot + channel name,
  IRL layout, hook text.
- *Per clip*: preview, score and breakdown, copy title/description,
  **Upload to YouTube** (Unlisted by default, optional trim; uploads over
  60 s are refused because they wouldn't become Shorts), **Thumbnail**
  (pick one of several real frames, edit the text, download), **Switch to
  facecam / Adjust facecam**, delete.
- *Per job*: "Generate more clips" reuses the downloaded source with no
  re-download; stop/save a running job; an "Active & saved jobs" list.
- A link to the long-form page.

**Analytics (`/analytics`)**: channel insights and best day to post, "Is it
growing?", "What your Shorts actually do" (real retention curves compared
across what your clips had in common), your top 10 videos, comparison
against other channels clipping the same streamers, and an AI strategy
overview. The retention findings and the latest saved overview are fed
into clip selection, which is how the app learns from your channel. Connect
your YouTube account here (see [YouTube OAuth](#youtube-oauth-analytics-and-upload)).

**Hook Line (`/hook-line`)**: pick a finished clip, have Claude write a
spoiler-style one-liner (or write your own), and render it flashed at the
very start of the clip as a separate pass. The original clip isn't touched.

**Long-form (`/long-form`)**: "The Story Of <streamer>", a bi-weekly
documentary series for the same channel.

1. Pick a streamer (a schedule slot every 14 days, or start one now). It
   researches their Twitch profile, their most-viewed clips (all-time and
   per year, downloaded and transcribed), Wikipedia, any article links and
   your notes.
2. Claude writes the story as scenes: narration (you read it, over a
   clip), moments (a clip playing at full volume with subtitles) and
   chapter cards. It's editable, and facts and quotes must come from the
   research.
3. Record each narrated scene in the browser. Misreads are flagged. "Use my
   AI voice" can patch a line with a clone of your own voice; the page
   warns if you lean on it for more than a third of the scenes.
4. Keyword visuals change on the words you say: emoji, stats, stock video,
   photos, posts, headlines, timelines. Optional background music has
   quieter/normal/louder levels.
5. Render (1080p), write a title and a description with chapters and
   credits, upload to YouTube as a regular video (private by default), and
   make two promo Shorts that link back to it.

Editing lessons (seeded from YouTube's review of the first episode, and
anything you paste in) are kept and fed into the next scripts.

Other background behaviour: a Telegram message when a job finishes (and
how many clips need a facecam placed), and a Sunday reminder to post.

### Environment variables

All configuration is via environment variables (Railway service variables,
or a local `.env`), never hardcoded.

| Variable | Needed for | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | everything | Claude API key |
| `CLIPPER_SELECT_MODEL` | — | Claude model for picking and cutting clips (optional; default `claude-opus-5`) |
| `CLIPPER_MODEL` | — | Claude model for everything else, and the fallback when the clip-picking model is unavailable (optional) |
| `APP_PASSWORD` | — | HTTP Basic auth password; unset = no auth (local dev only) |
| `CLIPPER_JOBS_DIR` | production | where jobs, clips, the clip registry and long-form projects live; point it at a mounted volume |
| `CLIPPER_BRAND_NAME` | — | channel name stamped beside the mascot (optional; default `Caught On Stream`) |
| `YTDLP_COOKIES` / `YTDLP_COOKIES_FILE` | YouTube downloads from a cloud IP | exported browser cookies (contents, or a path) |
| `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET` | Twitch lookups | trending, VOD suggestions, viewer-clip signal for long VODs, long-form research |
| `TRENDING_TWITCH_LOGINS` | tracked streamers | comma-separated Twitch logins |
| `TRENDING_YOUTUBE_CHANNELS` | — | comma-separated YouTube channel IDs / `@handles` for the "latest uploads" row |
| `YOUTUBE_API_KEY` | trending, channel insights, competitor search | public YouTube Data API v3 key |
| `YOUTUBE_OWN_CHANNEL` | no-auth channel insights | your channel's ID / `@handle` |
| `YOUTUBE_OAUTH_CLIENT_ID` / `YOUTUBE_OAUTH_CLIENT_SECRET` | Analytics + uploads | Google Cloud OAuth web client (see below) |
| `RAILWAY_PUBLIC_DOMAIN` | OAuth, keepalive | set by Railway automatically; used for the OAuth redirect URI and the self-ping |
| `TELEGRAM_BOT_TOKEN` (or `CLIPPER_BOT_API`) | Telegram notifications | bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | optional; discovered from the bot's recent messages if unset |
| `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) | long-form AI voice | Hugging Face token from an account that accepted the terms at huggingface.co/kyutai/pocket-tts |
| `PEXELS_API_KEY` / `PIXABAY_API_KEY` | long-form stock video | free keys; without them stock video is skipped |
| `CLIPPER_FFMPEG` | — | ffmpeg binary for long-form rendering (optional; default `ffmpeg`) |

### Channel profiles

Channel-specific settings live in `CHANNEL_PROFILES` in `webapp/main.py`:
which Twitch logins to track, the on-clip brand name and mascot colour,
the YouTube token file and the language Claude writes clip text in. Today
there is one profile, `main` (Caught On Stream). A second channel can be
added as another entry. Jobs remember which profile they were made for, and
a clip only ever uploads through that profile's connected YouTube account.

### YouTube OAuth (analytics and upload)

Channel insights has two sources:

- **A no-auth heuristic** works immediately. Set `YOUTUBE_OWN_CHANNEL` to
  your channel's ID (starts with `UC...`), `@handle` or legacy username,
  and it uses the public Data API (`YOUTUBE_API_KEY`) to guess a best day
  from your recent uploads' view counts, normalized by video age. Noisy,
  especially with few videos.
- **Real YouTube Analytics** (day-of-week views, retention, traffic
  sources) for your connected channel. This is also what the upload
  buttons and the retention learning use. One-time setup:

  1. Open the [Google Cloud Console](https://console.cloud.google.com/) and
     create a project (or pick an existing one) via the project selector at
     the top of the page.
  2. Go to [APIs & Services → Library](https://console.cloud.google.com/apis/library)
     and enable both **YouTube Data API v3** and **YouTube Analytics API**.
  3. Go to [APIs & Services → OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent)
     and configure it: User type **External** is fine for personal use.
     Leave it in **Testing** mode (no Google verification needed) and add
     your own Google account under **Test users** — only test users can
     complete the login while it's in Testing mode.
  4. Go to [APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials),
     click **Create Credentials → OAuth client ID**, choose **Web
     application**, and add this as an **Authorized redirect URI** (swap in
     your actual Railway domain):
     ```
     https://<your-app>.up.railway.app/auth/youtube/callback
     ```
  5. Copy the generated **Client ID** and **Client secret**, and set them as
     Railway service variables: `YOUTUBE_OAUTH_CLIENT_ID` and
     `YOUTUBE_OAUTH_CLIENT_SECRET`.
  6. Redeploy, open the app, and click **Connect YouTube** (on Home or
     Analytics). It walks you through Google's consent screen and back.

  Note: even with a connected account, YouTube's Analytics API has no
  "hour of day" dimension on regular reports, so the audience-activity
  heatmap YouTube Studio shows isn't available. "Best day to post" means
  best *day of the week*, which is the most the official APIs expose.
