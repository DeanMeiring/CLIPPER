# clipper

A personal project: an AI-powered video clipping tool that turns a long
video (YouTube link, Twitch VOD, or a local file) into a batch of short,
vertical (9:16), captioned highlight clips — the kind of tool Opus Clip /
Vizard / Klap sell as a subscription, except this one runs on your own
machine (or your own Railway deployment), for the cost of a few cents of
Claude API usage per video. Built to run my own clipping workflow — I post
a couple of clips a day, picked and rendered through this instead of a
paid SaaS.

Usable two ways: as a **command-line tool** (`clipper/`) for one-off local
runs, or as a small **web app** (`webapp/`) — see
[Web app (Railway)](#web-app-railway) below — that wraps the whole
pipeline plus a few extra features (weekly/per-game recaps, channel
analytics, competitor discovery, direct YouTube upload) behind a browser
UI you can deploy and check from your phone.

## Key features

- **AI clip selection** — Claude reads the full transcript and picks the
  most "clippable" moments (funny lines, hot takes, payoffs, hooks),
  steerable with a `--focus` prompt, and explains *why* it picked each one.
- **Smart reframing** — face detection (OpenCV, with an optional Claude
  vision fallback for tricky facecam overlays) picks a 9:16 crop, and
  splits gameplay-plus-facecam footage into a stacked layout automatically.
- **Karaoke-style burned-in captions** — word-by-word highlighted
  subtitles rendered via ffmpeg/libass, not just a plain subtitle track.
- **Long-form Twitch VOD support** — finds candidate highlight windows
  from chat-activity spikes and existing viewer clips *before* downloading
  anything, so an hours-long stream doesn't need to be fully fetched and
  transcribed just to find the good parts.
- **Weekly and per-game recap videos** — auto-assembles a landscape
  countdown compilation of the week's best Twitch clips (across tracked
  streamers, or across every streamer playing one game), with intro/outro
  cards and rank badges.
- **Channel insights & strategy** — a no-auth heuristic (public YouTube
  Data API) or, with OAuth, real YouTube Analytics (day-of-week
  performance, retention, traffic sources), plus a Claude-written
  plain-language take on what to post more of.
- **Competitor discovery** — finds other channels clipping the same
  streamer and pulls their top-performing shorts' transcripts and loud
  moments as comparison signal.
- **One-click YouTube upload** and **auto-generated thumbnails** for any
  rendered clip, plus optional **Telegram notifications** when a job
  finishes.

## What it does

1. Downloads the source video (`yt-dlp`) — or just points at a local file
   you already have — and grabs YouTube's auto-captions if available.
2. Gets a transcript with word-level timing (from the captions, or from
   local Whisper if you pass `--whisper` / the source has no captions).
3. Sends the transcript to Claude, which picks the N most "clippable"
   moments (funny lines, hot takes, payoffs, hooks — you can steer this
   with `--focus`) and returns start/end times, a title, and a hook
   caption for each.
4. For each picked moment: detects faces to choose a smart 9:16 crop
   (falls back to a center crop if no face is found), builds burned-in
   karaoke-style word-by-word captions, and renders the final clip with
   ffmpeg.
5. Writes all clips plus a `clips.json` (titles, timestamps, hook text,
   why Claude picked each one) to your output folder.

## Tech stack

- **Python 3.9+** — `clipper/` (CLI + pipeline logic) and `webapp/`
  (FastAPI web app), same codebase for both.
- **[Claude](https://www.anthropic.com/claude)** (Anthropic API) — moment
  selection, content-strategy summaries, and optional facecam
  vision-detection fallback.
- **yt-dlp** — YouTube/Twitch video + caption download.
- **faster-whisper** — local speech-to-text fallback for word-level timing
  when a source has no captions.
- **OpenCV** — Haar-cascade face detection for the 9:16 crop/facecam split.
- **ffmpeg** (via subprocess) — cutting, cropping, and burning in `.ass`
  subtitles for every rendered clip.
- **chat-downloader** — Twitch VOD chat replay, for highlight detection on
  long streams.
- **FastAPI + Uvicorn** — the web app, deployed as a **Docker** container
  on **Railway**.

## Setup (Windows)

You'll need three things installed once:

1. **Python 3.9+** — https://www.python.org/downloads/ (tick "Add python.exe
   to PATH" during install).
2. **ffmpeg** — https://www.gyan.dev/ffmpeg/builds/ (grab the "essentials"
   build, unzip it, add the `bin` folder to your PATH). Confirm with
   `ffmpeg -version` in a new terminal.
3. **This tool's Python dependencies**:

   ```
   cd clipper
   pip install -r requirements.txt
   ```

   (or `pip install -e .` if you want the `clipper` command available
   globally instead of running it as `python -m clipper.cli`).

### Anthropic API key

Clip selection is done by Claude, so you need an API key:

1. Get one at https://console.anthropic.com/ (pay-as-you-go — this tool's
   prompts are text-only and cheap, typically a few cents per video even
   on long ones).
2. Set it as an environment variable before running:

   ```
   setx ANTHROPIC_API_KEY "sk-ant-your-key-here"
   ```
   (close and reopen your terminal after `setx`, or just set it for the
   current session with `set ANTHROPIC_API_KEY=sk-ant-...`)

   Or copy `.env.example` to `.env`, fill in your key, and load it however
   you prefer (e.g. `python-dotenv`, or just `set /p` it in a batch file).

## Usage

Basic — 5 clips from a YouTube video, using its own captions:

```
python -m clipper.cli "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Clips land in `./clips/` by default: `clip_01.mp4`, `clip_02.mp4`, ...,
plus `clips.json`.

More options:

```
python -m clipper.cli SOURCE [options]

  SOURCE                  YouTube URL or path to a local video file

  -o, --out PATH          Output folder (default: ./clips)
  -n, --clips N           Number of clips to produce (default: 5)
  --min-len SECONDS       Minimum clip length (default: 20)
  --max-len SECONDS       Maximum clip length (default: 90)
  --focus "TEXT"          Steer what Claude looks for,
                          e.g. --focus "funniest moments"
                          e.g. --focus "moments that explain a concept simply"
  --whisper               Force local Whisper transcription instead of
                          YouTube's captions (slower, but works on videos
                          with no captions, and is more accurate)
  --whisper-model SIZE     tiny / base / small (default) / medium / large-v3
                          -- bigger = more accurate, slower, more RAM
  --api-key KEY           Anthropic API key (defaults to ANTHROPIC_API_KEY)
  --width / --height      Output resolution (default: 1080x1920, i.e. 9:16)
  --no-captions           Skip burning in captions
```

Examples:

```
# Local file, 3 short punchy clips, no captions
python -m clipper.cli "C:\Videos\podcast_ep12.mp4" -n 3 --max-len 45 --no-captions

# YouTube video with no captions -- use Whisper, look for funny bits
python -m clipper.cli "https://youtu.be/XXXXXXXXXXX" --whisper --focus "funniest moments"
```

## A note on whose videos to clip

This tool works the same way on your own uploads and on other people's
public videos -- it's just software, it doesn't know the difference. The
difference that matters is legal/platform, not technical: clipping your
own content is obviously fine; clipping someone else's and re-publishing
it can run into copyright and platform ToS issues depending on how much
you use, how you use it (commentary/criticism/fair use vs. straight
re-upload), and where you post it. Worth a quick gut-check per video
before you publish, not before you experiment locally.

## How it's organized (if you want to poke at it)

```
clipper/
  clipper/                    # core pipeline + CLI, importable on its own
    download.py                # yt-dlp wrapper + caption fetch (incl. ranged downloads)
    transcribe.py               # VTT parsing / Whisper fallback -> word-level timing
    select_moments.py           # Claude call -> which parts to clip, and why
    loud_moments.py              # loudness-spike detection, a secondary signal for selection
    reframe.py                   # face detection -> smart 9:16 crop / facecam split layout
    facecam_vision.py             # Claude-vision fallback for tricky facecam overlays
    captions.py                    # word-level timing -> burned-in .ass karaoke captions
    render.py                       # ffmpeg crop+scale+caption render
    thumbnail.py                     # frame + burned caption -> downloadable thumbnail
    cli.py                             # command-line entry point, ties it all together
    highlights.py               # chat-spike + viewer-clip signals for long Twitch VODs
    long_vod.py                  # long-VOD pipeline: candidate windows -> download -> select
    trending.py                   # tracked streamers'/channels' latest content lookups
    competitor_discovery.py        # find channels clipping a given streamer
    competitor_content.py           # pull a competitor short's transcript + loud moments
    weekly_recap.py              # cross-streamer weekly countdown recap render
    game_recap.py                 # per-game weekly recap (reuses weekly_recap's pipeline)
    channel_insights.py          # no-auth best-day-to-post heuristic (public Data API)
    youtube_analytics.py          # real Analytics data for the OAuth-connected channel
    youtube_oauth.py               # Google OAuth flow for the connected channel
    channel_strategy.py             # Claude-written content-strategy summary
    youtube_upload.py            # resumable upload of a rendered clip to YouTube
    notify.py                     # Telegram notification on job completion
  webapp/
    main.py               # FastAPI app: job queue, all API routes, HTML pages
  pyproject.toml
  requirements.txt
  .env.example
```

Each `clipper/` module works standalone too, if you ever want to script
around just one piece of it (e.g. use `reframe.py` on its own to
auto-crop existing clips you already made elsewhere).

## Web app (Railway)

`webapp/main.py` is a small FastAPI wrapper around the same pipeline: paste
a video URL in the browser, it downloads/transcribes/picks/renders in the
background, and you download the resulting clips from the page. One job
runs at a time.

Deploy: push this repo to Railway (the included `Dockerfile` installs
ffmpeg and the Python deps), set `ANTHROPIC_API_KEY` as a service
variable, and generate a domain. Enable "sleep on idle" in the Railway
service settings if you want it to spin down between uses and wake on the
next visit.

**Password-protect it** (recommended for a public Railway domain) by
setting an `APP_PASSWORD` variable — every route except `/healthz` then
requires HTTP Basic auth (any username, that password) before it'll run.
Leave `APP_PASSWORD` unset for local dev to skip auth entirely.

**YouTube downloads from a cloud IP** — Railway's IPs (like most datacenter
IPs) regularly hit YouTube's "sign in to confirm you're not a bot" wall,
which only real session cookies get past. If you hit that, export cookies
from a logged-in browser (e.g. the "Get cookies.txt LOCALLY" extension) and
set the file's contents as the `YTDLP_COOKIES` variable — it's written to
a temp file and passed to yt-dlp automatically. (Or point `YTDLP_COOKIES_FILE`
at a path already on disk, e.g. a mounted volume.)

Runs locally too:

```
pip install -r requirements.txt
uvicorn webapp.main:app --reload
```

### Other web app features

Beyond the core "paste a link, get clips" flow, the web app also has:

- **Weekly recap** — assembles a landscape, long-form countdown video of
  the week's best Twitch clips across whichever streamers you track (see
  `TRENDING_TWITCH_LOGINS` below), from weakest to biggest hit, with a
  rank badge burned into each clip.
- **Game recap** — the same idea, but across every streamer playing one
  game, discovered directly from Twitch rather than only your tracked list.
- **Competitor discovery** — search for a streamer and see which other
  channels are actively clipping them, with each one's top short's
  transcript and loudness signal pulled in as context.
- **Upload to YouTube** — a button on any rendered clip that uploads it
  straight to your connected channel via the Data API's resumable upload
  (needs the same YouTube OAuth setup as Channel insights, below).
- **Auto-generated thumbnails** — pulls a real frame from a clip and burns
  one bold caption line over it.
- **Telegram notifications** — optional ping to a Telegram bot when a job
  finishes (or fails), so you don't have to keep the tab open.

### Environment variables

All configuration is via environment variables (Railway service variables,
or a local `.env`) — never hardcoded. Names only below; get real values
from each service's own dashboard.

| Variable | Required for | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | everything | Claude API key, used for clip selection everywhere |
| `CLIPPER_SELECT_MODEL` | — | Claude model id for picking and cutting clips (optional; default `claude-opus-5`) |
| `CLIPPER_MODEL` | — | Claude model id for everything else -- facecam checks, hook lines, recaps, channel analysis -- and the backup when the clip-picking model is unavailable (optional) |
| `APP_PASSWORD` | — | HTTP Basic auth password for the whole app; unset = no auth (local dev only) |
| `CLIPPER_JOBS_DIR` | — | where job state/output lives; point at a mounted volume in production |
| `YTDLP_COOKIES` / `YTDLP_COOKIES_FILE` | YouTube downloads from a cloud IP | exported browser cookies, to get past YouTube's bot check |
| `TRENDING_TWITCH_LOGINS` | trending / weekly recap | comma-separated Twitch logins to track |
| `TRENDING_YOUTUBE_CHANNELS` | trending | comma-separated YouTube channel IDs / `@handles` |
| `YOUTUBE_API_KEY` | trending, channel insights, competitor discovery | public YouTube Data API v3 key |
| `YOUTUBE_OWN_CHANNEL` | no-auth channel insights | your own channel's ID / `@handle` |
| `YOUTUBE_OAUTH_CLIENT_ID` / `YOUTUBE_OAUTH_CLIENT_SECRET` | real Analytics + YouTube upload | Google Cloud OAuth Web client, see below |
| `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET` | long-VOD viewer-clip signal | Twitch developer app credentials |
| `TELEGRAM_BOT_TOKEN` (or `CLIPPER_BOT_API`) | Telegram notifications | bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | optional; auto-discovered from the bot's recent messages if unset |

### Channel insights — best day to post

The web app has a "Channel insights" panel showing a best-day-to-post
suggestion and channel-performance signals, from two independent sources:

- **A no-auth heuristic** — works immediately, no setup. Set the
  `YOUTUBE_OWN_CHANNEL` variable to your channel's ID (starts with `UC...`),
  `@handle`, or legacy username, and it uses the public YouTube Data API
  (the same `YOUTUBE_API_KEY` already used for trending lookups) to guess a
  best day from your recent uploads' view counts, normalized by video age.
  Noisy, especially with few videos.
- **Real YouTube Analytics** (day-of-week views, retention, traffic
  sources) for your own connected channel — much more accurate, needs a
  one-time OAuth setup:

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
  6. Redeploy, open the app, and click **Connect YouTube account** in the
     Channel insights panel — it'll walk you through Google's consent
     screen and back.

  Note: even with a real connected account, YouTube's Analytics API has no
  "hour of day" dimension on regular reports — the audience-activity
  heatmap YouTube Studio shows isn't exposed via any public API. "Best day
  to post" here means best *day of the week*, from your channel's real
  views/watch-time data, which is the accurate ceiling of what's obtainable
  through official APIs.
