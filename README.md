# clipper

Also deployable as a small web app (`webapp/`) — see [Web app (Railway)](#web-app-railway)
below for a browser UI on top of the same pipeline.

Turn a long video (YouTube link or a local file) into a batch of short,
vertical (9:16), captioned highlight clips — the kind of tool Opus Clip /
Vizard / Klap sell as a subscription, except this one runs on your own
machine, for the cost of a few cents of Claude API usage per video.

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
  clipper/
    download.py       # yt-dlp wrapper + caption fetch
    transcribe.py      # VTT parsing / Whisper fallback -> word-level timing
    select_moments.py  # Claude call -> which parts to clip, and why
    reframe.py         # face-detection -> smart 9:16 crop window
    captions.py         # word-level timing -> burned-in .ass karaoke captions
    render.py            # ffmpeg crop+scale+caption render
    cli.py                # ties it all together
  pyproject.toml
  requirements.txt
  .env.example
```

Each module works standalone too, if you ever want to script around just
one piece of it (e.g. use `reframe.py` on its own to auto-crop existing
clips you already made elsewhere).

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

Runs locally too:

```
pip install -r requirements.txt
uvicorn webapp.main:app --reload
```
