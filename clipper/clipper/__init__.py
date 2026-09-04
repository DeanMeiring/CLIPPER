"""clipper — turn a long video into vertical, captioned highlight clips.

Modules:
    download        fetch a YouTube video (or accept a local file) + captions
    transcribe      local Whisper fallback for word-level timestamps
    select_moments  ask Claude which segments are worth clipping
    reframe         smart 9:16 crop using face detection
    captions        build burned-in word-highlight subtitles
    render          ffmpeg cut + crop + caption pipeline
    cli             command-line entry point
"""

__version__ = "0.1.0"
