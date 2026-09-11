"""Command-line entry point: clipper <youtube-url-or-file> [options]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .download import download_video, is_url, probe_video
from .loud_moments import find_loud_moments
from .transcribe import get_transcript
from .select_moments import select_clips
from .long_vod import gather_candidates, is_long_vod, select_and_map
from .reframe import compute_layout
from .captions import build_ass
from .render import render_clip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clipper",
        description="Turn a long video into vertical, captioned highlight clips.",
    )
    p.add_argument("source", help="YouTube/Twitch URL or path to a local video file")
    p.add_argument("-o", "--out", default="clips", help="Output folder (default: ./clips)")
    p.add_argument("-n", "--clips", type=int, default=5, help="Number of clips to produce (default: 5)")
    p.add_argument("--min-len", type=float, default=20.0, help="Minimum clip length in seconds")
    p.add_argument("--max-len", type=float, default=90.0, help="Maximum clip length in seconds")
    p.add_argument("--focus", default=None, help='Steer clip selection, e.g. "funniest moments"')
    p.add_argument("--whisper", action="store_true", help="Force local Whisper transcription instead of YouTube captions")
    p.add_argument("--whisper-model", default="small", help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    p.add_argument("--api-key", default=None, help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)")
    p.add_argument("--width", type=int, default=1080, help="Output width (default: 1080)")
    p.add_argument("--height", type=int, default=1920, help="Output height (default: 1920, i.e. 9:16)")
    p.add_argument("--no-captions", action="store_true", help="Skip burning in captions")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_source"

    print(f"[1/5] Checking source: {args.source}")
    info = None
    if is_url(args.source):
        try:
            info = probe_video(args.source)
        except Exception as e:
            print(f"      Could not probe source metadata ({e}); trying a normal download.", file=sys.stderr)

    # Both pipelines below normalize to: source_title, and a list of
    # (video_path, words, pick) tuples ready for the shared render loop.
    if info and is_long_vod(info):
        print(f"      -> long Twitch VOD ({info.duration / 3600:.1f}h) -- using the chat-highlight "
              f"pipeline instead of downloading the whole stream")
        source_title = info.title

        candidates = gather_candidates(args.source, info, raw_dir, on_progress=print)
        if not candidates:
            print("      No candidate highlight moments found.", file=sys.stderr)
            return 1

        print(f"[3/5] Asking Claude to pick up to {args.clips} clip-worthy moments "
              f"from {len(candidates)} candidates...")
        mapped = select_and_map(
            candidates, n_clips=args.clips, min_len=args.min_len, max_len=args.max_len,
            focus=args.focus, api_key=args.api_key, source_title=source_title,
        )
        cand_words = {c["index"]: c["words"] for c in candidates}
        render_items = [(video_path, cand_words[pick.window_index], pick) for video_path, pick in mapped]
    else:
        print(f"      -> {args.source}")
        dl = download_video(args.source, raw_dir)
        print(f"      Fetched {dl.video_path.name}  ({dl.duration:.0f}s)  \"{dl.title}\"")
        source_title = dl.title

        print("[2/5] Getting transcript...")
        words = get_transcript(
            dl.video_path, dl.captions_path,
            prefer_whisper=args.whisper, whisper_model=args.whisper_model,
        )
        if not words:
            print("      No speech/captions found -- nothing to clip.", file=sys.stderr)
            return 1
        print(f"      -> {len(words)} words")

        print("      Scanning audio for loud/high-energy moments...")
        loud_moments = find_loud_moments(dl.video_path, dl.duration)
        if loud_moments:
            print(f"      -> {len(loud_moments)} loud moment(s) flagged")

        print(f"[3/5] Asking Claude to pick up to {args.clips} clip-worthy moments...")
        picks = select_clips(
            words, dl.duration,
            n_clips=args.clips, min_len=args.min_len, max_len=args.max_len,
            focus=args.focus, api_key=args.api_key, source_title=dl.title,
            loud_moments=loud_moments,
        )
        render_items = [(dl.video_path, words, pick) for pick in picks]

    if not render_items:
        print("      Model returned no usable picks.", file=sys.stderr)
        return 1
    print(f"      -> {len(render_items)} clips picked")

    metadata = []
    for i, (video_path, words, pick) in enumerate(render_items, start=1):
        print(f"[4/5] Rendering clip {i}/{len(render_items)}: \"{pick.title}\" ({pick.start:.0f}s-{pick.end:.0f}s)")

        clip_words = [w for w in words if w.start >= pick.start and w.end <= pick.end]

        layout = compute_layout(
            video_path, pick.start, pick.end,
            target_w=args.width, target_h=args.height,
        )

        out_path = out_dir / f"clip_{i:02d}.mp4"
        if args.no_captions or not clip_words:
            ass_path = out_dir / f"_clip_{i:02d}_empty.ass"
            build_ass([], pick.start, ass_path)
        else:
            ass_path = out_dir / f"_clip_{i:02d}.ass"
            build_ass(clip_words, pick.start, ass_path)

        render_clip(
            video_path, pick.start, pick.end, layout, ass_path, out_path,
            out_w=args.width, out_h=args.height,
        )
        print(f"      -> {out_path}")

        metadata.append({
            "file": out_path.name,
            "start": pick.start,
            "end": pick.end,
            "duration": round(pick.end - pick.start, 2),
            "title": pick.title,
            "hook_caption": pick.hook_caption,
            "upload_title": pick.upload_title,
            "description": pick.description,
            "reason": pick.reason,
        })

    meta_path = out_dir / "clips.json"
    meta_path.write_text(json.dumps({"source_title": source_title, "clips": metadata}, indent=2), encoding="utf-8")
    print(f"[5/5] Done. {len(metadata)} clip(s) in {out_dir}/  (details in clips.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
