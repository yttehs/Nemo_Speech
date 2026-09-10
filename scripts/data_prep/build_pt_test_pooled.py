#!/usr/bin/env python3
"""
Converts one pt_avprep video (mp4 + *_speaker_attribution_final.json) into
chunked, test_pooled-style output (wav/rttm/transcript-json per chunk),
matching the format evaluate_checkpoint_sortformer_conditioned.py expects for
--ref_dir.

Reuses the exact three-tier chunking algorithm validated for MagicData/AliMeeting
(clean-pause preferred -> relaxed adjacent-boundary fallback -> hard duration
backstop) -- the underlying problem is identical: a long session needs cutting at
good boundaries without splitting a segment's own audio.

Usage:
    python build_pt_test_pooled.py \
        --video Data_Portuguese/video/Brazil/Brazil_portuguese_01.mp4 \
        --attribution Data_Portuguese/transcript/Brazil/Brazil_portuguese_01_speaker_attribution_final.json \
        --output_dir Data_Portuguese/test_pooled
"""
import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf

from parse_pt_attribution import parse_attribution_file

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--video", required=True, type=Path)
ap.add_argument("--attribution", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path)
ap.add_argument("--target_duration", type=float, default=30.0)
ap.add_argument("--pause_threshold", type=float, default=0.2)
ap.add_argument("--soft_max_duration", type=float, default=45.0)
ap.add_argument("--hard_max_duration", type=float, default=60.0)
ap.add_argument("--max_speakers", type=int, default=4,
                 help="Hard ceiling on distinct speakers per chunk -- Sortformer supports at "
                      "most 4 speakers per input, a fixed architectural limit, not a tuning "
                      "preference. Unlike MagicData/AliMeeting (each source meeting itself had "
                      "at most 4 participants, so this could never be exceeded), a single long "
                      "street-interview video can feature many more than 4 distinct people over "
                      "its full runtime -- nothing previously stopped a chunk from spanning more "
                      "than 4 of them if several interviews happened close together in time. "
                      "Checked with higher priority than even --hard_max_duration: exceeding "
                      "this isn't a quality tradeoff to accept under time pressure, it's a hard "
                      "ceiling the model literally cannot process past.")
ap.add_argument("--language", default="por")


def find_chunk_boundaries(segments, target_duration, pause_threshold, soft_max_duration,
                            hard_max_duration, max_speakers=4):
    """Identical duration-based logic validated for MagicData/AliMeeting, with one addition:
    distinct speaker count per chunk is capped at max_speakers, checked BEFORE the
    duration-tier logic at every candidate extension point -- this is a hard architectural
    ceiling (Sortformer's own speaker-count limit), not a soft preference like duration, so
    it can force a cut even before --target_duration is reached, and overrides even the
    --hard_max_duration backstop if the two would otherwise conflict."""
    boundaries = []
    chunk_start_idx = 0
    n = len(segments)
    while chunk_start_idx < n:
        chunk_start_time = segments[chunk_start_idx]["start"]
        speakers_in_chunk = {segments[chunk_start_idx]["speaker"]}

        # Phase 1: accumulate until target_duration reached OR max_speakers would be exceeded
        i = chunk_start_idx
        max_end_so_far = segments[i]["end"]
        speaker_limit_hit = False
        while i < n - 1 and (max_end_so_far - chunk_start_time) < target_duration:
            next_speaker = segments[i + 1]["speaker"]
            if next_speaker not in speakers_in_chunk and len(speakers_in_chunk) >= max_speakers:
                speaker_limit_hit = True
                break
            speakers_in_chunk.add(next_speaker)
            i += 1
            max_end_so_far = max(max_end_so_far, segments[i]["end"])

        if speaker_limit_hit:
            # Hard ceiling reached before target_duration -- cut here regardless of
            # how short this leaves the chunk. No duration-tier logic applies.
            boundaries.append((chunk_start_idx, i))
            chunk_start_idx = i + 1
            continue

        if i >= n - 1:
            boundaries.append((chunk_start_idx, n - 1))
            break

        # Phase 2: look for a good cut point, starting from i, ALSO respecting max_speakers
        cut_after_idx = None
        j = i
        while j < n - 1:
            max_end_so_far = max(max_end_so_far, segments[j]["end"])
            next_speaker = segments[j + 1]["speaker"]
            if next_speaker not in speakers_in_chunk and len(speakers_in_chunk) >= max_speakers:
                cut_after_idx = j
                break
            gap = segments[j + 1]["start"] - segments[j]["end"]
            elapsed = max_end_so_far - chunk_start_time
            if elapsed < soft_max_duration:
                if gap >= pause_threshold:
                    cut_after_idx = j
                    break
            elif elapsed < hard_max_duration:
                if gap >= 0:
                    cut_after_idx = j
                    break
            else:
                cut_after_idx = j
                break
            speakers_in_chunk.add(next_speaker)
            j += 1
        if cut_after_idx is None:
            boundaries.append((chunk_start_idx, n - 1))
            break
        boundaries.append((chunk_start_idx, cut_after_idx))
        chunk_start_idx = cut_after_idx + 1
    return boundaries


def extract_audio(video_path: Path, wav_path: Path):
    """Mono 16kHz PCM, matching pt_avprep's own established convention."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le", str(wav_path), "-loglevel", "panic"],
        check=True,
    )


def main():
    args = ap.parse_args()
    segments, video_id = parse_attribution_file(args.attribution)
    print(f"{video_id}: {len(segments)} segments parsed")
    n_flagged = sum(1 for s in segments if s["note"])
    if n_flagged:
        print(f"  {n_flagged} segment(s) flagged for potential unmarked overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_wav_path = args.output_dir / f"{args.video.stem}_full.wav"
    print("Extracting audio...")
    extract_audio(args.video, full_wav_path)
    audio, sr = sf.read(full_wav_path, dtype="float32")
    print(f"  {len(audio)/sr:.1f}s at {sr}Hz")

    boundaries = find_chunk_boundaries(
        segments, args.target_duration, args.pause_threshold,
        args.soft_max_duration, args.hard_max_duration, args.max_speakers,
    )
    print(f"{len(boundaries)} chunk(s)")

    n_chunks_written = 0
    for chunk_idx, (start_idx, end_idx) in enumerate(boundaries):
        chunk_segments = segments[start_idx:end_idx + 1]
        chunk_abs_start = chunk_segments[0]["start"]
        chunk_abs_end = max(s["end"] for s in chunk_segments)

        sample_start = int(chunk_abs_start * sr)
        sample_end = int(chunk_abs_end * sr)
        chunk_audio = audio[sample_start:sample_end]

        chunk_uri = f"{args.video.stem}-chunk{chunk_idx:04d}"
        sf.write(args.output_dir / f"{chunk_uri}.wav", chunk_audio, sr)

        rttm_path = args.output_dir / f"{chunk_uri}.rttm"
        with rttm_path.open("w", encoding="utf-8") as f:
            for seg in chunk_segments:
                rel_start = seg["start"] - chunk_abs_start
                dur = seg["end"] - seg["start"]
                f.write(f"SPEAKER {chunk_uri} 1 {rel_start:.3f} {dur:.3f} "
                        f"<NA> <NA> {seg['speaker']} <NA> <NA>\n")

        # JSONL, one segment per line -- confirmed against evaluate_checkpoint_sortformer_
        # conditioned.py's ground_truth_text_per_speaker() directly: it parses this file
        # line-by-line via json.loads(l) per line, needing "offset" (not "start") and
        # "label" (not "speaker") specifically. A single, pretty-printed nested JSON object
        # (this script's original approach) fails immediately -- confirmed the hard way:
        # line 1 of a pretty-printed dump is just "{", not valid JSON on its own, so
        # per-line json.loads() throws on the very first line every time.
        lines = []
        for s in chunk_segments:
            entry = {
                "offset": s["start"] - chunk_abs_start,
                "duration": s["end"] - s["start"],
                "label": s["speaker"],
                "text": s["text"],
                "language": args.language,
            }
            if s["note"]:
                entry["note"] = s["note"]
            lines.append(json.dumps(entry, ensure_ascii=False))
        (args.output_dir / f"{chunk_uri}.json").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_chunks_written += 1

    full_wav_path.unlink()  # only the chunks are needed downstream
    print(f"\n{n_chunks_written} chunk(s) written to {args.output_dir}")


if __name__ == "__main__":
    main()
