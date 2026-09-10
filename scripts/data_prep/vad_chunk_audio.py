#!/usr/bin/env python3
"""
Chunks a raw audio file into 30-60s pieces using ONLY voice activity detection --
no ground-truth transcript, speaker labels, or segment boundaries used at all. This
is the genuine test-time simulation: at real deployment, only the audio is available.

Reuses the exact three-tier chunking algorithm already validated for MagicData/
AliMeeting/Portuguese (prefer a clean pause under a soft max, relax to any gap under
a hard max, force a cut at a hard backstop) -- the only change is WHERE candidate cut
points come from: VAD-detected silence gaps instead of ground-truth segment gaps. The
algorithm itself doesn't know or care about the difference.

VAD segments get a single, constant dummy "speaker" label (VAD carries no speaker
identity at all) purely so find_chunk_boundaries' shared signature is satisfiable --
its --max_speakers check never fires here, since every segment shares that one label.

Usage:
    python vad_chunk_audio.py \
        --audio path/to/long_recording.wav \
        --output_dir path/to/vad_chunks
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from build_pt_test_pooled import find_chunk_boundaries

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--audio", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path)
ap.add_argument("--target_duration", type=float, default=30.0)
ap.add_argument("--pause_threshold", type=float, default=0.2)
ap.add_argument("--soft_max_duration", type=float, default=45.0)
ap.add_argument("--hard_max_duration", type=float, default=60.0)
ap.add_argument("--vad_threshold", type=float, default=0.5,
                 help="Silero VAD's own speech-probability threshold (its default).")
ap.add_argument("--min_speech_duration_ms", type=int, default=250,
                 help="VAD segments shorter than this are discarded as spurious (Silero's own default).")
ap.add_argument("--min_silence_duration_ms", type=int, default=100,
                 help="Minimum silence before VAD splits into separate speech regions (Silero's own "
                      "default) -- kept below --pause_threshold (200ms) deliberately, so VAD doesn't "
                      "pre-merge away short pauses the chunker would otherwise want to consider as "
                      "candidate cut points.")


def main():
    args = ap.parse_args()

    print("Loading VAD model...")
    vad_model = load_silero_vad()

    # Load via soundfile, NOT silero_vad's own read_audio() helper -- confirmed the
    # hard way that read_audio() depends on a torchaudio/torchcodec backend combo
    # that isn't reliably available; soundfile is what this entire project already
    # uses everywhere else for audio I/O.
    audio_np, sr = sf.read(args.audio, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"Expected 16kHz audio, got {sr}Hz -- resample first "
                          f"(e.g. via ffmpeg -ar 16000) before running VAD.")
    audio_tensor = torch.from_numpy(audio_np)
    total_duration = len(audio_np) / sr
    print(f"Audio duration: {total_duration:.1f}s")

    vad_timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=sr,
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        return_seconds=True,
    )
    print(f"VAD detected {len(vad_timestamps)} speech region(s)")
    if not vad_timestamps:
        raise SystemExit("VAD detected no speech at all in this file -- nothing to chunk.")

    # A single, constant dummy speaker label -- see module docstring for why.
    segments = [{"start": ts["start"], "end": ts["end"], "speaker": "vad_speech"} for ts in vad_timestamps]

    boundaries = find_chunk_boundaries(
        segments, args.target_duration, args.pause_threshold,
        args.soft_max_duration, args.hard_max_duration, max_speakers=999999,
    )
    print(f"{len(boundaries)} chunk(s)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for chunk_idx, (start_idx, end_idx) in enumerate(boundaries):
        chunk_segments = segments[start_idx:end_idx + 1]
        chunk_abs_start = chunk_segments[0]["start"]
        chunk_abs_end = max(s["end"] for s in chunk_segments)

        sample_start = int(chunk_abs_start * sr)
        sample_end = int(chunk_abs_end * sr)
        chunk_audio = audio_np[sample_start:sample_end]

        chunk_uri = f"{args.audio.stem}-chunk{chunk_idx:04d}"
        chunk_wav_path = args.output_dir / f"{chunk_uri}.wav"
        sf.write(chunk_wav_path, chunk_audio, sr)

        manifest.append({
            "chunk_id": chunk_uri,
            "wav_path": str(chunk_wav_path),
            # Absolute time in the ORIGINAL, full audio file -- required later to
            # convert each chunk's own locally-timed diarization/ASR output back
            # into session-wide, absolute time when stitching everything together.
            "absolute_start": chunk_abs_start,
            "absolute_end": chunk_abs_end,
            "duration": chunk_abs_end - chunk_abs_start,
        })
        print(f"  {chunk_uri}: [{chunk_abs_start:.1f}, {chunk_abs_end:.1f}] "
              f"({chunk_abs_end - chunk_abs_start:.1f}s)")

    manifest_path = args.output_dir / f"{args.audio.stem}_chunk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} chunk(s) written to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
