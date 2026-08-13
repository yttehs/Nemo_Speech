#!/usr/bin/env python3
"""
Extract and concatenate just ONE speaker's segments from a session's final
audio (using offset/duration from the session's .json), so you can listen
to only their part in isolation -- confirms whether a flagged "quiet"
speaker is genuinely silent (near-zero samples throughout) or just quiet
but audible.

Usage:
    python extract_speaker_audio.py \
        --session_dir /path/to/test_pooled \
        --session_name test_3spk_sess15 \
        --speaker persona \
        --out_wav /tmp/persona_sess15_only.wav
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--session_dir", required=True, type=Path)
ap.add_argument("--session_name", required=True)
ap.add_argument("--speaker", required=True)
ap.add_argument("--out_wav", required=True, type=Path)
args = ap.parse_args()

json_path = args.session_dir / f"{args.session_name}.json"
wav_path = args.session_dir / f"{args.session_name}.wav"

audio, sr = sf.read(wav_path)
if audio.ndim > 1:
    audio = audio.mean(axis=1)

entries = [json.loads(l) for l in json_path.read_text(encoding="utf-8").splitlines() if l.strip()]
speaker_entries = [e for e in entries if e["label"] == args.speaker]

if not speaker_entries:
    raise ValueError(f"No entries found for speaker '{args.speaker}' in {json_path} "
                      f"(available: {sorted({e['label'] for e in entries})})")

print(f"Found {len(speaker_entries)} entries for {args.speaker}")
chunks = []
peaks, rmses = [], []
for e in speaker_entries:
    lo = int(e["offset"] * sr)
    hi = int((e["offset"] + e["duration"]) * sr)
    chunk = audio[lo:hi]
    chunks.append(chunk)
    peaks.append(float(np.max(np.abs(chunk))) if len(chunk) else 0.0)
    rmses.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if len(chunk) else 0.0)
    print(f"  offset={e['offset']:.3f}s dur={e['duration']:.3f}s text={e['text']!r} "
          f"peak={peaks[-1]:.5f} rms={rmses[-1]:.5f}")

combined = np.concatenate(chunks) if chunks else np.array([])
sf.write(args.out_wav, combined, sr)
print(f"\nWrote {len(combined)/sr:.2f}s of isolated audio to {args.out_wav}")
print(f"Overall peak: {max(peaks):.5f}   Overall RMS: {np.sqrt(np.mean(np.array(rmses)**2)):.5f}")
print("\nIf peak is near 0.0000 across every entry, this confirms genuine digital")
print("silence -- the source draw itself was empty, not just quiet.")
