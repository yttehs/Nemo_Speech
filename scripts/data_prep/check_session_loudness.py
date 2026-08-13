#!/usr/bin/env python3
"""
Quick check: what does the FINAL simulated session audio actually look like,
loudness-wise? Compares against what a typical, healthy speech recording
looks like, to test whether the whole-session normalize_audio() pass is
leaving sessions unexpectedly quiet -- which would explain a near-zero
false-alarm / dominant-miss DER pattern (Sortformer's speech-detection
threshold not firing on quiet audio, while never hallucinating speech in
silence).

Usage:
    python check_session_loudness.py --wav_dir /path/to/test_pooled --n 10
"""
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wav_dir", required=True, type=Path)
ap.add_argument("--n", type=int, default=10, help="How many random files to sample")
args = ap.parse_args()

wav_files = list(args.wav_dir.glob("*.wav"))
sample = random.sample(wav_files, min(args.n, len(wav_files)))

print(f"{'file':<30s} {'peak':>8s} {'rms':>8s} {'sr':>7s}")
peaks, rmses = [], []
for wav_path in sample:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peaks.append(peak)
    rmses.append(rms)
    print(f"{wav_path.name:<30s} {peak:>8.4f} {rms:>8.4f} {sr:>7d}")

print()
print(f"Mean peak: {np.mean(peaks):.4f}   (healthy speech is typically 0.3-1.0)")
print(f"Mean RMS:  {np.mean(rmses):.4f}   (healthy speech is typically 0.03-0.15)")
print()
print("If mean peak is well under 0.1, the final session audio is likely too quiet")
print("for Sortformer's speech-detection threshold to fire reliably -- worth checking")
print("normalize_audio()'s actual implementation in data_simulation_utils.py next.")
