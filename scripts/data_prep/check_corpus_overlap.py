#!/usr/bin/env python3
"""
Compute the actual overlap fraction for EVERY session in a directory
(fraction of total speech time where 2+ speakers are simultaneously
labeled active), and compare against the simulator's configured target
(~10% mean_overlap). Checks whether the heavy-overlap case found in one
session is a rare outlier or a systemic pattern across the corpus.

Usage:
    python check_corpus_overlap.py --wav_dir /path/to/test_pooled
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wav_dir", required=True, type=Path)
ap.add_argument("--frame_size", type=float, default=0.01)
ap.add_argument("--target_overlap", type=float, default=0.10)
args = ap.parse_args()

rows = []
for rttm_path in sorted(args.wav_dir.glob("*.rttm")):
    segs = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        f = line.strip().split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            segs.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    if not segs:
        continue

    speakers = {s for _, _, s in segs}
    if len(speakers) < 2:
        continue  # 1-speaker sessions can't overlap

    duration = max(e for _, e, _ in segs)
    n_frames = int(duration / args.frame_size) + 1
    frame_speaker_counts = [0] * n_frames
    for start, end, _ in segs:
        lo = max(0, int(round(start / args.frame_size)))
        hi = min(n_frames, int(round(end / args.frame_size)))
        for i in range(lo, hi):
            frame_speaker_counts[i] += 1

    speech_frames = sum(1 for c in frame_speaker_counts if c >= 1)
    overlap_frames = sum(1 for c in frame_speaker_counts if c >= 2)
    if speech_frames == 0:
        continue

    overlap_frac = overlap_frames / speech_frames
    rows.append((rttm_path.stem, overlap_frac))

rows.sort(key=lambda r: -r[1])

print(f"{'session':<25s} {'overlap_frac':>13s}")
for name, frac in rows:
    flag = "  <-- well above target" if frac > 3 * args.target_overlap else ""
    print(f"{name:<25s} {frac*100:>12.1f}%{flag}")

fracs = [r[1] for r in rows]
if fracs:
    print(f"\nTarget (config mean_overlap): {args.target_overlap*100:.0f}%")
    print(f"Median actual overlap:        {np.median(fracs)*100:.1f}%")
    print(f"Mean actual overlap:          {np.mean(fracs)*100:.1f}%")
    print(f"Sessions > 3x target ({3*args.target_overlap*100:.0f}%+): "
          f"{sum(1 for f in fracs if f > 3*args.target_overlap)} / {len(fracs)}")
