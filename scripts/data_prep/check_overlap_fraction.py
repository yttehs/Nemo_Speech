#!/usr/bin/env python3
"""
For one speaker in one session, compute exactly how much of their labeled
speaking time overlaps with another speaker vs. how much is solo -- tests
whether a speaker being imperceptible in the full mix (despite sounding
completely normal in isolation) is explained by unusually heavy overlap
with a co-speaker in this specific session.

Usage:
    python check_overlap_fraction.py \
        --session_dir /path/to/test_pooled \
        --session_name test_2spk_sess0 \
        --speaker waldyrious
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--session_dir", required=True, type=Path)
ap.add_argument("--session_name", required=True)
ap.add_argument("--speaker", required=True)
ap.add_argument("--frame_size", type=float, default=0.01)
args = ap.parse_args()

rttm_path = args.session_dir / f"{args.session_name}.rttm"
segs = []
for line in rttm_path.read_text(encoding="utf-8").splitlines():
    f = line.strip().split()
    if len(f) >= 8 and f[0] == "SPEAKER":
        segs.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))

duration = max(e for _, e, _ in segs)
n_frames = int(duration / args.frame_size) + 1
frame_speakers = [set() for _ in range(n_frames)]
for start, end, spk in segs:
    lo = max(0, int(round(start / args.frame_size)))
    hi = min(n_frames, int(round(end / args.frame_size)))
    for i in range(lo, hi):
        frame_speakers[i].add(spk)

solo_frames = sum(1 for spks in frame_speakers if spks == {args.speaker})
overlap_frames = sum(1 for spks in frame_speakers if args.speaker in spks and len(spks) > 1)
total_frames = solo_frames + overlap_frames

solo_s = solo_frames * args.frame_size
overlap_s = overlap_frames * args.frame_size
total_s = total_frames * args.frame_size

print(f"Speaker: {args.speaker}")
print(f"Total labeled speaking time: {total_s:.2f}s")
print(f"  Solo (no one else active):     {solo_s:.2f}s  ({100*solo_s/total_s:.1f}%)")
print(f"  Overlapping with someone else: {overlap_s:.2f}s  ({100*overlap_s/total_s:.1f}%)")
print()
print("For reference, the simulator's config targets ~10% mean overlap session-wide.")
print("If this speaker's overlap fraction is much higher than that, heavy overlap")
print("with their co-speaker is a plausible explanation for why they're hard to")
print("perceive in the full mix despite sounding normal in isolation.")
