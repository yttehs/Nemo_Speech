#!/usr/bin/env python3
"""
Compute turn-duration and speaker-switching-rate statistics across every
session, from ground-truth RTTM. Tests whether the simulator's turn_prob
config is producing unrealistically choppy, fast-switching conversations --
a pace mismatch that could degrade diarization broadly and evenly across the
whole corpus, unlike the session-specific "vanishing speaker" issue.

Usage:
    python check_turn_pacing.py --wav_dir /path/to/test_pooled
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wav_dir", required=True, type=Path)
args = ap.parse_args()

all_turn_durations = []
switches_per_session = []

for rttm_path in sorted(args.wav_dir.glob("*.rttm")):
    segs = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        f = line.strip().split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            segs.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    if len(segs) < 2:
        continue

    segs.sort(key=lambda s: s[0])
    for start, end, _ in segs:
        all_turn_durations.append(end - start)

    # Count switches: consecutive turns (by start time) with a different speaker.
    n_switches = sum(1 for i in range(1, len(segs)) if segs[i][2] != segs[i - 1][2])
    duration = max(e for _, e, _ in segs)
    switches_per_session.append(n_switches / duration * 60)  # switches per minute

turn_durations = np.array(all_turn_durations)
switch_rates = np.array(switches_per_session)

print(f"Turn durations (n={len(turn_durations)}):")
print(f"  Median: {np.median(turn_durations):.2f}s")
print(f"  Mean:   {np.mean(turn_durations):.2f}s")
print(f"  10th percentile: {np.percentile(turn_durations, 10):.2f}s")
print(f"  Fraction under 1.0s: {100*np.mean(turn_durations < 1.0):.1f}%")
print(f"  Fraction under 0.5s: {100*np.mean(turn_durations < 0.5):.1f}%")
print()
print(f"Speaker switches per minute (n={len(switch_rates)} sessions):")
print(f"  Median: {np.median(switch_rates):.1f}")
print(f"  Mean:   {np.mean(switch_rates):.1f}")
print()
print("For reference, natural conversational turn-taking research typically reports")
print("turns in the 1-3s range on average, with switches on the order of 10-20 per")
print("minute in fast-paced dialogue. Much shorter turns / much higher switch rates")
print("here would indicate unrealistically choppy simulated conversations.")
