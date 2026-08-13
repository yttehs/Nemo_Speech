#!/usr/bin/env python3
"""
Within a single session, compute each speaker's own RMS level using ONLY
their solo (non-overlapping) time -- moments where no other speaker is also
labeled active. This matters because the final audio is a mix: measuring a
speaker's RMS across ALL their labeled time (including overlap) actually
measures the SUM of whoever is simultaneously active, contaminating a quiet
speaker's true level with whatever louder speaker they happen to overlap
with. Solo-only time gives an uncontaminated read on each speaker's actual
level, which is what per_speaker_normalize is supposed to control.

Usage:
    python check_relative_speaker_loudness.py --wav_dir /path/to/test_pooled
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wav_dir", required=True, type=Path)
ap.add_argument("--min_speakers", type=int, default=2, help="Skip 1-speaker sessions, nothing to compare")
ap.add_argument("--frame_size", type=float, default=0.01)
args = ap.parse_args()


def parse_rttm(path):
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.strip().split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            segs.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return segs


rows = []
for rttm_path in sorted(args.wav_dir.glob("*.rttm")):
    wav_path = rttm_path.with_suffix(".wav")
    if not wav_path.exists():
        continue

    segs = parse_rttm(rttm_path)
    speakers = sorted({s for _, _, s in segs})
    if len(speakers) < args.min_speakers:
        continue

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Frame-level active-speaker-count map, to find genuinely solo stretches.
    duration = max(e for _, e, _ in segs)
    n_frames = int(duration / args.frame_size) + 1
    frame_speakers = [set() for _ in range(n_frames)]
    for start, end, spk in segs:
        lo = max(0, int(round(start / args.frame_size)))
        hi = min(n_frames, int(round(end / args.frame_size)))
        for i in range(lo, hi):
            frame_speakers[i].add(spk)

    speaker_solo_samples = defaultdict(list)
    for i, spks in enumerate(frame_speakers):
        if len(spks) == 1:  # solo -- exactly one speaker active, no contamination
            spk = next(iter(spks))
            lo_s = int(i * args.frame_size * sr)
            hi_s = int((i + 1) * args.frame_size * sr)
            speaker_solo_samples[spk].append(audio[lo_s:hi_s])

    speaker_rms = {}
    speaker_solo_dur = {}
    for spk, chunks in speaker_solo_samples.items():
        combined = np.concatenate(chunks) if chunks else np.array([])
        if len(combined) > 0:
            speaker_rms[spk] = float(np.sqrt(np.mean(combined.astype(np.float64) ** 2)))
            speaker_solo_dur[spk] = len(combined) / sr

    if len(speaker_rms) < 2:
        continue  # not enough solo time for at least 2 speakers to compare

    loudest_spk = max(speaker_rms, key=speaker_rms.get)
    quietest_spk = min(speaker_rms, key=speaker_rms.get)
    loudest = speaker_rms[loudest_spk]
    quietest = speaker_rms[quietest_spk]
    ratio = loudest / quietest if quietest > 0 else float("inf")
    rows.append((
        rttm_path.stem, ratio,
        loudest_spk, loudest, speaker_solo_dur[loudest_spk],
        quietest_spk, quietest, speaker_solo_dur[quietest_spk],
    ))

rows.sort(key=lambda r: -r[1])

print(f"{'session':<25s} {'ratio':>8s} {'loudest_spk':>14s} {'loud_rms':>9s} {'loud_dur_s':>10s} "
      f"{'quietest_spk':>14s} {'quiet_rms':>10s} {'quiet_dur_s':>11s}")
for name, ratio, loud_spk, loud_rms, loud_dur, quiet_spk, quiet_rms, quiet_dur in rows:
    warn = "  <-- TINY solo duration, likely a measurement artifact" if quiet_dur < 0.3 else (
        "  <-- check this one" if ratio > 5 else "")
    print(f"{name:<25s} {ratio:>7.2f}x {loud_spk:>14s} {loud_rms:>9.4f} {loud_dur:>10.2f} "
          f"{quiet_spk:>14s} {quiet_rms:>10.4f} {quiet_dur:>11.2f}{warn}")

if rows:
    ratios = [r[1] for r in rows]
    print(f"\nMedian ratio: {np.median(ratios):.2f}   Max ratio: {max(ratios):.2f}")
    print("Healthy sessions should mostly sit close to 1.0. Ratios above ~5 mean")
    print("one speaker is at least 5x louder than another within the same session --")
    print("that's the 'faint background speaker' pattern you saw in the viewer.")
