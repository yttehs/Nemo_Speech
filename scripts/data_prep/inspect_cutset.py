#!/usr/bin/env python3
"""
Inspect a Lhotse CutSet built by build_lhotse_cutset.py -- summary stats
across the whole split, plus a detailed look at a few sample cuts (speaker
count, per-speaker duration, sample text), so you can eyeball correctness
before pointing a training run at it.

Usage:
    python inspect_cutset.py --cuts_path /path/to/train_cuts.jsonl.gz --n_samples 3
"""
import argparse
from collections import Counter

from lhotse import CutSet

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cuts_path", required=True)
ap.add_argument("--n_samples", type=int, default=3, help="How many cuts to show in full detail")
args = ap.parse_args()

cuts = list(CutSet.from_file(args.cuts_path))

total_duration = sum(c.duration for c in cuts)
speaker_counts = Counter(len({s.speaker for s in c.supervisions}) for c in cuts)
empty_text_count = sum(1 for c in cuts for s in c.supervisions if not s.text or not s.text.strip())

print(f"Total cuts: {len(cuts)}")
print(f"Total duration: {total_duration/60:.1f} min")
print(f"Speaker-count distribution across cuts: {dict(sorted(speaker_counts.items()))}")
print(f"Supervisions with empty/missing text: {empty_text_count}")
print()

print(f"=== Detailed view of first {min(args.n_samples, len(cuts))} cuts ===")
for cut in cuts[: args.n_samples]:
    print(f"\nCut id: {cut.id}   duration: {cut.duration:.2f}s   audio: {cut.recording.sources[0].source}")
    by_speaker = {}
    for s in cut.supervisions:
        by_speaker.setdefault(s.speaker, []).append(s)
    for spk, segs in by_speaker.items():
        total_spk_dur = sum(s.duration for s in segs)
        sample_text = segs[0].text
        print(f"  speaker={spk:<20s} turns={len(segs):<3d} total_dur={total_spk_dur:.2f}s  "
              f"first_text={sample_text!r}")
