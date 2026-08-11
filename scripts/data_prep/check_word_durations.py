#!/usr/bin/env python3
"""
Scan an alignment manifest (output of create_alignment_manifest.py) for
implausibly long single-word spans -- a signature of forced-alignment
failure where NFA mis-attributes a stretch of silence to an adjacent word,
rather than correctly marking it as its own silence segment. This tends to
happen on clips with long dead-air stretches relative to how little is
actually said.

Since word-level timing accuracy matters a lot for how the speech data
simulator splices clips together, entries with any word exceeding
--max_word_duration are flagged, and can optionally be filtered out
entirely -- useful when you can't manually QC the audio yourself (e.g.
you don't speak the language), since this acts as a language-independent
proxy check.

Usage:
    # Just report, don't filter:
    python check_word_durations.py --manifest voxforge_pt_alignment_manifest.json

    # Report AND write a cleaned manifest with flagged entries removed:
    python check_word_durations.py \
        --manifest voxforge_pt_alignment_manifest.json \
        --max_word_duration 2.0 \
        --filtered_manifest_path voxforge_pt_alignment_manifest_clean.json
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--max_word_duration", type=float, default=2.0,
                 help="Flag any single word spanning longer than this many seconds "
                      "(default: 2.0 -- generous even for slow, deliberate speech).")
ap.add_argument("--filtered_manifest_path", type=Path, default=None,
                 help="Optional: write a cleaned manifest with flagged entries removed.")
args = ap.parse_args()

entries = []
with args.manifest.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

flagged = []
kept = []

for entry in entries:
    words = entry["words"]
    alignments = entry["alignments"]

    worst_word, worst_dur = None, 0.0
    prev_end = 0.0
    for word, end in zip(words, alignments):
        start = prev_end
        dur = end - start
        if word != "" and dur > worst_dur:
            worst_word, worst_dur = word, dur
        prev_end = end

    if worst_dur > args.max_word_duration:
        flagged.append((entry, worst_word, worst_dur))
    else:
        kept.append(entry)

print(f"Total entries:   {len(entries)}")
print(f"Flagged (a word > {args.max_word_duration}s): {len(flagged)}")
if flagged:
    print("\nWorst offenders:")
    for entry, word, dur in sorted(flagged, key=lambda x: -x[2])[:15]:
        print(f"  {dur:6.2f}s  {word!r:15s}  speaker={entry['speaker_id']:20s}  "
              f"file={Path(entry['audio_filepath']).name}")
    if len(flagged) > 15:
        print(f"  ... ({len(flagged) - 15} more)")

    from collections import Counter
    total_by_speaker = Counter(e["speaker_id"] for e in entries)
    flagged_by_speaker = Counter(e["speaker_id"] for e, _, _ in flagged)
    print("\nPer-speaker flag rate (speakers with >=3 flagged entries, sorted by rate):")
    rows = []
    for spk, n_flagged in flagged_by_speaker.items():
        n_total = total_by_speaker[spk]
        if n_flagged >= 3:
            rows.append((spk, n_flagged, n_total, n_flagged / n_total))
    rows.sort(key=lambda r: -r[3])
    for spk, n_flagged, n_total, rate in rows[:20]:
        print(f"  {spk:30s} {n_flagged:>4d}/{n_total:<4d} flagged  ({rate*100:5.1f}%)")
    if len(rows) > 20:
        print(f"  ... ({len(rows) - 20} more speakers with >=3 flagged entries)")

if args.filtered_manifest_path:
    with args.filtered_manifest_path.open("w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nCleaned manifest ({len(kept)} entries) written to: {args.filtered_manifest_path}")

    from collections import Counter
    before_counts = Counter(e["speaker_id"] for e in entries)
    after_counts = Counter(e["speaker_id"] for e in kept)
    zeroed_out = sorted(set(before_counts) - set(after_counts))
    thin = sorted(
        [(spk, after_counts[spk], before_counts[spk]) for spk in after_counts
         if after_counts[spk] <= 2 and before_counts[spk] >= 5],
        key=lambda r: r[1],
    )

    if zeroed_out:
        print(f"\nWARNING: {len(zeroed_out)} speaker(s) lost ALL utterances to filtering:")
        for spk in zeroed_out:
            print(f"  - {spk} (had {before_counts[spk]})")
    if thin:
        print(f"\nWARNING: {len(thin)} speaker(s) left with only 1-2 utterances after filtering "
              f"(started with 5+) -- barely represented, worth a look before proceeding:")
        for spk, n_after, n_before in thin[:20]:
            print(f"  - {spk}: {n_before} -> {n_after}")
        if len(thin) > 20:
            print(f"  ... ({len(thin) - 20} more)")
