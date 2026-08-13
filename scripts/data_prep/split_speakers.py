#!/usr/bin/env python3
"""
Split speakers (not sessions) into disjoint train/val/test pools, then write
one filtered manifest per split.

Splitting by SPEAKER rather than by session is what actually prevents
train/eval leakage -- session-level splitting would let the same speaker's
voice appear in both training and evaluation, just paired with different
co-speakers, which doesn't test generalization to genuinely new voices the
way real deployment will require.

Val/test pools are drawn from the speakers with the MOST distinct utterances
available (so held-out sessions can be closer to full-length and
representative), while train absorbs the thinner speakers, which can lean on
the simulator's adaptive session-length fallback without affecting how eval
numbers are interpreted. Ranking is by utterance COUNT rather than total
duration: the anti-repetition sampling in the simulator tracks which files
have been used, cycling through distinct clips before resetting, so count is
what actually determines how much a speaker can be drawn from before
repeating -- not raw seconds, which can be skewed by a few unusually long
individual clips. Which "rich" speakers land in val vs. test specifically is
randomized, so val and test aren't systematically different from each other.

Usage:
    python split_speakers.py \
        --manifest voxforge_pt_alignment_manifest_clean.json \
        --output_dir ./splits \
        --val_speakers 30 \
        --test_speakers 30 \
        --seed 42
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path)
ap.add_argument("--val_speakers", type=int, default=30, help="Number of speakers held out for validation")
ap.add_argument("--test_speakers", type=int, default=30, help="Number of speakers held out for test")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)

entries = []
with args.manifest.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

# Group entries and total duration by speaker.
by_speaker = defaultdict(list)
duration_by_speaker = defaultdict(float)
for e in entries:
    spk = e["speaker_id"]
    by_speaker[spk].append(e)
    duration_by_speaker[spk] += e["duration"]

all_speakers = list(by_speaker.keys())
random.shuffle(all_speakers)  # so ties in utterance count break randomly, not by incidental manifest order
n_total = len(all_speakers)
n_holdout = args.val_speakers + args.test_speakers

if n_holdout >= n_total:
    raise ValueError(
        f"Requested {n_holdout} held-out speakers (val+test), but only {n_total} "
        f"speakers exist in the manifest."
    )

# Speakers with the MOST distinct utterances become the candidate pool for
# val/test -- count, not total duration, since that's what the simulator's
# no-repeat sampling actually draws down before it has to reset and repeat.
speakers_by_count_desc = sorted(all_speakers, key=lambda s: len(by_speaker[s]), reverse=True)
holdout_pool = speakers_by_count_desc[:n_holdout]
train_speakers = speakers_by_count_desc[n_holdout:]

# Randomize WHICH of the "rich" speakers go to val vs. test, so the two
# aren't systematically different from each other -- both are drawn from
# the same high-material subpopulation, just randomly divided.
random.shuffle(holdout_pool)
val_speakers = holdout_pool[: args.val_speakers]
test_speakers = holdout_pool[args.val_speakers :]

splits = {"train": train_speakers, "val": val_speakers, "test": test_speakers}
speaker_to_split = {spk: split_name for split_name, speakers in splits.items() for spk in speakers}

args.output_dir.mkdir(parents=True, exist_ok=True)

print(f"Total speakers in manifest:   {n_total}")
print(f"Total utterances in manifest: {len(entries)}")
print()

for split_name, speakers in splits.items():
    split_entries = [e for spk in speakers for e in by_speaker[spk]]
    total_dur = sum(duration_by_speaker[spk] for spk in speakers)

    out_path = args.output_dir / f"voxforge_pt_{split_name}_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        for e in split_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(
        f"{split_name:5s}: {len(speakers):4d} speakers, {len(split_entries):5d} utterances, "
        f"{total_dur / 60:7.1f} min total  ->  {out_path}"
    )

# Sanity check: confirm disjointness before trusting the output.
train_set, val_set, test_set = set(train_speakers), set(val_speakers), set(test_speakers)
assert not (train_set & val_set), "Train/val speaker overlap detected!"
assert not (train_set & test_set), "Train/test speaker overlap detected!"
assert not (val_set & test_set), "Val/test speaker overlap detected!"
print("\nDisjointness verified: no speaker appears in more than one split.")

# Full per-speaker detail, sorted by utterance count descending, so the
# val/test cutoff is directly visible -- this is what actually lets you
# cross-check that the highest-count speakers landed in val/test.
print(f"\n=== All {n_total} speakers, sorted by utterance count (descending) ===")
print(f"{'speaker_id':<35s} {'utterances':>10s} {'duration_min':>13s} {'split':>7s}")
for spk in speakers_by_count_desc:
    n_utts = len(by_speaker[spk])
    dur_min = duration_by_speaker[spk] / 60
    print(f"{spk:<35s} {n_utts:>10d} {dur_min:>13.2f} {speaker_to_split[spk]:>7s}")

# Same detail, written to CSV for easier sorting/filtering outside the terminal.
csv_path = args.output_dir / "speaker_split_details.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["speaker_id", "utterance_count", "total_duration_sec", "split"])
    for spk in speakers_by_count_desc:
        writer.writerow([spk, len(by_speaker[spk]), round(duration_by_speaker[spk], 2), speaker_to_split[spk]])
print(f"\nFull per-speaker detail written to: {csv_path}")

# Explicit split membership lists, each sorted by count descending, so it's
# immediately visible that val/test cluster at the high-count end.
for split_name in ("val", "test", "train"):
    ordered = [spk for spk in speakers_by_count_desc if spk in set(splits[split_name])]
    print(f"\n{split_name} speakers ({len(ordered)}), by utterance count descending:")
    print("  " + ", ".join(f"{spk} ({len(by_speaker[spk])})" for spk in ordered))
