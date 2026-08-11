#!/usr/bin/env python3
"""
Filter a manifest down to only the utterances NeMo Forced Aligner (NFA)
actually produced a word-level CTM file for, dropping the rest.

Warns explicitly if dropping unaligned utterances would remove a speaker
from the corpus entirely (as opposed to just reducing their utterance
count) -- that's a meaningfully different situation worth knowing about
before moving on, since it changes speaker diversity rather than just
trimming volume.

Usage:
    python filter_manifest_by_alignment.py \
        --manifest /path/to/voxforge_pt_manifest.json \
        --output_dir /path/to/voxforge_pt_alignments \
        --filtered_manifest_path /path/to/voxforge_pt_manifest_aligned.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path,
                 help="The output_dir= you passed to align.py")
ap.add_argument("--filtered_manifest_path", required=True, type=Path)
args = ap.parse_args()

entries = []
with args.manifest.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

words_dir = args.output_dir / "ctm" / "words"
aligned_ids = {p.stem for p in words_dir.glob("*.ctm")} if words_dir.exists() else set()

before_counts = Counter(e["speaker_id"] for e in entries)

kept = [e for e in entries if Path(e["audio_filepath"]).stem in aligned_ids]
dropped = [e for e in entries if Path(e["audio_filepath"]).stem not in aligned_ids]

after_counts = Counter(e["speaker_id"] for e in kept)

zeroed_out = sorted(set(before_counts) - set(after_counts))

args.filtered_manifest_path.parent.mkdir(parents=True, exist_ok=True)
with args.filtered_manifest_path.open("w", encoding="utf-8") as f:
    for e in kept:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Utterances before filtering: {len(entries)}")
print(f"Utterances dropped (unaligned): {len(dropped)}")
print(f"Utterances kept:             {len(kept)}")
print(f"Speakers before filtering:    {len(before_counts)}")
print(f"Speakers after filtering:     {len(after_counts)}")
print(f"Filtered manifest written to: {args.filtered_manifest_path}")

if zeroed_out:
    print(f"\nWARNING: {len(zeroed_out)} speaker(s) had ALL their utterances dropped "
          f"(not just reduced) -- they no longer appear in the corpus at all:")
    for spk in zeroed_out:
        print(f"  - {spk} (had {before_counts[spk]} utterance(s), all unaligned)")
else:
    print("\nNo speaker was entirely removed -- every speaker still has at least one "
          "aligned utterance remaining.")
