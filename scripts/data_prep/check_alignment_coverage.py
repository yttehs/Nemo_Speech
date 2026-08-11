#!/usr/bin/env python3
"""
Check that NeMo Forced Aligner (NFA) produced a word-level CTM file for every
utterance in the input manifest, and report exactly which ones are missing
if not (NFA can silently skip individual utterances without failing the run).

Usage:
    python check_alignment_coverage.py \
        --manifest /path/to/voxforge_pt_manifest.json \
        --output_dir /path/to/voxforge_pt_alignments \
        [--dump_missing_csv /path/to/missing_details.csv]
"""

import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path,
                 help="The manifest you fed into align.py (manifest_filepath=)")
ap.add_argument("--output_dir", required=True, type=Path,
                 help="The output_dir= you passed to align.py")
ap.add_argument("--dump_missing_csv", type=Path, default=None,
                 help="Optional: write full manifest details (duration, text, speaker) "
                      "for every missing utterance to this CSV, sorted by speaker so "
                      "any speaker-specific failure pattern is easy to spot.")
args = ap.parse_args()

# NFA's default utt_id is the stem of audio_filepath.
manifest_entries = {}  # utt_id -> full manifest entry
with args.manifest.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        utt_id = Path(entry["audio_filepath"]).stem
        manifest_entries[utt_id] = entry

expected_ids = set(manifest_entries.keys())

words_dir = args.output_dir / "ctm" / "words"
found_ids = {p.stem for p in words_dir.glob("*.ctm")} if words_dir.exists() else set()

missing = expected_ids - found_ids
extra = found_ids - expected_ids  # shouldn't normally happen, but worth surfacing if it does

print(f"Utterances in manifest:        {len(expected_ids)}")
print(f"Word-level CTM files found:    {len(found_ids)}")
print(f"Missing (in manifest, no CTM): {len(missing)}")
if missing:
    shown = sorted(missing)[:20]
    print(f"  -> {shown}{' ... (truncated)' if len(missing) > 20 else ''}")
if extra:
    print(f"CTM files with no manifest entry (unexpected): {len(extra)}")
    print(f"  -> {sorted(extra)[:20]}")

if not missing and not extra:
    print("\nAll utterances aligned. Coverage is complete.")
else:
    print("\nCoverage incomplete -- see missing IDs above before moving on to create_alignment_manifest.py.")

if missing and args.dump_missing_csv:
    rows = []
    for utt_id in missing:
        entry = manifest_entries[utt_id]
        duration = entry.get("duration")
        text = entry.get("text", "")
        rows.append({
            "speaker_id": entry.get("speaker_id", ""),
            "utt_id": utt_id,
            "duration": duration,
            "text_len_chars": len(text),
            "chars_per_sec": round(len(text) / duration, 1) if duration else None,
            "text": text,
            "audio_filepath": entry.get("audio_filepath", ""),
        })
    rows.sort(key=lambda r: (r["speaker_id"], r["utt_id"]))

    with args.dump_missing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull details for all {len(rows)} missing utterances written to: {args.dump_missing_csv}")
    print("Sorted by speaker_id -- scan for any speaker with a disproportionate share of failures,")
    print("and check chars_per_sec for outliers (very high = text likely too long for the audio's")
    print("duration, which can make CTC alignment infeasible).")
