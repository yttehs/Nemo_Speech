#!/usr/bin/env python3
"""
Verify that every generated simulated session only contains speakers that
actually belong to its intended split -- catches exactly the kind of mistake
where a session directory gets generated from the wrong source manifest
(e.g. "test" sessions accidentally built from the val manifest).

Reads speaker_split_details.csv (from split_speakers.py) for the ground-truth
speaker->split mapping, then scans every .rttm file under each provided
session directory and checks every speaker_id found against the expected
split for that directory.

Usage:
    python check_split_leakage.py \
        --split_csv /path/to/splits/speaker_split_details.csv \
        --train_dirs train_simulated_sessions_1spk train_simulated_sessions_2spk ... \
        --val_dirs   val_simulated_sessions_1spk   val_simulated_sessions_2spk   ... \
        --test_dirs  test_simulated_sessions_1spk  test_simulated_sessions_2spk  ...
"""
import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--split_csv", required=True, type=Path)
ap.add_argument("--train_dirs", nargs="*", type=Path, default=[])
ap.add_argument("--val_dirs", nargs="*", type=Path, default=[])
ap.add_argument("--test_dirs", nargs="*", type=Path, default=[])
args = ap.parse_args()

speaker_split = {}
with args.split_csv.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        speaker_split[row["speaker_id"]] = row["split"]


def speakers_in_rttm_dir(directory: Path):
    """All distinct speaker_ids appearing across every .rttm file in a directory."""
    found = set()
    for rttm_path in directory.glob("*.rttm"):
        for line in rttm_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if len(fields) >= 8 and fields[0] == "SPEAKER":
                found.add(fields[7])
    return found


any_leak = False
for expected_split, dirs in [("train", args.train_dirs), ("val", args.val_dirs), ("test", args.test_dirs)]:
    for d in dirs:
        if not d.exists():
            print(f"  SKIP (not found): {d}")
            continue
        found_speakers = speakers_in_rttm_dir(d)
        leaked = {
            spk: speaker_split.get(spk, "UNKNOWN (not in split_csv at all)")
            for spk in found_speakers
            if speaker_split.get(spk) != expected_split
        }
        if leaked:
            any_leak = True
            print(f"LEAK in {d}  (expected all speakers to be '{expected_split}'):")
            for spk, actual_split in sorted(leaked.items()):
                print(f"    {spk}  ->  actually belongs to: {actual_split}")
        else:
            print(f"OK    {d}  ({len(found_speakers)} distinct speakers, all correctly '{expected_split}')")

print()
if any_leak:
    print("LEAKAGE DETECTED -- see above. Do not proceed to training until this is resolved.")
else:
    print("No cross-split leakage detected. All checked session directories are clean.")
