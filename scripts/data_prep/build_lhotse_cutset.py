#!/usr/bin/env python3
"""
Convert pooled simulator sessions (audio_filepath + per-segment .json entries
with label/text/offset/duration) into a Lhotse CutSet manifest -- the native
format LhotseSpeechToTextSpkBpeDataset / speaker_to_target() actually consume.

Why not just point at the .rttm files directly: speaker_to_target()'s bare-RTTM
path (via cut.custom['rttm_filepath']) works for speaker timing, but RTTM has
no text field, so return_text=True (which the real training dataset calls)
would fail. Building real SupervisionSegment objects with text attached,
directly from the pooled .json files, avoids that entirely.

Usage:
    python build_lhotse_cutset.py \
        --pooled_dir /path/to/train_pooled \
        --output_cuts /path/to/train_cuts.jsonl.gz
"""
import argparse
import json
from pathlib import Path

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pooled_dir", required=True, type=Path)
ap.add_argument("--output_cuts", required=True, type=Path)
args = ap.parse_args()

json_files = sorted(
    p for p in args.pooled_dir.glob("*.json")
    if not p.name.endswith("_combined_manifest.json")  # pool_sessions.py's aggregate file, not per-session
)

cuts = []
n_skipped_no_wav = 0
n_skipped_empty_cut = 0
n_entries_dropped_empty_text = 0

for json_path in json_files:
    session_name = json_path.stem
    wav_path = args.pooled_dir / f"{session_name}.wav"
    if not wav_path.exists():
        n_skipped_no_wav += 1
        continue

    entries = [json.loads(l) for l in json_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not entries:
        n_skipped_no_wav += 1
        continue

    # Drop entries with empty/whitespace-only text -- an empty transcript is a
    # bad training example either way, and if this speaker is ever picked as
    # the random target for this cut, the model would be asked to predict
    # nothing from real audio. Removing the supervision also correctly
    # removes that stretch from the speaker's activity mask, which is right
    # here since there's no real speech content to represent.
    kept_entries = [e for e in entries if e.get("text", "").strip()]
    n_entries_dropped_empty_text += len(entries) - len(kept_entries)

    if not kept_entries:
        n_skipped_empty_cut += 1
        continue  # every entry in this session was empty-text; nothing usable left

    recording = Recording.from_file(wav_path, recording_id=session_name)

    supervisions = [
        SupervisionSegment(
            id=f"{session_name}-sup{i:05d}",
            recording_id=session_name,
            start=e["offset"],
            duration=e["duration"],
            text=e["text"],
            speaker=e["label"],
        )
        for i, e in enumerate(kept_entries)
    ]

    cut = MonoCut(
        id=session_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=supervisions,
        custom={},  # speaker_to_target() calls cut.custom.get(...) unconditionally; None would crash it
    )
    cuts.append(cut)

cutset = CutSet.from_cuts(cuts)
args.output_cuts.parent.mkdir(parents=True, exist_ok=True)
cutset.to_file(args.output_cuts)
print(f"Wrote {len(cuts)} cuts to {args.output_cuts}")
print(f"Sessions skipped (missing/empty json or wav): {n_skipped_no_wav}")
print(f"Sessions skipped (all entries were empty-text): {n_skipped_empty_cut}")
print(f"Individual empty-text entries dropped (includes entries from skipped sessions above): {n_entries_dropped_empty_text}")
