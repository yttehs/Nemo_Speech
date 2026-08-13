#!/usr/bin/env python3
"""
Pool sessions from multiple per-bucket simulator output directories (e.g.
train_simulated_sessions_1spk, _2spk, _3spk, _4spk) into a single directory
per split, with globally unique session names and internally-consistent
path/session-name references throughout every file type.

Why this is needed: the simulator numbers sessions independently within each
output directory, starting from 0 -- so multispeaker_session_0.wav exists in
every bucket directory and would collide if just copied into one place.
Beyond the filename collision, the session name is also embedded INSIDE the
.json (audio_filepath / rttm_filepath / ctm_filepath), .rttm (column 2), and
.ctm (column 1) content. A plain rename without rewriting those references
would leave every file pointing at stale, wrong paths.

.wav, .meta, .txt have no internal path/session-name references and are
just copied under the new name. .json/.rttm/.ctm are rewritten. .list files
are rebuilt fresh from the pooled, renamed set. A single concatenated
per-segment manifest is also produced, since that's what a NeMo training
script will actually load via manifest_filepath=.

Usage:
    python pool_sessions.py \
        --split_name train \
        --bucket_dirs \
            /path/train_simulated_sessions_1spk \
            /path/train_simulated_sessions_2spk \
            /path/train_simulated_sessions_3spk \
            /path/train_simulated_sessions_4spk \
        --output_dir /path/train_pooled
"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--split_name", required=True, help="e.g. train, val, test -- used in the new session names")
ap.add_argument("--bucket_dirs", required=True, nargs="+", type=Path,
                 help="One or more per-bucket simulator output directories to pool together")
ap.add_argument("--output_dir", required=True, type=Path)
args = ap.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)

FILE_TYPES = ["wav", "rttm", "ctm", "json", "meta", "txt"]
all_new_paths = {ft: [] for ft in FILE_TYPES}
combined_manifest_lines = []

n_sessions_total = 0
for bucket_dir in args.bucket_dirs:
    m = re.search(r"(\d+)spk", bucket_dir.name)
    if not m:
        raise ValueError(f"Could not find '<N>spk' in directory name: {bucket_dir}")
    spk_tag = f"{m.group(1)}spk"

    # Discover session indices from the wav files actually present, rather
    # than assuming a session count.
    session_indices = sorted(
        int(mm.group(1))
        for wf in bucket_dir.glob("multispeaker_session_*.wav")
        if (mm := re.match(r"multispeaker_session_(\d+)\.wav$", wf.name))
    )

    for idx in session_indices:
        old_base = f"multispeaker_session_{idx}"
        new_base = f"{args.split_name}_{spk_tag}_sess{idx}"
        n_sessions_total += 1

        # --- .wav, .meta, .txt: no internal references, just copy under the new name ---
        for ext in ("wav", "meta", "txt"):
            src = bucket_dir / f"{old_base}.{ext}"
            if not src.exists():
                raise FileNotFoundError(f"Expected file missing: {src}")
            dst = args.output_dir / f"{new_base}.{ext}"
            dst.write_bytes(src.read_bytes())
            all_new_paths[ext].append(str(dst.resolve()))

        # --- .rttm: rewrite column 2 (session name) on every SPEAKER line ---
        rttm_src = bucket_dir / f"{old_base}.rttm"
        rttm_dst = args.output_dir / f"{new_base}.rttm"
        new_lines = []
        for line in rttm_src.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[0] == "SPEAKER":
                fields[1] = new_base
            new_lines.append(" ".join(fields))
        rttm_dst.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        all_new_paths["rttm"].append(str(rttm_dst.resolve()))

        # --- .ctm: rewrite column 1 (session name) on every line ---
        ctm_src = bucket_dir / f"{old_base}.ctm"
        ctm_dst = args.output_dir / f"{new_base}.ctm"
        new_lines = []
        for line in ctm_src.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                fields[0] = new_base
            new_lines.append(" ".join(fields))
        ctm_dst.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        all_new_paths["ctm"].append(str(ctm_dst.resolve()))

        # --- .json: rewrite audio_filepath / rttm_filepath / ctm_filepath on every line ---
        json_src = bucket_dir / f"{old_base}.json"
        json_dst = args.output_dir / f"{new_base}.json"
        new_lines = []
        for line in json_src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry["audio_filepath"] = str((args.output_dir / f"{new_base}.wav").resolve())
            entry["rttm_filepath"] = str((args.output_dir / f"{new_base}.rttm").resolve())
            entry["ctm_filepath"] = str((args.output_dir / f"{new_base}.ctm").resolve())
            new_line = json.dumps(entry, ensure_ascii=False)
            new_lines.append(new_line)
            combined_manifest_lines.append(new_line)
        json_dst.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        all_new_paths["json"].append(str(json_dst.resolve()))

# --- Rebuild .list files fresh, from the pooled/renamed set ---
for ext, paths in all_new_paths.items():
    list_path = args.output_dir / f"synthetic_{ext}.list"
    list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")

# --- Combined per-segment manifest: the single file a training script loads ---
combined_manifest_path = args.output_dir / f"{args.split_name}_combined_manifest.json"
combined_manifest_path.write_text("\n".join(combined_manifest_lines) + "\n", encoding="utf-8")

print(f"Pooled {n_sessions_total} sessions from {len(args.bucket_dirs)} bucket(s) into: {args.output_dir}")
print(f"Combined manifest ({len(combined_manifest_lines)} segments): {combined_manifest_path}")
