#!/usr/bin/env python3
"""
Reformat NeMo Forced Aligner's word-level CTM output into the CTM format
`scripts/speaker_tasks/create_alignment_manifest.py` actually expects.

Why this is needed: NFA's CTM lines have 5 fields --
    <utt_id> <channel=1> <start> <duration> <token>
but create_alignment_manifest.py's reader (get_seg_info_from_ctm_line) expects
the 8-field RT09 rich-transcription format --
    <SOURCE> <CHANNEL> <BEG-TIME> <DURATION> <TOKEN> <CONF> <TYPE> <SPEAKER>
and reads the speaker ID from field index 7, which NFA's output doesn't have
at all. Feeding NFA's CTM files in directly throws an IndexError.

Since each of our utterance files is a single speaker throughout, the fix is
simple: look up that utterance's speaker_id (already known, from the filtered
manifest -- no diarization needed) and inject it as a constant into every
line of that file's CTM, along with placeholder confidence/type fields.

Usage:
    python reformat_ctm_for_alignment_manifest.py \
        --manifest /path/to/voxforge_pt_manifest_aligned.json \
        --nfa_ctm_words_dir /path/to/voxforge_pt_alignments/ctm/words \
        --output_ctm_dir /path/to/voxforge_pt_alignments/ctm_words_reformatted
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path,
                 help="The filtered manifest (e.g. voxforge_pt_manifest_aligned.json)")
ap.add_argument("--nfa_ctm_words_dir", required=True, type=Path,
                 help="NFA's output_dir/ctm/words directory")
ap.add_argument("--output_ctm_dir", required=True, type=Path,
                 help="Where to write the reformatted CTM files")
args = ap.parse_args()

# utt_id -> speaker_id, from the manifest
speaker_by_utt_id = {}
with args.manifest.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        utt_id = Path(entry["audio_filepath"]).stem
        speaker_by_utt_id[utt_id] = entry["speaker_id"]

args.output_ctm_dir.mkdir(parents=True, exist_ok=True)

n_written = 0
n_no_speaker = 0
for ctm_path in sorted(args.nfa_ctm_words_dir.glob("*.ctm")):
    utt_id = ctm_path.stem
    speaker_id = speaker_by_utt_id.get(utt_id)
    if speaker_id is None:
        # CTM exists but this utt_id isn't in our (filtered) manifest -- skip it
        # rather than guess a speaker, so it doesn't silently end up mislabeled.
        n_no_speaker += 1
        continue

    out_lines = []
    for raw_line in ctm_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        fields = raw_line.split(" ")
        # fields: [utt_id, channel, start, duration, token] (NFA's 5-field format)
        source, channel, start, duration, token = fields[0], fields[1], fields[2], fields[3], fields[4]
        out_lines.append(f"{source} {channel} {start} {duration} {token} 1.00 lex {speaker_id}")

    (args.output_ctm_dir / ctm_path.name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    n_written += 1

print(f"Reformatted CTM files written: {n_written}")
if n_no_speaker:
    print(f"Skipped (CTM present but utt_id not in manifest): {n_no_speaker}")
print(f"Output directory: {args.output_ctm_dir}")
