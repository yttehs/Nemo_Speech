#!/usr/bin/env python3
"""
Parses one pt_avprep *_speaker_attribution_final.json file into a flat list of
segments: {start, end, speaker, text, note}.

final_speaker is treated as the ground-truth speaker label (the reconciled
TalkNet + voice-identity decision -- see final_status for how each was decided).

Segments whose text_pt contains " - " are flagged: confirmed with the pipeline's
author that this marks captions where the OCR source bundled a quick multi-speaker
exchange into a single subtitle frame (e.g. "Hoje... - Ou pretende entrevistar."),
meaning the reference text for that segment may contain words from more than one
speaker even though it's credited to a single final_speaker. Deliberately NOT
excluded or split (splitting would require guessing at where one speaker's words
end and the other's begin, and per-half timing isn't recoverable from a single
caption timestamp) -- flagged instead, so results can be interpreted with this
important caveat rather than silently treating this text as clean, single-speaker
ground truth.
"""
import json
from pathlib import Path

OVERLAP_NOTE = "Potential overlap here - GT not captured properly"


def parse_attribution_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    n_null_speaker = 0
    for entry in data["attribution"]:
        if entry["final_speaker"] is None:
            # The attribution pipeline itself couldn't confidently resolve a
            # speaker for this segment (neither TalkNet nor voice-matching
            # succeeded) -- confirmed this happens in real data (Brazil_portuguese_04
            # crashed on exactly this before this fix). Excluded rather than given
            # a placeholder label: an unresolved segment kept with any label would
            # inject unreliable ground truth, which is worse than losing that small
            # slice of audio from the eval set entirely.
            n_null_speaker += 1
            continue
        seg = {
            "start": entry["start_sec"],
            "end": entry["end_sec"],
            "speaker": entry["final_speaker"],
            "text": entry["text_pt"],
            "note": None,
        }
        if " - " in seg["text"]:
            seg["note"] = OVERLAP_NOTE
        segments.append(seg)
    if n_null_speaker:
        print(f"  WARNING: {n_null_speaker} segment(s) had final_speaker=null "
              f"(unresolved by the attribution pipeline), excluded")
    return segments, data.get("video_id")


if __name__ == "__main__":
    import sys
    segs, video_id = parse_attribution_file(Path(sys.argv[1]))
    print(f"video_id: {video_id}")
    print(f"{len(segs)} segments parsed")
    n_flagged = sum(1 for s in segs if s["note"])
    print(f"{n_flagged} segment(s) flagged for potential unmarked overlap "
          f"({100*n_flagged/len(segs):.1f}%)")
    print("\nFirst flagged example:")
    for s in segs:
        if s["note"]:
            print(f"  [{s['start']:.2f},{s['end']:.2f}] {s['speaker']}: {s['text']!r}")
            break
