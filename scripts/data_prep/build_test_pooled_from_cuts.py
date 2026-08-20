#!/usr/bin/env python3
"""
Builds a "test_pooled"-equivalent directory from a combined Lhotse CutSet (e.g.
Data/multitalker_train_data/test_cuts.jsonl.gz) -- the flat, per-session
.wav/.rttm/.json layout that run_sortformer_batch.py and
evaluate_checkpoint_sortformer_conditioned.py actually expect as --wav_dir/--ref_dir.

Necessary because our CML-TTS-PT/FastMSS pipeline never produced that layout --
pool_sessions.py is hard-coded to NeMo's OWN MultiSpeakerSimulator's file naming
(multispeaker_session_N.{wav,rttm,ctm,json,meta,txt}), a completely different
convention than FastMSS's UUID-named output, and wouldn't find anything to pool
here. This script goes straight from the Lhotse CutSet instead.

Per cut, writes:
    <uri>.wav   symlink to the cut's real underlying audio file (no need to
                duplicate it -- same reasoning as prepare_mfa_corpus.py earlier)
    <uri>.rttm  ground truth, built directly from the cut's SupervisionSegments
    <uri>.json  JSONL, one line per supervision, with the offset/label/text keys
                ground_truth_text_per_speaker() actually reads

Usage:
    python build_test_pooled_from_cuts.py \
        --cuts Data/multitalker_train_data/test_cuts.jsonl.gz \
        --output_dir Data/test_pooled
"""
import argparse
import json
from pathlib import Path

from lhotse import CutSet

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cuts", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path)
args = ap.parse_args()


def main():
    cuts = CutSet.from_file(args.cuts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_no_audio_source = 0

    for cut in cuts:
        uri = cut.id

        try:
            audio_source = Path(cut.recording.sources[0].source).resolve()
        except (AttributeError, IndexError):
            n_no_audio_source += 1
            print(f"  WARNING: {uri} has no resolvable audio source, skipping")
            continue

        wav_dst = args.output_dir / f"{uri}.wav"
        if not wav_dst.exists():
            wav_dst.symlink_to(audio_source)

        rttm_dst = args.output_dir / f"{uri}.rttm"
        with rttm_dst.open("w", encoding="utf-8") as f:
            for sup in cut.supervisions:
                f.write(f"SPEAKER {uri} 1 {sup.start:.3f} {sup.duration:.3f} "
                        f"<NA> <NA> {sup.speaker} <NA> <NA>\n")

        json_dst = args.output_dir / f"{uri}.json"
        with json_dst.open("w", encoding="utf-8") as f:
            for sup in cut.supervisions:
                entry = {
                    "audio_filepath": str(wav_dst.resolve()),
                    "offset": sup.start,
                    "duration": sup.duration,
                    "label": sup.speaker,
                    "text": sup.text or "",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        n_written += 1

    print(f"{n_written} sessions written to {args.output_dir} "
          f"({n_no_audio_source} skipped for missing audio source)")


if __name__ == "__main__":
    main()
