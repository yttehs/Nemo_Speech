#!/usr/bin/env python3
"""
Tier 2: Sortformer-conditioned WER on your simulated test set. For each session:
(1) find the optimal ground-truth-speaker <-> Sortformer-speaker mapping via
score_der's own overlap-based matching (reused directly, not reimplemented),
(2) for each matched pair, transcribe conditioned on SORTFORMER's predicted
timing for that speaker, (3) score against the GROUND-TRUTH speaker's real
reference text.

Interpret cautiously: a gap between this and Tier 1 (ground-truth-conditioned)
could reflect the ASR model's own quality OR Sortformer's known struggles with
simulated/spliced audio specifically (see the earlier DER investigation) --
these aren't distinguishable from this number alone. Tier 3 (real audio) is
the one that isolates ASR quality from that confound.

Run run_sortformer_batch.py on test_pooled FIRST to get predicted RTTMs.

Usage:
    python evaluate_checkpoint_sortformer_conditioned.py \
        --overrides conf/multitalker_finetune_overrides_aws.yaml \
        --checkpoint ".../multitalker_pt_adapter_aws--val_wer=0.6957-epoch=46.ckpt" \
        --ref_dir /path/to/test_pooled \
        --pred_dir /path/to/test_pooled_sortformer_preds
"""
import argparse
import json
from pathlib import Path

from scripts.data_prep.der_scorer import score_der
from scripts.data_prep.multispeaker_inference import load_model, parse_rttm, build_mask_for_speaker, transcribe_with_mask
from scripts.data_prep.wer_scorer import word_error_rate

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--overrides", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--ref_dir", required=True, type=Path, help="test_pooled directory (ground truth wav/rttm/json)")
ap.add_argument("--pred_dir", required=True, type=Path, help="Directory of Sortformer's predicted RTTMs")
ap.add_argument("--pretrained_model", default="nvidia/parakeet-tdt-0.6b-v3")
ap.add_argument("--dummy_cuts_path", required=True,
                 help="Any real, valid *_cuts.jsonl.gz file from training (e.g. test_cuts.jsonl.gz) -- "
                      "needed only because NeMo validates train_ds/validation_ds/test_ds as real "
                      "manifests at model construction time, even though this script never uses them.")
ap.add_argument("--collar", type=float, default=0.25)
args = ap.parse_args()


def ground_truth_text_per_speaker(json_path):
    """Reconstruct each speaker's full reference text, in time order -- same
    aggregation build_lhotse_cutset.py / speaker_to_target() use."""
    entries = [json.loads(l) for l in json_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries.sort(key=lambda e: e["offset"])
    text_by_speaker = {}
    for e in entries:
        text_by_speaker.setdefault(e["label"], []).append(e["text"])
    return {spk: " ".join(words) for spk, words in text_by_speaker.items()}


def main():
    print("Loading model...")
    model = load_model(args.overrides, args.checkpoint, args.pretrained_model,
                        dummy_cuts_path=args.dummy_cuts_path)

    pred_rttms = sorted(args.pred_dir.glob("*.rttm"))
    print(f"Found {len(pred_rttms)} predicted RTTMs to evaluate\n")

    total_sub = total_del = total_ins = total_ref_words = 0
    n_pairs_scored = 0
    n_sessions_skipped = 0

    for pred_rttm in pred_rttms:
        session_name = pred_rttm.stem
        ref_rttm = args.ref_dir / f"{session_name}.rttm"
        ref_json = args.ref_dir / f"{session_name}.json"
        wav_path = args.ref_dir / f"{session_name}.wav"
        if not (ref_rttm.exists() and ref_json.exists() and wav_path.exists()):
            n_sessions_skipped += 1
            continue

        der_result = score_der(ref_rttm, pred_rttm, collar=args.collar)
        mapping = der_result.get("speaker_mapping", {})
        if not mapping:
            n_sessions_skipped += 1
            continue

        ref_texts = ground_truth_text_per_speaker(ref_json)
        sortformer_segments = parse_rttm(pred_rttm)

        for ref_speaker, sortformer_speaker in mapping.items():
            reference_text = ref_texts.get(ref_speaker, "")
            if not reference_text.strip():
                continue
            try:
                spk_target, bg_spk_target = build_mask_for_speaker(
                    wav_path, sortformer_segments, target_speaker=sortformer_speaker
                )
            except ValueError:
                continue

            hypothesis_text = transcribe_with_mask(model, wav_path, spk_target, bg_spk_target)
            wer, n_sub, n_del, n_ins, n_ref = word_error_rate(reference_text, hypothesis_text)

            total_sub += n_sub
            total_del += n_del
            total_ins += n_ins
            total_ref_words += n_ref
            n_pairs_scored += 1

            print(f"[{session_name}] ref_speaker={ref_speaker} -> sortformer={sortformer_speaker}  WER={wer:.3f}")
            print(f"    reference:  {reference_text}")
            print(f"    hypothesis: {hypothesis_text}\n")

    print(f"\nScored {n_pairs_scored} speaker-pairs across sessions ({n_sessions_skipped} sessions skipped)")
    overall_wer = (
        (total_sub + total_del + total_ins) / total_ref_words if total_ref_words > 0 else float("nan")
    )
    print(f"Overall Tier 2 WER: {overall_wer:.4f}  "
          f"(sub={total_sub}, del={total_del}, ins={total_ins}, ref_words={total_ref_words})")


if __name__ == "__main__":
    main()
