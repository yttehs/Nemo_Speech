#!/usr/bin/env python3
"""
Tier 2: Sortformer-conditioned WER on your simulated test set. For each session:
(1) find the optimal ground-truth-speaker <-> Sortformer-speaker mapping via
score_der's own overlap-based matching (reused directly, not reimplemented),
(2) for each matched pair, transcribe conditioned on SORTFORMER's predicted
timing for that speaker, (3) score against the GROUND-TRUTH speaker's real
reference text.

Also reports Sortformer's own Diarization Error Rate against the MFA-derived
ground truth, aggregated (time-weighted, not a naive per-session average) and
broken down per-session -- this was previously computed internally by score_der()
purely to build the speaker mapping, then discarded without ever being reported.
DER is accumulated for EVERY session with valid files, independent of whether a
usable speaker mapping comes out of it: a session where Sortformer predicted no
speech at all has an empty mapping (and gets skipped for WER scoring, since
there's nothing to condition transcription on) but still has a real, meaningful
DER (100% missed) -- silently excluding it from the DER report would hide exactly
the kind of total-failure case this report exists to catch.

Interpret the WER number cautiously: a gap between this and Tier 1 (ground-truth-
conditioned) could reflect the ASR model's own quality OR Sortformer's known
struggles with simulated/spliced audio specifically (see the earlier DER
investigation) -- these aren't distinguishable from the WER number alone. That's
exactly what the new DER report is for: a poor DER here, on clean simulated audio
with real ground truth, points at diarization as a contributing factor before ever
touching the noise/duration confounds real audio introduces. Tier 3 (real audio)
remains the one that isolates ASR quality from the diarization confound entirely.

Run run_sortformer_batch.py on test_pooled FIRST to get predicted RTTMs.

Usage:
    python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
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
ap.add_argument("--collar", type=float, default=0.01)
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
    n_missing_files = 0
    n_no_wer_mapping = 0

    total_missed = total_false_alarm = total_confusion = total_correct = 0.0
    total_ref_time_der = 0.0
    per_session_der = []
    per_session_wer_stats = {}  # session_name -> [sub, del, ins, ref_words], accumulated across
                                  # that session's own speaker-pairs -- separate from the DER dict
                                  # above (keyed the same way) so the two can be joined afterward

    for pred_rttm in pred_rttms:
        session_name = pred_rttm.stem
        ref_rttm = args.ref_dir / f"{session_name}.rttm"
        ref_json = args.ref_dir / f"{session_name}.json"
        wav_path = args.ref_dir / f"{session_name}.wav"
        if not (ref_rttm.exists() and ref_json.exists() and wav_path.exists()):
            n_missing_files += 1
            continue

        der_result = score_der(ref_rttm, pred_rttm, collar=args.collar)

        if der_result.get("der") is not None:
            total_missed += der_result["missed"]
            total_false_alarm += der_result["false_alarm"]
            total_confusion += der_result["confusion"]
            total_correct += der_result["correct"]
            total_ref_time_der += der_result["total_ref_time"]
            per_session_der.append((session_name, der_result["der"], der_result["total_ref_time"]))

        mapping = der_result.get("speaker_mapping", {})
        if not mapping:
            n_no_wer_mapping += 1
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

            stats = per_session_wer_stats.setdefault(session_name, [0, 0, 0, 0])
            stats[0] += n_sub
            stats[1] += n_del
            stats[2] += n_ins
            stats[3] += n_ref

            print(f"[{session_name}] ref_speaker={ref_speaker} -> sortformer={sortformer_speaker}  WER={wer:.3f}")
            print(f"    reference:  {reference_text}")
            print(f"    hypothesis: {hypothesis_text}\n")

    print(f"\nScored {n_pairs_scored} speaker-pairs across sessions "
          f"({n_missing_files} sessions missing files, "
          f"{n_no_wer_mapping} sessions had no usable speaker mapping for WER)")
    overall_wer = (
        (total_sub + total_del + total_ins) / total_ref_words if total_ref_words > 0 else float("nan")
    )
    print(f"Overall Tier 2 WER: {overall_wer:.4f}  "
          f"(sub={total_sub}, del={total_del}, ins={total_ins}, ref_words={total_ref_words})")

    print(f"\n--- Diarization Error Rate (Sortformer vs. MFA ground truth, collar={args.collar}s) ---")
    if total_ref_time_der > 0:
        overall_der = (total_missed + total_false_alarm + total_confusion) / total_ref_time_der
        # Time-weighted, not a naive average of per-session DER values -- a session
        # with 5s of reference speech and one with 50s shouldn't count equally.
        print(f"Overall DER: {overall_der:.4f}  (time-weighted across {len(per_session_der)} sessions, "
              f"{total_ref_time_der:.1f}s total reference speech)")
        print(f"  missed={total_missed/total_ref_time_der:.4f}  "
              f"false_alarm={total_false_alarm/total_ref_time_der:.4f}  "
              f"confusion={total_confusion/total_ref_time_der:.4f}  "
              f"(fractions of total reference speech time)")

        print(f"\nPer-session DER, worst to best:")
        for session_name, der, ref_time in sorted(per_session_der, key=lambda x: -x[1]):
            print(f"  {der:.4f}  {session_name}  (ref_time={ref_time:.1f}s)")
    else:
        print("No sessions had a computable DER -- check that ref/pred RTTMs contain real speech.")

    # --- Does DER actually explain the WER gap, or is something else going on? ---
    # Joins the two per-session dicts above by session_name. Only sessions with BOTH a
    # computed DER and at least one scored WER pair are included -- a session that got
    # skipped for WER (empty mapping) has no WER value to correlate against, even though
    # it has a DER value from the block above.
    joined = []
    for session_name, der, ref_time in per_session_der:
        stats = per_session_wer_stats.get(session_name)
        if stats is None:
            continue
        sub, del_, ins, ref_words = stats
        if ref_words == 0:
            continue
        session_wer = (sub + del_ + ins) / ref_words
        joined.append((session_name, der, session_wer))

    print(f"\n--- DER vs. WER correlation ({len(joined)} sessions with both) ---")
    if len(joined) >= 2:
        ders = [d for _, d, _ in joined]
        wers = [w for _, _, w in joined]
        mean_der = sum(ders) / len(ders)
        mean_wer = sum(wers) / len(wers)
        cov = sum((d - mean_der) * (w - mean_wer) for d, w in zip(ders, wers))
        std_der = (sum((d - mean_der) ** 2 for d in ders)) ** 0.5
        std_wer = (sum((w - mean_wer) ** 2 for w in wers)) ** 0.5
        if std_der > 0 and std_wer > 0:
            pearson_r = cov / (std_der * std_wer)
            print(f"Pearson correlation (session DER, session WER): {pearson_r:.3f}")
            print("  (near 0 -> DER doesn't explain session-to-session WER variation, something else "
                  "does; closer to 1 -> sessions with worse diarization consistently show worse WER, "
                  "i.e. diarization IS a real driver of the WER gap)")
        else:
            print("Can't compute correlation -- no variance in DER or WER across these sessions.")

        print(f"\nPer-session DER and WER side by side, worst DER first:")
        for session_name, der, wer in sorted(joined, key=lambda x: -x[1]):
            print(f"  DER={der:.4f}  WER={wer:.4f}  {session_name}")
    else:
        print("Not enough sessions with both a DER and a scored WER to compute a correlation.")


if __name__ == "__main__":
    main()
