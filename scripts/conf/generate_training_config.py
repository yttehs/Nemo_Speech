#!/usr/bin/env python3
"""
Generates a dataset-specific multitalker training override config from
multitalker_finetune_overrides_template.yaml, by measuring the REAL session
duration distribution and session count from your actual combined train/dev/test
Lhotse cutsets -- rather than reusing another language/dataset's numbers, which
has already caused two real bugs in this project's history:
  - train_ds.max_duration copied unchanged from an earlier, differently-scaled
    dataset silently filtered out 100% of the new dataset's sessions (every
    session exceeded the reused cap), producing a 50-epoch run that trained
    on nothing at all.
  - optim.sched.max_steps copied unchanged under-scheduled training for a
    larger dataset with a different session-duration profile.

Computes:
  max_duration / quadratic_duration:
    ceil(observed_max_duration * duration_headroom / round_to) * round_to
    -- headroom above the observed max, not just clearing it exactly, in case
    a future re-generation of the dataset slightly increases it. Defaults
    (10% headroom, rounded up to the nearest 10) exactly reproduce both prior
    baselines' own hand-computed values: Portuguese (52.3s real max -> 60.0),
    Spanish (81.1s real max -> 90.0).

  max_steps:
    cuts_per_batch = batch_duration / mean_train_session_duration
    steps_per_epoch = n_train_sessions / cuts_per_batch
    max_steps = round(steps_per_epoch * max_epochs / accumulate_grad_batches)
    -- same formula used by hand for both prior baselines, generalized. Still
    an ESTIMATE, not a guarantee -- Lhotse's dynamic batching means the real
    cuts/batch varies. Watch NeMo's own reported steps/epoch early in
    training and re-generate if it's meaningfully different from what this
    script prints.

  warmup_steps:
    round(max_steps * warmup_ratio). Default ratio matches both prior
    baselines' own ~6.35% (400/6300 for the Portuguese-CML run) -- not
    independently re-tuned per dataset, just carried forward as a convention.

max_duration is computed from the OVERALL max across train+dev+test combined,
not train alone -- train happened to have the highest max in both prior
datasets, but that's not guaranteed, and all three dataloaders share the same
max_duration value in the template.

Usage:
    python generate_training_config.py \
        --template multitalker_finetune_overrides_template.yaml \
        --train_cuts Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz \
        --dev_cuts Data_Spanish/multitalker_train_data/dev_cuts.jsonl.gz \
        --test_cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
        --exp_name multitalker_es_cmltts_aws \
        --output conf/multitalker_finetune_overrides_aws_cmltts_es.yaml
"""
import argparse
import math
from pathlib import Path

from lhotse import CutSet

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--template", required=True, type=Path)
ap.add_argument("--train_cuts", required=True, type=Path)
ap.add_argument("--dev_cuts", required=True, type=Path)
ap.add_argument("--test_cuts", required=True, type=Path)
ap.add_argument("--exp_name", required=True,
                 help="Unique exp_manager.name for this run -- must not collide with any other "
                      "baseline's name, or checkpoints could get written into the same directory.")
ap.add_argument("--output", required=True, type=Path)
ap.add_argument("--batch_duration", type=float, default=90.0,
                 help="Must match train_ds.batch_duration in the template -- this script doesn't "
                      "read it back out of the template, so keep them in sync if you ever change it.")
ap.add_argument("--max_epochs", type=int, default=50,
                 help="Must match trainer.max_epochs in the template, same sync caveat as batch_duration.")
ap.add_argument("--accumulate_grad_batches", type=int, default=2,
                 help="Must match trainer.accumulate_grad_batches in the template, same caveat.")
ap.add_argument("--duration_headroom", type=float, default=1.1,
                 help="max_duration = ceil(observed_max * this / --round_to) * --round_to")
ap.add_argument("--round_to", type=float, default=10.0)
ap.add_argument("--warmup_ratio", type=float, default=400 / 6300,
                 help="warmup_steps = max_steps * this ratio. Default matches both prior baselines' "
                      "own ratio, not independently re-tuned.")
ap.add_argument("--resume_from_dir", default=None,
                 help="Only needed when resuming a run that was ALREADY interrupted before this "
                      "template's resume_if_exists=true was active -- such a run has a timestamped "
                      "version folder (e.g. multitalker_finetune_experiments/<name>/2026-08-20_02-02-26) "
                      "because resume_if_exists being off is what caused NeMo to create one. Pass that "
                      "exact directory here to set exp_manager.explicit_log_dir, pointing resumption "
                      "directly at it. Leave unset for a normal fresh run -- resume_if_exists is "
                      "already on by default in this template, so a fresh run's own eventual "
                      "interruption won't need this at all; NeMo will find its own checkpoints "
                      "automatically next time.")
ap.add_argument("--use_purity_weighted_targets", action="store_true",
                 help="Replace the hard {0,1} spk_target/bg_spk_target mask with a continuous "
                      "STNO-derived purity weight (see audio_to_text_lhotse_speaker.py). Off by "
                      "default, preserving exact original behavior -- this is a real architecture "
                      "change to what the model trains against, not a tuning knob to leave on by "
                      "habit. Applied consistently to train_ds/validation_ds/test_ds so a run can't "
                      "end up training with purity weighting but evaluating without it.")
ap.add_argument("--lambda_overlap_weight", type=float, default=0.5,
                 help="Only has any effect when --use_purity_weighted_targets is set. "
                      "speaker_target = P_T + lambda*P_O, bg_target = (1-lambda)*P_O + P_N. "
                      "0.5 is an arbitrary starting point, not a tuned value -- treat as a real "
                      "hyperparameter to sweep, same as any other.")
ap.add_argument("--use_cer", action="store_true",
                 help="Set NeMo's own internal use_cer config flag, switching the live, "
                      "during-training val_wer metric (and trainer.test()) from word-level to "
                      "character-level scoring. REQUIRED for CJK languages -- confirmed directly "
                      "that word-level scoring on unspaced text like Chinese treats an entire "
                      "utterance as one 'word', so any single wrong character reports the WHOLE "
                      "utterance as 100% wrong. Off by default, matching every existing "
                      "European-language config, where word-level IS the correct metric.")
ap.add_argument("--num_mel_frame_per_asr_frame", type=int, default=8,
                 help="MUST equal the pretrained backbone's actual encoder subsampling_factor -- "
                      "check via model.cfg.encoder.subsampling_factor before assuming the "
                      "default. Default (8) is correct for every FastConformer backbone used so "
                      "far (all European languages via Parakeet-TDT); confirmed WRONG (should be "
                      "4) for nvidia/stt_zh_conformer_transducer_large, a plain Conformer model "
                      "-- this single mismatch silently corrupted every Chinese training run's "
                      "mask supervision, since the mask ends up built at half the encoder's "
                      "real time resolution.")
args = ap.parse_args()


def main():
    expected_placeholders = ["__MAX_DURATION_FLOAT__", "__MAX_DURATION_INT__", "__MAX_STEPS__",
                              "__WARMUP_STEPS__", "__EXP_NAME__", "__RESUME_LOG_DIR__",
                              "__BATCH_DURATION__", "__USE_PURITY_WEIGHTED_TARGETS__",
                              "__LAMBDA_OVERLAP_WEIGHT__", "__USE_CER__",
                              "__NUM_MEL_FRAME_PER_ASR_FRAME__"]
    template_text = args.template.read_text(encoding="utf-8")
    missing_from_template = [tok for tok in expected_placeholders if tok not in template_text]
    if missing_from_template:
        raise SystemExit(
            f"ERROR: {args.template} is missing placeholder(s) {missing_from_template} -- this "
            f"looks like a stale copy of the template predating one or more of these fields. "
            f"Substituting into it would silently succeed while leaving the actual YAML value "
            f"unchanged from whatever's hardcoded in this file, rather than using your "
            f"--batch_duration (or other) argument. Get the current template before re-running."
        )

    print("Loading cutsets to measure the real duration distribution...")
    train_cuts = list(CutSet.from_file(args.train_cuts))
    dev_cuts = list(CutSet.from_file(args.dev_cuts))
    test_cuts = list(CutSet.from_file(args.test_cuts))

    all_durations = [c.duration for c in train_cuts + dev_cuts + test_cuts]
    train_durations = [c.duration for c in train_cuts]

    observed_max = max(all_durations)
    n_train = len(train_cuts)
    train_mean = sum(train_durations) / n_train

    print(f"  train: n={n_train}, mean={train_mean:.2f}s")
    print(f"  overall (train+dev+test): min={min(all_durations):.2f}s, max={observed_max:.2f}s")

    max_duration = math.ceil(observed_max * args.duration_headroom / args.round_to) * args.round_to
    cuts_per_batch = args.batch_duration / train_mean
    steps_per_epoch = n_train / cuts_per_batch
    max_steps = round(steps_per_epoch * args.max_epochs / args.accumulate_grad_batches)
    warmup_steps = round(max_steps * args.warmup_ratio)

    print(f"\nComputed values:")
    print(f"  max_duration / quadratic_duration: {max_duration}")
    print(f"  max_steps: {max_steps}  (~{steps_per_epoch:.0f} steps/epoch)")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  exp_manager.name: {args.exp_name}")
    resume_log_dir_value = "null" if args.resume_from_dir is None else args.resume_from_dir
    print(f"  exp_manager.explicit_log_dir: {resume_log_dir_value}")
    print(f"  batch_duration (all three data blocks): {args.batch_duration:g}")
    print(f"  use_purity_weighted_targets (all three data blocks): {args.use_purity_weighted_targets}")
    print(f"  lambda_overlap_weight (all three data blocks): {args.lambda_overlap_weight:g}")

    filled = (
        template_text
        .replace("__MAX_DURATION_FLOAT__", f"{max_duration:.1f}")
        .replace("__MAX_DURATION_INT__", f"{int(max_duration)}")
        .replace("__MAX_STEPS__", str(max_steps))
        .replace("__WARMUP_STEPS__", str(warmup_steps))
        .replace("__EXP_NAME__", args.exp_name)
        .replace("__RESUME_LOG_DIR__", resume_log_dir_value)
        .replace("__BATCH_DURATION__", f"{args.batch_duration:g}")
        .replace("__USE_PURITY_WEIGHTED_TARGETS__", "true" if args.use_purity_weighted_targets else "false")
        .replace("__LAMBDA_OVERLAP_WEIGHT__", f"{args.lambda_overlap_weight:g}")
        .replace("__USE_CER__", "true" if args.use_cer else "false")
        .replace("__NUM_MEL_FRAME_PER_ASR_FRAME__", str(args.num_mel_frame_per_asr_frame))
    )

    remaining = [tok for tok in expected_placeholders if tok in filled]
    if remaining:
        raise SystemExit(f"ERROR: placeholder(s) {remaining} still present after substitution -- "
                          f"this shouldn't be reachable given the upfront check already passed; "
                          f"please report this as a bug in the script.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(filled, encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
