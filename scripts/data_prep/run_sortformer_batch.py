#!/usr/bin/env python3
"""
Run Sortformer diarization over every .wav in a pooled session directory
(e.g. test_pooled/) and write its predicted RTTM into a separate output
directory -- never overwrites the ground-truth RTTM sitting next to the wav.

This is step 1 of measuring Sortformer's real Portuguese performance before
committing to a training-data strategy: run this first, then score the
output against ground truth with der_scorer.py.

Usage:
    python run_sortformer_batch.py \
        --wav_dir /path/to/test_pooled \
        --output_dir /path/to/test_pooled_sortformer_preds \
        --model_name nvidia/diar_streaming_sortformer_4spk-v2.1
"""
import argparse
from pathlib import Path

import torch
from nemo.collections.asr.models import SortformerEncLabelModel

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--wav_dir", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path)
ap.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
ap.add_argument("--streaming", action="store_true", default=True,
                 help="Set the streaming cache params (chunk_len etc). Use --no-streaming "
                      "for the offline checkpoint, which has no cache/warm-up regime at all.")
ap.add_argument("--no-streaming", dest="streaming", action="store_false")
ap.add_argument("--limit", type=int, default=None,
                 help="Only process this many files (for a quick first look before running everything)")
args = ap.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading {args.model_name} ...")
diar_model = SortformerEncLabelModel.from_pretrained(args.model_name)
diar_model.eval()
if torch.cuda.is_available():
    diar_model = diar_model.cuda()
    print("Running on GPU")
else:
    print("Running on CPU")

if args.streaming:
    # Streaming cache params, matching NVIDIA's own documented usage example.
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300
else:
    print("Streaming cache params NOT set -- expecting the offline checkpoint, "
          "which processes each clip in one full-attention pass with no cache.")


def parse_segment(seg):
    """Normalize diarize()'s per-segment output (string or tuple) into (start, end, speaker)."""
    parts = seg.strip().split() if isinstance(seg, str) else list(seg)
    start, end = float(parts[0]), float(parts[1])
    speaker = str(parts[2])
    if speaker.isdigit():
        speaker = f"speaker_{speaker}"
    return start, end, speaker


wav_files = sorted(args.wav_dir.glob("*.wav"))
if args.limit:
    wav_files = wav_files[: args.limit]

print(f"Found {len(wav_files)} wav files to process")

n_done, n_failed = 0, 0
for wav_path in wav_files:
    uri = wav_path.stem
    out_rttm = args.output_dir / f"{uri}.rttm"
    if out_rttm.exists():
        continue  # allows safely resuming an interrupted batch run

    try:
        predicted = diar_model.diarize(audio=[str(wav_path)], batch_size=1)
        segments_raw = predicted[0]
        segments = sorted((parse_segment(s) for s in segments_raw), key=lambda s: s[0])

        with out_rttm.open("w", encoding="utf-8") as f:
            for start, end, speaker in segments:
                dur = end - start
                f.write(f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
        n_done += 1
    except Exception as e:
        print(f"  FAILED on {wav_path.name}: {e!r}")
        n_failed += 1

    if (n_done + n_failed) % 20 == 0:
        print(f"  ... {n_done + n_failed}/{len(wav_files)} processed")

print(f"\nDone. {n_done} predicted RTTMs written to {args.output_dir}, {n_failed} failed.")
