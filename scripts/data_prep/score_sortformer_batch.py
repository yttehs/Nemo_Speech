#!/usr/bin/env python3
"""
Score every predicted RTTM (from run_sortformer_batch.py) against its
matching ground-truth RTTM (from pool_sessions.py), reporting both an
overall DER and a breakdown by speaker-count bucket -- since "does this get
worse with more speakers" is exactly the question worth answering before
deciding whether Sortformer needs attention before the ASR fine-tuning
phase.

Usage:
    python score_sortformer_batch.py \
        --ref_dir /path/to/test_pooled \
        --pred_dir /path/to/test_pooled_sortformer_preds \
        --collar 0.25
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

from der_scorer import score_der

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ref_dir", required=True, type=Path)
ap.add_argument("--pred_dir", required=True, type=Path)
ap.add_argument("--collar", type=float, default=0.25)
args = ap.parse_args()

pred_rttms = sorted(args.pred_dir.glob("*.rttm"))
print(f"Found {len(pred_rttms)} predicted RTTMs to score")

per_bucket = defaultdict(lambda: {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total_ref_time": 0.0})
overall = {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total_ref_time": 0.0}
n_scored, n_skipped = 0, 0

for pred_path in pred_rttms:
    ref_path = args.ref_dir / pred_path.name
    if not ref_path.exists():
        print(f"  SKIP (no matching ground-truth RTTM): {pred_path.name}")
        n_skipped += 1
        continue

    result = score_der(ref_path, pred_path, collar=args.collar)
    if result["der"] is None:
        n_skipped += 1
        continue

    m = re.search(r"(\d+)spk", pred_path.stem)
    bucket = f"{m.group(1)}spk" if m else "unknown"

    for key in ("missed", "false_alarm", "confusion", "total_ref_time"):
        per_bucket[bucket][key] += result[key]
        overall[key] += result[key]
    n_scored += 1

print(f"\nScored {n_scored} sessions ({n_skipped} skipped)\n")

print(f"{'bucket':<10s} {'DER':>8s} {'miss':>8s} {'false_alarm':>12s} {'confusion':>10s} {'ref_time_s':>11s}")
for bucket in sorted(per_bucket.keys()):
    b = per_bucket[bucket]
    der = (b["missed"] + b["false_alarm"] + b["confusion"]) / b["total_ref_time"] if b["total_ref_time"] > 0 else float("nan")
    print(f"{bucket:<10s} {der*100:>7.2f}% {b['missed']:>8.1f} {b['false_alarm']:>12.1f} {b['confusion']:>10.1f} {b['total_ref_time']:>11.1f}")

overall_der = (
    (overall["missed"] + overall["false_alarm"] + overall["confusion"]) / overall["total_ref_time"]
    if overall["total_ref_time"] > 0
    else float("nan")
)
print(f"\n{'OVERALL':<10s} {overall_der*100:>7.2f}%")
