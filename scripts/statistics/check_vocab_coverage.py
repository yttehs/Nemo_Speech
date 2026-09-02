#!/usr/bin/env python3
"""
Checks whether a character-based pretrained model's fixed vocabulary (cfg.labels)
actually covers the characters that appear in a given training cutset's real text.
Any character used in training but NOT in the model's vocabulary is structurally
impossible for the model to ever output correctly, regardless of how much training
happens -- this creates a hard error floor, not a slow-convergence problem.

Usage:
    python check_vocab_coverage.py \
        --pretrained_model nvidia/stt_zh_conformer_transducer_large \
        --cuts Data_alimeeting/lhotse_cuts/train_cuts.jsonl.gz
"""
import argparse
from collections import Counter

from lhotse import CutSet
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pretrained_model", required=True)
ap.add_argument("--cuts", required=True)
ap.add_argument("--top_n", type=int, default=40)
args = ap.parse_args()


def main():
    print(f"Loading {args.pretrained_model} ...")
    model = EncDecRNNTModel.from_pretrained(args.pretrained_model, map_location="cpu")

    labels = model.cfg.get("labels", None)
    if labels is None:
        raise SystemExit("This model's config has no 'labels' field -- it may not actually "
                          "be a character-based model, or labels live somewhere unexpected "
                          "in its config. Check model.cfg directly to see its real structure.")
    label_set = set(labels)
    print(f"Model vocabulary: {len(label_set)} characters/labels")

    print(f"Loading cuts from {args.cuts} ...")
    cuts = CutSet.from_file(args.cuts)

    char_counts = Counter()
    n_supervisions = 0
    for cut in cuts:
        for sup in cut.supervisions:
            n_supervisions += 1
            char_counts.update(sup.text)

    total_chars_in_text = sum(char_counts.values())
    unique_chars_in_text = set(char_counts.keys())
    missing = unique_chars_in_text - label_set

    n_missing_tokens = sum(char_counts[c] for c in missing)

    print(f"\n{n_supervisions} supervisions checked, {total_chars_in_text} total characters, "
          f"{len(unique_chars_in_text)} unique characters used")
    print(f"{len(missing)} unique character(s) NOT in the model's vocabulary "
          f"({100 * len(missing) / len(unique_chars_in_text):.1f}% of unique characters used)")
    print(f"{n_missing_tokens} total character INSTANCES are out-of-vocabulary "
          f"({100 * n_missing_tokens / total_chars_in_text:.2f}% of all characters in the "
          f"training text)")

    if missing:
        print(f"\nTop {min(args.top_n, len(missing))} most frequent missing characters:")
        missing_sorted = sorted(missing, key=lambda c: -char_counts[c])
        for c in missing_sorted[:args.top_n]:
            print(f"  {c!r}: {char_counts[c]} occurrences")


if __name__ == "__main__":
    main()
