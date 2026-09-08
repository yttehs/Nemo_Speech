#!/usr/bin/env python3
"""
Creates a duration-targeted subset of a cutset, stratified by each chunk's maximum
simultaneous speaker count (1-4), rather than a plain random sample.

Classification is by the PEAK overlap any chunk reaches at any instant -- the same
event-sweep concept compute_overlap_stats.py uses -- not by how many participants were
in the originating meeting, and not by whichever speaker count happens to dominate the
chunk's total duration. A chunk that's mostly solo speech but briefly hits 4-way
overlap counts as a "4-speaker" chunk here: the goal is guaranteeing real examples of
each overlap regime exist in the subset, not modeling the corpus's natural (heavily
1-speaker-skewed) distribution.

Default proportions are EQUAL across all four buckets (25% each), a deliberate choice,
not a neutral default: AliMeeting's natural distribution is roughly 75%/20%/5%/0.6% by
speaking time (confirmed via compute_overlap_stats.py on the full training set), so a
plain random sample at any reasonable subset size would contain only minutes of true
4-way overlap. Override --proportions if the natural skew is actually what's wanted.

Usage:
    python create_alimeeting_subset.py \
        --cuts Data_alimeeting/lhotse_cuts/train_cuts.jsonl.gz \
        --output Data_alimeeting/lhotse_cuts/train_cuts_subset20h.jsonl.gz \
        --target_hours 20.0
"""
import argparse
import random

from lhotse import CutSet

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cuts", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--target_hours", type=float, required=True)
ap.add_argument("--proportions", default="0.25,0.25,0.25,0.25",
                 help="Comma-separated target share of --target_hours for the 1/2/3/4-speaker "
                      "buckets respectively, in that order. Must sum to 1.0. Default is equal "
                      "across all four -- see module docstring for why that's the deliberate "
                      "choice, not the corpus's natural distribution.")
ap.add_argument("--seed", type=int, default=42)


def max_simultaneous_speakers(cut):
    """Event-sweep over one cut's supervisions: +1 at each segment start, -1 at each
    end, sorted by time (ends before starts on ties, so a segment ending exactly when
    another begins doesn't get counted as briefly overlapping). Returns the highest
    running count reached. Same underlying concept as compute_overlap_stats.py, but
    per-cut classification rather than corpus-wide aggregate statistics."""
    events = []
    for sup in cut.supervisions:
        start = sup.start
        end = sup.start + sup.duration
        events.append((start, 1))
        events.append((end, -1))
    # -1 (end) sorts before +1 (start) at the same timestamp
    events.sort(key=lambda e: (e[0], e[1]))

    running = 0
    peak = 0
    for _time, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


def main():
    args = ap.parse_args()
    rng = random.Random(args.seed)

    proportions = [float(p) for p in args.proportions.split(",")]
    if len(proportions) != 4:
        raise SystemExit(f"--proportions must have exactly 4 values, got {len(proportions)}")
    if abs(sum(proportions) - 1.0) > 1e-6:
        raise SystemExit(f"--proportions must sum to 1.0, got {sum(proportions)}")

    print(f"Loading cuts from {args.cuts} ...")
    cuts = CutSet.from_file(args.cuts)

    buckets = {1: [], 2: [], 3: [], 4: []}
    n_skipped_empty = 0
    n_skipped_over4 = 0
    for cut in cuts:
        peak = max_simultaneous_speakers(cut)
        if peak == 0:
            n_skipped_empty += 1
            continue
        if peak > 4:
            # Defensive: AliMeeting sessions have at most 4 participants, so this
            # shouldn't happen, but cap rather than silently mis-bucket if it does.
            n_skipped_over4 += 1
            continue
        buckets[peak].append(cut)

    if n_skipped_empty:
        print(f"WARNING: {n_skipped_empty} cut(s) had no supervisions at all, skipped")
    if n_skipped_over4:
        print(f"WARNING: {n_skipped_over4} cut(s) had peak overlap >4 speakers "
              f"(unexpected for AliMeeting), skipped")

    print("\nAvailable duration per bucket (peak simultaneous speakers):")
    available_hours = {}
    for n_spk in (1, 2, 3, 4):
        rng.shuffle(buckets[n_spk])  # shuffle now so later sequential-fill is a random sample
        total_seconds = sum(c.duration for c in buckets[n_spk])
        available_hours[n_spk] = total_seconds / 3600
        print(f"  {n_spk}-speaker: {len(buckets[n_spk])} chunks, {available_hours[n_spk]:.2f}h available")

    print(f"\nTarget: {args.target_hours:.2f}h total, proportions {proportions}")
    selected = []
    for n_spk, proportion in zip((1, 2, 3, 4), proportions):
        target_seconds = args.target_hours * 3600 * proportion
        picked = []
        accumulated = 0.0
        for cut in buckets[n_spk]:
            if accumulated >= target_seconds:
                break
            picked.append(cut)
            accumulated += cut.duration
        selected.extend(picked)
        target_h = target_seconds / 3600
        got_h = accumulated / 3600
        shortfall_note = ""
        if got_h < target_h - 1e-6:
            shortfall_note = f"  <-- SHORTFALL: only {available_hours[n_spk]:.2f}h available in this bucket at all"
        print(f"  {n_spk}-speaker: wanted {target_h:.2f}h, got {got_h:.2f}h "
              f"({len(picked)} chunks){shortfall_note}")

    rng.shuffle(selected)  # mix buckets together -- don't leave training batches
                            # clustered by speaker count in sequence
    total_selected_hours = sum(c.duration for c in selected) / 3600
    print(f"\nTotal selected: {len(selected)} chunks, {total_selected_hours:.2f}h "
          f"(target was {args.target_hours:.2f}h)")

    CutSet.from_cuts(selected).to_file(args.output)
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
