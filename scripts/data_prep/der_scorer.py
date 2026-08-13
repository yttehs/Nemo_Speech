#!/usr/bin/env python3
"""
Diarization Error Rate (DER) scorer -- compares a hypothesis RTTM (e.g.
Sortformer's predicted output) against a reference RTTM (ground truth),
with no external dependencies beyond the standard library.

DER = (missed_speech + false_alarm + speaker_confusion) / total_reference_speech_time

Approach: discretize time into small frames, determine which reference and
hypothesis speakers are active in each frame, find the reference<->hypothesis
speaker mapping that maximizes total overlap (exhaustive permutation search
-- fine given at most 4 speakers per session), then classify every frame of
reference speech as correctly attributed, confused, or missed, and every
frame of non-reference-speech hypothesis activity as false alarm.

A collar (default 0.25s) is excluded around every reference speaker-turn
boundary, since exact boundary-timing disagreement is a much less meaningful
error than misattributing a whole segment -- this matches standard practice
(NVIDIA's own CALLHOME evaluation uses the same collar width).

This is a solid diagnostic scorer, not a drop-in replacement for the NIST
reference implementation (md-eval.pl) -- good enough to judge "is this a
serious problem," not for publication-exact numbers.

Usage as a library:
    from der_scorer import score_der
    result = score_der("ref.rttm", "hyp.rttm", collar=0.25)
    print(result["der"])
"""
import itertools
import math
from pathlib import Path


def parse_rttm(path):
    """Returns list of (start, end, speaker) tuples."""
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start = float(fields[3])
            dur = float(fields[4])
            speaker = fields[7]
            segments.append((start, start + dur, speaker))
    return segments


def _build_collar_mask(ref_segments, total_duration, collar, frame_size):
    """True at frames within `collar` seconds of any reference turn boundary."""
    n_frames = int(round(total_duration / frame_size)) + 1
    mask = [False] * n_frames
    boundaries = set()
    for start, end, _ in ref_segments:
        boundaries.add(start)
        boundaries.add(end)
    for b in boundaries:
        # Use the same round()-based convention as _frame_speaker_sets, with
        # no "+1" padding -- when collar=0 this must exclude zero frames,
        # not a minimum of one frame per boundary.
        lo = max(0, int(round((b - collar) / frame_size)))
        hi = min(n_frames, int(round((b + collar) / frame_size)))
        for i in range(lo, hi):
            mask[i] = True
    return mask


def _frame_speaker_sets(segments, n_frames, frame_size):
    """List (length n_frames) of sets of speakers active in that frame."""
    frames = [set() for _ in range(n_frames)]
    for start, end, speaker in segments:
        # round rather than truncate/ceil -- frame_size like 0.01 isn't exactly
        # representable in binary float, so division can land at e.g.
        # 499.9999999997 instead of exactly 500; rounding is robust to that
        # noise, truncation/ceiling are not.
        lo = max(0, int(round(start / frame_size)))
        hi = min(n_frames, int(round(end / frame_size)))
        for i in range(lo, hi):
            frames[i].add(speaker)
    return frames


def score_der(ref_rttm_path, hyp_rttm_path, collar=0.25, frame_size=0.01):
    ref_segments = parse_rttm(ref_rttm_path)
    hyp_segments = parse_rttm(hyp_rttm_path)

    if not ref_segments:
        return {"der": None, "reason": "empty reference RTTM"}

    total_duration = max(
        max(e for _, e, _ in ref_segments), max((e for _, e, _ in hyp_segments), default=0.0)
    )
    n_frames = int(round(total_duration / frame_size)) + 1

    ref_frames = _frame_speaker_sets(ref_segments, n_frames, frame_size)
    hyp_frames = _frame_speaker_sets(hyp_segments, n_frames, frame_size)
    collar_mask = _build_collar_mask(ref_segments, total_duration, collar, frame_size)

    ref_speakers = sorted({spk for _, _, spk in ref_segments})
    hyp_speakers = sorted({spk for _, _, spk in hyp_segments})

    # Optimal ref<->hyp speaker mapping: maximize total overlapping frame-time.
    # Exhaustive permutation search -- fine for <=4 speakers per side.
    overlap_count = {(r, h): 0 for r in ref_speakers for h in hyp_speakers}
    for i in range(n_frames):
        if collar_mask[i]:
            continue
        for r in ref_frames[i]:
            for h in hyp_frames[i]:
                overlap_count[(r, h)] += 1

    best_mapping = {}
    if ref_speakers and hyp_speakers:
        n = min(len(ref_speakers), len(hyp_speakers))
        best_score = -1
        # Try every way to pick n hyp speakers (in order) to pair with the first n ref speakers
        for hyp_subset in itertools.permutations(hyp_speakers, n):
            score = sum(overlap_count[(ref_speakers[i], hyp_subset[i])] for i in range(n))
            if score > best_score:
                best_score = score
                best_mapping = {ref_speakers[i]: hyp_subset[i] for i in range(n)}

    missed = false_alarm = confusion = correct = 0
    total_ref_time = 0
    for i in range(n_frames):
        if collar_mask[i]:
            continue
        r_set, h_set = ref_frames[i], hyp_frames[i]
        total_ref_time += len(r_set)  # counts each simultaneously-active ref speaker separately

        for r in r_set:
            mapped_h = best_mapping.get(r)
            if mapped_h is not None and mapped_h in h_set:
                correct += 1
            elif h_set:
                confusion += 1  # ref speaker active, but not correctly matched to any hyp speaker present
            else:
                missed += 1  # ref speaker active, hyp predicts total silence here

        mapped_hyp_values = set(best_mapping.values())
        if not r_set:
            false_alarm += len(h_set)  # any hyp speech with zero reference speech present is false alarm

    total_ref_time *= frame_size
    missed *= frame_size
    false_alarm *= frame_size
    confusion *= frame_size
    correct *= frame_size

    der = (missed + false_alarm + confusion) / total_ref_time if total_ref_time > 0 else None

    return {
        "der": der,
        "missed": missed,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "correct": correct,
        "total_ref_time": total_ref_time,
        "speaker_mapping": best_mapping,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score a single hypothesis RTTM against a reference RTTM")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--collar", type=float, default=0.25)
    args = ap.parse_args()

    result = score_der(args.ref, args.hyp, collar=args.collar)
    for k, v in result.items():
        print(f"{k}: {v}")
