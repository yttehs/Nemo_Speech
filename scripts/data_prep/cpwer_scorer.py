#!/usr/bin/env python3
"""
cpWER (concatenated minimum-permutation Word Error Rate) for multi-speaker
transcription evaluation, where hypothesis speaker labels have no known
correspondence to reference speaker labels (e.g. "linked_speaker_3" from
embedding-based cross-chunk clustering vs "Speaker7" in the ground truth).

Reference JSON format: a flat list of {start, end, speaker, text} dicts (confirmed
directly against real Clipto_Data samples -- not wrapped in any outer object, unlike
pt_avprep's {"video_id":..., "attribution": [...]} format).

Speaker mapping uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment),
an exact, polynomial-time solution to the optimal-assignment problem -- not brute-force
permutations. Matters concretely here: real files in this corpus have up to 10
speakers, and a naive itertools.permutations approach would need up to 10! (~3.6M)
WER computations, growing further still if hypothesis and reference speaker counts
differ (e.g. imperfect clustering splitting one real speaker into two IDs).

Mismatched speaker counts are handled by padding the smaller side with empty-string
dummy speakers before building the cost matrix -- an unmatched reference speaker's
full text becomes pure deletions (a real speaker the pipeline missed entirely); an
unmatched hypothesis speaker's full text becomes pure insertions (a hallucinated or
duplicate speaker identity), with zero contribution to the reference-word denominator,
matching standard cpWER convention.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from wer_scorer import word_error_rate


def parse_reference_json(path: Path):
    """Returns a flat list of {start, end, speaker, text} dicts, as-is from the file."""
    return json.loads(path.read_text(encoding="utf-8"))


def concatenate_by_speaker(segments):
    """Groups segments by speaker, sorts each speaker's own segments by start time,
    joins their text with spaces. Returns {speaker_label: concatenated_text}."""
    by_speaker = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker"], []).append(seg)
    result = {}
    for speaker, segs in by_speaker.items():
        segs_sorted = sorted(segs, key=lambda s: s["start"])
        result[speaker] = " ".join(s["text"] for s in segs_sorted)
    return result


def compute_cpwer(ref_by_speaker: dict, hyp_by_speaker: dict, char_level: bool = False):
    """
    ref_by_speaker / hyp_by_speaker: {speaker_label: concatenated_text}

    Returns a dict: {
        "cpwer": float,
        "total_errors": int, "total_ref_words": int,
        "mapping": [(ref_speaker, hyp_speaker_or_None, n_sub, n_del, n_ins, n_ref), ...],
    }
    mapping entries with hyp_speaker=None are reference speakers matched to a padded
    dummy (i.e. no real hypothesis speaker was assigned to them -- fully deleted).
    """
    ref_speakers = list(ref_by_speaker.keys())
    hyp_speakers = list(hyp_by_speaker.keys())
    n_ref, n_hyp = len(ref_speakers), len(hyp_speakers)
    n = max(n_ref, n_hyp)

    # Pad the smaller side with empty-string dummy speakers so the cost matrix is square.
    padded_ref_texts = [ref_by_speaker[s] for s in ref_speakers] + [""] * (n - n_ref)
    padded_hyp_texts = [hyp_by_speaker[s] for s in hyp_speakers] + [""] * (n - n_hyp)

    cost = np.zeros((n, n), dtype=np.int64)
    details = {}  # (i, j) -> (n_sub, n_del, n_ins, n_ref_words)
    for i in range(n):
        for j in range(n):
            _wer, n_sub, n_del, n_ins, n_ref_words = word_error_rate(
                padded_ref_texts[i], padded_hyp_texts[j], char_level=char_level
            )
            cost[i, j] = n_sub + n_del + n_ins
            details[(i, j)] = (n_sub, n_del, n_ins, n_ref_words)

    row_ind, col_ind = linear_sum_assignment(cost)

    total_errors = 0
    total_ref_words = 0
    mapping = []
    for i, j in zip(row_ind, col_ind):
        n_sub, n_del, n_ins, n_ref_words = details[(i, j)]
        total_errors += n_sub + n_del + n_ins
        total_ref_words += n_ref_words
        ref_label = ref_speakers[i] if i < n_ref else None
        hyp_label = hyp_speakers[j] if j < n_hyp else None
        # Include EVERY pair, even ref_label=None (an extra/hallucinated hypothesis
        # speaker matched to a padded dummy reference slot) -- these still contribute
        # real insertion errors to total_errors above, so excluding them from the
        # returned mapping would make that part of the total invisible to anyone
        # inspecting per-pair results to understand where errors came from.
        mapping.append((ref_label, hyp_label, n_sub, n_del, n_ins, n_ref_words))

    cpwer = total_errors / total_ref_words if total_ref_words > 0 else float("nan")
    return {
        "cpwer": cpwer,
        "total_errors": total_errors,
        "total_ref_words": total_ref_words,
        "mapping": mapping,
    }


if __name__ == "__main__":
    import sys
    segments = parse_reference_json(Path(sys.argv[1]))
    by_speaker = concatenate_by_speaker(segments)
    print(f"{len(segments)} segments, {len(by_speaker)} speaker(s)")
    for speaker, text in by_speaker.items():
        print(f"  {speaker}: {len(text.split())} words, {text[:60]!r}...")
