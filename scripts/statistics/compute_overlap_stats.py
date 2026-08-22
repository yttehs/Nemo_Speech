#!/usr/bin/env python3
"""
Computes the same speech-overlap statistics FastMSS's own sim.py prints during
simulation (total speech/speaking-time/silence duration, single-speaker vs
overlapped speech, and a breakdown by number of concurrently active speakers) --
computed directly from a Lhotse CutSet's own supervisions, so it can be run on
ANY cutset (final combined training data, a specific split, Portuguese vs
Spanish, etc.) for direct comparison against FastMSS's own reported numbers or
against Tier 1/2/3 evaluation results.

Uses an exact event-sweep (not frame discretization) -- processes each session's
speaker-segment start/end times as events in chronological order, tracking how
many speakers are concurrently active between consecutive events, and accumulates
exact durations per active-speaker-count bucket. No discretization error.

Definitions match FastMSS's own convention:
    Total speech duration    = wall-clock time with >=1 speaker active (union,
                                not double-counted during overlap)
    Total speaking time      = SUM of every individual speaker segment's duration
                                (DOES double-count overlapped time -- this is why
                                speaking time can exceed 100% of recording duration
                                when overlap is heavy, same as FastMSS's own
                                >100% figures we saw during the Portuguese dry run)
    Silence duration         = recording duration - total speech duration
    Single-speaker duration  = wall-clock time with EXACTLY 1 speaker active
    Overlapped speech        = wall-clock time with >=2 speakers active
                              = total speech duration - single-speaker duration

Usage (one or more labeled cutsets, for a side-by-side comparison table):
    python compute_overlap_stats.py \
        --cuts Portuguese_train=Data/multitalker_train_data/train_cuts.jsonl.gz \
        --cuts Spanish_train=Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz
"""
import argparse
from pathlib import Path

from lhotse import CutSet

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cuts", required=True, action="append",
                 help="LABEL=path/to/cuts.jsonl.gz -- repeat this flag to compare multiple cutsets "
                      "side by side, e.g. --cuts Portuguese=... --cuts Spanish=...")
ap.add_argument("--max_speakers_table", type=int, default=6,
                 help="Report the by-speaker-count breakdown up to this many concurrent speakers; "
                      "anything above gets folded into a final N+ bucket rather than crashing.")
args = ap.parse_args()


def fmt_hhmmss(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_stats(cuts, max_speakers_table: int):
    """Event-sweep over every cut's supervisions. Returns a dict of aggregate stats."""
    total_recording_duration = 0.0
    total_speaking_time = 0.0  # sum of individual segment durations, double-counts overlap
    duration_by_n_speakers = [0.0] * (max_speakers_table + 2)  # index = n_speakers active

    for cut in cuts:
        total_recording_duration += cut.duration
        events = []  # (time, +1 for segment start, -1 for segment end)
        for sup in cut.supervisions:
            if sup.duration <= 0:
                continue
            events.append((sup.start, 1))
            events.append((sup.start + sup.duration, -1))
            total_speaking_time += sup.duration

        if not events:
            continue
        events.sort()

        active = 0
        prev_time = events[0][0]
        for time, delta in events:
            if time > prev_time and active > 0:
                seg_dur = time - prev_time
                bucket = min(active, max_speakers_table + 1)
                duration_by_n_speakers[bucket] += seg_dur
            active += delta
            prev_time = time

    total_speech_duration = sum(duration_by_n_speakers)
    single_speaker_duration = duration_by_n_speakers[1] if len(duration_by_n_speakers) > 1 else 0.0
    overlapped_duration = total_speech_duration - single_speaker_duration
    silence_duration = total_recording_duration - total_speech_duration

    return {
        "total_recording_duration": total_recording_duration,
        "total_speech_duration": total_speech_duration,
        "total_speaking_time": total_speaking_time,
        "silence_duration": silence_duration,
        "single_speaker_duration": single_speaker_duration,
        "overlapped_duration": overlapped_duration,
        "duration_by_n_speakers": duration_by_n_speakers,
    }


def print_full_breakdown(label: str, stats: dict, max_speakers_table: int):
    rec = stats["total_recording_duration"]
    speech = stats["total_speech_duration"]
    speaking = stats["total_speaking_time"]
    silence = stats["silence_duration"]
    single = stats["single_speaker_duration"]
    overlap = stats["overlapped_duration"]

    print(f"\n=== {label} ===")
    print(f"Total recording duration: {fmt_hhmmss(rec)} ({rec:.1f}s)")
    print(f"{'Total speech duration':30} {fmt_hhmmss(speech):>10}  {100*speech/rec:6.2f}% of recording")
    print(f"{'Total speaking time':30} {fmt_hhmmss(speaking):>10}  {100*speaking/rec:6.2f}% of recording")
    print(f"{'Total silence duration':30} {fmt_hhmmss(silence):>10}  {100*silence/rec:6.2f}% of recording")
    if speech > 0:
        print(f"{'Single-speaker duration':30} {fmt_hhmmss(single):>10}  {100*single/rec:6.2f}% "
              f"({100*single/speech:.2f}% of speech)")
        print(f"{'Overlapped speech duration':30} {fmt_hhmmss(overlap):>10}  {100*overlap/rec:6.2f}% "
              f"({100*overlap/speech:.2f}% of speech)")

    print(f"\n  {'N speakers':12} {'Duration':>10} {'Speaking time':>14} {'% of speech':>12} {'% of speaking time':>20}")
    for n in range(1, max_speakers_table + 2):
        d = stats["duration_by_n_speakers"][n]
        if d <= 0:
            continue
        label_n = f"{n}" if n <= max_speakers_table else f"{n}+"
        speaking_n = d * n
        pct_speech = 100 * d / speech if speech > 0 else 0.0
        pct_speaking = 100 * speaking_n / speaking if speaking > 0 else 0.0
        print(f"  {label_n:12} {fmt_hhmmss(d):>10} {fmt_hhmmss(speaking_n):>14} {pct_speech:11.2f}% {pct_speaking:19.2f}%")


def main():
    parsed = []
    for entry in args.cuts:
        if "=" not in entry:
            raise SystemExit(f"--cuts entries must be LABEL=path, got: {entry!r}")
        label, path_str = entry.split("=", 1)
        parsed.append((label, Path(path_str)))

    all_stats = {}
    for label, path in parsed:
        print(f"Loading {label} from {path}...")
        cuts = list(CutSet.from_file(path))
        stats = compute_stats(cuts, args.max_speakers_table)
        all_stats[label] = stats
        print_full_breakdown(label, stats, args.max_speakers_table)

    if len(all_stats) > 1:
        print(f"\n\n=== Comparison summary ===")
        print(f"{'Dataset':20} {'Rec. dur.':>10} {'% overlap of speech':>20} {'% speaking of rec.':>20}")
        for label, stats in all_stats.items():
            rec = stats["total_recording_duration"]
            speech = stats["total_speech_duration"]
            speaking = stats["total_speaking_time"]
            overlap = stats["overlapped_duration"]
            pct_overlap = 100 * overlap / speech if speech > 0 else 0.0
            pct_speaking_of_rec = 100 * speaking / rec if rec > 0 else 0.0
            print(f"{label:20} {fmt_hhmmss(rec):>10} {pct_overlap:19.2f}% {pct_speaking_of_rec:19.2f}%")


if __name__ == "__main__":
    main()
