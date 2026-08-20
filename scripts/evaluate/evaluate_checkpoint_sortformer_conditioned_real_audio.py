#!/usr/bin/env python3
"""
Tier 3: transcribe a REAL (non-simulated) audio file, conditioned on Sortformer's
own predicted speaker activity -- the truest representation of deployment, since
it sidesteps the simulated-audio splicing artifacts Sortformer struggles with (see
the earlier DER investigation). No ground truth exists for real audio, so this is
qualitative (read/listen and judge), not a WER number.

Run run_sortformer_batch.py on your real audio FIRST to get its predicted RTTM,
then point this script at both.

Usage:
    python evaluate_checkpoint_sortformer_conditioned_real_audio.py \
        --overrides conf/multitalker_finetune_overrides_aws.yaml \
        --checkpoint ".../multitalker_pt_adapter_aws--val_wer=0.6957-epoch=46.ckpt" \
        --wav_path /path/to/real_conversation.wav \
        --sortformer_rttm /path/to/real_conversation.rttm
"""
import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf

from scripts.data_prep.multispeaker_inference import (
    ENCODER_FRAME_SECONDS,
    arrival_order,
    build_mask_for_speaker,
    load_model,
    parse_rttm,
    transcribe_with_mask,
)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--overrides", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--wav_path", required=True)
ap.add_argument("--sortformer_rttm", required=True)
ap.add_argument("--pretrained_model", default="nvidia/parakeet-tdt-0.6b-v3")
ap.add_argument("--dummy_cuts_path", required=True,
                 help="Any real, valid *_cuts.jsonl.gz file from training (e.g. test_cuts.jsonl.gz) -- "
                      "needed only because NeMo validates train_ds/validation_ds/test_ds as real "
                      "manifests at model construction time, even though this script never uses them.")
ap.add_argument("--output_json", default=None,
                 help="Where to write the per-segment transcript JSON (speaker + start/end + text, "
                      "one entry per RTTM segment), consumable by diarization_review.html. "
                      "Defaults to <wav_path stem>_transcript.json next to --wav_path.")
args = ap.parse_args()


def ensure_16k_mono(wav_path):
    """
    build_mask_for_speaker() and transcribe_with_mask() both implicitly assume 16kHz
    audio -- speaker_to_target()'s frame rate is derived from the file's OWN sampling
    rate (a_cut.sampling_rate / num_sample_per_mel_frame), while the model's own
    preprocessor always expects 16kHz. Simulated training/Tier-1/Tier-2 audio is
    uniformly 16kHz already, so this mismatch never surfaced there. Real recordings
    (e.g. podcast/conversation clips) are frequently 44.1kHz/48kHz, and feeding that
    straight through would misalign the speaker-activity mask against what the model
    actually consumes -- silently, since solve_length_mismatch() pads/truncates
    rather than erroring. Resample once up front and use the resampled path
    everywhere downstream, rather than patching sample-rate handling in two places.

    Returns the original path unchanged if it's already 16kHz mono; otherwise writes
    a resampled temp copy and returns that path instead.
    """
    info = sf.info(wav_path)
    if info.samplerate == 16000 and info.channels == 1:
        return wav_path

    print(f"NOTE: {wav_path} is {info.samplerate}Hz/{info.channels}ch, not 16kHz mono -- "
          f"resampling to a temp copy before running (mask frame-rate math assumes 16kHz).")
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    tmp_path = Path(tempfile.mkdtemp(prefix="tier3_resampled_")) / (Path(wav_path).stem + "_16k.wav")
    sf.write(tmp_path, audio, 16000, subtype="PCM_16")
    return str(tmp_path)


def bucket_words_into_segments(word_entries, speaker_segments):
    """
    Assigns each decoded word (with its own start/end time from the RNNT decoder's
    word-level timestamps) to whichever ONE of this speaker's own RTTM segments it
    best belongs to, then joins per-segment -- reconstructing which words landed in
    which turn, for a given speaker's already-generated full-session transcript.

    speaker_segments: list of (start, end) tuples for ONE speaker, same segments
    used to build that speaker's mask (so segment count/order already matches what
    conditioned the transcription -- this only re-attributes words after the fact).

    Overlap-first: a word goes wherever it overlaps most with a segment. Words the
    decoder emitted outside any of this speaker's segments (turn-taking slack --
    the model's own output timing isn't guaranteed to land exactly inside RTTM
    boundaries) fall back to the nearest segment by time, so nothing silently drops.
    Returns {segment_index: joined_text}.
    """
    buckets = defaultdict(list)
    for w in word_entries:
        w_start, w_end = w["start"], w["end"]
        best_idx, best_overlap = None, 0.0
        for i, (s_start, s_end) in enumerate(speaker_segments):
            overlap = min(w_end, s_end) - max(w_start, s_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx is None:
            best_idx = min(
                range(len(speaker_segments)),
                key=lambda i: min(abs(w_start - speaker_segments[i][0]), abs(w_start - speaker_segments[i][1])),
            )
        buckets[best_idx].append(w["word"])
    return {i: " ".join(words) for i, words in buckets.items()}


def main():
    print("Loading model...")
    model = load_model(args.overrides, args.checkpoint, args.pretrained_model,
                        dummy_cuts_path=args.dummy_cuts_path)

    wav_path = ensure_16k_mono(args.wav_path)

    segments = parse_rttm(args.sortformer_rttm)
    speakers = arrival_order(segments)
    print(f"\nSortformer detected {len(speakers)} speaker(s): {speakers}\n")

    output_segments = []

    for spk in speakers:
        try:
            spk_target, bg_spk_target = build_mask_for_speaker(wav_path, segments, target_speaker=spk)
        except ValueError as e:
            print(f"--- {spk}: skipped ({e}) ---\n")
            continue
        total_active_s = spk_target.sum().item() * ENCODER_FRAME_SECONDS
        text, word_entries = transcribe_with_mask(
            model, wav_path, spk_target, bg_spk_target, return_word_timestamps=True
        )
        print(f"--- {spk} (~{total_active_s:.1f}s active) ---")
        print(f"{text}\n")

        speaker_segments = sorted((start, end) for start, end, s in segments if s == spk)
        text_by_idx = bucket_words_into_segments(word_entries, speaker_segments)
        for i, (start, end) in enumerate(speaker_segments):
            output_segments.append({
                "speaker": spk,
                "start": start,
                "end": end,
                "text": text_by_idx.get(i, ""),
            })

    output_segments.sort(key=lambda s: s["start"])
    output_path = args.output_json or str(Path(args.wav_path).with_name(Path(args.wav_path).stem + "_transcript.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "wav_path": str(args.wav_path),
            "sortformer_rttm": str(args.sortformer_rttm),
            "speakers": speakers,
            "segments": output_segments,
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote per-segment transcript to {output_path}")


if __name__ == "__main__":
    main()
