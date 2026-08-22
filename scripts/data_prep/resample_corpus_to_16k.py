#!/usr/bin/env python3
"""
Resamples every .wav file under a CML-TTS-style corpus root to 16kHz, writing to a
NEW directory while preserving the exact relative path structure (train/audio/<spk>/
<chapter>/<file>.wav etc.) -- so downstream scripts (audit_transcript_audio_mismatch.py,
prepare_mfa_corpus.py) just need --audio_root pointed at the output directory instead
of the original, with zero other changes. The CSVs (train.csv/dev.csv/test.csv) don't
need copying or editing -- their wav_filename column is a relative path, valid against
either root.

Confirmed compatible with the whole existing pipeline: audit_transcript_audio_mismatch.py
and prepare_mfa_corpus.py have no sample-rate-dependent logic at all (duration is
sample-rate-independent; symlinking doesn't care what's in the file), MFA's own feature
extraction has an 8kHz ceiling regardless of input rate (confirmed via the MFA Interspeech
paper), and build_lhotse_cutset.py's ensure_target_rate() already has a "matches target,
just symlink" fast path that pre-resampled input will hit automatically.

Usage:
    python resample_corpus_to_16k.py \
        --source_dir /path/to/cml_tts_dataset_spanish_v0.1 \
        --output_dir /path/to/cml_tts_dataset_spanish_v0.1_16k
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--source_dir", required=True, type=Path)
ap.add_argument("--output_dir", required=True, type=Path)
ap.add_argument("--target_sample_rate", type=int, default=16000)
ap.add_argument("--workers", type=int, default=4,
                 help="Process-based (not thread-based) parallelism -- resampling is real CPU work, "
                      "not I/O-bound like the audit script's duration checks, so this needs true "
                      "multiprocessing to actually use multiple cores. Defaults to 4, matching this "
                      "project's known AWS box constraint (4 CPU cores) -- raise it if you're running "
                      "this somewhere with more.")
args = ap.parse_args()


def resample_one(src_path_str: str, dst_path_str: str, target_sr: int):
    """Runs in a worker process. Returns (dst_path_str, error_or_None)."""
    dst_path = Path(dst_path_str)
    if dst_path.exists():
        return dst_path_str, None  # already done -- safe to resume a partial run
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        audio, _ = librosa.load(src_path_str, sr=target_sr, mono=True)
        sf.write(dst_path_str, audio, target_sr, subtype="PCM_16")
        return dst_path_str, None
    except Exception as e:
        return dst_path_str, str(e)


def main():
    if not args.source_dir.exists():
        raise SystemExit(f"source_dir not found: {args.source_dir}")

    wav_paths = sorted(args.source_dir.rglob("*.wav"))
    print(f"Found {len(wav_paths)} wav files under {args.source_dir}")
    if not wav_paths:
        return

    jobs = []
    for src_path in wav_paths:
        rel_path = src_path.relative_to(args.source_dir)
        dst_path = args.output_dir / rel_path
        jobs.append((str(src_path), str(dst_path)))

    n_done = n_skipped_existing = n_errors = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(resample_one, src, dst, args.target_sample_rate): (src, dst)
            for src, dst in jobs
        }
        for i, future in enumerate(as_completed(futures)):
            dst_path_str, error = future.result()
            if error is not None:
                n_errors += 1
                errors.append((dst_path_str, error))
            else:
                n_done += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)} processed...")

    print(f"\n{n_done} files resampled (or already present) -> {args.output_dir}")
    if n_errors:
        print(f"{n_errors} errors:")
        for path, error in errors[:20]:
            print(f"  {path}: {error}")
        if n_errors > 20:
            print(f"  ...and {n_errors - 20} more")


if __name__ == "__main__":
    main()
