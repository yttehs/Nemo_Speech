#!/usr/bin/env python3
"""
Build a NeMo-style per-utterance manifest from a VoxForge submissions directory.

VoxForge layout (per submission folder, e.g. vchoi-20091024-bif/):
    etc/README   - contains "User Name: <username>" plus speaker metadata
    etc/PROMPTS  - lines like "<folder>/mfc/<num> <TRANSCRIPT TEXT>"
    wav/<num>.wav

Speaker ID assignment:
    - If README's User Name is present and not "anonymous", that username is
      used as the speaker ID. Multiple submission folders from the same named
      user are merged into one speaker, since they're the same real person.
    - If the user is "anonymous" (or README is missing/unreadable), the full
      submission folder name is used as the speaker ID instead, so each such
      submission is treated as its own distinct, unverifiable speaker.

Also resamples audio to 16kHz mono into a parallel output directory, since
VoxForge PT audio ships at 48kHz and the rest of the pipeline (forced
alignment, the speech data simulator, Sortformer) expects 16kHz.

Output filenames are FLAT and include the source folder name (e.g.
"vchoi-20091024-bif__083.wav"), not nested per-folder. This matters: VoxForge
numbers utterances *within* each submission folder, so "001.wav" exists in
dozens of different folders. Several downstream tools (including NeMo Forced
Aligner) derive an utterance ID from the bare filename stem alone, ignoring
the directory -- if the filename isn't already globally unique, those tools
will silently collide and overwrite each other's output.

Usage:
    python voxforge_to_manifest.py \
        --voxforge_dir /path/to/voxforge-pt \
        --resampled_dir /path/to/voxforge-pt-16k \
        --manifest_path /path/to/voxforge_pt_manifest.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import soundfile as sf
import librosa

PROMPTS_LINE_RE = re.compile(r"^\S+/mfc/(\S+)\s+(.+)$")


def parse_readme_username(readme_path: Path) -> Optional[str]:
    """Pull the 'User Name:' field out of a VoxForge etc/README file."""
    if not readme_path.exists():
        return None
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if line.strip().lower().startswith("user name:"):
            name = line.split(":", 1)[1].strip()
            return name or None
    return None


def resolve_speaker_id(
    folder_name: str, readme_username: Optional[str], anonymous_mode: str = "per_folder"
) -> str:
    """Named users merge across sessions.

    Anonymous/unknown users are handled per `anonymous_mode`:
      - "per_folder" (default): each anonymous submission is its own distinct
        speaker, since there's no way to verify two anonymous submissions are
        the same real person. More conservative -- avoids the simulator ever
        treating two acoustically different voices as "the same speaker."
      - "single": every anonymous submission collapses into one shared
        "anonymous" speaker ID, matching VoxForge's own official SPK_List
        convention (where "anonymous" appears as a single roster entry).
    """
    if readme_username and readme_username.strip().lower() != "anonymous":
        return readme_username.strip()
    if anonymous_mode == "single":
        return "anonymous"
    return folder_name


def resample_to_16k(src_wav: Path, dst_wav: Path, target_sr: int = 16000) -> None:
    if dst_wav.exists():
        return  # allows safely re-running the script without redoing work
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(src_wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # VoxForge PT is documented mono; guard anyway
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(dst_wav, audio, target_sr, subtype="PCM_16")


def build_manifest(
    voxforge_dir: Path, resampled_dir: Path, manifest_path: Path, anonymous_mode: str = "per_folder"
) -> None:
    n_utts = 0
    n_folders = 0
    n_folders_skipped = 0
    n_lines_skipped = 0
    speakers_seen = set()

    with manifest_path.open("w", encoding="utf-8") as out_f:
        for folder in sorted(voxforge_dir.iterdir()):
            if not folder.is_dir():
                continue

            prompts_path = folder / "etc" / "PROMPTS"
            wav_dir = folder / "wav"
            readme_path = folder / "etc" / "README"

            if not prompts_path.exists() or not wav_dir.exists():
                n_folders_skipped += 1
                continue

            username = parse_readme_username(readme_path)
            speaker_id = resolve_speaker_id(folder.name, username, anonymous_mode)
            speakers_seen.add(speaker_id)
            n_folders += 1

            for line in prompts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                m = PROMPTS_LINE_RE.match(line)
                if not m:
                    n_lines_skipped += 1
                    continue  # malformed line -- skip it, don't crash the whole run
                utt_id, text = m.group(1), m.group(2)

                src_wav = wav_dir / f"{utt_id}.wav"
                if not src_wav.exists():
                    n_lines_skipped += 1
                    continue  # PROMPTS references a file that isn't actually there

                dst_wav = resampled_dir / f"{folder.name}__{utt_id}.wav"
                resample_to_16k(src_wav, dst_wav)
                duration = sf.info(dst_wav).duration

                out_f.write(
                    json.dumps(
                        {
                            "audio_filepath": str(dst_wav.resolve()),
                            "duration": round(float(duration), 3),
                            "text": text,
                            "speaker_id": speaker_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_utts += 1

    print(f"Folders processed:     {n_folders}  (skipped, no PROMPTS/wav: {n_folders_skipped})")
    print(f"Utterances written:    {n_utts}  (skipped lines: {n_lines_skipped})")
    print(f"Distinct speaker IDs:  {len(speakers_seen)}")
    print(f"Manifest written to:   {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxforge_dir", required=True, type=Path,
                     help="Path to voxforge-pt/ containing submission folders")
    ap.add_argument("--resampled_dir", required=True, type=Path,
                     help="Where to write 16kHz mono copies of the audio")
    ap.add_argument("--manifest_path", required=True, type=Path,
                     help="Output manifest .json path (one JSON object per line)")
    ap.add_argument("--anonymous_mode", choices=["per_folder", "single"], default="per_folder",
                     help="'per_folder' (default): each anonymous submission is its own speaker. "
                          "'single': all anonymous submissions collapse into one 'anonymous' ID, "
                          "matching VoxForge's own SPK_List convention.")
    args = ap.parse_args()

    args.resampled_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    build_manifest(args.voxforge_dir, args.resampled_dir, args.manifest_path, args.anonymous_mode)


if __name__ == "__main__":
    main()
