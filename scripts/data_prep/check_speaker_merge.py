"""
Run this on your server against your real manifest + SPK_List to check whether
named speakers merged correctly, before we decide anything about the anonymous
count.

Usage:
    python check_speaker_merge.py \
        --manifest /path/to/voxforge_pt_manifest.json \
        --spk_list /path/to/SPK_List
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--spk_list", required=True)
args = ap.parse_args()

spk_list = set(open(args.spk_list, encoding="utf-8").read().split())

manifest_speakers = set()
with open(args.manifest, encoding="utf-8") as f:
    for line in f:
        manifest_speakers.add(json.loads(line)["speaker_id"])

exact_matches = manifest_speakers & spk_list
not_in_spk_list = manifest_speakers - spk_list  # anonymous-folder IDs + possible mismatches
missing_named = spk_list - manifest_speakers    # SPK_List entries with NO matching speaker_id at all

print(f"SPK_List entries:                         {len(spk_list)}")
print(f"Manifest speaker IDs matching exactly:     {len(exact_matches)}")
print(f"Manifest speaker IDs NOT in SPK_List:      {len(not_in_spk_list)}  (expected: mostly anonymous-* folder names)")
print(f"SPK_List entries with NO match at all:     {len(missing_named)}")
if missing_named:
    print(f"  -> {sorted(missing_named)}")
    print("  These didn't merge -- check for case/spelling drift across that speaker's submission folders.")
