#!/usr/bin/env python3
"""
Load a SPECIFIC training checkpoint (not the end-of-training .nemo bundle, which
reflects the LAST epoch, not necessarily the BEST one) and run test-set evaluation.

Rebuilds the exact same architecture train_multitalker_aws.py used (encoder/decoder/
joint shapes must match exactly for the checkpoint's weights to load), using the
pretrained model purely for its CONFIG this time -- the checkpoint already has fully
trained weights, so we don't need the pretrained WEIGHTS or the zero-init step at all.

Usage:
    python evaluate_checkpoint.py \
        --overrides conf/multitalker_finetune_overrides_aws.yaml \
        --checkpoint "multitalker_finetune_experiments/.../checkpoints/multitalker_pt_adapter_aws--val_wer=0.6957-epoch=46.ckpt" \
        --test_cuts /path/to/test_cuts.jsonl.gz \
        --pretrained_model nvidia/parakeet-tdt-0.6b-v3
"""
import numpy as np
if not hasattr(np, "row_stack"):
    # Same numba-cuda / NumPy 2.0 compatibility patch as train_multitalker_aws.py --
    # needed here too, since this script also triggers the RNNT loss kernel via
    # trainer.test().
    np.row_stack = np.vstack
import argparse
import tarfile
import tempfile
from pathlib import Path

import lightning.pytorch as pl
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.multitalker_asr_models import EncDecMultiTalkerRNNTBPEModel

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--overrides", required=True)
ap.add_argument("--checkpoint", required=True, help="Path to a specific .ckpt file (NOT the .nemo bundle)")
ap.add_argument("--test_cuts", required=True)
ap.add_argument("--pretrained_model", default="nvidia/parakeet-tdt-0.6b-v3")
args = ap.parse_args()


def main():
    # --- Rebuild the same architecture used in training. We only need the pretrained
    # model's CONFIG here (to get encoder/decoder/joint shapes exactly right) -- not
    # its weights, since the checkpoint below has fully trained weights already. ---
    print(f"Loading base architecture config from: {args.pretrained_model} ...")
    base_model = EncDecRNNTBPEModel.from_pretrained(args.pretrained_model, map_location="cpu")
    base_cfg = base_model.cfg
    del base_model

    override_cfg = OmegaConf.load(args.overrides)
    with open_dict(base_cfg):
        merged_cfg = OmegaConf.merge(base_cfg, override_cfg.model)
    merged_cfg.test_ds.cuts_path = args.test_cuts
    # train_ds/validation_ds aren't used for evaluation, but merged_cfg still has the
    # ??? placeholders from the overrides file -- fill them with the same test_cuts
    # so nothing downstream chokes on an unresolved mandatory value.
    merged_cfg.train_ds.cuts_path = args.test_cuts
    merged_cfg.validation_ds.cuts_path = args.test_cuts

    # --- Resolve internal "nemo:" artifact references (tokenizer path etc.), same
    # approach as train_multitalker_aws.py. ---
    nemo_file_path = hf_hub_download(
        repo_id=args.pretrained_model,
        filename=f"{args.pretrained_model.split('/')[-1]}.nemo",
    )
    extract_dir = Path(tempfile.mkdtemp(prefix="eval_artifacts_"))
    with tarfile.open(nemo_file_path) as tar:
        tar.extractall(extract_dir)

    def resolve_nemo_artifact_refs(cfg, base_dir):
        def _walk(node):
            if OmegaConf.is_config(node):
                keys = list(node.keys()) if hasattr(node, "keys") else range(len(node))
                for key in keys:
                    val = node[key]
                    if isinstance(val, str) and val.startswith("nemo:"):
                        node[key] = str(base_dir / val[len("nemo:"):])
                    elif OmegaConf.is_config(val):
                        _walk(val)
        _walk(cfg)

    with open_dict(merged_cfg):
        resolve_nemo_artifact_refs(merged_cfg, extract_dir)

    # --- Construct the model. Weights will be entirely overwritten by the checkpoint
    # next, so what gets loaded here (pretrained-init + zero-init kernels) doesn't
    # matter -- only the architecture SHAPE matters at this point. ---
    trainer = pl.Trainer(devices=1, accelerator="gpu", logger=False, enable_checkpointing=False)
    model = EncDecMultiTalkerRNNTBPEModel(cfg=merged_cfg, trainer=trainer)

    # --- Load the ACTUAL trained checkpoint weights, overwriting everything above. ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Checkpoint loaded cleanly. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        print("WARNING: strict loading reported mismatches -- the rebuilt architecture may not "
              "exactly match what was trained. This should NOT happen if --overrides matches the "
              "config actually used for training. Missing:", missing, "Unexpected:", unexpected)
    del ckpt

    model.eval()

    # --- Run test-set evaluation using Lightning's own test loop, the same machinery
    # that already worked correctly for validation during training. ---
    print("\nRunning test-set evaluation...")
    results = trainer.test(model)
    print("\nTest results:")
    for r in results:
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
