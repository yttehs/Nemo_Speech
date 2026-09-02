#!/usr/bin/env python3
"""
Fine-tune EncDecMultiTalkerRNNTBPEModel from a pretrained Parakeet-TDT-0.6B-v3
checkpoint, with the backbone frozen and only the speaker kernels trainable.

Design choice worth explaining: rather than relying on the config-driven
`init_from_pretrained_model` mechanism (ModelPT.maybe_init_from_pretrained_checkpoint),
this script loads the pretrained weights explicitly. It's the same underlying
operation, but doing it directly here sidesteps a real uncertainty we couldn't
verify without running it -- whether `self.from_pretrained()` tolerates being
called with a different target class than the checkpoint was saved under. Loading
the base model in its own native class first avoids that question entirely, and
lets us print a real diagnostic of exactly what loaded vs. what didn't, rather
than trusting something that happens implicitly during __init__.

Usage:
    python train_multitalker.py \
        --overrides conf/multitalker_finetune_overrides.yaml \
        --train_cuts /path/to/train_cuts.jsonl.gz \
        --val_cuts /path/to/val_cuts.jsonl.gz \
        --test_cuts /path/to/test_cuts.jsonl.gz \
        --pretrained_model nvidia/parakeet-tdt-0.6b-v3
"""
import numpy as np
if not hasattr(np, "row_stack"):
    # numba-cuda's arrayobj.py still references the deprecated NumPy 1.x alias
    # row_stack (removed entirely in NumPy 2.0), causing an AttributeError the
    # first time a numba CUDA kernel gets JIT-compiled -- lazy compilation means
    # this doesn't crash at import time, only when the RNNT loss kernel first
    # actually runs (the first real training step). row_stack was always a pure
    # alias for vstack (verified identical behavior), so restoring it is a safe,
    # zero-risk patch -- avoids downgrading numpy globally, which would break
    # scipy/ml-dtypes/librosa's own numpy>=2.0/2.1 requirements.
    np.row_stack = np.vstack

import argparse
import atexit
import shutil
import tarfile
import tempfile
from pathlib import Path

import lightning.pytorch as pl
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.models.multitalker_asr_models import (
    EncDecMultiTalkerRNNTBPEModel,
    EncDecMultiTalkerRNNTModel,
)
from nemo.utils.exp_manager import exp_manager

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--overrides", required=True, help="Path to multitalker_finetune_overrides.yaml")
ap.add_argument("--train_cuts", required=True)
ap.add_argument("--val_cuts", required=True)
ap.add_argument("--test_cuts", required=True)
ap.add_argument("--pretrained_model", default="nvidia/parakeet-tdt-0.6b-v3")
ap.add_argument("--character_based", action="store_true",
                 help="Set this for pretrained backbones with a character vocabulary instead "
                      "of a subword (BPE/SentencePiece) tokenizer -- e.g. "
                      "nvidia/stt_zh_conformer_transducer_large, trained on AISHELL-2 Mandarin. "
                      "Loading such a model via the default (BPE) path fails immediately with "
                      "'`cfg` must have `tokenizer` config', since that config block doesn't "
                      "exist for a character-based checkpoint. Off by default -- every model "
                      "used so far (Parakeet-TDT for the European languages) is BPE-based.")
args = ap.parse_args()


def main():
    # --- Step 1: load the pretrained model in its OWN native class. ---
    # This gives us both its exact architecture config (so our target model's
    # encoder/decoder/joint shapes match precisely) and its trained weights.
    print(f"Loading pretrained model: {args.pretrained_model} "
          f"({'character-based' if args.character_based else 'BPE-tokenized'}) ...")
    if args.character_based:
        base_model = EncDecRNNTModel.from_pretrained(args.pretrained_model, map_location="cpu")
    else:
        base_model = EncDecRNNTBPEModel.from_pretrained(args.pretrained_model, map_location="cpu")
    base_cfg = base_model.cfg
    base_state_dict = base_model.state_dict()
    print(f"Pretrained model loaded. {len(base_state_dict)} parameter tensors.")

    # --- Step 2: merge multitalker-specific overrides on TOP of the pretrained
    # architecture config, so nothing about encoder/decoder/joint shapes changes. ---
    override_cfg = OmegaConf.load(args.overrides)
    with open_dict(base_cfg):
        merged_cfg = OmegaConf.merge(base_cfg, override_cfg.model)
    merged_cfg.train_ds.cuts_path = args.train_cuts
    merged_cfg.validation_ds.cuts_path = args.val_cuts
    merged_cfg.test_ds.cuts_path = args.test_cuts

    # --- Step 2c: if the override switches to a DIFFERENT scheduler type than the
    # pretrained model's own native one, replace optim.sched entirely rather than
    # merging into it. OmegaConf.merge is key-by-key, so a stale, scheduler-specific
    # key from the pretrained model's own config can survive the merge even when the
    # override switches schedulers -- confirmed via a real failure: a Noam-scheduled
    # backbone's own d_model=512 survived a merge into a WarmupAnnealing override and
    # crashed with "WarmupPolicy.__init__() got an unexpected keyword argument
    # 'd_model'". Every prior language used Parakeet-TDT, whose own native scheduler
    # didn't happen to collide this way -- first backbone where this had a chance to
    # surface, not something specific to Mandarin or to character-based models.
    if "sched" in override_cfg.model.get("optim", {}):
        override_sched_name = override_cfg.model.optim.sched.get("name")
        base_sched_name = base_cfg.optim.sched.get("name") if "sched" in base_cfg.get("optim", {}) else None
        if override_sched_name != base_sched_name:
            with open_dict(merged_cfg):
                merged_cfg.optim.sched = override_cfg.model.optim.sched
            print(f"Scheduler type changed ({base_sched_name} -> {override_sched_name}) -- "
                  f"replaced optim.sched entirely rather than merging it, to avoid leaking "
                  f"stale, scheduler-specific keys from the pretrained model's own config.")

    # --- Step 2b: resolve NeMo's internal "nemo:<relative_path>" artifact
    # references (e.g. tokenizer.model_path) to real, persistent local paths.
    # These references only resolve automatically during an ACTIVE restore_from()
    # call (via a temporary app_state.nemo_file_folder) -- base_cfg was copied out
    # of that context, so constructing a SECOND model from it needs the real paths
    # resolved ourselves. A .nemo file is just a tar archive, so we extract it
    # directly rather than relying on NeMo's transient internal state.
    nemo_file_path = hf_hub_download(
        repo_id=args.pretrained_model,
        filename=f"{args.pretrained_model.split('/')[-1]}.nemo",
    )
    extract_dir = Path(tempfile.mkdtemp(prefix="pretrained_artifacts_"))
    atexit.register(shutil.rmtree, extract_dir, ignore_errors=True)  # clean up on exit (normal or
                                                                       # crash) instead of leaking
                                                                       # this every run -- the tokenizer
                                                                       # file may still be referenced
                                                                       # later in training, so it needs
                                                                       # to survive for the process's
                                                                       # lifetime, just not forever after
    with tarfile.open(nemo_file_path) as tar:
        tar.extractall(extract_dir)

    def resolve_nemo_artifact_refs(cfg, base_dir):
        n_resolved = 0
        def _walk(node):
            nonlocal n_resolved
            if OmegaConf.is_config(node):
                keys = list(node.keys()) if hasattr(node, "keys") else range(len(node))
                for key in keys:
                    val = node[key]
                    if isinstance(val, str) and val.startswith("nemo:"):
                        node[key] = str(base_dir / val[len("nemo:"):])
                        n_resolved += 1
                    elif OmegaConf.is_config(val):
                        _walk(val)
        _walk(cfg)
        return n_resolved

    with open_dict(merged_cfg):
        n_resolved = resolve_nemo_artifact_refs(merged_cfg, extract_dir)
    print(f"Resolved {n_resolved} internal 'nemo:' artifact reference(s) to real paths under {extract_dir}")

    trainer_cfg = override_cfg.trainer
    exp_manager_cfg = override_cfg.exp_manager

    # --- Step 3: build the trainer, then the multitalker model with the merged config. ---
    trainer = pl.Trainer(**OmegaConf.to_container(trainer_cfg, resolve=True))
    if args.character_based:
        model = EncDecMultiTalkerRNNTModel(cfg=merged_cfg, trainer=trainer)
    else:
        model = EncDecMultiTalkerRNNTBPEModel(cfg=merged_cfg, trainer=trainer)

    # --- Step 4: load the pretrained weights explicitly, and report exactly what happened. ---
    missing_keys, unexpected_keys = model.load_state_dict(base_state_dict, strict=False)
    print(f"\nWeight loading report:")
    print(f"  Keys present in target model but NOT in pretrained checkpoint "
          f"(expected: spk_kernels / bg_spk_kernels, freshly initialized): {len(missing_keys)}")
    non_kernel_missing = [k for k in missing_keys if "spk_kernel" not in k]
    if non_kernel_missing:
        print(f"  WARNING: {len(non_kernel_missing)} missing keys are NOT speaker-kernel related -- "
              f"this may indicate an architecture mismatch, worth investigating before training:")
        for k in non_kernel_missing[:10]:
            print(f"    {k}")
    else:
        print("  All missing keys are speaker-kernel related, as expected.")
    print(f"  Keys in checkpoint but not used by target model: {len(unexpected_keys)}")
    if unexpected_keys:
        for k in unexpected_keys[:10]:
            print(f"    {k}")
    del base_model, base_state_dict

    # --- Step 5: freeze the backbone, leave only the speaker kernels trainable. ---
    # Confirmed sufficient on its own: NeMo's default optimizer setup
    # (ModelPT.setup_optimization) passes list(self.parameters()) unfiltered into the
    # optimizer, but a parameter with requires_grad=False never accumulates a gradient,
    # so standard optimizers skip updating it regardless of whether it's in the param list.
    for module in (model.encoder, model.decoder, model.joint):
        for param in module.parameters():
            param.requires_grad = False
    for module in (model.spk_kernels, model.bg_spk_kernels):
        for param in module.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")
    if n_trainable == n_total:
        raise RuntimeError(
            "No parameters were frozen -- 100% of parameters are trainable. "
            "This means the freezing step did not work as expected; do not proceed "
            "to training until this is fixed, since it would silently full-fine-tune "
            "the entire backbone rather than just the speaker kernels."
        )

    # --- Step 5b: zero-init each kernel's final layer, so injection starts as a true
    # no-op (x + 0 = x) rather than random noise perturbing the pretrained encoder's
    # representations from step 0. Standard technique (same as LoRA's zero-init B
    # matrix) -- the final layer receives real gradient and learns immediately;
    # earlier layers in the same kernel start receiving gradient only once the final
    # layer's weights move away from zero. Verified this doesn't dead-end training.
    for kernel_dict in (model.spk_kernels, model.bg_spk_kernels):
        for kernel in kernel_dict.values():
            torch.nn.init.zeros_(kernel[-1].weight)
            torch.nn.init.zeros_(kernel[-1].bias)
    print("Zero-initialized final layer of every speaker kernel (spk_kernels + bg_spk_kernels).")

    # --- Step 6: set up experiment tracking/checkpointing, then train. ---
    exp_manager(trainer, exp_manager_cfg)
    trainer.fit(model)


if __name__ == "__main__":
    main()
