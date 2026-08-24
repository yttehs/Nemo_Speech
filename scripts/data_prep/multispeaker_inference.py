"""
Shared logic for Tier 2/3 evaluation: load a trained checkpoint, and transcribe
audio conditioned on an ARBITRARY speaker-activity segment list (ground truth OR
Sortformer's predictions) rather than the training dataloader's own targets.

Mask construction reuses speaker_to_target() itself -- the same function training
used -- by building a throwaway Lhotse Cut whose supervisions are the segments we
want a mask for. This guarantees identical mask-building logic to training, rather
than risking a subtly different reimplementation.
"""
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from lhotse import MonoCut, Recording, SupervisionSegment
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import speaker_to_target
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.multitalker_asr_models import EncDecMultiTalkerRNNTBPEModel

# The RNNT decoder's own word-timestamp offsets are indices into the SAME
# encoder-output frame sequence that speaker_to_target() builds its masks on
# (num_sample_per_mel_frame * num_mel_frame_per_asr_frame / sampling_rate =
# 160 * 8 / 16000 = 0.08s -- the "80ms/frame" convention already used
# elsewhere in this codebase, e.g. the ~active-seconds estimate in the Tier 2/3
# scripts). Centralized here so both the mask math and the timestamp math stay
# using one shared constant instead of two independently-hardcoded literals.
ENCODER_FRAME_SECONDS = 0.08


def parse_rttm(path):
    """Returns list of (start, end, speaker) tuples."""
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.strip().split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            segments.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return segments


def arrival_order(segments):
    """Unique speaker labels, ordered by first appearance -- same convention
    speaker_to_target() uses internally, so mask row i corresponds to the
    i-th speaker in THIS list."""
    first_seen = {}
    for start, end, spk in segments:
        if spk not in first_seen or start < first_seen[spk]:
            first_seen[spk] = start
    return sorted(first_seen, key=lambda s: first_seen[s])


def load_model(overrides_path, checkpoint_path, pretrained_model="nvidia/parakeet-tdt-0.6b-v3",
                dummy_cuts_path=None):
    """Rebuilds the trained architecture and loads a specific checkpoint's weights.
    Same construction sequence as evaluate_checkpoint.py.

    dummy_cuts_path: a real, valid *_cuts.jsonl.gz file (e.g. your test_cuts.jsonl.gz
    from training). NeMo's ModelPT.__init__ eagerly calls setup_training_data() the
    moment train_ds has ANY non-'???' cuts_path set -- it doesn't matter that we never
    actually use this dataloader, it still gets constructed and validated at model
    construction time. Pointing it at a real manifest avoids a crash; pointing it at
    an arbitrary placeholder string (e.g. the checkpoint path itself) does not, since
    NeMo will try to parse whatever's there as an actual Lhotse manifest.
    """
    base_model = EncDecRNNTBPEModel.from_pretrained(pretrained_model, map_location="cpu")
    base_cfg = base_model.cfg
    del base_model

    override_cfg = OmegaConf.load(overrides_path)
    with open_dict(base_cfg):
        merged_cfg = OmegaConf.merge(base_cfg, override_cfg.model)

    if dummy_cuts_path is None:
        raise ValueError(
            "dummy_cuts_path is required -- point it at any real, valid *_cuts.jsonl.gz "
            "file (e.g. your test_cuts.jsonl.gz from training). NeMo's ModelPT.__init__ "
            "eagerly validates train_ds/validation_ds/test_ds as real manifests at "
            "construction time, even though this script never actually uses them."
        )
    for key in ("train_ds", "validation_ds", "test_ds"):
        merged_cfg[key].cuts_path = str(dummy_cuts_path)

    nemo_file_path = hf_hub_download(repo_id=pretrained_model, filename=f"{pretrained_model.split('/')[-1]}.nemo")
    extract_dir = Path(tempfile.mkdtemp(prefix="eval_artifacts_"))
    with tarfile.open(nemo_file_path) as tar:
        tar.extractall(extract_dir)

    def _resolve(cfg, base_dir):
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
        _resolve(merged_cfg, extract_dir)

    trainer = pl.Trainer(devices=1, accelerator="gpu", logger=False, enable_checkpointing=False)
    model = EncDecMultiTalkerRNNTBPEModel(cfg=merged_cfg, trainer=trainer)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Checkpoint loaded. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        print("WARNING: architecture mismatch detected -- results below should not be trusted "
              "until this is resolved. Missing:", missing, "Unexpected:", unexpected)
    del ckpt

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def enable_word_timestamps(model):
    """
    Reconfigures the model's decoding strategy to emit word-level timestamps
    alongside the hypothesis text (NeMo's `compute_timestamps` decoding option).
    Idempotent -- safe to call more than once on the same model instance.

    Deliberately word-level only, not segment-level: NeMo's segment-level
    timestamps key off sentence-ending punctuation (segment_seperators), and
    this model's training targets (VoxForge-derived, per wer_scorer.py's own
    docstring) never contain punctuation -- segment-level splitting wouldn't
    find any boundaries to split on. Word-level offsets are reliable
    regardless, and are all Tier 3's RTTM-segment bucketing needs.

    Only call this when you actually need timestamps (e.g. Tier 3, for
    building the per-segment transcript view) -- `preserve_alignments=True`
    keeps per-frame alignment data around for the full decode, which costs
    extra memory on longer real-audio clips. Tier 2 does not need this and
    should not enable it.
    """
    if getattr(model, "_word_timestamps_enabled", False):
        return
    decoding_cfg = deepcopy(model.cfg.decoding)
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
    model.change_decoding_strategy(decoding_cfg)
    model._word_timestamps_enabled = True


def build_mask_for_speaker(wav_path, segments, target_speaker, num_speakers=4,
                            use_purity_weighted_targets=False, lambda_overlap_weight=0.5):
    """
    Builds (spk_target, bg_spk_target) 1D tensors for ONE target speaker, using
    speaker_to_target() itself so mask construction exactly matches training.

    segments: list of (start, end, speaker) tuples -- ground truth OR Sortformer's
    predictions, doesn't matter which; this function doesn't care about the source.

    use_purity_weighted_targets/lambda_overlap_weight: MUST match whatever the model
    was actually trained with (see audio_to_text_lhotse_speaker.py's identically-named
    train_ds/validation_ds/test_ds config keys) -- evaluating a purity-trained model
    with hard binary masks (or vice versa) is a genuine train/test mismatch, not just a
    missed optimization. Note: RTTM segments (Sortformer's predictions) carry no
    continuous per-frame confidence, only discrete start/end boundaries -- soft_label's
    continuous values here reflect boundary-alignment softening only (a segment edge
    landing mid-ASR-frame), the same mechanism as training, just without genuine
    diarization-confidence gradation since RTTM itself doesn't carry that.
    """
    recording = Recording.from_file(wav_path, recording_id="eval")
    supervisions = [
        SupervisionSegment(id=f"eval-sup{i:05d}", recording_id="eval",
                            start=start, duration=end - start, text="", speaker=spk)
        for i, (start, end, spk) in enumerate(segments)
    ]
    cut = MonoCut(id="eval", start=0.0, duration=recording.duration, channel=0,
                  recording=recording, supervisions=supervisions, custom={})

    mask, texts = speaker_to_target(cut, num_speakers=num_speakers, return_text=True,
                                     soft_label=use_purity_weighted_targets)
    mask = mask.transpose(0, 1)[: len(texts)]  # [num_real_speakers, num_frames], arrival order

    order = arrival_order(segments)
    if target_speaker not in order:
        raise ValueError(f"Speaker {target_speaker!r} not found in segments (available: {order})")
    idx = order.index(target_speaker)

    if idx >= mask.shape[0]:
        # speaker_to_target()'s own internal mask-building can end up with FEWER
        # rows than our independently-computed arrival order suggests -- typically
        # because a predicted speaker's activity is so brief it never crosses the
        # frame-level activity threshold internally. Not a bug to chase down; just
        # needs to be skipped gracefully rather than crash the whole evaluation run.
        raise ValueError(
            f"Speaker {target_speaker!r} is at arrival-order index {idx}, but "
            f"speaker_to_target()'s own mask only has {mask.shape[0]} speaker row(s) -- "
            f"likely a predicted speaker whose activity was too brief to register "
            f"internally. Skipping this pair."
        )

    if use_purity_weighted_targets:
        # Identical derivation to audio_to_text_lhotse_speaker.py's training-time logic --
        # see test_purity_weighting.py for the shared correctness tests both paths rely on.
        non_target_idx = [i for i in range(mask.shape[0]) if i != idx]
        d_target = mask[idx]
        if non_target_idx:
            prod_others_silent = torch.prod(1 - mask[non_target_idx], dim=0)
        else:
            prod_others_silent = torch.ones_like(d_target)  # no other speakers at all in this session
        p_t = d_target * prod_others_silent
        p_o = d_target - p_t
        p_silence = prod_others_silent * (1 - d_target)
        p_n = 1 - p_silence - d_target

        lam = lambda_overlap_weight
        spk_target = p_t + lam * p_o
        bg_spk_target = (1 - lam) * p_o + p_n
    else:
        spk_target = mask[idx]
        bg_spk_target = (mask.sum(dim=0) - mask[idx] > 0).float()  # same OR-of-others logic as
                                                                      # the dataset's bg_speaker_target
    return spk_target, bg_spk_target


def transcribe_with_mask(model, wav_path, spk_target, bg_spk_target, return_word_timestamps=False):
    """Runs the model conditioned on a manually-built speaker-activity mask,
    bypassing the training dataloader entirely.

    return_word_timestamps: when False (default -- Tier 2's usage), returns just
    the hypothesis text string, unchanged from before. When True (Tier 3), also
    enables word-level timestamp decoding (see enable_word_timestamps()) and
    returns (text, word_entries), where each entry is
    {"word": str, "start": float seconds, "end": float seconds}, session-relative
    (same time axis as the RTTM segments passed into build_mask_for_speaker).
    """
    if return_word_timestamps:
        enable_word_timestamps(model)

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    device = next(model.parameters()).device
    signal = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    signal_len = torch.tensor([signal.shape[1]], device=device)

    spk_target = spk_target.unsqueeze(0).to(device)
    bg_spk_target = bg_spk_target.unsqueeze(0).to(device)

    with torch.no_grad():
        model.set_speaker_targets(spk_target, bg_spk_target)
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        hyps = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )
        model.clear_speaker_targets()

    hyp = hyps[0]
    text = hyp.text if hasattr(hyp, "text") else str(hyp)

    if not return_word_timestamps:
        return text

    word_entries = []
    timestamp = getattr(hyp, "timestamp", None) or {}
    for w in timestamp.get("word", []):
        word_entries.append({
            "word": w["word"],
            "start": w["start_offset"] * ENCODER_FRAME_SECONDS,
            "end": w["end_offset"] * ENCODER_FRAME_SECONDS,
        })
    return text, word_entries
