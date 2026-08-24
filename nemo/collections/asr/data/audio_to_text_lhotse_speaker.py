# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import speaker_to_target
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextSpkBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py. It has the same functionality of LhotseSpeechToTextBpeDataset but also yield speaker target tensor.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'spk_targets': NeuralType(('B', 'T'), LabelsType()),
            'bg_spk_targets': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(self, cfg, tokenizer: TokenizerSpec):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.fixed_spk_id = self.cfg.get('fixed_spk_id', None)
        self.inference_mode = self.cfg.get('inference_mode', False)
        # Opt-in, defaults preserve exact current (binary mask) behavior -- critical since
        # already-trained checkpoints (Portuguese, Spanish CML-TTS baselines) were trained
        # and evaluated against the binary mask; silently changing the default here would
        # invalidate those as a comparison baseline for anyone re-running their eval.
        self.use_purity_weighted_targets = self.cfg.get('use_purity_weighted_targets', False)
        self.lambda_overlap_weight = self.cfg.get('lambda_overlap_weight', 0.5)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:

        audio, audio_lens, cuts = self.load_audio(cuts)

        tokens = []
        spk_targets = []
        bg_spk_targets = []

        if self.inference_mode:
            return audio, audio_lens, None, None, None, None

        for idx, cut in enumerate(cuts):

            speaker_targets, texts = speaker_to_target(
                a_cut=cut,
                num_speakers=self.num_speakers,
                num_sample_per_mel_frame=self.num_sample_per_mel_frame,
                num_mel_frame_per_asr_frame=self.num_mel_frame_per_asr_frame,
                return_text=True,
                # soft_label=True is REQUIRED for purity weighting below -- without it,
                # speaker_targets arrives already hard-binarized (0/1) and there is no
                # real per-speaker activity probability left to derive STNO values from.
                # When purity weighting is off, this has no effect on training: the
                # binary-vs-soft-thresholded mask is later reproduced by construction
                # (see the else branch below), so existing runs are unaffected.
                soft_label=self.use_purity_weighted_targets,
            )
            speaker_targets = speaker_targets.transpose(0, 1)[: len(texts)]

            target_speaker_id = random.choice(range(len(texts)))
            non_target_speaker_ids = [i for i in range(len(texts)) if i != target_speaker_id]
            text = texts[target_speaker_id]

            if self.use_purity_weighted_targets:
                # Derives STNO-style (Silence/Target/Non-target/Overlap) frame probabilities
                # from each speaker's own boundary-softened activity probability d(s,t) =
                # speaker_targets[s], using the same independence-assumption formula as
                # DiCoW (Polok et al., "Diarization-Conditioned Whisper"):
                #   P_S = prod_s(1 - d(s,t))                      -- nobody active
                #   P_T = d(target,t) * prod_{s!=target}(1-d(s,t)) -- target active, alone
                #   P_O = d(target,t) - P_T                        -- target active, not alone
                #   P_N = (1 - P_S) - d(target,t)                  -- someone else active, target isn't
                # These four sum to 1 by construction (verified in test_purity_weighting.py).
                d_target = speaker_targets[target_speaker_id]
                prod_others_silent = torch.prod(1 - speaker_targets[non_target_speaker_ids], dim=0)
                p_t = d_target * prod_others_silent
                p_o = d_target - p_t
                p_silence = prod_others_silent * (1 - d_target)
                p_n = 1 - p_silence - d_target

                lam = self.lambda_overlap_weight
                speaker_target = p_t + lam * p_o
                bg_speaker_target = (1 - lam) * p_o + p_n
            else:
                speaker_target = speaker_targets[target_speaker_id]
                bg_speaker_target = speaker_targets[non_target_speaker_ids].sum(dim=0) > 0

            tokens.append(torch.as_tensor(self.tokenizer(text, cut.supervisions[0].language)))
            spk_targets.append(speaker_target)
            bg_spk_targets.append(bg_speaker_target)

        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_vectors(spk_targets, padding_value=0)
        bg_spk_targets = collate_vectors(bg_spk_targets, padding_value=0)

        return audio, audio_lens, tokens, token_lens, spk_targets, bg_spk_targets
