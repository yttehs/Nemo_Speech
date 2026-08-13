Steps to train a Sortformer using VoxForge Portuguese database

The first thing to be done is that we need to simulation "conversation" data for portuguese. Even otherwise, Sortformer expects word level aligned data to train the models. Hence, we have to first force align all the data we have. Next simulate converstaions (Note: Even if we had conversation data, we would need them to be force aligned at word level)

The code below first accumulates all the spekaer wise data and created a json file with audio file and speaker id mapping.

This script was generated with the help of Claude
1) python scripts/data_prep/voxforge_to_manifest.py \
  --voxforge_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge-pt \
  --resampled_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge-pt-16k \
  --manifest_path /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest.json 

Next, we have to force align the data. Since we are working with Portuguese data, we will use "nvidia/stt_pt_fastconformer_hybrid_large_pc" a dedicated Portuguese ASR model to force align

The aligning script has been provided by the offcial Nemo repository.
2) python tools/nemo_forced_aligner/align.py \
    pretrained_name="nvidia/stt_pt_fastconformer_hybrid_large_pc" \
    manifest_filepath=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest.json \
    output_dir=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments

2b) Check if all the utterances have been force aligned: This script has been generated with the help of Claude
python check_alignment_coverage.py \
  --manifest /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest.json \
  --output_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments \
  --dump_missing_csv /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments/missing_details.csv

2c)Once you have checked the missing alignments, check the missing_details.csv to identify if there is anything can be done to recover those files. If not, proceed to filter out the missing alignments.
python scripts/data_prep/filter_manifest_by_alignment.py \
  --manifest /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest.json \
  --output_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments \
  --filtered_manifest_path /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest_aligned.json


Next, the CTM files have to be reformatted so that they can be processed by the Nemo Script: scripts/speaker_tasks/create_alignment_manifest.py
3a) 
python scripts/data_prep/reformat_ctm_for_alignment_manifest.py \
  --manifest /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest_aligned.json \
  --nfa_ctm_words_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments/ctm/words \
  --output_ctm_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments/ctm_words_reformatted

3b) Run the Nemo provided script against the reformatted CTM directory 
python scripts/speaker_tasks/create_alignment_manifest.py \
  --input_manifest_filepath /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_manifest_aligned.json \
  --output_manifest_filepath /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignment_manifest.json \
  --base_alignment_path /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments/ctm_words_reformatted \
  --ctm_output_directory /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignments/unused_scratch \
  --use_ctm_alignment_source

Now, on analysing the alignment files, I found that there are a couple of words that have been assigned a very long time segement by the NFA (Nemo Forced Aligner). This happens generally when the audio file is very long but has few words in it. In order to see how many such cases are present, Claude suggested the following verification script. This script also filters out those words which have been assigned a segemnet longer than they ideally should.

python scripts/data_prep/check_word_durations.py \
  --manifest  /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignment_manifest.json \
  --filtered_manifest_path  /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignment_manifest_clean.json

After filtering out the suspected erroneous alignments, we lost complete data from one speaker () and there were a couple of speakers with either one or two audio files. But this is okay since the whole pipeliine is to check for a Proof of Concept (POC) for ur Multilingual EEND + ASR pipeline.


Now, we have to simulate conversations. For this, we will use the existing Nemo scripts.

1) Since we do not have as much as data as was used in the official Nemo repo, we will use shorter session lenth of 60 sec (original 600 seconds). Also, we will work with two speaker scenario for now.
python tools/speech_data_simulator/multispeaker_simulator.py \
    data_simulator.manifest_filepath=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/voxforge_pt_alignment_manifest_clean.json \
    data_simulator.outputs.output_dir=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/simulated_sessions_2spk \
    data_simulator.session_config.num_speakers=2 \
    data_simulator.session_config.num_sessions=10 \
    data_simulator.session_config.session_length=60


Train-Val-Test Data Split Preparation
Now that we have set up the pipeline to simulate conversation sessions, we will work towards creating train-val-test speaker splits. We want to make sure that there is no speaker leakage across the three sets. So the first step is to create three disjoint sets.

For the VoxForge Portuguese data we are following the stratergy stated below:
a) Split 241 speakers → train (~180) / val (~30) / test (~30), disjoint.
b) Per split, generate: 2spk, 3spk, 4spk sessions (150/25/25 train/val/test each), plus a smaller 1spk allocation (say 75/15/15) — respecting the same speaker pool boundaries.
c) When real data arrives, fold it into test only, keeping the synthetic test set as a secondary signal rather than discarding it.
d) The following script also ensures that speakers with higher number of audio files go into val and test splits, so that we can simulate better conversations.

1) This script was generated using Claude
python scripts/data_prep/split_speakers.py \
  --manifest /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/voxforge_pt_alignment_manifest_clean.json \
  --output_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/splits \
  --val_speakers 30 \
  --test_speakers 30 

Example: python tools/speech_data_simulator/multispeaker_simulator.py data_simulator.manifest_filepath=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/splits/voxforge_pt_val_manifest.json data_simulator.outputs.output_dir=/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/val_simulated_sessions_1spk data_simulator.session_config.num_speakers=1 data_simulator.session_config.num_sessions=15 data_simulator.session_config.session_length=20

2) Once the train-test-val folders have been created, let us do one final check to ensure that there is no speaker leakage:
python scripts/data_prep/check_split_leakage.py \
  --split_csv /media/.../Portuguese/splits/speaker_split_details.csv \
  --train_dirs /media/.../train_simulated_sessions_1spk /media/.../train_simulated_sessions_2spk /media/.../train_simulated_sessions_3spk /media/.../train_simulated_sessions_4spk \
  --val_dirs   /media/.../val_simulated_sessions_1spk   /media/.../val_simulated_sessions_2spk   /media/.../val_simulated_sessions_3spk   /media/.../val_simulated_sessions_4spk \
  --test_dirs  /media/.../test_simulated_sessions_1spk  /media/.../test_simulated_sessions_2spk  /media/.../test_simulated_sessions_3spk  /media/.../test_simulated_sessions_4spk

3) Now that we have generated invidual sessions for Train, test and validation splits, we have to combine them all into a pooled train, test and val folder. However, when doing so, we will have to make sure we name the files appropriately becuase the files across these folders share the same names. For example, check the files in train_simulated_sessions_2spk and train_simulated_sessions_4spk folders.

python scripts/data_prep/pool_sessions.py \
  --split_name train \
  --bucket_dirs \
    /media/.../train_simulated_sessions_1spk \
    /media/.../train_simulated_sessions_2spk \
    /media/.../train_simulated_sessions_3spk \
    /media/.../train_simulated_sessions_4spk \
  --output_dir /media/.../train_pooled

python scripts/data_prep/pool_sessions.py \
  --split_name val \
  --bucket_dirs \
    /media/.../val_simulated_sessions_1spk \
    /media/.../val_simulated_sessions_2spk \
    /media/.../val_simulated_sessions_3spk \
    /media/.../val_simulated_sessions_4spk \
  --output_dir /media/.../val_pooled

python scripts/data_prep/pool_sessions.py \
  --split_name test \
  --bucket_dirs \
    /media/.../test_simulated_sessions_1spk \
    /media/.../test_simulated_sessions_2spk \
    /media/.../test_simulated_sessions_3spk \
    /media/.../test_simulated_sessions_4spk \
  --output_dir /media/.../test_pooled

Or you can also run "pool_all_splits.sh" script


As a final Check, check the loudenss of the created session wav files. This script was also created by Claude

python scripts/data_prep/check_session_loudness.py --wav_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/test_pooled --n 10

Okay!
Now we have simulated all possible conversation scenarios. The next step is to adapt the existing models for Postuguese MultiSpeaker ASR.

We will use a prtrained multilingual ASR model for the ASR section of the End to End Diarization module. Now we have to check how well does the speech diarization section of the Sortformer works on our simulated conversation data. For this, I have generated the following scripts using Claude.


# Using Streaming version
# 1. Run Sortformer on your simulated audio (start with a small sample to move fast)
python scripts/data_prep/run_sortformer_batch.py \
  --wav_dir /media/.../test_pooled \
  --output_dir results/test_pooled_sortformer_preds \
  --limit 40

#Using offline version
python scripts/data_prep/run_sortformer_batch.py \
  --wav_dir .../test_pooled \
  --output_dir results/test_pooled_sortformer_offline_preds \
  --model_name nvidia/diar_sortformer_4spk-v1 \
  --no-streaming \
  --limit 40

# 2. Score predictions against your simulator's own ground-truth RTTMs; the following script also uses another script der_scorer.py (which was also generated using Claude). I have copied der_scorer.py script to "scripts/data_prep/" folder.
python scripts/data_prep/score_sortformer_batch.py \
  --ref_dir /media/.../test_pooled \
  --pred_dir results/test_pooled_sortformer_preds \
  --collar 0.25

The above check gave very high DERs. This was becuase there was mismatch between the sortformer rttm predictions and the Nemo force aligned(NFA) rttms. The NFA data has longer segments including even silece sections. We have clenaed up extreme case earlier using: scripts/data_prep/check_word_durations.py (see above). On the other hand Sortformer gives better VAD. But then, when there is overallping speech and one of the speaker's audio is faint, Sortformer fails to identify that speaker. Due to these reasons, we got a very hugh DER when we compared Sortformer rttm and NFA rttm. We then checked the sortformer performacne on portuguese video in general byt using two speech segments from a Portuguese podcast downloaded from Youtube. ANd Sortformer performed quite well in identifying the voiced segments (including overlap speech). So the isse is not Sortformer doesn't work on Portuguese. The issue is with the simulated conversations. But, this should not be an issue in our ASR finetuning, becuase the simulated conversation is more indicative of a very difficult test scenario. Also, NFA choosing longer audio segments is not an issue, becuase the underlying ASR model is CTC based. SO it is alignment free. A trailing or preceeding silence segment doesn't harm the ASR performance (ideally!)

Let us start with the steps to finetune the ASR model now:

1) Prepare the train, test, valid data in the required format; script generated using Claude
python scripts/data_prep/build_lhotse_cutset.py \
  --pooled_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/train_pooled --output_cuts /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/train_pooled/train_cuts.jsonl.gz

python scripts/data_prep/build_lhotse_cutset.py \
  --pooled_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/val_pooled --output_cuts /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/val_pooled/val_cuts.jsonl.gz

python scripts/data_prep/build_lhotse_cutset.py \
  --pooled_dir /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/test_pooled --output_cuts /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/test_pooled/test_cuts.jsonl.gz

Verify the above generated cut files:

python scripts/data_prep/inspect_cutset.py --cuts_path /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/train_pooled/train_cuts.jsonl.gz --n_samples 5
python scripts/data_prep/inspect_cutset.py --cuts_path /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/val_pooled/val_cuts.jsonl.gz --n_samples 3
python scripts/data_prep/inspect_cutset.py --cuts_path /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/test_pooled/test_cuts.jsonl.gz --n_samples 3

Train/Fine-tune the model
Script and the yaml file has been generated using Claude after looking into all the scripts in the repository
1)python train_multitalker.py \
  --overrides conf/multitalker_finetune_overrides.yaml \
  --train_cuts /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/train_pooled/train_cuts.jsonl.gz \
  --val_cuts /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/val_pooled/val_cuts.jsonl.gz \
  --test_cuts /media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese/test_pooled/test_cuts.jsonl.gz
