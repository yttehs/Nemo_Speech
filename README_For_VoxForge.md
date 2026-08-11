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
