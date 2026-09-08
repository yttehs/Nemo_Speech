## Copy FastMSS prepared data

```bash
cp -r /data/home/vishwas/Workspace/AliMeeting/Data_alimeeting/lhotse_cuts/*jsonl.gz Data_alimeeting/multitalker_train_data/. 
ln -s /data/home/vishwas/Workspace/AliMeeting/Data_alimeeting/chunked_audio Data_alimeeting/.
```

## Generate the configuration yaml file
```bash
Note: We will have to generate the yaml file based on the data being used. Few hyperparamters need to be set after checking the data sample duration. Here we will use "cmn" to represent Alimeeting Chinese

python scripts/conf/generate_training_config.py \
    --template conf/multitalker_finetune_overrides_template.yaml \
    --train_cuts Data_alimeeting/multitalker_train_data/train_cuts.jsonl.gz \
    --dev_cuts Data_alimeeting/multitalker_train_data/eval_cuts.jsonl.gz \
    --test_cuts Data_alimeeting/multitalker_train_data/test_cuts.jsonl.gz \
    --exp_name multitalker_cmn_purity_lam025_aws \
    --output conf/multitalker_finetune_overrides_aws_purity_lam025_cmn.yaml \
    --batch_duration 90 \
    --use_cer \
    --num_mel_frame_per_asr_frame 4 \
    --use_purity_weighted_targets --lambda_overlap_weight 0.25

# You can update the batch duration if you get GPU memeory issues.
```

## If needed, you can recalculate the Train/Dev/Test statistics 

```bash
python scripts/statistics/compute_overlap_stats.py \
    --cuts Chinese_train=Data_alimeeting/multitalker_train_data/train_cuts.jsonl.gz \
    --cuts Chinese_test=Data_alimeeting/multitalker_train_data/test_cuts.jsonl.gz \
    --cuts Chinese_dev=Data_alimeeting/multitalker_train_data/eval_cuts.jsonl.gz
```

## Check the vocab difference between the pretrained model and the Alimeeting text vocabulary

```bash
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/statistics/check_vocab_coverage.py \
    --pretrained_model nvidia/stt_zh_conformer_transducer_large \
    --cuts Data_alimeeting/multitalker_train_data/train_cuts.jsonl.gz
```


## I wanted to train a model with a subset of Alimeeting data. So here is the script to extract a 20 hour subset from the train data

```bash
python scripts/data_prep/create_alimeeting_subset.py \
    --cuts Data_alimeeting/multitalker_train_data/train_cuts.jsonl.gz \
    --output Data_alimeeting/multitalker_train_data/train_cuts_subset20h.jsonl.gz \
    --target_hours 20.0

Note: Once the subset is created, I would suggest you verify the overlap stats using scripts/statistics/compute_overlap_stats.py
Note: If the train set changes, make sure to regenerate the config file - scripts/conf/generate_training_config.py
```


## Train the model
```bash
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/train/train_multitalker_aws.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmn.yaml \
    --train_cuts Data_alimeeting/multitalker_train_data/train_cuts.jsonl.gz \
    --val_cuts Data_alimeeting/multitalker_train_data/eval_cuts.jsonl.gz \
    --test_cuts Data_alimeeting/multitalker_train_data/test_cuts.jsonl.gz \
    --pretrained_model nvidia/stt_zh_conformer_transducer_large \
    --character_based
```

## Evaluation


### Tier 1: With respect to MFA ground truth

```bash
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint.py 
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_fr.yaml 
    --checkpoint "multitalker_finetune_experiments/multitalker_fr_cmltts_aws/checkpoints/multitalker_fr_cmltts_aws--val_wer=0.2798-epoch=36.ckpt"  
    --test_cuts Data_French/multitalker_train_data/test_cuts.jsonl.gz 

Note: Repeat this step on dev set as well.
```

### Tier 2: Sortformer Diarization labels, compared with MFA ground truth labels for DER, and also used for ASR

```bash
1. From the FastMSS pipeline we have generated the "test_cuts.jsonl.gz" file. In order to get speaker segments from Sortformer (run_sortformer_batch.py), we would need the folder to be in a particular format. The following step served that purpose.
python scripts/data_prep/build_test_pooled_from_cuts.py \
    --cuts Data_alimeeting/multitalker_train_data/test_cuts.jsonl.gz \
    --output_dir Data_alimeeting/multitalker_train_data/test_pooled

# 2. Now run Sortformer over the "test_pooled" folder from the previous step.
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/data_prep/run_sortformer_batch.py \
    --wav_dir Data_alimeeting/multitalker_train_data/test_pooled \
    --output_dir Data_alimeeting/multitalker_train_data/test_pooled_sortformer_preds

# 3. Then the Tier 2 DER -reporting evaluation script.
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmn.yaml  \
    --checkpoint "multitalker_finetune_experiments/multitalker_cmn_aws/checkpoints/multitalker_cmn_aws--val_wer=0.8030-epoch=22.ckpt" \
    --pretrained_model nvidia/stt_zh_conformer_transducer_large \
    --ref_dir Data_alimeeting/multitalker_train_data/test_pooled \  
    --pred_dir Data_alimeeting/multitalker_train_data/test_pooled_sortformer_preds \   
    --dummy_cuts_path Data_alimeeting/multitalker_train_data/test_cuts.jsonl.gz \
    --char_level_scoring

Note: Repeat this step on dev set as well.
```

### Tier 3: Sortformer Diarization labels on unseen real audio (Youtube downloaded); No groundtruth MFA labels available

```bash

```

Note: The imports - PYTHONPATH=. and LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib", were quick fixes to work on my system. You may not need them. There would be a better and neater way to deal with this. Feel free to update accordingly.
