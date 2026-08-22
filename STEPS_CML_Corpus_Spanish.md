## Copy FastMSS prepared data

```bash
cp -r /data/home/vishwas/Workspace/FastMSS/Data_Spanish/multitalker_train_data Data_Spanish/.
cp -r /data/home/vishwas/Workspace/FastMSS/Data_Spanish/fastmss_final Data_Spanish/.
```

## Generate the configuration yaml file
```bash
Note: We will have to generate the yaml file based on the data being used. Few hyperparamters need to be set after checking the data sample duration.

python scripts/conf/generate_training_config.py \
    --template conf/multitalker_finetune_overrides_template.yaml \
    --train_cuts Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz \
    --dev_cuts Data_Spanish/multitalker_train_data/dev_cuts.jsonl.gz \
    --test_cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
    --exp_name multitalker_es_cmltts_aws \
    --output conf/multitalker_finetune_overrides_aws_cmltts_es.yaml \
    --batch_duration 60

# You can update the batch duration if you get GPU memeory issues.
```

## If needed, you can recalculate the Train/Dev/Test statistics 

```bash
python scripts/statistics/compute_overlap_stats.py \
    --cuts Spanish_train=Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz \
    --cuts Spanish_test=Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
    --cuts Spanish_dev=Data_Spanish/multitalker_train_data/dev_cuts.jsonl.gz
```

## Train the model
```bash
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/train/train_multitalker_aws.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_es.yaml \
    --train_cuts Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz \
    --val_cuts Data_Spanish/multitalker_train_data/dev_cuts.jsonl.gz \
    --test_cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
    --pretrained_model nvidia/parakeet-tdt-0.6b-v3
```

## Evaluation


### Tier 1: With respect to MFA ground truth

```bash
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint.py 
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_es.yaml 
    --checkpoint "multitalker_finetune_experiments/multitalker_es_cmltts_aws/checkpoints/multitalker_es_cmltts_aws--val_wer=0.2798-epoch=36.ckpt"  
    --test_cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz 

Note: Repeat this step on dev set as well.
```

### Tier 2: Sortformer Diarization labels, compared with MFA ground truth labels for DER, and also used for ASR

```bash
1. From the FastMSS pipeline we have generated the "test_cuts.jsonl.gz" file. In order to get speaker segments from Sortformer (run_sortformer_batch.py), we would need the folder to be in a particular format. The following step served that purpose.
python scripts/data_prep/build_test_pooled_from_cuts.py \
    --cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
    --output_dir Data_Spanish/multitalker_train_data/test_pooled

# 2. Now run Sortformer over the "test_pooled" folder from the previous step.
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/data_prep/run_sortformer_batch.py \
    --wav_dir Data_Spanish/multitalker_train_data/test_pooled \
    --output_dir Data_Spanish/multitalker_train_data/test_pooled_sortformer_preds

# 3. Then the Tier 2 DER -reporting evaluation script.
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_es.yaml \
    --checkpoint "multitalker_finetune_experiments/multitalker_es_cmltts_aws/checkpoints/multitalker_es_cmltts_aws--val_wer=0.2798-epoch=36.ckpt" \
    --ref_dir Data_Spanish/multitalker_train_data/test_pooled \  
    --pred_dir Data_Spanish/multitalker_train_data/test_pooled_sortformer_preds \   
    --dummy_cuts_path Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz

Note: Repeat this step on dev set as well.
```

### Tier 3: Sortformer Diarization labels on unseen real audio (Youtube downloaded); No groundtruth MFA labels available

```bash

```

Note: The imports - PYTHONPATH=. and LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib", were quick fixes to work on my system. You may not need them. There would be a better and neater way to deal with this. Feel free to update accordingly.
