# Note
```bash
I guess I have by mistake deleted the "Data" folder with the portuguese train data. For now I am recreating "Data_Portuguese", but with only the newly created (youtube based) test set. You will have to re create the other folders in the Data folder. Since the models trained using the initial data folder are present, for now creating the data folder is not a priority.
```

## Copy FastMSS prepared data

```bash
cp -r /data/home/vishwas/Workspace/FastMSS/Data/multitalker_train_data Data/CML_Corpus/.
cp -r /data/home/vishwas/Workspace/FastMSS/Data/fastmss_final Data/.
```

## Generate the configuration yaml file
```bash
Note: We will have to generate the yaml file based on the data being used. Few hyperparamters need to be set after checking the data sample duration.

python scripts/conf/generate_training_config.py \
    --template conf/multitalker_finetune_overrides_template.yaml \
    --train_cuts Data/CML_Corpus/multitalker_train_data/train_cuts.jsonl.gz \
    --dev_cuts Data/CML_Corpus/multitalker_train_data/dev_cuts.jsonl.gz \
    --test_cuts Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz \
    --exp_name multitalker_pt_cmltts_aws \
    --output conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \ 
    --batch_duration 90 \
    --use_purity_weighted_targets --lambda_overlap_weight 0.5

# You can update the batch duration if you get GPU memeory issues.
```

## If needed, you can recalculate the Train/Dev/Test statistics 

```bash
python scripts/statistics/compute_overlap_stats.py \
    --cuts Portuguese_train=Data/CML_Corpus/multitalker_train_data/train_cuts.jsonl.gz \
    --cuts Portuguese_test=Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz \
    --cuts Portuguese_dev=Data/CML_Corpus/multitalker_train_data/dev_cuts.jsonl.gz
```


## Train the model
```bash
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/train/train_multitalker_aws.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \
    --train_cuts Data/CML_Corpus/multitalker_train_data/train_cuts.jsonl.gz \
    --val_cuts Data/CML_Corpus/multitalker_train_data/dev_cuts.jsonl.gz \
    --test_cuts Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz \
    --pretrained_model nvidia/parakeet-tdt-0.6b-v3
```

## Evaluation


### Tier 1: With respect to MFA ground truth

```bash
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint.py 
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml 
    --checkpoint "multitalker_finetune_experiments/multitalker_pt_cmltts_aws/2026-08-18_02-52-30/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.5244-epoch=44.ckpt"  
    --test_cuts Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz 
```

### Tier 2: Sortformer Diarization labels, compared with MFA ground truth labels for DER, and also used for ASR

```bash
1. From the FastMSS pipeline we have generated the "test_cuts.jsonl.gz" file. In order to get speaker segments from Sortformer (run_sortformer_batch.py), we would need the folder to be in a particular format. The following step served that purpose.
python scripts/data_prep/build_test_pooled_from_cuts.py \
    --cuts Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz \
    --output_dir Data/CML_Corpus/multitalker_train_data/test_pooled

# 2. Now run Sortformer over the "test_pooled" folder from the previous step.
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/data_prep/run_sortformer_batch.py \
    --wav_dir Data/CML_Corpus/multitalker_train_data/test_pooled \
    --output_dir Data/CML_Corpus/multitalker_train_data/test_pooled_sortformer_preds

# 3. Then the Tier 2 DER -reporting evaluation script.
PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \
    --checkpoint "multitalker_finetune_experiments/multitalker_pt_cmltts_aws/2026-08-18_02-52-30/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.5244-epoch=44.ckpt" \
    --ref_dir Data/CML_Corpus/multitalker_train_data/test_pooled \  
    --pred_dir Data/CML_Corpus/multitalker_train_data/test_pooled_sortformer_preds \   
    --dummy_cuts_path Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz
```

### Tier 3: Sortformer Diarization labels on unseen real audio (Youtube downloaded); No groundtruth MFA labels available

```bash

```

Note: The imports - PYTHONPATH=. and LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib", were quick fixes to work on my system. You may not need them. There would be a better and neater way to deal with this. Feel free to update accordingly.
