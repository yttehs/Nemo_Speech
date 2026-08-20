## Copy FastMSS prepared data
cp -r /data/home/vishwas/Workspace/FastMSS/Data_Spanish/multitalker_train_data Data_Spanish/.
cp -r /data/home/vishwas/Workspace/FastMSS/Data_Spanish/fastmss_final Data_Spanish/.

## Generate the configuration yaml file
```bash
Note: We will have to generate the yaml file based on the data being used. Few hyperparamters need to be set after checking the data sample duration.

python generate_training_config.py \
    --template conf/multitalker_finetune_overrides_template.yaml \
    --train_cuts Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz \
    --dev_cuts Data_Spanish/multitalker_train_data/dev_cuts.jsonl.gz \
    --test_cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
    --exp_name multitalker_es_cmltts_aws \
    --output conf/multitalker_finetune_overrides_aws_cmltts_es.yaml    
```

## Train the model
```bash
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python train_multitalker_aws.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_es.yaml \
    --train_cuts Data_Spanish/multitalker_train_data/train_cuts.jsonl.gz \
    --val_cuts Data_Spanish/multitalker_train_data/dev_cuts.jsonl.gz \
    --test_cuts Data_Spanish/multitalker_train_data/test_cuts.jsonl.gz \
    --pretrained_model nvidia/parakeet-tdt-0.6b-v3
```

## Evaluation


### Tier 1: With respect to MFA ground truth

```bash
LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python evaluate_checkpoint.py 
    --overrides conf/multitalker_finetune_overrides_aws_cmltts.yaml 
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
python evaluate_checkpoint_sortformer_conditioned.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts.yaml \
    --checkpoint "multitalker_finetune_experiments/multitalker_pt_cmltts_aws/2026-08-18_02-52-30/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.5244-epoch=44.ckpt" \
    --ref_dir Data/CML_Corpus/multitalker_train_data/test_pooled \  
    --pred_dir Data/CML_Corpus/multitalker_train_data/test_pooled_sortformer_preds \   
    --dummy_cuts_path Data/CML_Corpus/multitalker_train_data/test_cuts.jsonl.gz
```

### Tier 3: Sortformer Diarization labels on unseen real audio (Youtube downloaded); No groundtruth MFA labels available

```bash

```
