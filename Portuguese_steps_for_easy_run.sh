#python scripts/data_prep/build_pt_test_pooled.py --video Data_Portuguese/video/Brazil/Brazil_portuguese_01.mp4 --attribution Data_Portuguese/transcript/Brazil/Brazil_portuguese_01_speaker_attribution_final.json --output_dir Data_Portuguese/test_pooled

#python scripts/data_prep/build_pt_test_pooled.py --video Data_Portuguese/video/Brazil/Brazil_portuguese_02.mp4 --attribution Data_Portuguese/transcript/Brazil/Brazil_portuguese_02_speaker_attribution_final.json --output_dir Data_Portuguese/test_pooled

#python scripts/data_prep/build_pt_test_pooled.py --video Data_Portuguese/video/Brazil/Brazil_portuguese_03.mp4 --attribution Data_Portuguese/transcript/Brazil/Brazil_portuguese_03_speaker_attribution_final.json --output_dir Data_Portuguese/test_pooled

#python scripts/data_prep/build_pt_test_pooled.py --video Data_Portuguese/video/Brazil/Brazil_portuguese_04.mp4 --attribution Data_Portuguese/transcript/Brazil/Brazil_portuguese_04_speaker_attribution_final.json --output_dir Data_Portuguese/test_pooled

#python scripts/data_prep/build_pt_test_pooled.py --video Data_Portuguese/video/Brazil/Brazil_portuguese_05.mp4 --attribution Data_Portuguese/transcript/Brazil/Brazil_portuguese_05_speaker_attribution_final.json --output_dir Data_Portuguese/test_pooled

#PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
#    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \
#    --checkpoint multitalker_finetune_experiments/multitalker_pt_cmltts_aws/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.4855-epoch=42.ckpt \
#    --ref_dir Data_Portuguese/test_pooled \
#    --pred_dir Data_Portuguese/test_pooled_oracle_preds \
#    --dummy_cuts_path Data_Portuguese/multitalker_train_data_small/test_cuts.jsonl.gz \
#    --pretrained_model nvidia/parakeet-tdt-0.6b-v3

#PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/data_prep/run_sortformer_batch.py \
#    --wav_dir Data_Portuguese/test_pooled \
#    --output_dir Data_Portuguese/test_pooled_sortformer_preds

#PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
#    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \
#    --checkpoint multitalker_finetune_experiments/multitalker_pt_cmltts_aws/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.4855-epoch=42.ckpt \
#    --ref_dir Data_Portuguese/test_pooled \
#    --pred_dir Data_Portuguese/test_pooled_sortformer_preds \
#    --dummy_cuts_path Data_Portuguese/multitalker_train_data_small/test_cuts.jsonl.gz \
#    --pretrained_model nvidia/parakeet-tdt-0.6b-v3

PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \
    --checkpoint multitalker_finetune_experiments/multitalker_pt_cmltts_aws/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.4855-epoch=42.ckpt \
    --ref_dir Data_Portuguese/test_Europe_pooled \
    --pred_dir Data_Portuguese/test_Europe_pooled_oracle_preds \
    --dummy_cuts_path Data_Portuguese/multitalker_train_data_small/test_cuts.jsonl.gz \
    --pretrained_model nvidia/parakeet-tdt-0.6b-v3

#PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/data_prep/run_sortformer_batch.py \
#    --wav_dir Data_Portuguese/test_Europe_pooled \
#    --output_dir Data_Portuguese/test_Europe_pooled_sortformer_preds

#PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/evaluate/evaluate_checkpoint_sortformer_conditioned.py \
#    --overrides conf/multitalker_finetune_overrides_aws_cmltts_pt.yaml \
#    --checkpoint multitalker_finetune_experiments/multitalker_pt_cmltts_aws/checkpoints/multitalker_pt_cmltts_aws--val_wer=0.4855-epoch=42.ckpt \
#    --ref_dir Data_Portuguese/test_Europe_pooled \
#    --pred_dir Data_Portuguese/test_Europe_pooled_sortformer_preds \
#    --dummy_cuts_path Data_Portuguese/multitalker_train_data_small/test_cuts.jsonl.gz \
#    --pretrained_model nvidia/parakeet-tdt-0.6b-v3

#PYTHONPATH=. LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python scripts/train/train_multitalker_aws.py \
#    --overrides conf/multitalker_finetune_overrides_aws_cmltts_large_pt.yaml \
#    --train_cuts Data_Portuguese/multitalker_train_data_large/train_cuts.jsonl.gz \
#    --val_cuts Data_Portuguese/multitalker_train_data_large/dev_cuts.jsonl.gz \
#    --test_cuts Data_Portuguese/multitalker_train_data_large/test_cuts.jsonl.gz \
#    --pretrained_model nvidia/parakeet-tdt-0.6b-v3
