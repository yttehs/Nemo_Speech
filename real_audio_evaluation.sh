for wav in Data/real_audio/*.wav; do
    clip=$(basename "$wav" .wav)
    rttm="Data/real_audio_sortformer_preds/${clip}.rttm"
    echo "=== $clip ==="
    LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib" python evaluate_checkpoint_sortformer_conditioned_real_audio.py \
        --overrides conf/multitalker_finetune_overrides_aws.yaml \
        --checkpoint "multitalker_finetune_experiments/multitalker_pt_adapter_aws/2026-08-13_10-22-31/checkpoints/multitalker_pt_adapter_aws--val_wer=0.6957-epoch=46.ckpt" \
        --wav_path "$wav" \
        --sortformer_rttm "$rttm" \
        --dummy_cuts_path "Data/test_pooled/test_cuts.jsonl.gz"
done
