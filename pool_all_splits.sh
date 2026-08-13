#!/bin/bash
set -e          # stop immediately if any command fails
set -u          # treat unset variables as errors, catches typos in paths below

# --- Adjust these once, used everywhere below ---
DATA_ROOT="/media/amber/charizard/vishwas/Workspace/Clipto/Data/Voxforge/Portuguese"
SCRIPT="scripts/data_prep/pool_sessions.py"  # adjust to wherever you saved it

for SPLIT in train val test; do
    echo ""
    echo "=== Pooling ${SPLIT} ==="
    python "${SCRIPT}" \
        --split_name "${SPLIT}" \
        --bucket_dirs \
            "${DATA_ROOT}/${SPLIT}_simulated_sessions_1spk" \
            "${DATA_ROOT}/${SPLIT}_simulated_sessions_2spk" \
            "${DATA_ROOT}/${SPLIT}_simulated_sessions_3spk" \
            "${DATA_ROOT}/${SPLIT}_simulated_sessions_4spk" \
        --output_dir "${DATA_ROOT}/${SPLIT}_pooled"
done

echo ""
echo "=== All three splits pooled. Running leakage check on the result ==="
python scripts/data_prep/check_split_leakage.py \
    --split_csv "${DATA_ROOT}/splits/speaker_split_details.csv" \
    --train_dirs "${DATA_ROOT}/train_pooled" \
    --val_dirs   "${DATA_ROOT}/val_pooled" \
    --test_dirs  "${DATA_ROOT}/test_pooled"

echo ""
echo "Done."
