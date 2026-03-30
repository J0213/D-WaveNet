#!/bin/bash
# ============================================================================
# Reproduce D-WaveNet results in Table 2 across all datasets and horizons
# ============================================================================
# Usage: bash scripts/run_all_experiments.sh
#
# This script trains D-WaveNet on all three datasets with all four prediction
# horizons (24h, 48h, 96h, 168h), reproducing the D-WaveNet rows in Table 2.
#
# NOTE: Baseline models (ARIMA, LSTM, GRU, Informer, Autoformer, FEDformer,
#       TimesNet, MultiWaveNet) follow their respective original codebases
#       and are not bundled in this repository. See README.md for details.
#
# Prerequisites:
#   - Place real buoy data in data/ directory (see data/README_DATA.md)
#   - Activate the conda environment: conda activate dwavenet
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "  D-WaveNet: Reproducing Table 2 Results"
echo "============================================"

# Datasets and their mean depths
declare -A DEPTHS
DEPTHS[ShiDao]=5.0
DEPTHS[XiaoMaiDao]=8.0
DEPTHS[LianYunGang]=8.0

# Prediction horizons
HORIZONS=(24 48 96 168)

# GPU device
GPU=0

for dataset in ShiDao XiaoMaiDao LianYunGang; do
    for pred_len in "${HORIZONS[@]}"; do
        echo ""
        echo "--------------------------------------------"
        echo "  Dataset: ${dataset} | Horizon: ${pred_len}h"
        echo "--------------------------------------------"

        python run.py \
            --data_path "./data/${dataset}.csv" \
            --dataset_name "${dataset}" \
            --mean_depth "${DEPTHS[$dataset]}" \
            --seq_len 96 \
            --pred_len "${pred_len}" \
            --d_model 512 \
            --n_heads 8 \
            --e_layers 3 \
            --dropout 0.1 \
            --gamma 0.5 \
            --lambda_smooth 0.01 \
            --batch_size 32 \
            --learning_rate 0.0001 \
            --train_epochs 50 \
            --patience 5 \
            --gpu "${GPU}" \
            --seed 42

    done
done

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results saved in: ./outputs/results/"
echo "============================================"
