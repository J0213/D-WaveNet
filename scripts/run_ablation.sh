#!/bin/bash
# ============================================================================
# Reproduce Table 5: Ablation Study on LianYunGang Dataset
# ============================================================================
# Usage: bash scripts/run_ablation.sh
#
# Runs 5 model variants:
#   - Model A: w/o λ (Energy Dissipation Factor)
#   - Model B: w/o WCFT (Cross-Scale Interaction)
#   - Model C: w/o KDCM (Kinematic-Dynamic Module)
#   - Model D: w/o L_PHY (Physics-Guided Loss)
#   - Full D-WaveNet
# ============================================================================

set -e

echo "============================================"
echo "  D-WaveNet: Reproducing Table 5 (Ablation)"
echo "============================================"

DATASET="LianYunGang"
DATA_PATH="./data/LianYunGang.csv"
DEPTH=8.0
GPU=0
HORIZONS=(24 48 96 168)

# ---- Model A: w/o lambda (Energy Dissipation Factor) ----
for pred_len in "${HORIZONS[@]}"; do
    echo ""
    echo "  [Model A] w/o λ | Horizon: ${pred_len}h"
    python run.py \
        --data_path "${DATA_PATH}" --dataset_name "${DATASET}_ablation_A" \
        --mean_depth "${DEPTH}" --pred_len "${pred_len}" \
        --ablation no_lambda --gpu "${GPU}" --seed 42
done

# ---- Model B: w/o WCFT (Decoupled Encoders) ----
for pred_len in "${HORIZONS[@]}"; do
    echo ""
    echo "  [Model B] w/o WCFT | Horizon: ${pred_len}h"
    python run.py \
        --data_path "${DATA_PATH}" --dataset_name "${DATASET}_ablation_B" \
        --mean_depth "${DEPTH}" --pred_len "${pred_len}" \
        --ablation no_wcft --gpu "${GPU}" --seed 42
done

# ---- Model C: w/o KDCM (Pure Regression) ----
for pred_len in "${HORIZONS[@]}"; do
    echo ""
    echo "  [Model C] w/o KDCM | Horizon: ${pred_len}h"
    python run.py \
        --data_path "${DATA_PATH}" --dataset_name "${DATASET}_ablation_C" \
        --mean_depth "${DEPTH}" --pred_len "${pred_len}" \
        --ablation no_kdcm --gpu "${GPU}" --seed 42
done

# ---- Model D: w/o Physics Loss (Unconstrained) ----
for pred_len in "${HORIZONS[@]}"; do
    echo ""
    echo "  [Model D] w/o L_PHY | Horizon: ${pred_len}h"
    python run.py \
        --data_path "${DATA_PATH}" --dataset_name "${DATASET}_ablation_D" \
        --mean_depth "${DEPTH}" --pred_len "${pred_len}" \
        --ablation no_phy --gpu "${GPU}" --seed 42
done

# ---- Full D-WaveNet ----
for pred_len in "${HORIZONS[@]}"; do
    echo ""
    echo "  [Full Model] D-WaveNet | Horizon: ${pred_len}h"
    python run.py \
        --data_path "${DATA_PATH}" --dataset_name "${DATASET}_full" \
        --mean_depth "${DEPTH}" --pred_len "${pred_len}" \
        --gpu "${GPU}" --seed 42
done

echo ""
echo "============================================"
echo "  Ablation study complete!"
echo "  Results saved in: ./outputs/results/"
echo "============================================"
