#!/bin/bash
# ============================================================================
# Extreme Event Robustness Evaluation
# ============================================================================
# Usage: bash scripts/run_typhoon_eval.sh
#
# This script trains D-WaveNet on each dataset and evaluates robustness
# during extreme sea states. The automated evaluation in exp_main.py
# extracts samples exceeding the 99th percentile SWH threshold and
# computes event-level metrics (Event MSE, peak error, peak lag).
#
# NOTE: The specific Typhoon In-Fa event window analysis in Table 4
# of the manuscript was performed via manual time-window extraction
# using IBTrACS records (July 2021). The automated p99 extraction
# here serves as a statistical proxy for extreme event robustness.
#
# The test set (2021-2022) contains the Typhoon In-Fa period.
# ============================================================================

set -e

echo "============================================"
echo "  D-WaveNet: Typhoon Event Evaluation"
echo "============================================"

GPU=0

# Train with 96h horizon on each dataset, then evaluate
for dataset in ShiDao XiaoMaiDao LianYunGang; do
    echo ""
    echo "  Evaluating Typhoon response: ${dataset}"

    DEPTH=8.0
    if [ "${dataset}" = "ShiDao" ]; then
        DEPTH=5.0
    fi

    python run.py \
        --data_path "./data/${dataset}.csv" \
        --dataset_name "${dataset}_typhoon" \
        --mean_depth "${DEPTH}" \
        --seq_len 96 \
        --pred_len 96 \
        --gpu "${GPU}" \
        --seed 42

done

echo ""
echo "============================================"
echo "  Typhoon evaluation complete!"
echo "  Use predictions.npy and ground_truth.npy"
echo "  to generate Figure 9 (typhoon time series)"
echo "============================================"
