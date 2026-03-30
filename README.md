# D-WaveNet: A Physics-Guided Transformer with Cross-Scale Interaction for Significant Wave Height Forecasting

## Overview

This repository provides the PyTorch implementation of **D-WaveNet**, a physics-guided constrained Transformer framework for Significant Wave Height (SWH) forecasting. The framework integrates:

- **Wavelet-Component Fusion Transformer (WCFT):** Explicitly models the non-linear energy modulation between high-frequency wind-seas and low-frequency swells via a Cross-Scale Attention mechanism.
- **Kinematic-Dynamic Coupled Module (KDCM):** Internalizes wave acceleration as a precursor signal to eliminate hysteresis, and enforces physics-guided regularization (depth-induced breaking limits & Stokes steepness constraints) to prevent physically invalid predictions.

## Repository Structure

```
D-WaveNet_Project/
├── data/                       # Data directory (place datasets here)
│   └── README_DATA.md          # Instructions for obtaining real buoy data
├── data_provider/              # Data loading & strictly causal preprocessing
│   ├── __init__.py
│   └── data_loader.py          # Causal moving average, DWT decomposition, Z-score normalization
├── models/                     # Core network architectures
│   ├── __init__.py
│   └── D_WaveNet.py            # Full implementation of WCFT (with Cross-Scale Attention) and KDCM
├── exp/                        # Experiment runner with full train/val/test pipeline
│   ├── __init__.py
│   └── exp_main.py             # Training loop, validation, testing, metric logging
├── utils/                      # Evaluation metrics & custom losses
│   ├── __init__.py
│   ├── metrics.py              # MSE, MAE, NSE/R², Skill Scores, RMSE
│   ├── physics_loss.py         # Physics-Guided Regularization Loss (depth + steepness + energy smoothness)
│   └── tools.py                # EarlyStopping, learning rate adjustment utilities
├── scripts/                    # Scripts to reproduce paper results
│   ├── run_all_experiments.sh  # Reproduces D-WaveNet rows in Table 2
│   ├── run_ablation.sh         # Reproduces Table 5 (ablation study)
│   └── run_typhoon_eval.sh     # Reproduces Table 4 (extreme event evaluation)
├── run.py                      # Main entry point with full training pipeline
├── requirements.txt            # Python environment dependencies
└── README.md                   # This file
```

## Environment Setup

```bash
# Create conda environment
conda create -n dwavenet python=3.8 -y
conda activate dwavenet

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.12+ (with CUDA 11.3 for GPU acceleration)
- NVIDIA GPU with ≥8GB VRAM (RTX 3090 recommended)

## Data Preparation

The datasets used in this study are from the **National Marine Data Center** (http://mds.nmdis.org.cn/). See `data/README_DATA.md` for detailed instructions on data acquisition and preprocessing.

**Quick start with provided sample data:**
```bash
# Place your CSV files in the data/ directory with the following format:
# Column: 'SWH' (Significant Wave Height in meters)
# Temporal resolution: hourly
# Example: data/ShiDao.csv, data/XiaoMaiDao.csv, data/LianYunGang.csv
```

## Quick Start

### Single Experiment
```bash
python run.py \
    --data_path ./data/LianYunGang.csv \
    --dataset_name LianYunGang \
    --seq_len 96 \
    --pred_len 168 \
    --train_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --gamma 0.5 \
    --gpu 0
```

### Reproduce All Paper Results
```bash
# Table 2 (D-WaveNet rows): Full results across all datasets and horizons
bash scripts/run_all_experiments.sh

# Table 5: Ablation study on LianYunGang
bash scripts/run_ablation.sh

# Table 4: Typhoon event evaluation
bash scripts/run_typhoon_eval.sh
```

**Note on Baselines:** The eight baseline models (ARIMA, LSTM, GRU, Informer, Autoformer, FEDformer, TimesNet, MultiWaveNet) referenced in Table 2 follow their respective original codebases and are not bundled in this repository. Informer: https://github.com/zhouhaoyi/Informer2020; Autoformer: https://github.com/thuml/Autoformer; FEDformer: https://github.com/MAZiqing/FEDformer; TimesNet: https://github.com/thuml/Time-Series-Library; MultiWaveNet: https://github.com/TianG21/MultiWaveNet.

## Key Design Choices Addressing Reproducibility

### 1. Strictly Causal Preprocessing (No Data Leakage)
All preprocessing operations are **strictly causal**, operating only within each sample's historical look-back window:
- **Moving average decomposition**: Uses one-sided (causal) convolution — only past and present values are used.
- **Wavelet decomposition**: Uses `mode='zero'` boundary extension to prevent future information leakage.
- **Z-score normalization**: Statistics (mean, std) are computed **exclusively from the training set** and applied to validation/test sets.

### 2. Cross-Scale Interaction (Not Simple Summation)
The WCFT implements a genuine dual-stage attention mechanism:
- **Intra-Scale Self-Attention**: Each wavelet sub-band (D1, D2, D3, A3) first captures its own temporal dynamics independently.
- **Inter-Scale Cross-Attention**: Each sub-band's features serve as queries, while concatenated features from all other sub-bands serve as keys and values, enabling explicit cross-frequency energy modulation modeling.

### 3. Complete Physics-Guided Regularization
The physics loss (Eq. 12 in the paper) includes all three terms:
- **Depth-induced breaking penalty**: `ReLU(H_pred - 0.78d)²`
- **Stokes steepness penalty**: `ReLU(ξ_pred - 1/7)²`
- **Energy smoothness regularization**: `‖ΔE_sys‖²` to discourage physically impossible energy jumps

**Important:** The depth and steepness penalties operate in **physical space (meters)**, not in the Z-score normalized space. The loss function inverse-transforms predictions before computing these constraints, ensuring the physical thresholds are meaningful.

### 4. Terminology Clarification
Following best practices in the physics-informed ML literature, the loss function is more precisely described as **"physics-guided regularization"** rather than a fully physics-informed approach that incorporates governing PDEs. The approach applies soft physical constraints to penalize predictions exceeding known hydrodynamic limits.

### 5. NSE / R² Equivalence
The R² metric reported in this study uses the formulation `1 - SS_res/SS_tot` (Eq. 15), which is **mathematically identical** to the Nash-Sutcliffe Efficiency (NSE). Both metrics are computed and reported in the evaluation output.

## Hyperparameter Configuration (Table 7 in Paper)

| Module | Hyperparameter | Optimal Value |
|--------|---------------|---------------|
| Data | Look-back Window (L_in) | 96 hours |
| Data | Batch Size | 32 |
| Decomposition | Wavelet Base | db4 |
| Decomposition | Decomposition Level (J) | 3 |
| WCFT | Encoder Depth (N) | 3 |
| WCFT | Latent Dimension (d_model) | 512 |
| WCFT | Attention Heads (h) | 8 |
| WCFT | Dropout Rate | 0.1 |
| Physics | Loss Weight (γ) | 0.5 |
| Physics | Steepness Limit (ξ_limit) | 1/7 |
| Optimization | Learning Rate | 1e-4 |
| Optimization | Training Epochs | 50 (with Early Stopping, patience=5) |

## Citation

If you find this work useful, please cite:
```bibtex
@article{jiang2026dwavenet,
  title={D-WaveNet: A Physics-Guided Transformer with Cross-Scale Interaction for Significant Wave Height Forecasting},
  author={Jiang, Wentao and Zhang, Dabin and Zhao, Ming},
  journal={Ocean Engineering},
  year={2026}
}
```

## License

This project is released under the MIT License.
