"""
D-WaveNet: Main Entry Point
=============================
Provides the complete execution pipeline for training, validation, and testing.
All hyperparameters are aligned with Table 7 (Appendix A) of the paper.

Usage Examples:
--------------
# Standard training and evaluation
python run.py --data_path ./data/LianYunGang.csv --dataset_name LianYunGang --pred_len 168

# Quick test with synthetic data (for code verification only)
python run.py --synthetic --pred_len 24

# Ablation: remove physics loss
python run.py --data_path ./data/LianYunGang.csv --dataset_name LianYunGang --gamma 0.0 --ablation no_phy

# Ablation: remove KDCM (use --ablation no_kdcm)
python run.py --data_path ./data/LianYunGang.csv --dataset_name LianYunGang --ablation no_kdcm
"""

import argparse
import torch
import random
import numpy as np
from exp.exp_main import ExpMain


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='D-WaveNet: Physics-Guided Transformer for SWH Forecasting'
    )

    # ----------------------------------------------------------------
    # Data Configuration
    # ----------------------------------------------------------------
    parser.add_argument('--data_path', type=str, default='./data/LianYunGang.csv',
                        help='Path to CSV file containing SWH observations')
    parser.add_argument('--dataset_name', type=str, default='LianYunGang',
                        help='Dataset identifier (ShiDao / XiaoMaiDao / LianYunGang)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for code verification (will NOT reproduce paper results)')
    parser.add_argument('--mean_depth', type=float, default=8.0,
                        help='Mean water depth at station in meters (ShiDao:5, XiaoMaiDao:8, LianYunGang:8)')

    # ----------------------------------------------------------------
    # Forecasting Task
    # ----------------------------------------------------------------
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input look-back window (hours). Default: 96 (4 days)')
    parser.add_argument('--pred_len', type=int, default=168,
                        help='Prediction horizon (hours). Options: 24, 48, 96, 168')

    # ----------------------------------------------------------------
    # Decomposition Module (Section 3.2.1)
    # ----------------------------------------------------------------
    parser.add_argument('--ma_window', type=int, default=12,
                        help='Causal moving average window for wind-sea/swell separation')
    parser.add_argument('--wavelet', type=str, default='db4',
                        help='Wavelet basis function (default: db4)')
    parser.add_argument('--decomp_level', type=int, default=3,
                        help='Wavelet decomposition level J (default: 3)')

    # ----------------------------------------------------------------
    # WCFT Architecture (Section 3.2.2)
    # ----------------------------------------------------------------
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer latent dimension (default: 512)')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='Number of encoder blocks (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (default: 0.1)')

    # ----------------------------------------------------------------
    # KDCM Configuration (Section 3.2.3)
    # ----------------------------------------------------------------
    parser.add_argument('--kdcm_dim', type=int, default=128,
                        help='Hidden dimension for KDCM 1D-Conv channels')

    # ----------------------------------------------------------------
    # Physics-Guided Regularization (Eq. 12)
    # ----------------------------------------------------------------
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Physics loss weight γ (default: 0.5). Set 0 for pure data-driven.')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                        help='Energy smoothness regularization weight')

    # ----------------------------------------------------------------
    # Training Protocol (Section 4.2.3)
    # ----------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='Maximum training epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader num_workers')

    # ----------------------------------------------------------------
    # Device Configuration
    # ----------------------------------------------------------------
    parser.add_argument('--use_gpu', action='store_true', dest='use_gpu', default=True,
                        help='Use GPU for training (default: True)')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                        help='Force CPU-only execution')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')

    # ----------------------------------------------------------------
    # Output Configuration
    # ----------------------------------------------------------------
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory for checkpoints and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # ----------------------------------------------------------------
    # Ablation Study Flags (Section 4.5)
    # ----------------------------------------------------------------
    parser.add_argument('--ablation', type=str, default=None,
                        choices=[None, 'no_lambda', 'no_wcft', 'no_kdcm', 'no_phy'],
                        help='Ablation mode: '
                             'no_lambda = Model A (remove λ factor), '
                             'no_wcft = Model B (decoupled encoders), '
                             'no_kdcm = Model C (pure regression, no kinematic features), '
                             'no_phy = Model D (remove physics loss)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # ----------------------------------------------------------------
    # Handle ablation configurations
    # ----------------------------------------------------------------
    if args.ablation == 'no_phy':
        print("[ABLATION] Model D: Removing physics-guided loss (γ = 0)")
        args.gamma = 0.0
        args.lambda_smooth = 0.0
    elif args.ablation == 'no_lambda':
        print("[ABLATION] Model A: Energy Dissipation Factor will be frozen at 0")
    elif args.ablation == 'no_wcft':
        print("[ABLATION] Model B: Using decoupled encoders (no cross-scale attention)")
    elif args.ablation == 'no_kdcm':
        print("[ABLATION] Model C: Using scalar regression (no kinematic features)")

    # ----------------------------------------------------------------
    # Print configuration
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  D-WaveNet Configuration")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:25s}: {value}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Run experiment
    # ----------------------------------------------------------------
    exp = ExpMain(args)
    results = exp.run()

    print("\n" + "=" * 60)
    print("  Experiment Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
