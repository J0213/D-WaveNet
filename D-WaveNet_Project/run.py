import argparse
import torch
from models import D_WaveNet
from utils.physics_loss import PhysicsInformedLoss

def main():
    parser = argparse.ArgumentParser(description='D-WaveNet for Significant Wave Height Forecasting')
    
    # Basic Configs
    parser.add_argument('--model', type=str, required=False, default='D_WaveNet', help='Model name')
    parser.add_argument('--data', type=str, default='LianYunGang', help='Dataset name')
    
    # Forecasting Task
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=168, help='Prediction sequence length')
    
    # Model Hyperparameters (Aligned with Paper Appendix A)
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='Num of attention heads')
    parser.add_argument('--e_layers', type=int, default=3, help='Num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Physics Constraints
    parser.add_argument('--gamma', type=float, default=0.5, help='Physics loss weight')
    parser.add_argument('--wavelet', type=str, default='db4', help='Wavelet base function')
    parser.add_argument('--decomp_level', type=int, default=3, help='Wavelet decomposition level')
    
    # Optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size of training input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer initial learning rate')
    parser.add_argument('--train_epochs', type=int, default=50, help='Train epochs')
    
    args = parser.parse_args()
    print("Args in experiment:")
    print(args)

    # Initialize Model & Loss
    model = D_WaveNet.Model(args)
    criterion = PhysicsInformedLoss(gamma=args.gamma)
    
    print(f"Model {args.model} built successfully. Total Parameters: {sum(p.numel() for p in model.parameters())}")
    print("Training pipeline ready (Placeholder in open-source version).")

if __name__ == '__main__':
    main()