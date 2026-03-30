"""
Utility Tools for D-WaveNet Training
======================================
Includes EarlyStopping, learning rate scheduling helpers, and logging utilities.
"""

import os
import numpy as np
import torch


class EarlyStopping:
    """
    Early Stopping to terminate training when validation loss stops improving.
    Saves the best model checkpoint.

    Parameters
    ----------
    patience : int
        Number of epochs to wait without improvement before stopping (default: 5).
    delta : float
        Minimum change to qualify as an improvement (default: 0.0).
    verbose : bool
        If True, prints a message when early stopping is triggered.
    """

    def __init__(self, patience=5, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, checkpoint_path):
        """
        Check if validation loss improved.

        Parameters
        ----------
        val_loss : float - Current epoch's validation loss
        model : nn.Module - Model to save if improved
        checkpoint_path : str - Path to save the best model

        Returns
        -------
        bool : True if training should stop
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(self, val_loss, model, path):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    """
    Cosine Annealing learning rate scheduler.
    Dynamically adjusts learning rate during training (Section 4.2.3).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    epoch : int - Current epoch
    args : Namespace - Must contain: learning_rate, train_epochs
    """
    lr_min = args.learning_rate * 0.01
    lr = lr_min + 0.5 * (args.learning_rate - lr_min) * (
        1 + np.cos(np.pi * epoch / args.train_epochs)
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_parameters(model):
    """Count total trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def print_model_summary(model, args):
    """Print a summary of the model architecture and parameters."""
    total_params = count_parameters(model)
    print("=" * 60)
    print(f"  D-WaveNet Model Summary")
    print("=" * 60)
    print(f"  Input length (seq_len):    {args.seq_len} hours")
    print(f"  Prediction horizon:        {args.pred_len} hours")
    print(f"  WCFT d_model:              {args.d_model}")
    print(f"  WCFT n_heads:              {args.n_heads}")
    print(f"  WCFT encoder layers:       {args.e_layers}")
    print(f"  Physics loss weight (γ):   {args.gamma}")
    print(f"  Total trainable params:    {total_params:,}")
    print(f"  Approx. params (M):        {total_params / 1e6:.2f}M")
    print("=" * 60)
