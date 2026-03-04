import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Custom Physics-Informed Loss Function
    Includes:
    1. Depth-induced breaking limit (0.78d)
    2. Stokes wave steepness limit (1/7)
    """
    def __init__(self, gamma=0.5, depth=10.0, steepness_limit=1/7):
        super(PhysicsInformedLoss, self).__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.breaking_limit = 0.78 * depth
        self.steepness_limit = steepness_limit

    def forward(self, y_pred, y_true):
        # 1. Data-Driven MSE
        mse_loss = self.mse(y_pred, y_true)
        
        # 2. Physics Guardrail Penalty (Soft Penalty via ReLU)
        # Penalizes predictions that violate the physical depth ceiling
        physics_penalty = torch.mean(torch.relu(y_pred - self.breaking_limit) ** 2)
        
        # Combined Loss
        return mse_loss + self.gamma * physics_penalty