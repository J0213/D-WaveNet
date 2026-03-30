"""
Physics-Guided Regularization Loss (Eq. 12)
=============================================
Implements the complete physics-guided loss function that augments MSE
with three physical constraint terms:

1. Depth-induced breaking penalty: ReLU(H_pred - 0.78*d)²
   - Prevents predictions exceeding the shallow-water breaking limit
   
2. Stokes steepness penalty: ReLU(ξ_pred - 1/7)²
   - Penalizes predictions with excessive wave steepness
   
3. Energy smoothness regularization: ‖ΔE_sys‖²
   - Discourages physically impossible sudden energy jumps

CRITICAL IMPLEMENTATION NOTE:
    The MSE loss is computed in normalized (Z-score) space, consistent with
    the model's training targets. However, the physics constraint terms
    (depth-breaking and steepness penalties) MUST operate in physical space
    (meters) to be meaningful. Therefore, predictions are inverse-transformed
    before computing these penalty terms.

Terminology Note:
    This loss applies "physics-guided regularization" via soft constraints
    that penalize predictions exceeding known hydrodynamic limits. It does
    NOT incorporate governing PDEs (e.g., spectral energy balance equations),
    and therefore should not be described as a fully "physics-informed" loss
    in the strict PINN sense.

Total Loss = MSE(normalized) + γ * (L_depth(physical) + L_steepness(physical)
                                     + λ_smooth * L_smoothness(physical))
"""

import torch
import torch.nn as nn


class PhysicsGuidedLoss(nn.Module):
    """
    Physics-Guided Regularization Loss.

    Parameters
    ----------
    gamma : float
        Overall weight for physics regularization terms (default: 0.5).
    depth : float
        Mean water depth at monitoring station in meters (default: 8.0).
    steepness_limit : float
        Stokes wave steepness limit H/L (default: 1/7 ≈ 0.142).
    lambda_smooth : float
        Weight for energy smoothness regularization (default: 0.01).
    wave_period_approx : float
        Approximate dominant wave period in seconds for steepness estimation
        (default: 8.0 seconds, typical for coastal areas in the study).
    train_mean : float
        Training set mean SWH (meters) for inverse Z-score transform.
    train_std : float
        Training set std SWH (meters) for inverse Z-score transform.
    """

    def __init__(self, gamma=0.5, depth=8.0, steepness_limit=1.0 / 7.0,
                 lambda_smooth=0.01, wave_period_approx=8.0,
                 train_mean=0.0, train_std=1.0):
        super(PhysicsGuidedLoss, self).__init__()
        self.gamma = gamma
        self.lambda_smooth = lambda_smooth
        self.mse = nn.MSELoss()

        # Store normalization stats for inverse transform
        self.train_mean = train_mean
        self.train_std = train_std

        # Depth-induced breaking limit: H_max = 0.78 * d (in meters)
        self.breaking_limit = 0.78 * depth

        # Stokes steepness limit
        self.steepness_limit = steepness_limit

        # Approximate wavelength for steepness estimation
        # In shallow water: L ≈ T * sqrt(g * d)
        g = 9.81  # gravitational acceleration (m/s²)
        self.approx_wavelength = wave_period_approx * (g * depth) ** 0.5

    def _inverse_normalize(self, y_normalized):
        """Convert Z-score normalized predictions back to physical meters."""
        return y_normalized * self.train_std + self.train_mean

    def forward(self, y_pred, y_true, swell_pred=None):
        """
        Compute the total physics-guided loss.

        Parameters
        ----------
        y_pred : Tensor [batch, pred_len]
            Total predicted SWH in NORMALIZED (Z-score) space.
        y_true : Tensor [batch, pred_len]
            Ground truth SWH in NORMALIZED (Z-score) space.
        swell_pred : Tensor [batch, pred_len], optional
            Swell component prediction in normalized space.

        Returns
        -------
        total_loss : Tensor (scalar)
            MSE(normalized) + γ * physics_penalty(physical)
        loss_components : dict
            Individual loss component values for logging.
        """
        # ----------------------------------------------------------------
        # 1. Data-driven MSE loss (computed in normalized space)
        # ----------------------------------------------------------------
        mse_loss = self.mse(y_pred, y_true)

        # ----------------------------------------------------------------
        # 2. Inverse-transform to physical space for physics constraints
        #    This is CRITICAL: depth/steepness limits are defined in meters,
        #    so predictions must be in meters for the penalties to activate.
        # ----------------------------------------------------------------
        y_pred_phys = self._inverse_normalize(y_pred)

        # ----------------------------------------------------------------
        # 3. Depth-induced breaking penalty (first term in Eq. 12)
        #    ReLU ensures penalty is applied ONLY when limit is exceeded
        #    Operates in physical space (meters)
        # ----------------------------------------------------------------
        depth_penalty = torch.mean(
            torch.relu(y_pred_phys - self.breaking_limit) ** 2
        )

        # ----------------------------------------------------------------
        # 4. Stokes steepness penalty (first term in Eq. 12)
        #    Estimated steepness: ξ ≈ H / L_approx
        #    Penalty when ξ exceeds the theoretical 1/7 limit
        #    Operates in physical space (meters)
        # ----------------------------------------------------------------
        estimated_steepness = y_pred_phys / self.approx_wavelength
        steepness_penalty = torch.mean(
            torch.relu(estimated_steepness - self.steepness_limit) ** 2
        )

        # ----------------------------------------------------------------
        # 5. Energy smoothness regularization (second term in Eq. 12)
        #    ‖ΔE_sys‖² — penalizes physically impossible sudden energy jumps
        #    Wave energy ∝ H², so ΔE ∝ Δ(H²) along the time axis
        #    Operates in physical space for interpretable energy units
        # ----------------------------------------------------------------
        energy_phys = y_pred_phys ** 2  # proxy for wave energy density
        energy_diff = energy_phys[:, 1:] - energy_phys[:, :-1]
        smoothness_penalty = torch.mean(energy_diff ** 2)

        # ----------------------------------------------------------------
        # Total loss: MSE(norm) + γ * PHY(physical)
        # ----------------------------------------------------------------
        physics_penalty = (depth_penalty + steepness_penalty
                           + self.lambda_smooth * smoothness_penalty)
        total_loss = mse_loss + self.gamma * physics_penalty

        # Loss component dictionary for logging and diagnostics
        loss_components = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'depth_penalty': depth_penalty.item(),
            'steepness_penalty': steepness_penalty.item(),
            'smoothness_penalty': smoothness_penalty.item(),
            'gamma_weighted_physics': (self.gamma * physics_penalty).item(),
        }

        return total_loss, loss_components
