"""
Evaluation Metrics for D-WaveNet
=================================
Includes standard regression metrics and oceanographic forecast skill scores.

Key Clarification (Section 4.2.1):
    The R² metric computed here uses the formulation:
        R² = 1 - SS_res / SS_tot = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
    This is MATHEMATICALLY IDENTICAL to the Nash-Sutcliffe Efficiency (NSE)
    coefficient used in hydrology and ocean engineering. Both metrics are
    computed and reported for accessibility across communities.
"""

import numpy as np


def MSE(pred, true):
    """Mean Squared Error (Eq. 13)"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))


def MAE(pred, true):
    """Mean Absolute Error (Eq. 14)"""
    return np.mean(np.abs(pred - true))


def NSE(pred, true):
    """
    Nash-Sutcliffe Efficiency (Eq. 15).
    Mathematically identical to the R² formulation used in this study.

    NSE = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

    Interpretation ranges (Moriasi et al., 2007):
        NSE > 0.75 : Very good
        0.65 < NSE ≤ 0.75 : Good
        0.50 < NSE ≤ 0.65 : Satisfactory
        NSE ≤ 0.50 : Unsatisfactory

    Returns
    -------
    float : NSE value in (-inf, 1.0]. Value of 1.0 indicates perfect agreement.
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)

    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def R2(pred, true):
    """
    Coefficient of Determination R² (Eq. 15).
    Under our formulation, this is identical to NSE.
    Provided as a separate function for API compatibility with
    scikit-learn conventions.
    """
    return NSE(pred, true)


def persistence_skill_score(pred, true, horizon_values):
    """
    Skill Score relative to Persistence Baseline.
    Persistence assumes y_{t+H} = y_t (last known value persists).

    SS_pers = 1 - MSE_model / MSE_persistence

    Parameters
    ----------
    pred : np.ndarray - Model predictions
    true : np.ndarray - Ground truth
    horizon_values : np.ndarray - Persistence baseline (y_t for each sample)

    Returns
    -------
    float : Skill score. > 0 means better than persistence.
    """
    mse_model = MSE(pred, true)
    mse_persist = MSE(horizon_values, true)

    if mse_persist < 1e-10:
        return 0.0
    return 1.0 - (mse_model / mse_persist)


def climatology_skill_score(pred, true, climatology_mean):
    """
    Skill Score relative to Climatological Baseline.
    Climatology predicts the historical monthly mean.

    SS_clim = 1 - MSE_model / MSE_climatology

    Parameters
    ----------
    pred : np.ndarray - Model predictions
    true : np.ndarray - Ground truth
    climatology_mean : float - Historical mean SWH

    Returns
    -------
    float : Skill score. > 0 means better than climatology.
    """
    mse_model = MSE(pred, true)
    clim_pred = np.full_like(true, climatology_mean)
    mse_clim = MSE(clim_pred, true)

    if mse_clim < 1e-10:
        return 0.0
    return 1.0 - (mse_model / mse_clim)


def metric(pred, true):
    """
    Compute all evaluation metrics used in the D-WaveNet manuscript.

    Returns
    -------
    dict : Dictionary containing MSE, RMSE, MAE, R2, NSE values
    """
    results = {
        'MSE': MSE(pred, true),
        'RMSE': RMSE(pred, true),
        'MAE': MAE(pred, true),
        'R2': R2(pred, true),
        'NSE': NSE(pred, true),  # Explicitly reported for ocean engineering community
    }
    return results
