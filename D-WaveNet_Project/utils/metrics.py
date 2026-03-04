import numpy as np

def MAE(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)

def R2_Pearson(pred, true):
    """
    Pearson's Coefficient of Determination (R^2)
    Evaluates the linear collinearity and proportion of variance explained.
    """
    # Flatten the arrays to compute global correlation
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    correlation_matrix = np.corrcoef(true_flat, pred_flat)
    correlation_xy = correlation_matrix[0, 1]
    return correlation_xy ** 2

def NSE(pred, true):
    """
    Nash-Sutcliffe Efficiency (NSE)
    The gold standard in ocean engineering for evaluating actual 1:1 alignment 
    and penalizing systematic over/under-prediction during extreme events.
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    numerator = np.sum((true_flat - pred_flat) ** 2)
    denominator = np.sum((true_flat - np.mean(true_flat)) ** 2)
    
    # Add a small epsilon to avoid division by zero
    if denominator == 0:
        return 0.0
    return 1 - (numerator / denominator)

def metric(pred, true):
    """
    Returns all evaluation metrics used in the D-WaveNet manuscript.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    r2 = R2_Pearson(pred, true)
    nse = NSE(pred, true)
    
    return mae, mse, r2, nse