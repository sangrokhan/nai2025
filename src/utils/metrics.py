"""
Evaluation Metrics

Computes various regression metrics for model evaluation.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics

    Args:
        y_true: True values (log-transformed)
        y_pred: Predicted values (log-transformed)

    Returns:
        Dictionary with metrics:
            - r2: R² score
            - mae: Mean Absolute Error (log space)
            - rmse: Root Mean Squared Error (log space)
            - mape: Mean Absolute Percentage Error (original space)
    """
    metrics = {}

    # R² score
    metrics['r2'] = r2_score(y_true, y_pred)

    # MAE and RMSE in log space
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE in original space
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)

    mape = np.mean(np.abs((y_true_original - y_pred_original) / (y_true_original + 1e-8))) * 100
    metrics['mape'] = mape

    return metrics


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute residuals and error distributions

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with residual statistics
    """
    residuals = y_pred - y_true

    return {
        'residuals': residuals,
        'abs_residuals': np.abs(residuals),
        'squared_residuals': residuals ** 2,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
    }
