from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def calculate_robust_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Regression metrics used across benchmarks."""
    return {
        "Model": model_name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    """Classification metrics used across benchmarks."""
    return {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred)),
    }

