import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def compute_ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov (KS) statistic for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(tpr - fpr))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard credit scoring metrics on predicted probabilities."""
    auc = float(roc_auc_score(y_true, y_pred))
    return {
        "roc_auc": round(auc, 5),
        "gini": round(2 * auc - 1, 5),
        "ks_stat": round(compute_ks_statistic(y_true, y_pred), 5),
        "pr_auc": round(float(average_precision_score(y_true, y_pred)), 5),
        "brier_score": round(float(brier_score_loss(y_true, y_pred)), 5),
    }