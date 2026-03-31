"""Evaluation metrics for few-shot EEG decoding."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and Cohen's Kappa.

    Args:
        y_true: Ground-truth labels, shape [N].
        y_pred: Predicted labels, shape [N].

    Returns:
        Dict containing `accuracy`, `macro_f1`, and `kappa`.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
