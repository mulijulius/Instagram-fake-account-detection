import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict


def evaluate_models(scorer, X: np.ndarray, y: np.ndarray, alpha: float, beta: float, threshold: float) -> Dict[str, float]:
    scores = scorer.score(X, alpha=alpha, beta=beta)
    preds = (scores >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y, scores)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics
