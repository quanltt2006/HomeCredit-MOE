"""Evaluation metrics for Home Credit Default Risk."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def binary_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC for binary classification.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        AUC-ROC score
    """
    return roc_auc_score(y_true, y_pred)


def multiclass_auc(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Compute macro AUC-ROC for multiclass classification.
    
    Args:
        y_true: True labels (0, 1, ..., n_classes-1)
        y_pred: Predicted probabilities [N, n_classes]
        n_classes: Number of classes
        
    Returns:
        Macro-averaged AUC-ROC score
    """
    if y_pred.ndim == 1:
        # Convert to one-vs-all format
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        
    return roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "binary",
) -> dict:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        task: 'binary' or 'multiclass'
        
    Returns:
        Dictionary with AUC, AP, accuracy, etc.
    """
    if task == "binary":
        auc = binary_auc(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        preds_binary = (y_pred > 0.5).astype(int)
        accuracy = (preds_binary == y_true).mean()
        
        return {
            "auc_roc": auc,
            "average_precision": ap,
            "accuracy": accuracy,
        }
    else:
        # Multiclass
        if y_pred.ndim == 1:
            y_pred = np.stack([1 - y_pred, y_pred], axis=1)
            
        auc = multiclass_auc(y_true, y_pred)
        preds_class = y_pred.argmax(axis=1)
        accuracy = (preds_class == y_true).mean()
        
        return {
            "auc_roc_macro": auc,
            "accuracy": accuracy,
        }