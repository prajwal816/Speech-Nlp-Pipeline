from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates classification metrics.
    - y_true: list of true labels (strings or ints)
    - y_pred: list of predicted labels
    - y_prob: list of probabilities for positive class (useful for binary or one-vs-rest ROC)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_prob is not None:
        try:
            # Multi-class or binary depending on dimensions
            if len(np.array(y_prob).shape) > 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            print(f"Warning: Could not calculate ROC-AUC. {e}")
            
    return metrics
