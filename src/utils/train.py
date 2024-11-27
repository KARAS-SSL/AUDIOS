import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_scores):
    """
    Compute the Equal Error Rate (EER) given true labels and prediction scores.
    
    Args:
        y_true (numpy.ndarray): Ground truth binary labels (0 or 1).
        y_scores (numpy.ndarray): Predicted scores (logits or probabilities).

    Returns:
        eer (float): The Equal Error Rate.
        threshold (float): The threshold at which EER is achieved.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # False negative rate is 1 - true positive rate
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold

