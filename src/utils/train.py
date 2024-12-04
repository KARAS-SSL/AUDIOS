import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute the Equal Error Rate (EER) given true labels and prediction scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores (logits or probabilities).

    Returns
    -------
    eer : float
        The Equal Error Rate.
    threshold : float
        The threshold at which EER is achieved.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # False negative rate is 1 - true positive rate
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_index]
    eer = fpr[eer_index]
    return eer, eer_threshold
