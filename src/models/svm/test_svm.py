import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils.embeddings import load_embeddings
from src.utils.eer import compute_eer

# ----------------------------------------------------------------

def test_svm(test_embeddings_folder_path: str, model_folder: str, gender: str = "") -> None:
    """
    Test the model and compute evaluation metrics (accuracy, precision, recall, F1-score and EER).

    Parameters
    ----------
    test_embeddings_folder_path : str
        The path to the folder containing the test embeddings.
    model_folder : str
        The path to the folder containing the model and scaler.
    gender : str
        Filter embeddings by gender. Default is "" (no filter).

    Returns
    -------
    None
    """
    # Load test embeddings
    test_loader = load_embeddings(test_embeddings_folder_path, gender)
    test_inputs, test_targets = zip(*test_loader)
    test_inputs = np.vstack(test_inputs)
    test_targets = np.hstack(test_targets)

    # Load model and scaler
    svm = joblib.load(os.path.join(model_folder, "svm_model.joblib"))
    scaler = joblib.load(os.path.join(model_folder, "scaler.joblib"))

    # Normalize test inputs
    test_inputs = scaler.transform(test_inputs)

    # Make predictions
    test_predictions = svm.predict(test_inputs)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_report = classification_report(test_targets, test_predictions)

    # Calculate Precision, Recall, F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average="binary")

    # Calculate EER
    eer, _ = compute_eer(test_targets, test_predictions)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_predictions)
    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.title("Confusion Matrix", fontsize=16)
    plt.savefig(os.path.join(model_folder, "confusion_matrix.png"))
    plt.show()

    metrics = {"test_accuracy": test_accuracy, "test_report": test_report, "eer": eer}

    # Save the metrics to a JSON file
    metrics["model"] = "SVM"
    metrics["test_dataset"] = test_embeddings_folder_path
    metrics_file_path = os.path.join(model_folder, "test_results.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f, indent=4)
