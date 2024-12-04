import json
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.train import compute_eer

# ----------------------------------------------------------------
# TRAINING FUNCTION

def train_svm(
    train_loader: list,
    val_loader: list,
    train_embeddings_folder_path: str,
    val_embeddings_folder_path: str,
    hyperparameters: dict,
    output_path: str,
    randomness_seed: int,
) -> float:
    """
    Train a SVM model.

    Parameters
    ----------
    train_loader : list
        A list of tuples containing the training inputs and targets.
    val_loader : list
        A list of tuples containing the validation inputs and targets.
    train_embeddings_folder_path : str
        The path to the folder containing the training embeddings.
    val_embeddings_folder_path : str
        The path to the folder containing the validation embeddings.
    hyperparameters : dict
        A dictionary containing the hyperparameters for training:
        - *batch_size* (int): Batch size for training.
        - *C* (float): Regularization parameter for the SVM.
        - *kernel* (str): Kernel type for the SVM ('linear', 'poly', 'rbf', 'sigmoid').
        - *gamma* (str): Kernel coefficient for the SVM ('scale', 'auto').
    output_path : str
        The path to the folder where the model will be saved.
    randomness_seed : int
        The seed value for reproducibility.

    Returns
    -------
    float
        The best validation EER.
    """
    # Set random seed for reproducibility
    np.random.seed(randomness_seed)

    # Create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    run_number = len(os.listdir(output_path))
    run_folder = os.path.join(output_path, "run" + str(run_number))
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    c = hyperparameters["C"]
    kernel = hyperparameters["kernel"]
    gamma = hyperparameters["gamma"]

    # Initialize lists for inputs and targets
    train_inputs = []
    train_targets = []
    val_inputs = []
    val_targets = []

    # Extract training data by iterating over the loader
    for inputs, targets in train_loader:
        train_inputs.append(inputs)
        train_targets.append(targets)

    # Extract validation data by iterating over the loader
    for inputs, targets in val_loader:
        val_inputs.append(inputs)
        val_targets.append(targets)

    # Convert lists to numpy arrays
    train_inputs = np.vstack(train_inputs)
    train_targets = np.hstack(train_targets)

    val_inputs = np.vstack(val_inputs)
    val_targets = np.hstack(val_targets)

    # Normalize the inputs using StandardScaler
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)
    val_inputs = scaler.transform(val_inputs)

    # Train the SVM
    svm = SVC(C=c, kernel=kernel, gamma=gamma, random_state=randomness_seed)
    svm.fit(train_inputs, train_targets)

    # Evaluate on validation set
    val_predictions = svm.predict(val_inputs)
    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_report = classification_report(val_targets, val_predictions, output_dict=True)
    val_eer, _ = compute_eer(val_targets, val_predictions)

    # Save model and scaler
    joblib.dump(svm, os.path.join(run_folder, "svm_model.joblib"))
    joblib.dump(scaler, os.path.join(run_folder, "scaler.joblib"))

    # Save hyperparameters and metrics
    hyperparameters["validation_eer"] = val_eer
    hyperparameters["validation_accuracy"] = val_accuracy
    hyperparameters["validation_report"] = val_report
    hyperparameters["model"] = "SVM"
    hyperparameters["train_dataset"] = train_embeddings_folder_path
    hyperparameters["validation_dataset"] = val_embeddings_folder_path
    with open(os.path.join(run_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"Training completed. Validation Accuracy: {val_accuracy:.4f}")
    print(f"Classification Report:\n {classification_report(val_targets, val_predictions)}")
    print(f"Model and logs saved in {run_folder}.")

    return val_eer
