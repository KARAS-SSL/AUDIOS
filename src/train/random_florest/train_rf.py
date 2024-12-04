import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from src.utils.dataset import load_embeddings
from src.utils.train import compute_eer


def train_rf(train_loader: list, val_loader: list, train_embeddings_folder_path: str, val_embeddings_folder_path: str, hyperparameters: dict, output_path: str, randomness_seed: int) -> float:
    # Set random seed for reproducibility
    np.random.seed(randomness_seed)
    
    # Create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    run_number = len(os.listdir(output_path))
    run_folder = os.path.join(output_path, 'run' + str(run_number))
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
   
    batch_size = hyperparameters['batch_size']
    n_estimators = hyperparameters['n_estimators']
    max_depth  = hyperparameters['max_depth']
    min_samples_split = hyperparameters['min_samples_split']
    min_samples_leaf = hyperparameters['min_samples_leaf']
    
    # Extract inputs and targets
    train_inputs, train_targets = zip(*train_loader)
    train_inputs = np.vstack(train_inputs)
    train_targets = np.hstack(train_targets)
    
    val_inputs, val_targets = zip(*val_loader)
    val_inputs = np.vstack(val_inputs)
    val_targets = np.hstack(val_targets)
    
    # Normalize the inputs using StandardScaler
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)
    val_inputs = scaler.transform(val_inputs)
    
    # Train the Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=randomness_seed
    )
    rf.fit(train_inputs, train_targets)
    
    # Evaluate on validation set
    val_predictions = rf.predict(val_inputs)
    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_report = classification_report(val_targets, val_predictions, output_dict=True)
    val_eer, val_eer_threshold = compute_eer(val_targets, val_predictions)
    
    # Save model and scaler
    joblib.dump(rf, os.path.join(run_folder, "random_forest_model.joblib"))
    joblib.dump(scaler, os.path.join(run_folder, "scaler.joblib"))
    
    # Save hyperparameters and metrics
    hyperparameters["validation_eer"] = val_eer
    hyperparameters["validation_accuracy"] = val_accuracy
    hyperparameters["validation_report"] = val_report
    hyperparameters["model"] = "Random Florest"
    hyperparameters["train_dataset"] = train_embeddings_folder_path
    hyperparameters["validation_dataset"] = val_embeddings_folder_path  
    with open(os.path.join(run_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)
    
    print(f"Training completed. Validation Accuracy: {val_accuracy:.4f}")
    print(f"Classification Report:\n {classification_report(val_targets, val_predictions)}")
    print(f"Model and logs saved in {run_folder}.")

    return val_eer
