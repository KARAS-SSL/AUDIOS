
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.train.mlp.mlp_model import MLP
from src.utils.dataset import load_embeddings

import json

# Function to load the model checkpoint based on the run path and flag
def load_model(run_path: str, use_best_model: bool, model: nn.Module, device: str):
    # Determine which checkpoint to load
    checkpoint_file = "best_model.pt" if use_best_model else "final_model.pt"
    checkpoint_path = os.path.join(run_path, checkpoint_file)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found in {run_path}")

    return model

# Function to test the model and compute evaluation metrics including Equal Error Rate
def test_mlp(test_embeddings_folder_path: str, run_path: str, use_best_model: bool = True, device: str = "cpu"):

    # Load the hyperparameters and batch size from the config file
    config_path = os.path.join(run_path, "hyperparameters.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract model hyperparameters from config
    input_dim    = config.get("input_dim", 512)
    hidden_dim_1 = config.get("hidden_dim_1", 256)
    output_dim   = config.get("output_dim", 2)
    batch_size   = config.get("batch_size", 32)

    # Initialize the model using the hyperparameters from config
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim_1, output_dim=output_dim).to(device)

    # Load the model
    model = load_model(run_path, use_best_model, model, device)
    model.eval()

    # Create the DataLoader for the test set
    test_loader = load_embeddings(test_embeddings_folder_path, batch_size=batch_size, shuffle=False)
    
    # Initialize lists to hold the true labels and predicted probabilities
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []

    # Loop through the test set
    with torch.no_grad():  # No gradients are needed for inference
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device
            outputs = model(inputs)  # Get model outputs
            
            # Get the probabilities (for calculating ROC and EER)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification, get the probability for class '1'
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)

            # Append true and predicted labels to the lists
            all_true_labels.extend(targets.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    # Convert lists to numpy arrays for metrics calculation
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    all_pred_probs = np.array(all_pred_probs)
    
    # Calculate Accuracy
    accuracy = accuracy_score(all_true_labels, all_pred_labels)

    # Calculate Precision, Recall, F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='binary')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute ROC curve and EER
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_pred_probs)
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - tpr))]  # Find the threshold where FPR equals TPR
    eer = fpr[np.nanargmin(np.abs(fpr - tpr))]  # Equal Error Rate is the value of FPR at that threshold

    # Prepare metrics for saving
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),  # Convert the confusion matrix to a list
        'eer': eer,  # Adding EER to the metrics
        'eer_threshold': eer_threshold  # Adding the threshold at which EER occurs
    }

    # Print the metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")

    # Save the metrics to a JSON file
    metrics_file_path = os.path.join(run_path, "metrics.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f, indent=4)
