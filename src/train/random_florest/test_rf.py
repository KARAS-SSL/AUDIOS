import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.dataset import load_embeddings

def test_rf(test_embeddings_folder_path: str, model_folder: str):
    # Load test embeddings
    test_loader = load_embeddings(test_embeddings_folder_path)
    test_inputs, test_targets = zip(*test_loader)
    test_inputs = np.vstack(test_inputs)
    test_targets = np.hstack(test_targets)
    
    # Load model and scaler
    rf = joblib.load(os.path.join(model_folder, 'random_forest_model.joblib'))
    scaler = joblib.load(os.path.join(model_folder, 'scaler.joblib')) 
 
    # Normalize test inputs
    test_inputs = scaler.transform(test_inputs)
    
    # Make predictions
    test_predictions = rf.predict(test_inputs)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_report = classification_report(test_targets, test_predictions)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Classification Report:\n {test_report}")

       # Compute ROC curve and EER
    fpr, tpr, thresholds = roc_curve(test_targets, test_predictions, pos_label=1)
    fnr = 1 - tpr 
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    
    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_predictions)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_targets), yticklabels=np.unique(test_targets))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(model_folder, 'confusion_matrix.png'))
    plt.show()

    metrics = {
        'test_accuracy': test_accuracy,
        'test_report': test_report,
        'eer': eer
    }
    
    # Save the metrics to a JSON file
    metrics['model'] = 'Random Forest'
    metrics['test_dataset'] = test_embeddings_folder_path
    metrics_file_path = os.path.join(model_folder, "test_results.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f, indent=4) 
    

