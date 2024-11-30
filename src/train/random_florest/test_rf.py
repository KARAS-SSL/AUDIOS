import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
    
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='binary') 
   
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # print(f"Test Accuracy: {test_accuracy:.4f}")
    # print(f"Classification Report:\n {test_report}")

       # Compute ROC curve and EER
    fpr, tpr, thresholds = roc_curve(test_targets, test_predictions, pos_label=1)
    fnr = 1 - tpr 
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
   
    print(f"EER: {eer:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_predictions)
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])  
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig(os.path.join(model_folder, "confusion_matrix.png"))
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
    

