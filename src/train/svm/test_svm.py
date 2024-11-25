
import os
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.dataset import load_embeddings

def test_svm(test_embeddings_folder_path: str, model_folder: str, gender: str = ""):
    # Load test embeddings
    test_loader = load_embeddings(test_embeddings_folder_path, gender)
    test_inputs, test_targets = zip(*test_loader)
    test_inputs = np.vstack(test_inputs)
    test_targets = np.hstack(test_targets)
    
    # Load model and scaler
    svm = joblib.load(os.path.join(model_folder, 'svm_model.joblib'))
    scaler = joblib.load(os.path.join(model_folder, 'scaler.joblib'))
    
    # Normalize test inputs
    test_inputs = scaler.transform(test_inputs)
    
    # Make predictions
    test_predictions = svm.predict(test_inputs)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_report = classification_report(test_targets, test_predictions)

    # Calculate Precision, Recall, F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='binary') 

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    # print(f"Classification Report:\n {test_report}")
    
    # Compute ROC curve and EER
    fpr, tpr, thresholds = roc_curve(test_targets, test_predictions, pos_label=1)
    fnr = 1 - tpr 
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
   
    print(f"EER: {eer:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_predictions)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(model_folder, "confusion_matrix.png"))
    plt.show()
    
    metrics = {
        'test_accuracy': test_accuracy,
        'test_report': test_report,
        'eer': eer
    }
   
    # Save the metrics to a JSON file
    metrics['model'] = 'SVM'
    metrics['test_dataset'] = test_embeddings_folder_path
    metrics_file_path = os.path.join(model_folder, "test_results.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f, indent=4) 
    

