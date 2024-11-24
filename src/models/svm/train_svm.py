import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.dataset import load_embeddings_and_labels_numpy
from svm_model import create_svm_model

#----------------------------------------------------------------
# TRAINING FUNCTION

def train_svm(embeddings_folder_path, labels_path, model_output_path, test_size=0.2, random_state=42):
    embeddings, labels = load_embeddings_and_labels_numpy(embeddings_folder_path, labels_path)
    
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)
    
    # Create and train the SVM model
    clf = create_svm_model()
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}")
    
    # Save the model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)

# Data paths
embeddings_folder_path = "path/to/embeddings"
labels_path = "path/to/labels.csv"
model_output_path = "models/svm_model.joblib"

# Model training
train_svm(embeddings_folder_path, labels_path, model_output_path)