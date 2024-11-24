import joblib
import torch

def predict(embedding_path, model_path):
    # Load the model
    clf = joblib.load(model_path)
    
    # Load the embedding
    embedding = torch.load(embedding_path).numpy().reshape(1, -1)
    
    # Make prediction
    prediction = clf.predict(embedding)
    
    return prediction[0]

# Data paths
embedding_path = "path/to/single/embedding.pt"
model_path = "models/svm_model.joblib"

# Prediction
prediction = predict(embedding_path, model_path)
print("Prediction:", "Real" if prediction == 0 else "Fake")
