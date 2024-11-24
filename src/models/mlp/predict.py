import torch
from mlp_model import MLP

def predict(embedding_path, model_path, input_dim, hidden_dim, output_dim):
    model = MLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    embedding = torch.load(embedding_path)
    with torch.no_grad():
        output = model(embedding)
        prediction = torch.argmax(output, dim=1)
    
    return prediction.item()

# Model parameters
input_dim = 768
hidden_dim = 256
output_dim = 2

# Data Paths
embedding_path = "path/to/single/embedding.pt"
model_path = "mlp_model.pth"

# Prediction
prediction = predict(embedding_path, model_path, input_dim, hidden_dim, output_dim)
print("Prediction:", "Real" if prediction == 0 else "Fake")