import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from mlp_model import MLP
from utils.dataset import load_embeddings_and_labels

#----------------------------------------------------------------
# TRAINING FUNCTION

def train_mlp(embeddings_folder_path, labels_path, input_dim, hidden_dim, output_dim, epochs=10, batch_size=32, learning_rate=0.001):
    embeddings, labels = load_embeddings_and_labels(embeddings_folder_path, labels_path)
    
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader)}")
    
    torch.save(model.state_dict(), "mlp_model.pth")

# Model parameters
input_dim = 768  # Embedding dimension
hidden_dim = 256
output_dim = 2  # Real or fake

# Data paths
embeddings_folder_path = "path/to/embeddings"
labels_path = "path/to/labels.csv"

# Model training
train_mlp(embeddings_folder_path, labels_path, input_dim, hidden_dim, output_dim)