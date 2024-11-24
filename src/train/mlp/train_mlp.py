import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split

from src.train.mlp.mlp_model import MLP
from src.utils.dataset import load_embeddings

import json
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# TRAINING FUNCTION

def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    np.random.seed(seed)

def train_mlp(train_embeddings_folder_path: str, val_embeddings_folder_path: str, hyperparameters: dict, output_path: str, 
              randomness_seed: int, device: str):

    set_random_seeds(randomness_seed)
    
    # Create output folder 
    if not os.path.exists(output_path): os.makedirs(output_path)
    run_number = len(os.listdir(output_path))    
    run_folder = os.path.join(output_path, 'run'+str(run_number))
    if not os.path.exists(run_folder): os.makedirs(run_folder)
    
    # Hyperparameters
    epochs        = hyperparameters['epochs']
    batch_size    = hyperparameters['batch_size']
    learning_rate = hyperparameters['learning_rate']
    patience      = hyperparameters['patience']
    hidden_dim_1  = hyperparameters['hidden_dim_1']
    output_dim    = hyperparameters['output_dim']
   
    # Load embeddings 
    train_loader = load_embeddings(train_embeddings_folder_path, batch_size=batch_size, shuffle=False)
    val_loader   = load_embeddings(val_embeddings_folder_path, batch_size=batch_size, shuffle=False)

    # Determine input dimension dynamically from the dataset
    sample_input, _         = next(iter(train_loader))
    input_dim               = sample_input.shape[1]
 
    # Define model, loss function, and optimizer
    model = MLP(input_dim, hidden_dim_1, output_dim).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                val_loss += loss.item()

        # Calculate average losses
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Logging
        epoch_log = f"Epoch {epoch+1}/{epochs}"
        train_log = f"Train Loss: {epoch_train_loss:.4f}"
        val_log = f"Validation Loss: {epoch_val_loss:.4f}"
        print(epoch_log, train_log, val_log)

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(run_folder, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(run_folder, "final_model.pth"))

    # Save training logs as NumPy arrays
    np.save(os.path.join(run_folder, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(run_folder, "val_losses.npy"), np.array(val_losses))

    # Save hyperparameters and best validation loss as JSON
    hyperparameters["best_val_loss"] = best_val_loss
    hyperparameters['input_dim'] = input_dim
    hyperparameters['model'] = 'MLP'
    with open(os.path.join(run_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)

    # Plot and save loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "loss_plot.png"))
    plt.show()

    print(f"Training completed. Logs, models, and hyperparameters saved in {run_folder}.")
    

