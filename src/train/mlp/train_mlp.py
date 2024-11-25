import os
import time
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

def train_mlp(
    train_embeddings_folder_path: str,
    val_embeddings_folder_path: str,
    hyperparameters: dict,
    output_path: str,
    randomness_seed: int,
    device: str,
):
    # Set random seed for reproducibility
    set_random_seeds(randomness_seed)

    # Create output folder for the training run
    os.makedirs(output_path, exist_ok=True)
    run_number = len(os.listdir(output_path))
    run_folder = os.path.join(output_path, f"run{run_number}")
    os.makedirs(run_folder, exist_ok=True)

    # Extract hyperparameters
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    patience = hyperparameters["patience"]
    hidden_dim_1 = hyperparameters["hidden_dim_1"]
    output_dim = hyperparameters["output_dim"]
    dropout_rate = hyperparameters["dropout"]
    weight_decay = hyperparameters["weight_decay"]

    # Load training and validation datasets
    train_loader = load_embeddings(
        train_embeddings_folder_path, batch_size=batch_size, shuffle=True
    )
    val_loader = load_embeddings(
        val_embeddings_folder_path, batch_size=batch_size, shuffle=False
    )

    # Dynamically determine input dimension from the dataset
    sample_input, _ = next(iter(train_loader))
    input_dim = sample_input.shape[1]

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = MLP(input_dim, hidden_dim_1, output_dim, dropout_rate).to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Initialize variables for training loop
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_func(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = model(inputs).squeeze()
                loss = loss_func(outputs, targets)
                val_loss += loss.item()

        # Compute average losses
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        # Log progress
        epoch_log = f"Epoch {epoch + 1}/{epochs}"
        train_log = f"Train Loss: {epoch_train_loss:.4f}"
        val_log = f"Validation Loss: {epoch_val_loss:.4f}"
        lr_log = f"Learning Rate: {optimizer.param_groups[0]['lr']:.6e}"
        print(f"{epoch_log} | {train_log} | {val_log} | {lr_log}")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_folder, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Log epoch time
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

    # Save final model and results
    torch.save(model.state_dict(), os.path.join(run_folder, "final_model.pth"))
    np.save(os.path.join(run_folder, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(run_folder, "val_losses.npy"), np.array(val_losses))

    # Save hyperparameters and best validation loss
    hyperparameters["best_val_loss"] = best_val_loss
    hyperparameters["input_dim"] = input_dim
    hyperparameters["model"] = "MLP"
    hyperparameters["train_dataset"] = train_embeddings_folder_path
    hyperparameters["validation_dataset"] = val_embeddings_folder_path
    with open(os.path.join(run_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "loss_plot.png"))
    plt.show()

    print(f"Training completed. Logs, models, and hyperparameters saved in {run_folder}.")