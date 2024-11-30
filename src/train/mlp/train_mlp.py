import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.train.mlp.mlp_model import MLP
from src.utils.dataset import load_embeddings
from src.utils.train import compute_eer

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
) -> float:
    print("Starting training...")
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

    # Dynamically calculate input_dim based on the first batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = MLP(input_dim, hidden_dim_1, output_dim, dropout_rate).to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Initialize variables for training loop
    train_losses, val_losses, val_accuracies = [], [], []
    val_eers = []
    best_val_eer = float("inf")
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
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_scores = []
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = model(inputs).squeeze()
                loss = loss_func(outputs, targets)
                val_loss += loss.item()

                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) >= 0.5).int()
                correct_predictions += (predictions == targets.int()).sum().item()
                total_samples += targets.size(0)

                # Store true labels and predicted scores for EER computation
                all_targets.extend(targets.cpu().numpy())
                all_scores.extend(outputs.cpu().numpy())

        # Compute average losses and accuracy
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = correct_predictions / total_samples
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        # Compute EER
        eer, eer_threshold = compute_eer(np.array(all_targets), np.array(all_scores))
        val_eers.append(eer)

        if eer < best_val_eer:
            best_val_eer = eer

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        # Log progress
        epoch_log = f"Epoch {epoch + 1}/{epochs}"
        train_log = f"Train Loss: {epoch_train_loss:.4f}"
        val_log = f"Validation Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_accuracy:.4f}"
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
    np.save(os.path.join(run_folder, "val_accuracies.npy"), np.array(val_accuracies))

    # Save hyperparameters and best validation loss
    hyperparameters["best_val_loss"] = best_val_loss
    hyperparameters["best_val_eer"] = best_val_eer
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
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "loss_plot.png"))
    # plt.show()

    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", marker="o", color="green")
    plt.title("Validation Accuracy", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "accuracy_plot.png"))
    # plt.show()
    
    print(f"Training completed. Logs, models, and hyperparameters saved in {run_folder}.")

    return best_val_eer
