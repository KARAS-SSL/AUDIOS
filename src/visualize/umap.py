
import numpy as np
import matplotlib.pyplot as plt
from src.utils.dataset import load_embeddings

import umap

def visualize_embeddings_umap(embeddings_folder_path: str, batch_size: int = 32, random_state: int = 42) -> None:

    # Load embeddings
    loader = load_embeddings(embeddings_folder_path, batch_size=batch_size, shuffle=True)
   
    # Extract embeddings and labels from the data loader
    embeddings = []
    labels = []
    for batch in loader:
        inputs, targets = batch
        embeddings.append(inputs.numpy())  # Convert tensor to numpy array
        labels.append(targets.numpy())    # Convert tensor to numpy array

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # Apply UMAP
    umap_model = umap.UMAP(n_components=2, random_state=random_state)
    umap_embeddings = umap_model.fit_transform(embeddings)

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title("UMAP Visualization of Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label="Labels")
    plt.show()
    
 
