
import numpy as np
import matplotlib.pyplot as plt
from src.utils.dataset import load_embeddings

from sklearn.manifold import TSNE

def visualize_embeddings_tsne(embeddings_folder_path: str, batch_size: int = 32, random_state: int = 42) -> None:

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

    # Check if number of samples is less than 30 and adjust perplexity
    n_samples = embeddings.shape[0]
    perplexity = 30 if n_samples > 30 else n_samples - 1

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label="Labels")
    plt.show() 
