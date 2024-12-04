import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_embeddings_tsne(embeddings_folder_path: str, embeddings_loader: list, random_state: int = 7) -> None:
    """
    Visualize embeddings using t-SNE.

    Parameters
    ----------
    embeddings_folder_path : str
        Path to the folder containing embeddings.
    embeddings_loader : list
        A list of tuples containing the embeddings and labels.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    None
    """
    # Extract embeddings and labels from the data loader
    embeddings = []
    labels = []
    for batch in embeddings_loader:
        inputs, targets = batch
        embeddings.append(inputs.numpy())  # Convert tensor to numpy array
        labels.append(targets.numpy())  # Convert tensor to numpy array

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # Check if number of samples is less than 30 and adjust perplexity
    n_samples = embeddings.shape[0]
    perplexity = 30 if n_samples > 30 else n_samples - 1

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_embeddings = tsne.fit_transform(embeddings)
    print("t-SNE applied.")

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )

    plt.title("t-SNE Visualization of Embeddings", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=16)
    plt.ylabel("t-SNE Dimension 2", fontsize=16)
    plt.colorbar(scatter, label="Labels")
    plt.savefig(os.path.join(embeddings_folder_path, "t-sne.png"))
    plt.show()
