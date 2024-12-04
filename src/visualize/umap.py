import os

import matplotlib.pyplot as plt
import numpy as np
import umap


def visualize_embeddings_umap(embeddings_folder_path: str, embeddings_loader: list, random_state: int = 7) -> None:
    """
    Visualize embeddings using UMAP.

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

    # Apply UMAP
    print("Applying UMAP...")
    umap_model = umap.UMAP(n_components=2, random_state=random_state)
    umap_embeddings = umap_model.fit_transform(embeddings)
    print("UMAP applied.")

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )

    plt.title("UMAP Visualization of Embeddings", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=16)
    plt.ylabel("UMAP Dimension 2", fontsize=16)
    plt.colorbar(scatter, label="Labels")
    plt.savefig(os.path.join(embeddings_folder_path, "umap.png"))
    plt.show()
