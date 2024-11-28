
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def visualize_embeddings_tsne(embeddings_loader: list, random_state: int = 42) -> None:

    # Extract embeddings and labels from the data loader
    embeddings = []
    labels = []
    for batch in embeddings_loader:
        inputs, targets = batch
        embeddings.append(inputs.numpy())  # Convert tensor to numpy array
        labels.append(targets.numpy())    # Convert tensor to numpy array

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
    for label, color, marker in zip([0, 1], ['#6699CC', '#893168'], ['o', 's']):
        idx = labels == label
        plt.scatter(
            tsne_embeddings[idx, 0],
            tsne_embeddings[idx, 1],
            color=color,
            label=f"Class {label} ({'Fake' if label == 0 else 'Real'})",
            alpha=0.7,
            marker=marker
        )
    
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Classes", loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
