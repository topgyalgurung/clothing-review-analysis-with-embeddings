# t-SNE and plots

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def apply_tsne(embeddings):
    if len(embeddings) < 2:
        raise ValueError("t-SNE needs at least two reviews.")

    tsne = TSNE(
        n_components=2,
        random_state=0,
        perplexity=min(30, len(embeddings) - 1),
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(embeddings)


def plot_tsne(points):
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.5)

    plt.title("Review Embeddings")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.show()