# Categorization and sentiment analysis

from scipy.spatial import distance


def categorize_feedback(embedding, category_embeddings, categories):
    distances = [
        distance.cosine(embedding, category_embedding)
        for category_embedding in category_embeddings
    ]

    closest_index = distances.index(min(distances))

    return categories[closest_index]


def categorize_reviews(embeddings, category_embeddings, categories):
    return [
        categorize_feedback(embedding, category_embeddings, categories)
        for embedding in embeddings
    ]
