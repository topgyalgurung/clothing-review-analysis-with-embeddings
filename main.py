from openai import OpenAI

from clothing_review_analysis.analysis import categorize_reviews
from clothing_review_analysis.data import load_reviews
from clothing_review_analysis.embeddings import create_embeddings
from clothing_review_analysis.storage import (
    create_collection,
    find_similar_reviews,
    store_reviews,
)
from clothing_review_analysis.visualization import apply_tsne, plot_tsne


def main():
    client = OpenAI()
    model = "text-embedding-3-small"

    # Load reviews. Start small while developing.
    reviews = load_reviews(
        "data/womens_clothing_e-commerce_reviews.csv"
    ).head(200).copy()

    texts = reviews["Review Text"].tolist()
    ids = [str(index) for index in reviews.index]

    # Create review embeddings.
    embeddings = create_embeddings(texts, client, model)

    # Create category embeddings.
    categories = ["Quality", "Fit", "Style", "Comfort"]

    category_embeddings = create_embeddings(
        categories, client, model
    )

    # Categorize reviews.
    reviews["Category"] = categorize_reviews(
        embeddings, category_embeddings, categories
    )

    print(reviews[["Review Text", "Category"]].head())

    # Store reviews and their existing embeddings.
    collection = create_collection()
    store_reviews(collection, ids, texts, embeddings)

    # Search for similar reviews.
    example_review = (
        "Absolutely wonderful - silky and sexy and comfortable"
    )

    query_embedding = create_embeddings(
        [example_review], client, model
    )[0]

    similar_reviews = find_similar_reviews(
        collection, query_embedding
    )

    for review in similar_reviews:
        print(review)

    # Visualize review embeddings.
    points = apply_tsne(embeddings)
    plot_tsne(points)


if __name__ == "__main__":
    main()