 # Chroma storage and similarity search


import chromadb


def create_collection():
    client = chromadb.PersistentClient(path="chroma")

    return client.get_or_create_collection(
        name="reviews_text_embedding_3_small",
        embedding_function=None,
    )


def store_reviews(collection, ids, texts, embeddings):
    for start in range(0, len(texts), 100):
        end = start + 100

        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
        )


def find_similar_reviews(collection, query_embedding, n=3):
    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n, count),
        include=["documents"],
    )

    return results["documents"][0]