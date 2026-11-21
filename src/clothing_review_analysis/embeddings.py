# Generate, cache, load embeddings

import numpy as np


def create_embeddings(texts, client, model):
    embeddings = []

    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]

        response = client.embeddings.create(
            model=model,
            input=batch,
        )

        results = sorted(response.data, key=lambda item: item.index)

        for result in results:
            embeddings.append(result.embedding)

    return np.array(embeddings)