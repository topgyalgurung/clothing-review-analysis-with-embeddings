# Load and clean reviews

import pandas as pd


def load_reviews(file_path):
    reviews = pd.read_csv(file_path)

    reviews = reviews.dropna(subset=["Review Text"]).copy()
    reviews["Review Text"] = reviews["Review Text"].str.strip()
    reviews = reviews[reviews["Review Text"] != ""].copy()

    return reviews