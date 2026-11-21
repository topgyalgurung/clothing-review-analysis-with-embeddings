# Clothing Review Analysis with Embeddings

A Python project for exploring clothing reviews using OpenAI embeddings, ChromaDB similarity search, and t-SNE visualization.

## What it does

- Loads and cleans review text from a CSV file.
- Uses `text-embedding-3-small` to represent reviews as vectors.
- Assigns reviews to Quality, Fit, Style, or Comfort categories.
- Stores embeddings locally in `chroma/` and finds similar reviews.
- Displays a 2D scatter plot of review embeddings.

## Setup

Requires Python 3.14+, [uv](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
uv sync --no-install-project
export OPENAI_API_KEY="your-api-key"
```

The dataset is located at `data/womens_clothing_e-commerce_reviews.csv` and must contain a `Review Text` column. The demo uses the first 200 cleaned reviews.

## Run

Once the missing functions below are implemented, run from the project root:

```bash
PYTHONPATH=src .venv/bin/python main.py
```

The script prints sample categories and similar reviews, then opens a plot. Embedding requests use the OpenAI API and incur usage costs.

## Current status

Work in progress: `create_embeddings` in `embeddings.py` and `categorize_reviews` in `analysis.py` still need implementation. The package configuration also references the old module name; setup above installs dependencies without building the package.
