# Model, paths, category descriptions

from pathlib import Path

# These paths are relative to where you run the command.
DATA_PATH = Path("data/womens_clothing_e-commerce_reviews.csv")
OUTPUT_DIR = Path("outputs")
CHROMA_PATH = Path("chroma")

EMBEDDING_MODEL = "text-embedding-3-small"

CATEGORIES = {
    "Quality": "Fabric quality, stitching, and durability.",
    "Fit": "Sizing, garment length, and how clothing fits.",
    "Style": "Appearance, design, colors, and fashion.",
    "Comfort": "Softness, breathability, and comfort when worn.",
}