"""Build merged training data and NLTK feature file."""

from __future__ import annotations

from pathlib import Path

from src.data_loader import load_merged, save_processed
from src.features import build_feature_frame, save_feature_file

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"


def main() -> None:
    merged = load_merged()
    train_path, _ = save_processed(merged, PROCESSED_DIR)
    featured = build_feature_frame(merged)
    feature_path = save_feature_file(featured, PROCESSED_DIR / "train_features.csv")

    label_counts = merged["sentiment"].value_counts()
    source_counts = merged["source"].value_counts()

    print(f"Saved training data: {train_path} ({len(merged):,} rows)")
    print(f"Saved feature file: {feature_path}")
    print("\nSentiment distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")
    print("\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count:,}")


if __name__ == "__main__":
    main()
