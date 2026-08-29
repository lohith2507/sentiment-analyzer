"""Clean the merged datasets, write train/val/test splits, and build the feature file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import (
    MIN_WORDS,
    PROCESSED_DIR,
    load_merged,
    save_splits,
    split_dataset,
)
from src.features import build_feature_frame, save_feature_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sentiment training data")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--drop-three-star",
        action="store_true",
        help="Exclude McDonald's 3-star reviews instead of labelling them neutral",
    )
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()

    merged, stats = load_merged(
        drop_three_star=args.drop_three_star,
        min_words=args.min_words,
        return_stats=True,
    )

    print("Cleaning:")
    print(f"  rows in                {stats['input']:>7,}")
    for key, label in (
        ("dropped_corrupted", "corrupted text"),
        ("dropped_too_short", f"under {args.min_words} words"),
        ("dropped_duplicates", "duplicates"),
        ("dropped_conflicting", "conflicting labels"),
    ):
        if key in stats:
            print(f"  dropped {label:<22}{stats[key]:>7,}")
    print(f"  rows out               {stats['output']:>7,}")

    splits = split_dataset(merged)
    paths = save_splits(splits, args.output_dir)

    print("\nSplits:")
    for name, frame in splits.items():
        print(f"  {name:<6}{len(frame):>7,}  ->  {paths[name].name}")

    print("\nSentiment distribution (train):")
    for label, count in splits["train"]["sentiment"].value_counts().items():
        share = count / len(splits["train"])
        print(f"  {label:<10}{count:>7,}  ({share:.1%})")

    print("\nSource distribution (train):")
    for source, count in splits["train"]["source"].value_counts().items():
        print(f"  {source:<10}{count:>7,}")

    if not args.skip_features:
        featured = build_feature_frame(splits["train"])
        feature_path = save_feature_file(featured, args.output_dir / "train_features.csv")
        print(f"\nFeature file: {feature_path}")


if __name__ == "__main__":
    main()
