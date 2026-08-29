"""Score a fine-tuned model on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import ID2LABEL, PROCESSED_DIR, SENTIMENT_LABELS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "sentiment-distilbert"


def predict_labels(
    texts: list[str],
    model_dir: Path,
    batch_size: int = 32,
    max_length: int = 192,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return np.array(preds)


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = list(range(len(SENTIMENT_LABELS)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(name) for name in SENTIMENT_LABELS) + 2

    header = " " * (width + 8) + "predicted"
    columns = "".join(f"{name:>{width}}" for name in SENTIMENT_LABELS)
    print(header)
    print(" " * (width + 6) + columns)
    for idx, name in enumerate(SENTIMENT_LABELS):
        row = "".join(f"{count:>{width},}" for count in matrix[idx])
        prefix = "actual" if idx == 1 else " " * 6
        print(f"{prefix}{name:>{width}}{row}")


def evaluate(
    test_path: Path = PROCESSED_DIR / "test.csv",
    model_dir: Path = DEFAULT_MODEL_DIR,
    batch_size: int = 32,
    max_length: int = 192,
) -> dict[str, float]:
    df = pd.read_csv(test_path)
    y_true = df["label_id"].to_numpy()
    y_pred = predict_labels(
        df["text"].astype(str).tolist(),
        model_dir=model_dir,
        batch_size=batch_size,
        max_length=max_length,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Model: {model_dir}")
    print(f"Test rows: {len(df):,}\n")
    print("Per-class metrics:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(SENTIMENT_LABELS))),
            target_names=list(SENTIMENT_LABELS),
            digits=3,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print_confusion_matrix(y_true, y_pred)

    print("\nBy source:")
    df = df.assign(predicted=y_pred)
    for source, group in df.groupby("source"):
        source_accuracy = accuracy_score(group["label_id"], group["predicted"])
        source_f1 = f1_score(group["label_id"], group["predicted"], average="macro")
        print(
            f"  {source:<10}n={len(group):>6,}  "
            f"accuracy={source_accuracy:.3f}  macro F1={source_f1:.3f}"
        )

    majority = df["label_id"].value_counts(normalize=True).max()
    print("\nOverall:")
    print(f"  accuracy            {accuracy:.3f}")
    print(f"  macro F1            {macro_f1:.3f}")
    print(f"  majority-class base {majority:.3f}  (always predict most common label)")

    return {"accuracy": accuracy, "f1_macro": macro_f1, "majority_baseline": majority}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument("--test-path", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    evaluate(
        test_path=args.test_path,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
