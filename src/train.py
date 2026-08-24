"""Fine-tune a Hugging Face transformer for 3-class sentiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_loader import ID2LABEL, LABEL2ID, load_merged

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CSV = ROOT / "processed" / "train.csv"
DEFAULT_MODEL_DIR = ROOT / "models" / "sentiment-distilbert"
DEFAULT_BASE_MODEL = "distilbert-base-uncased"


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def _load_training_frame(train_csv: Path | None) -> pd.DataFrame:
    if train_csv and train_csv.exists():
        df = pd.read_csv(train_csv)
    else:
        df = load_merged()
    return df[["text", "label_id"]].rename(columns={"label_id": "labels"})


def fine_tune(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    output_dir: Path = DEFAULT_MODEL_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    seed: int = 42,
    max_samples: int | None = None,
) -> Path:
    df = _load_training_frame(train_csv)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed)
    train_df, eval_df = train_test_split(
        df,
        test_size=0.1,
        random_state=seed,
        stratify=df["labels"],
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True)).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune sentiment classifier")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output = fine_tune(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    print(f"Model saved to {output}")


if __name__ == "__main__":
    main()
