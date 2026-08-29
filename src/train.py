"""Fine-tune a Hugging Face transformer for 3-class sentiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data_loader import (
    ID2LABEL,
    LABEL2ID,
    PROCESSED_DIR,
    SEED,
    load_merged,
    save_splits,
    split_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "sentiment-distilbert"
DEFAULT_BASE_MODEL = "distilbert-base-uncased"


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss to counter the positive-class skew."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None
            else None
        )
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    if train_path.exists() and val_path.exists():
        return {
            "train": pd.read_csv(train_path),
            "val": pd.read_csv(val_path),
        }

    splits = split_dataset(load_merged())
    save_splits(splits, processed_dir)
    return splits


def fine_tune(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = DEFAULT_MODEL_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 192,
    learning_rate: float = 2e-5,
    seed: int = SEED,
    max_samples: int | None = None,
    class_weighting: bool = True,
) -> Path:
    splits = _load_splits(processed_dir)
    train_df = splits["train"][["text", "label_id"]].rename(columns={"label_id": "labels"})
    val_df = splits["val"][["text", "label_id"]].rename(columns={"label_id": "labels"})

    if max_samples is not None:
        train_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=seed)
        val_cap = max(int(max_samples * 0.2), 1)
        val_df = val_df.sample(n=min(val_cap, len(val_df)), random_state=seed)

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
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    class_weights = None
    if class_weighting:
        classes = np.array(sorted(LABEL2ID.values()))
        weights = compute_class_weight(
            "balanced", classes=classes, y=train_df["labels"].to_numpy()
        )
        class_weights = torch.tensor(weights, dtype=torch.float)

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
        seed=seed,
        fp16=use_cuda,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_info = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "seed": seed,
        "class_weighting": class_weighting,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "validation_metrics": trainer.evaluate(),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune sentiment classifier")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-class-weighting", action="store_true")
    args = parser.parse_args()

    output = fine_tune(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        class_weighting=not args.no_class_weighting,
    )
    print(f"Model saved to {output}")


if __name__ == "__main__":
    main()
