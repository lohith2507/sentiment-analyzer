"""Run sentiment inference with the fine-tuned model."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import ID2LABEL, normalize_for_model
from src.features import extract_text_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "sentiment-distilbert"


class SentimentPredictor:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, max_length: int = 192):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> dict:
        raw_text = (text or "").strip()
        if not raw_text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {label: 0.0 for label in ID2LABEL.values()},
                "features": {},
            }

        # The model was trained on normalized text, so serving must match.
        model_text = normalize_for_model(raw_text)

        encoded = self.tokenizer(
            model_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        label_id = int(torch.argmax(logits, dim=-1).item())
        probabilities = {ID2LABEL[i]: round(probs[i], 4) for i in range(len(probs))}

        return {
            "sentiment": ID2LABEL[label_id],
            "confidence": round(probs[label_id], 4),
            "probabilities": probabilities,
            "features": extract_text_features(raw_text),
        }
