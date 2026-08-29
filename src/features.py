"""NLTK-based text feature extraction for sentiment analysis."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def ensure_nltk_data() -> None:
    for resource in ("vader_lexicon", "punkt", "punkt_tab"):
        try:
            nltk.data.find(
                f"sentiment/{resource}.zip"
                if resource == "vader_lexicon"
                else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    ensure_nltk_data()
    return SentimentIntensityAnalyzer()


def extract_text_features(text: str) -> dict[str, float | int]:
    """Extract numeric features from a single text sample."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    text = str(text)
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    vader_scores = _vader().polarity_scores(text)
    uppercase_chars = sum(1 for c in text if c.isupper())
    alpha_chars = sum(1 for c in text if c.isalpha()) or 1

    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": (char_count / word_count) if word_count else 0.0,
        "sentence_count": max(len(nltk.sent_tokenize(text)), 1),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": uppercase_chars / alpha_chars,
        "emoji_count": len(EMOJI_PATTERN.findall(text)),
        "has_url": int(bool(URL_PATTERN.search(text))),
        "vader_neg": vader_scores["neg"],
        "vader_neu": vader_scores["neu"],
        "vader_pos": vader_scores["pos"],
        "vader_compound": vader_scores["compound"],
    }


def build_feature_frame(df: pd.DataFrame, text_col: str = "text_raw") -> pd.DataFrame:
    """Add NLTK/VADER features to a dataframe that already has text labels.

    Features are computed on the unnormalized text so that casing, punctuation,
    and emoji signal survives; the model itself consumes the normalized column.
    """
    ensure_nltk_data()
    feature_rows = [extract_text_features(t) for t in df[text_col]]
    features = pd.DataFrame(feature_rows, index=df.index)

    enriched = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    if "source" in enriched.columns:
        enriched["source_youtube"] = (enriched["source"] == "youtube").astype(int)
        enriched["source_mcdonalds"] = (enriched["source"] == "mcdonalds").astype(int)
    return enriched


def save_feature_file(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
