"""Load and merge English sentiment datasets (YouTube + McDonald's)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
YOUTUBE_PATH = ROOT / "YoutubeCommentsDataSet.csv"
MCDONALDS_PATH = ROOT / "McDonald_s_Reviews.csv"

SENTIMENT_LABELS = ("negative", "neutral", "positive")
LABEL2ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

MCDONALDS_RATING_MAP = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def load_youtube(path: Path = YOUTUBE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"Comment": "text", "Sentiment": "sentiment"})
    df["text"] = _clean_text(df["text"])
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["source"] = "youtube"
    df = df[df["text"].ne("") & df["sentiment"].isin(SENTIMENT_LABELS) & df["text"].notna()]
    return df[["text", "sentiment", "source"]].reset_index(drop=True)


def load_mcdonalds(path: Path = MCDONALDS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"review": "text", "rating": "rating_raw"})
    df["text"] = _clean_text(df["text"])
    df["sentiment"] = df["rating_raw"].map(MCDONALDS_RATING_MAP)
    df["source"] = "mcdonalds"
    df = df[df["text"].ne("") & df["sentiment"].notna() & df["text"].notna()]
    return df[["text", "sentiment", "source"]].reset_index(drop=True)


def load_merged(
    youtube_path: Path = YOUTUBE_PATH,
    mcdonalds_path: Path = MCDONALDS_PATH,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Merge YouTube comments and McDonald's reviews into one training frame."""
    frames = [load_youtube(youtube_path), load_mcdonalds(mcdonalds_path)]
    merged = pd.concat(frames, ignore_index=True)
    if dedupe:
        merged = merged.drop_duplicates(subset=["text"], keep="first")
    merged["label_id"] = merged["sentiment"].map(LABEL2ID)
    return merged.reset_index(drop=True)


def save_processed(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    df[["text", "sentiment", "source", "label_id"]].to_csv(train_path, index=False)
    return train_path, output_dir
