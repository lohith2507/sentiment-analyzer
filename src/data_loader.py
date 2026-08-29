"""Load, clean, and split English sentiment datasets (YouTube + McDonald's)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
YOUTUBE_PATH = ROOT / "YoutubeCommentsDataSet.csv"
MCDONALDS_PATH = ROOT / "McDonald_s_Reviews.csv"
PROCESSED_DIR = ROOT / "processed"

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

SEED = 42
# Short reviews like "terrible" carry real sentiment, so only empty and
# single-word rows are discarded by default.
MIN_WORDS = 2
MAX_CORRUPTION_RATIO = 0.2

# The McDonald's export was saved through a broken encoding round-trip, leaving
# runs of replacement characters where the original text is unrecoverable.
_REPLACEMENT_RUN = re.compile("(?:\ufffd|\u00ef\u00bf\u00bd)+")
_APOSTROPHE = re.compile("['\u2019\u02bc\u2018`]")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_UNDERSCORE = re.compile(r"_+")
_WHITESPACE = re.compile(r"\s+")


def corruption_ratio(text: object) -> float:
    """Fraction of a string made up of replacement characters."""
    value = str(text)
    if not value:
        return 0.0
    corrupt = sum(len(match.group()) for match in _REPLACEMENT_RUN.finditer(value))
    return corrupt / len(value)


def repair_text(text: object) -> str:
    """Drop unrecoverable replacement-character runs and collapse whitespace."""
    return _WHITESPACE.sub(" ", _REPLACEMENT_RUN.sub(" ", str(text))).strip()


def normalize_for_model(text: object) -> str:
    """Reduce text to the lowest common style shared by both sources.

    The YouTube export arrived lowercased with punctuation stripped while the
    McDonald's reviews kept both, which lets a model identify the source (and
    therefore its label distribution) instead of reading sentiment. Inference
    must apply this same normalization.
    """
    value = repair_text(text).lower()
    # Collapse contractions ("don't" -> "dont") to match the YouTube export,
    # which already lost its apostrophes, rather than splitting them in two.
    value = _APOSTROPHE.sub("", value)
    value = _PUNCT.sub(" ", value)
    value = _UNDERSCORE.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def _base_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    original = df["text_raw"].astype(str)
    # Measured before repair, since repair removes the evidence of corruption.
    df["corruption"] = original.map(corruption_ratio)
    df["text_raw"] = original.map(repair_text)
    df["text"] = original.map(normalize_for_model)
    df["source"] = source
    return df[["text", "text_raw", "sentiment", "source", "corruption"]]


def load_youtube(path: Path = YOUTUBE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"Comment": "text_raw", "Sentiment": "sentiment"})
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["sentiment"].isin(SENTIMENT_LABELS)]
    return _base_frame(df, "youtube").reset_index(drop=True)


def load_mcdonalds(
    path: Path = MCDONALDS_PATH,
    drop_three_star: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"review": "text_raw"})
    ratings = df["rating"].astype(str).str.strip()
    if drop_three_star:
        df = df[ratings.ne("3 stars")]
        ratings = ratings[ratings.ne("3 stars")]
    df["sentiment"] = ratings.map(MCDONALDS_RATING_MAP)
    df = df[df["sentiment"].notna()]
    return _base_frame(df, "mcdonalds").reset_index(drop=True)


def clean_frame(
    df: pd.DataFrame,
    min_words: int = MIN_WORDS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove corrupted, redundant, and contradictory rows.

    Runs before splitting so near-duplicates cannot straddle train and test.
    Order matters: conflicts must be detected before deduplication, or
    collapsing each text to one row hides the disagreement.
    """
    stats = {"input": len(df)}

    keep = df["corruption"] <= MAX_CORRUPTION_RATIO
    stats["dropped_corrupted"] = int((~keep).sum())
    df = df[keep]

    word_counts = df["text"].str.split().str.len().fillna(0)
    keep = word_counts >= min_words
    stats["dropped_too_short"] = int((~keep).sum())
    df = df[keep]

    # A normalized text carrying more than one label cannot be resolved, so drop
    # every row in the group rather than picking a winner arbitrarily.
    label_counts = df.groupby("text")["sentiment"].transform("nunique")
    keep = label_counts.eq(1)
    stats["dropped_conflicting"] = int((~keep).sum())
    df = df[keep]

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    stats["dropped_duplicates"] = before - len(df)

    stats["output"] = len(df)
    return df.drop(columns=["corruption"]).reset_index(drop=True), stats


def load_merged(
    youtube_path: Path = YOUTUBE_PATH,
    mcdonalds_path: Path = MCDONALDS_PATH,
    clean: bool = True,
    drop_three_star: bool = False,
    min_words: int = MIN_WORDS,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Merge both sources into one labelled frame."""
    frames = [
        load_youtube(youtube_path),
        load_mcdonalds(mcdonalds_path, drop_three_star=drop_three_star),
    ]
    merged = pd.concat(frames, ignore_index=True)

    stats: dict[str, int] = {"input": len(merged), "output": len(merged)}
    if clean:
        merged, stats = clean_frame(merged, min_words=min_words)
    else:
        merged = merged.drop(columns=["corruption"])

    merged["label_id"] = merged["sentiment"].map(LABEL2ID)
    merged = merged.reset_index(drop=True)
    return (merged, stats) if return_stats else merged


def split_dataset(
    df: pd.DataFrame,
    seed: int = SEED,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Stratified three-way split; the test portion is never used for training."""
    train_df, holdout = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["sentiment"],
    )
    val_df, test_df = train_test_split(
        holdout,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=holdout["sentiment"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path = PROCESSED_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["text", "text_raw", "sentiment", "source", "label_id"]
    paths = {}
    for name, frame in splits.items():
        path = output_dir / f"{name}.csv"
        frame[columns].to_csv(path, index=False)
        paths[name] = path
    return paths
