# Social Media Sentiment Analyzer

Fine-tuned DistilBERT classifier for English social text (**positive**, **neutral**, **negative**), trained on YouTube comments and McDonald's reviews. Includes data cleaning, NLTK/VADER feature extraction, training with class-weighted loss, held-out evaluation, and a Gradio demo.

## Requirements

- Python 3.10+ (3.11 recommended)
- ~2 GB disk for the fine-tuned checkpoint under `models/`
- GPU optional — training and inference run on CPU; CUDA is used automatically when available
- On first feature build or Gradio prediction, NLTK downloads `vader_lexicon`, `punkt`, and `punkt_tab` into your local NLTK data dir (no manual step)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the source CSVs at the repo root (they are gitignored):

- `YoutubeCommentsDataSet.csv`
- `McDonald_s_Reviews.csv`

## Pipeline

### 1. Clean data and build features

```bash
python build_features.py
```

Writes stratified `processed/train.csv`, `val.csv`, `test.csv`, and optional `train_features.csv`. Useful flags: `--drop-three-star`, `--min-words`, `--skip-features`.

### 2. Fine-tune DistilBERT

```bash
python -m src.train
```

Saves the best checkpoint under `models/sentiment-distilbert/`. Both training sources are normalized the same way so the model cannot cheat by detecting which dataset a row came from.

### 3. Evaluate on the held-out test set

```bash
python -m src.evaluate
```

Reports per-class metrics, a confusion matrix, per-source accuracy/F1, and a majority-class baseline.

### 4. Launch the Gradio app

```bash
python app.py
```

Requires a trained model at `models/sentiment-distilbert/`.

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Gradio UI for live predictions |
| `build_features.py` | Cleaning, splits, and feature file |
| `src/data_loader.py` | Load, repair, normalize, and split data |
| `src/features.py` | NLTK/VADER text features |
| `src/train.py` | Fine-tuning with early stopping |
| `src/evaluate.py` | Test-set scoring |
| `src/predict.py` | Inference helper used by the app |

## Notes

- Feature extraction and the Gradio app call `ensure_nltk_data()` so lexicon/tokenizer assets download on first use.
- Inference applies the same `normalize_for_model` transform used in training.
- `processed/` and `models/` are intentionally ignored; regenerate them locally after pulling.
