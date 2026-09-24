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

- `YoutubeCommentsDataSet.csv` — columns `Comment`, `Sentiment` (`positive` / `neutral` / `negative`)
- `McDonald_s_Reviews.csv` — columns `review`, `rating` (`1 star` … `5 stars`)

### Label mapping

| Source | Raw value | Model label |
| --- | --- | --- |
| YouTube | `positive` / `neutral` / `negative` | same |
| McDonald's | 1–2 stars | `negative` |
| McDonald's | 3 stars | `neutral` (or dropped with `--drop-three-star`) |
| McDonald's | 4–5 stars | `positive` |

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
| `requirements.txt` | Python dependencies for training and the Gradio app |
| `build_features.py` | Cleaning, splits, and feature file |
| `src/data_loader.py` | Load, repair, normalize, and split data |
| `src/features.py` | NLTK/VADER text features |
| `src/train.py` | Fine-tuning with early stopping |
| `src/evaluate.py` | Test-set scoring |
| `src/predict.py` | Inference helper used by the app |

## Notes

- Feature extraction and the Gradio app call `ensure_nltk_data()` so lexicon/tokenizer assets download on first use.
- Inference applies the same `normalize_for_model` transform used in training.
- Empty or whitespace-only input returns `neutral` with confidence `0.0` and skips the model call.
- `python app.py` launches Gradio locally (`share=False` by default); pass `share=True` only if you want a temporary public link.
- `processed/` and `models/` are intentionally ignored; regenerate them locally after pulling.
