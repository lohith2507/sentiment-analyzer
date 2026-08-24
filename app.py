"""Gradio web app for social media sentiment analysis."""

from __future__ import annotations

import gradio as gr

from src.predict import SentimentPredictor

predictor = SentimentPredictor()

EXAMPLES = [
    "I love this product, absolutely amazing experience!",
    "Traffic was terrible this morning, worst commute ever.",
    "The update added some new settings, nothing special.",
    "Apple Pay is so convenient and easy to use at checkout.",
    "Why does it look like someone spit on my food? Never coming back.",
]


def analyze(text: str):
    result = predictor.predict(text)
    probs = result["probabilities"]
    features = result["features"]

    summary = (
        f"**Sentiment:** {result['sentiment'].title()}  \n"
        f"**Confidence:** {result['confidence']:.1%}"
    )
    # Gradio Label requires numeric confidence scores, not formatted strings.
    prob_table = {label.title(): float(score) for label, score in probs.items()}
    feature_table = {
        "Word count": int(features.get("word_count", 0)),
        "Emoji count": int(features.get("emoji_count", 0)),
        "VADER compound": round(float(features.get("vader_compound", 0.0)), 3),
        "Exclamations": int(features.get("exclamation_count", 0)),
        "Has URL": bool(features.get("has_url", 0)),
    }
    return summary, prob_table, feature_table


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="Social media text",
        placeholder="Paste a tweet, comment, or review...",
        lines=4,
    ),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Class probabilities"),
        gr.JSON(label="NLTK features"),
    ],
    title="Social Media Sentiment Analyzer",
    description=(
        "Classifies English social text as **positive**, **neutral**, or **negative** "
        "using a fine-tuned DistilBERT model trained on YouTube comments and McDonald's reviews."
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
