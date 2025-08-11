# core/sentiment/predictor.py
from typing import Tuple, Any

# EDIT THIS mapping to match how you trained the model
_NUM_TO_SENTIMENT = {
    "0": "negative",
    "1": "neutral",
    "2": "positive",
}

def predict_sentiment(model: Any, vectorizer: Any, text: str) -> Tuple[str, float]:
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())

    raw_label = model.classes_[idx]   # could be 'positive' or 0/1/2
    label = str(raw_label)

    # normalize numeric labels to human strings if needed
    label = _NUM_TO_SENTIMENT.get(label, label)  # leaves 'positive'/'negative' as-is

    return label, float(proba[idx])
