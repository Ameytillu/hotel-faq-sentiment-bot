# core/sentiment/loader.py
import joblib
from pathlib import Path
from typing import Tuple, Any

def load_sentiment_model(model_path: str, vectorizer_path: str) -> Tuple[Any, Any]:
    m = Path(model_path); v = Path(vectorizer_path)
    if not m.exists(): raise FileNotFoundError(f"Model not found: {m}")
    if not v.exists(): raise FileNotFoundError(f"Vectorizer not found: {v}")
    return joblib.load(m), joblib.load(v)
