# apps/streamlit/streamlit_app.py
import streamlit as st, sys, platform, json, traceback
from pathlib import Path

st.set_page_config(page_title="Hotel Bot — Diagnostics", layout="wide")
st.title("🩺 Hotel Bot — Cloud Diagnostics")

# ---- paths we expect
REPO_ROOT = Path(__file__).resolve().parents[2]
candidates = {
    "faq": [
        REPO_ROOT / "data" / "rag_data" / "hotel_faq.json",
        REPO_ROOT / "hotel_faq.json",
        REPO_ROOT / "data" / "hotel_faq.json",
    ],
    "model": [
        REPO_ROOT / "models" / "sentiment_model.pkl",
        REPO_ROOT / "model" / "sentiment_model.pkl",
    ],
    "vectorizer": [
        REPO_ROOT / "models" / "vectorizer.pkl",
        REPO_ROOT / "model" / "vectorizer.pkl",
    ],
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

faq_path = first_existing(candidates["faq"])
model_path = first_existing(candidates["model"])
vec_path = first_existing(candidates["vectorizer"])

# ---- show environment
st.subheader("Env")
st.code({
    "python": sys.version,
    "platform": platform.platform(),
    "repo_root": str(REPO_ROOT),
})

# ---- show file existence
st.subheader("Asset check")
st.code({
    "faq_candidates": [str(p) for p in candidates["faq"]],
    "faq_found": str(faq_path) if faq_path else None,
    "model_candidates": [str(p) for p in candidates["model"]],
    "model_found": str(model_path) if model_path else None,
    "vectorizer_candidates": [str(p) for p in candidates["vectorizer"]],
    "vectorizer_found": str(vec_path) if vec_path else None,
})

# ---- attempt imports with clear errors (no heavy loading)
errors = {}
try:
    from core.rag.retriever import FAQRetriever  # type: ignore
except Exception as e:
    errors["core.rag.retriever import"] = f"{type(e).__name__}: {e}"
try:
    from core.rag.router import HotelQARouter  # type: ignore
except Exception as e:
    errors["core.rag.router import"] = f"{type(e).__name__}: {e}"
try:
    from core.sentiment.loader import load_sentiment_model
except Exception as e:
    errors["core.sentiment.loader import"] = f"{type(e).__name__}: {e}"
try:
    from core.sentiment.predictor import predict_sentiment
except Exception as e:
    errors["core.sentiment.predictor import"] = f"{type(e).__name__}: {e}"
try:
    from core.policy.restaurant_actions import decide_action
except Exception as e:
    errors["core.policy.restaurant_actions import"] = f"{type(e).__name__}: {e}"
try:
    from services.coupons import create_free_coupon
except Exception as e:
    errors["services.coupons import"] = f"{type(e).__name__}: {e}"
try:
    from services.payments import calc_refund
except Exception as e:
    errors["services.payments import"] = f"{type(e).__name__}: {e}"

st.subheader("Import check")
st.code(errors or "✅ All core module imports succeeded")

# ---- quick coupon smoke test (proves app is responsive)
try:
    from services.coupons import create_free_coupon
    c = create_free_coupon()
    st.success(f"Coupon smoke test OK → Example: `{c['code']}` (expires {c['expires']})")
except Exception as e:
    st.error("Coupon smoke test failed")
    st.exception(e)

st.info("If this page renders, your Cloud build & entrypoint are OK. "
        "Next, we’ll re-enable the full app once assets/imports are confirmed.")
