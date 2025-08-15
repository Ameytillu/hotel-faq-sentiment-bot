# ------------------------------------------------------------
# Hotel FAQ + Sentiment (offline) — Streamlit Cloud version
# ------------------------------------------------------------
from __future__ import annotations

import re
import sys
from pathlib import Path
import streamlit as st

# ---------- Page config + heartbeat (renders instantly)
st.set_page_config(page_title="Hotel Chatbot (Offline)", page_icon="🛎️", layout="wide")
st.write("✅ Boot OK")  # If you don't see this, it's a build/entry issue.

# ---------- Repo root + import path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------- Fallback import by file path (if package imports fail)
def _import_by_path(module_name: str, rel_path: str):
    import importlib.util
    full = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader, f"Cannot import {module_name} from {full}"
    spec.loader.exec_module(mod)  # type: ignore
    return mod

IMPORT_MODE = {}

try:
    from core.rag.retriever import FAQRetriever  # type: ignore
    IMPORT_MODE["retriever"] = "package"
except Exception:
    FAQRetriever = _import_by_path("core.rag.retriever", "core/rag/retriever.py").FAQRetriever  # type: ignore
    IMPORT_MODE["retriever"] = "file"

try:
    from core.rag.router import HotelQARouter  # type: ignore
    IMPORT_MODE["router"] = "package"
except Exception:
    HotelQARouter = _import_by_path("core.rag.router", "core/rag/router.py").HotelQARouter  # type: ignore
    IMPORT_MODE["router"] = "file"

from core.sentiment.loader import load_sentiment_model
from core.sentiment.predictor import predict_sentiment
from core.policy.restaurant_actions import decide_action
from services.coupons import create_free_coupon
from services.payments import calc_refund

# ---------- File discovery (works whether files are at root or under data/)
def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

FAQ_PATH = _first_existing([
    REPO_ROOT / "data" / "rag_data" / "hotel_faq.json",
    REPO_ROOT / "hotel_faq.json",
    REPO_ROOT / "data" / "hotel_faq.json",
])

MODEL_PATH = _first_existing([
    REPO_ROOT / "models" / "sentiment_model.pkl",
    REPO_ROOT / "model" / "sentiment_model.pkl",
])

VECTORIZER_PATH = _first_existing([
    REPO_ROOT / "models" / "vectorizer.pkl",
    REPO_ROOT / "model" / "vectorizer.pkl",
])

# =========================================================
# Lazy, cached loaders (heavy stuff only runs once)
# bump _version to force refresh if needed
# =========================================================
@st.cache_resource(show_spinner="Loading FAQ retriever & router…")
def get_retriever_and_router(_version: str = "v1"):
    if not FAQ_PATH:
        raise FileNotFoundError("hotel_faq.json not found. Place it at data/rag_data/ or repo root.")
    retriever = FAQRetriever(str(FAQ_PATH))
    router = HotelQARouter(str(FAQ_PATH), retriever)
    return retriever, router

@st.cache_resource(show_spinner="Loading sentiment model…")
def get_model_and_vec(_version: str = "v1"):
    if not MODEL_PATH or not VECTORIZER_PATH:
        raise FileNotFoundError("sentiment_model.pkl/vectorizer.pkl not found under models/.")
    return load_sentiment_model(str(MODEL_PATH), str(VECTORIZER_PATH))

# =========================================================
# Intent detection (used in Auto mode)
# =========================================================
REVIEW_KW = {
    "food","dish","meal","restaurant","breakfast","lunch","dinner",
    "pizza","burger","pasta","fries","soup","salad","dessert","service",
    "waiter","chef","taste","portion","ambience","ambiance","fresh","cold","stale"
}
OPINION_KW = {
    "good","great","amazing","awesome","delicious","tasty","love","loved","excellent",
    "bad","terrible","awful","disappointed","hate","overpriced","ok","okay","mediocre",
    "cold","stale","undercooked","burnt","salty","sweet","bland","fresh","friendly","rude"
}
RATING_RE = re.compile(r"\b([1-5])\s*/\s*5\b|\b([1-5])\s*star", re.I)

def detect_intent(text: str) -> str:
    t = text.lower().strip()
    if not t:
        return "FAQ"
    rating_like = bool(RATING_RE.search(t)) or ("⭐" in t)
    review_hit  = any(k in t for k in REVIEW_KW)
    opinion_hit = any(k in t for k in OPINION_KW)
    longish     = len(t.split()) >= 6
    if (review_hit and (opinion_hit or longish)) or rating_like:
        return "REVIEW"
    return "FAQ"

# =========================================================
# Main UI
# =========================================================
def main():
    st.title("🛎️ Hotel Chatbot — Offline (FAQ + Sentiment)")

    # Sidebar + diagnostics
    with st.sidebar:
        faq_thr = st.slider("FAQ match threshold", 0.0, 1.0, 0.60, 0.01)
        mode = st.radio("Mode", ["Auto", "FAQ", "Review"], horizontal=True)
        st.caption("Auto detects intent. Positive → coupon • Negative → 15% refund.")

        with st.expander("Diagnostics"):
            st.write("Repo root:", REPO_ROOT)
            st.write("Import mode:", IMPORT_MODE)
            st.write(f"FAQ path: {FAQ_PATH} — exists: {bool(FAQ_PATH and FAQ_PATH.exists())}")
            st.write(f"Model path: {MODEL_PATH} — exists: {bool(MODEL_PATH and MODEL_PATH.exists())}")
            st.write(f"Vectorizer path: {VECTORIZER_PATH} — exists: {bool(VECTORIZER_PATH and VECTORIZER_PATH.exists())}")

    # Lazy-load heavy resources
    try:
        retriever, router = get_retriever_and_router()
    except Exception as e:
        st.error("Failed to load FAQ router/retriever. Ensure hotel_faq.json is committed.")
        st.exception(e)
        return

    try:
        model, vectorizer = get_model_and_vec()
    except Exception as e:
        st.error("Failed to load sentiment model/vectorizer (pkl files).")
        st.exception(e)
        return

    # Chat history
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["text"])

    # Input
    user = st.chat_input("Ask a hotel question or submit a restaurant review…")
    if not user:
        return

    st.session_state.chat.append({"role": "user", "text": user})
    st.chat_message("user").write(user)

    intent = {"Auto": detect_intent(user), "FAQ": "FAQ", "Review": "REVIEW"}[mode]
    st.caption(f"Detected intent: **{intent}**")

    blocks: list[str] = []

    # FAQ branch
    if intent == "FAQ":
        try:
            result = router.answer(user, threshold_retriever=faq_thr)
        except Exception as e:
            st.error("FAQ routing failed.")
            st.exception(e)
            return

        if result.get("found"):
            kind = result.get("kind", "rule")
            if kind == "rule":
                blocks.append(
                    f"**Answer:** {result['answer']}\n\n_Matched (rule):_ “{result.get('question','')}”"
                )
            else:
                sim = result.get("similarity", 0.0)
                blocks.append(
                    f"**FAQ:** {result['answer']}\n\n_Matched:_ “{result.get('question','')}” (sim={sim:.2f})"
                )
        else:
            blocks.append("I couldn’t find a close FAQ match.")
            sugg = result.get("suggestions", [])
            if sugg:
                items = []
                for s in sugg[:5]:
                    q = s.get("question") or s.get("q") or ""
                    sc = s.get("score", 0.0)
                    items.append(f"- {q} _(sim={sc:.2f})_")
                if items:
                    blocks.append("Did you mean:\n" + "\n".join(items))
            blocks.append("Try rephrasing or lower the FAQ threshold in the sidebar.")

    # Review (sentiment) branch
    else:
        try:
            label, score = predict_sentiment(model, vectorizer, user)
        except Exception as e:
            st.error("Sentiment prediction failed.")
            st.exception(e)
            return

        action, senti_msg = decide_action(label, score)

        if action == "COUPON_FREE":
            c = create_free_coupon()
            blocks.append(
                f"{senti_msg}\n\n**Your coupon:** `{c['code']}` (valid until {c['expires']})."
            )
            st.info(f"Coupon `{c['code']}` generated locally.")
        elif action == "REFUND_15":
            blocks.append(f"{senti_msg}\n\nPlease confirm your order details to compute the refund:")
            with st.chat_message("assistant"):
                base_key = f"refund_{len(st.session_state.chat)}"
                col1, col2 = st.columns(2)
                with col1:
                    order_id = st.text_input("Order ID", key=f"order_{base_key}")
                with col2:
                    amount = st.number_input("Order Amount ($)", min_value=0.0, step=0.5, key=f"amount_{base_key}")
                if st.button("Compute 15% refund", key=f"btn_{base_key}"):
                    if order_id and amount > 0:
                        r = calc_refund(amount, 15.0)
                        st.success(
                            f"Refund: **${r['refund_amount']:.2f}** "
                            f"(15% of ${amount:.2f}) for Order **{order_id}**."
                        )
                    else:
                        st.warning("Enter a valid Order ID and amount.")
        else:
            blocks.append(senti_msg)  # neutral acknowledgement

    # Output assistant message
    bot_text = "\n\n".join(blocks) if blocks else "Sorry, I couldn’t process that."
    st.session_state.chat.append({"role": "assistant", "text": bot_text})
    st.chat_message("assistant").write(bot_text)


if __name__ == "__main__":
    main()
