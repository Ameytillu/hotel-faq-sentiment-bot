# ------------------------------------------------------------
# Hotel FAQ + Sentiment (offline) — Streamlit app
# ------------------------------------------------------------
from __future__ import annotations

# ---------- robust path setup (works locally & on Streamlit Cloud)
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]  # …/hotel_faq_sentiment_bot
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _import_by_path(module_name: str, rel_path: str):
    """Import a module by a relative path as a fallback."""
    import importlib.util
    full = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader, f"Cannot import {module_name} from {full}"
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---------- try normal package imports, then fallback to path imports
_IMPORT_MODE = {}
try:
    from core.rag.retriever import FAQRetriever           # type: ignore
    _IMPORT_MODE["retriever"] = "package"
except Exception:
    FAQRetriever = _import_by_path("core.rag.retriever", "core/rag/retriever.py").FAQRetriever  # type: ignore
    _IMPORT_MODE["retriever"] = "file"

try:
    from core.rag.router import HotelQARouter             # type: ignore
    _IMPORT_MODE["router"] = "package"
except Exception:
    HotelQARouter = _import_by_path("core.rag.router", "core/rag/router.py").HotelQARouter      # type: ignore
    _IMPORT_MODE["router"] = "file"

# ---------- std / local imports
import re
import streamlit as st

from core.sentiment.loader import load_sentiment_model
from core.sentiment.predictor import predict_sentiment
from core.policy.restaurant_actions import decide_action
from services.coupons import create_free_coupon
from services.payments import calc_refund

# ---------- paths
FAQ_PATH        = REPO_ROOT / "data" / "rag_data" / "hotel_faq.json"
MODEL_PATH      = REPO_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = REPO_ROOT / "models" / "vectorizer.pkl"

# ---------- page config
st.set_page_config(page_title="Hotel Chatbot (Offline)", page_icon="🛎️")
st.title("🛎️ Hotel Chatbot — Offline (FAQ + Sentiment)")

# =========================================================
# Cache heavy resources
# bump versions below to force cloud rebuild if needed
# =========================================================
@st.cache_resource
def get_retriever_and_router(_version: str = "v8"):
    retriever = FAQRetriever(str(FAQ_PATH))
    router = HotelQARouter(str(FAQ_PATH), retriever)
    return retriever, router

@st.cache_resource
def get_model_and_vec(_version: str = "v2"):
    return load_sentiment_model(str(MODEL_PATH), str(VECTORIZER_PATH))

# ---------- load with friendly errors
try:
    retriever, router = get_retriever_and_router()
except Exception as e:
    st.error("Failed to load FAQ router/retriever. Check `data/rag_data/hotel_faq.json` formatting.")
    st.exception(e)
    st.stop()

try:
    model, vectorizer = get_model_and_vec()
except Exception as e:
    st.error("Failed to load sentiment model/vectorizer (.pkl files).")
    st.exception(e)
    st.stop()

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
# Sidebar controls + diagnostics
# =========================================================
with st.sidebar:
    faq_thr = st.slider("FAQ match threshold", 0.0, 1.0, 0.60, 0.01)
    st.caption("Mode: Auto detects FAQ vs Review. Positive → coupon • Negative → 15% refund.")
    mode = st.radio("Mode", ["Auto", "FAQ", "Review"], horizontal=True)

    with st.expander("Diagnostics"):
        st.write(f"FAQ exists: {FAQ_PATH.exists()}  →  {FAQ_PATH}")
        st.write(f"Model exists: {MODEL_PATH.exists()}  →  {MODEL_PATH}")
        st.write(f"Vectorizer exists: {VECTORIZER_PATH.exists()}  →  {VECTORIZER_PATH}")
        st.write("Import mode:", _IMPORT_MODE)
        st.write("FAQ backend:", getattr(retriever, "backend_name", "?"))
        try:
            st.write("Indexed entries:", len(getattr(retriever, "entries", [])))
        except Exception:
            pass
        try:
            st.write("Model classes_:", list(getattr(model, "classes_", [])))
        except Exception:
            pass

# =========================================================
# Chat history
# =========================================================
if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["text"])

# =========================================================
# Chat input
# =========================================================
user = st.chat_input("Ask a hotel question or submit a restaurant review…")
if not user:
    st.stop()

st.session_state.chat.append({"role": "user", "text": user})
st.chat_message("user").write(user)

intent = {"Auto": detect_intent(user), "FAQ": "FAQ", "Review": "REVIEW"}[mode]
st.caption(f"Detected intent: **{intent}**")

blocks: list[str] = []

# =========================================================
# FAQ branch (Router → Retriever)
# =========================================================
if intent == "FAQ":
    try:
        # router.answer is the main entry; it will use rules first then retrieval
        result = router.answer(user, threshold_retriever=faq_thr)
    except Exception as e:
        st.error("FAQ routing failed.")
        st.exception(e)
        st.stop()

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
        blocks.append("Try rephrasing or lowering the FAQ threshold in the sidebar.")

# =========================================================
# Review (sentiment) branch
# =========================================================
else:
    try:
        label, score = predict_sentiment(model, vectorizer, user)
    except Exception as e:
        st.error("Sentiment prediction failed.")
        st.exception(e)
        st.stop()

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

# ---------- output assistant message
bot_text = "\n\n".join(blocks) if blocks else "Sorry, I couldn’t process that."
st.session_state.chat.append({"role": "assistant", "text": bot_text})
st.chat_message("assistant").write(bot_text)
