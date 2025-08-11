# apps/streamlit/streamlit_app.py

# ---------- robust module imports header ----------
from pathlib import Path
import sys

# find project root (…/hotel_faq_sentiment_bot)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# helper: import a module by file path (fallback when package import fails)
def _import_by_path(module_name: str, rel_path: Path):
    import importlib.util
    full = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec.loader is not None, f"Cannot load {module_name} from {full}"
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# try normal imports; fallback to file imports if needed
_IMPORT_MODE = {}

try:
    from core.rag.retriever import FAQRetriever  # type: ignore
    _IMPORT_MODE["retriever"] = "package"
except Exception:
    FAQRetriever = _import_by_path("core.rag.retriever", Path("core/rag/retriever.py")).FAQRetriever
    _IMPORT_MODE["retriever"] = "file"

try:
    from core.rag.router import HotelQARouter  # type: ignore
    _IMPORT_MODE["router"] = "package"
except Exception:
    HotelQARouter = _import_by_path("core.rag.router", Path("core/rag/router.py")).HotelQARouter
    _IMPORT_MODE["router"] = "file"

# ---------- std / local imports ----------
import re
import streamlit as st
from core.sentiment.loader import load_sentiment_model
from core.sentiment.predictor import predict_sentiment
from core.policy.restaurant_actions import decide_action
from services.coupons import create_free_coupon
from services.payments import calc_refund

# ---------- paths ----------
FAQ_PATH        = REPO_ROOT / "data" / "rag_data" / "hotel_faq.json"
MODEL_PATH      = REPO_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = REPO_ROOT / "models" / "vectorizer.pkl"

# ---------- page config ----------
st.set_page_config(page_title="Hotel Chatbot (Offline)", page_icon="🛎️")
st.title("🛎️ Hotel Chatbot — Offline (FAQ + Sentiment)")

# =========================================================
# Cache heavy resources (with a version to bust Streamlit cache)
# =========================================================
@st.cache_resource
def get_retriever_and_router(version: str = "v10"):
    """Build + return retriever and router with a guaranteed fresh import."""
    import importlib
    import core.rag.retriever as retr_mod
    import core.rag.router as router_mod

    # avoid stale bytecode/module state in hosted envs
    importlib.invalidate_caches()
    retr_mod = importlib.reload(retr_mod)
    router_mod = importlib.reload(router_mod)

    retriever = retr_mod.FAQRetriever(str(FAQ_PATH))
    router = router_mod.HotelQARouter(str(FAQ_PATH), retriever)
    return retriever, router, retr_mod  # return module for diagnostics, too

@st.cache_resource
def get_model_and_vec():
    return load_sentiment_model(str(MODEL_PATH), str(VECTORIZER_PATH))

# ---------- load with friendly errors ----------
try:
    retriever, router, retr_mod = get_retriever_and_router(version="v10")
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

# ---------- diagnostics (AFTER retriever is built) ----------
import inspect
st.caption(f"retriever module path: {getattr(retr_mod, '__file__', 'unknown')}")
st.caption(f"HAVE_BM25: {getattr(retr_mod, 'HAVE_BM25', None)}")
st.caption(f"backend: {getattr(retriever, 'backend_name', getattr(retriever, 'backend', '?'))}")
# st.code("\n".join(inspect.getsource(retr_mod).splitlines()[:25]))  # optional

# =========================================================
# Simple intent detection (used in Auto mode)
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
        st.write("Indexed entries:", len(getattr(retriever, "entries", [])))
        st.write("Router room types:", getattr(router, "room_types", []))
        try:
            st.write("Sample questions:", [e["q"] for e in retriever.entries[:3]])
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

# echo user
st.session_state.chat.append({"role": "user", "text": user})
st.chat_message("user").write(user)

# choose intent
intent = {"Auto": detect_intent(user), "FAQ": "FAQ", "Review": "REVIEW"}[mode]
st.caption(f"Detected intent: **{intent}**")

blocks = []

# =========================================================
# FAQ branch (Router → Retriever)
# =========================================================
if intent == "FAQ":
    result = router.answer(user, threshold_retriever=faq_thr)

    if result.get("found"):
        kind = result.get("kind", "rule")
        if kind == "rule":
            blocks.append(
                f"**Answer:** {result['answer']}\n\n_Matched (rule):_ “{result['question']}”"
            )
        else:  # retrieval
            sim = result.get("similarity", 0)
            blocks.append(
                f"**FAQ:** {result['answer']}\n\n_Matched:_ “{result['question']}” (sim={sim:.2f})"
            )
    else:
        blocks.append("I couldn’t find a close FAQ match.")
        sugg = result.get("suggestions", [])
        if sugg:
            blocks.append(
                "Did you mean:\n" + "\n".join(
                    [f"- {s['question']} _(sim={s['score']:.2f})_" for s in sugg[:5]]
                )
            )
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

# output assistant message
bot_text = "\n\n".join(blocks)
st.session_state.chat.append({"role": "assistant", "text": bot_text})
st.chat_message("assistant").write(bot_text)
