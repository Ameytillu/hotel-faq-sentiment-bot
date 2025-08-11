# core/rag/retriever.py
import json, re
from pathlib import Path
from typing import Dict, List, Tuple

# --------- small helpers ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SEP = re.compile(r"[\s\-_\/]+")
TOK = re.compile(r"[a-z0-9]+")

def _norm(s: str) -> str:
    return _SEP.sub(" ", s.lower().strip())

def _tok(s: str) -> List[str]:
    return TOK.findall(_norm(s))

def _safe_norm(scores) -> List[float]:
    # normalize to 0..1 without div-by-zero or inf
    try:
        mx = float(max(scores)) if len(scores) else 0.0
    except ValueError:
        mx = 0.0
    if mx <= 1e-9:
        return [0.0 for _ in scores]
    out = [float(s) / mx for s in scores]
    # clamp just in case
    return [max(0.0, min(1.0, x)) for x in out]

# --------- flatten knowledge ----------
def _flatten(raw: Dict) -> Dict:
    """Return entries + light domain metadata (room types)."""
    if isinstance(raw, list):  # legacy: file is just a list of {q,a}
        raw = {"faq": raw}

    entries: List[Dict] = []

    # 1) Free-form FAQ
    for it in raw.get("faq", []) or []:
        entries.append({"q": it["question"].strip(), "a": it["answer"].strip(), "alts": it.get("alts", [])})

    # 2) Rooms
    rooms = raw.get("rooms", []) or []
    room_types = [r["room_type"] for r in rooms if r.get("room_type")]
    if room_types:
        entries.append({
            "q": "what room types do you have",
            "a": f"We offer: {', '.join(room_types)}.",
            "alts": ["room types", "room categories", "what rooms do you have", "available room types"]
        })
    for r in rooms:
        rt = (r.get("room_type") or "").strip()
        if not rt: 
            continue
        desc = r.get("description", "")
        feat = ", ".join(r.get("features", [])[:6])
        price = r.get("price_per_night")
        ans = f"{rt}: {desc}"
        if feat:  ans += f" Key features: {feat}."
        if price: ans += f" Price per night: ${price}."
        entries.append({
            "q": f"tell me about the {rt.lower()}",
            "a": ans,
            "alts": [f"do you have {rt.lower()}", f"{rt.lower()} details", f"{rt.lower()} price", f"{rt.lower()} room"]
        })

    # 3) Menus (minimal)
    menus = raw.get("menus", {}) or {}
    for meal in ["breakfast", "lunch", "dinner", "brunch"]:
        items = menus.get(meal, []) or []
        if items:
            listing = ", ".join(f"{i['name']} (${i['price']})" for i in items if "name" in i and "price" in i)
            entries.append({
                "q": f"what is the {meal} menu",
                "a": f"Our {meal} menu includes: {listing}.",
                "alts": [f"{meal} menu", f"{meal} options", f"what can i have for {meal}"]
            })
    if any(menus.get(m) for m in ["breakfast","lunch","dinner","brunch"]):
        entries.append({
            "q": "restaurant menu",
            "a": "You can view today’s restaurant menu at the front desk or via the QR code in the lobby.",
            "alts": ["food menu", "today's menu", "what is the menu"]
        })

    return {"entries": entries, "room_types": room_types}

# --------- retriever (Embeddings -> BM25/fuzzy fallback) ----------
class FAQRetriever:
    def __init__(self, path: str):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        flat = _flatten(raw)
        self.entries: List[Dict] = flat["entries"]
        self.room_types: List[str] = flat["room_types"]

        # index strings (q + alts)
        self.docs: List[str] = [" | ".join([e["q"], *e.get("alts", [])]) for e in self.entries]
        self.norm_docs: List[str] = [_norm(t) for t in self.docs]

        # try embeddings first
        self.embed = None
        try:
            from sentence_transformers import SentenceTransformer
            import faiss, numpy as np
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=str(REPO_ROOT / "models" / "embeddings")
            )
            embs = self.model.encode(self.docs, normalize_embeddings=True, convert_to_numpy=True)
            self.index = faiss.IndexFlatIP(embs.shape[1]); self.index.add(embs)
            self.embed = True
        except Exception:
            # fallback: BM25 + fuzzy
            from rank_bm25 import BM25Okapi
            self.embed = False
            self.bm25 = BM25Okapi([_tok(t) for t in self.docs])

    # --------- domain router (rooms) ----------
    def _room_route(self, query: str):
        q = _norm(query)
        if "room" not in q:
            return None
        # tokenize & look for known room keywords
        want = None
        for key in ["single","double","twin","queen","king"]:
            if key in q:
                want = key
                break
        if not want:
            # generic "rooms" question
            if self.room_types:
                return {
                    "found": True,
                    "similarity": 1.0,
                    "question": "what room types do you have",
                    "answer": f"We offer: {', '.join(self.room_types)}.",
                    "hits": []
                }
            return None

        # map user word to available type, or say not available
        avail = [rt.lower() for rt in self.room_types]
        if want in avail:
            # pick the entry describing that room
            for i, e in enumerate(self.entries):
                if f"tell me about the {want}" in e["q"]:
                    return {
                        "found": True,
                        "similarity": 1.0,
                        "question": e["q"],
                        "answer": e["a"],
                        "hits": [(i, 1.0)]
                    }
        else:
            if self.room_types:
                return {
                    "found": True,
                    "similarity": 1.0,
                    "question": f"{want} room availability",
                    "answer": f"We don’t have a {want.capitalize()} Room. Available types: {', '.join(self.room_types)}.",
                    "hits": []
                }
        return None

    # --------- generic search ----------
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        if self.embed:
            v = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            sims, idxs = self.index.search(v, k)
            return [(int(i), float(s)) for i, s in zip(idxs[0], sims[0])]

        # BM25 + fuzzy hybrid
        from rapidfuzz.fuzz import token_set_ratio
        bm = self.bm25.get_scores(_tok(query))
        bm = _safe_norm(bm)
        qn = _norm(query)
        fuzz = [token_set_ratio(qn, d) / 100.0 for d in self.norm_docs]
        # combine, clamp
        sc = [max(0.0, min(1.0, 0.6 * b + 0.4 * f)) for b, f in zip(bm, fuzz)]
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:k]
        return [(i, float(sc[i])) for i in order]

    def answer(self, query: str, k: int = 5, threshold: float = 0.60) -> Dict:
        # 1) hotel-specific routing first (rooms)
        routed = self._room_route(query)
        if routed:
            return routed

        # 2) retrieval
        hits = self.search(query, k=k)
        best_idx, best_sim = hits[0]
        if best_sim < threshold:
            # unique suggestions (top-k)
            seen, sugg = set(), []
            for i, s in hits:
                q = self.entries[i]["q"]
                if q in seen: 
                    continue
                seen.add(q)
                sugg.append({"question": q, "score": round(float(s), 2)})
            return {"found": False, "similarity": round(float(best_sim), 2), "suggestions": sugg}

        return {
            "found": True,
            "similarity": round(float(best_sim), 2),
            "question": self.entries[best_idx]["q"],
            "answer": self.entries[best_idx]["a"],
            "hits": [(i, round(float(s), 2)) for i, s in hits]
        }
