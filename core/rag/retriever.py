# core/rag/retriever.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---------- Optional backends ----------
# BM25 (best for short QA); optional
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

# TF-IDF (lightweight, great fallback)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize(text))


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Try to recover BOM/encoding quirks
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _q(q: str) -> str:
    # Normalize questions once for indexing
    return _normalize(q)


def _make_entry(q: str, a: str) -> Dict[str, str]:
    return {"q": _q(q), "a": a.strip()}


def _build_entries_from_data(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build a flat list of {q, a} from your rich hotel JSON.
    - Uses explicit 'faq' list when available.
    - Also synthesizes simple QAs from common sections so users get more hits.
    """
    entries: List[Dict[str, str]] = []

    # 1) Explicit FAQs
    if isinstance(raw.get("faq"), list):
        for item in raw["faq"]:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if q and a:
                entries.append(_make_entry(q, a))
            # optional alternates
            alts = item.get("alts") or item.get("alternates") or []
            if isinstance(alts, list):
                for alt in alts:
                    if isinstance(alt, str) and alt.strip():
                        entries.append(_make_entry(alt, a))

    # 2) Hotel policies → simple QAs
    pol = raw.get("hotel_policies", {})
    if isinstance(pol, dict):
        for k, v in pol.items():
            if isinstance(v, str) and v.strip():
                # generate a couple of light variants per policy key
                k_norm = k.replace("_", " ").strip()
                variants = [
                    f"{k_norm}",
                    f"what is {k_norm}",
                    f"{k_norm}?",
                ]
                for z in variants:
                    entries.append(_make_entry(z, v))

    # 3) Rooms → "Tell me about {room_type}"
    rooms = raw.get("rooms", [])
    if isinstance(rooms, list):
        for r in rooms:
            rt = (r.get("room_type") or "").strip()
            desc = r.get("description") or ""
            if rt and desc:
                entries.append(_make_entry(f"what is {rt}", desc))
                entries.append(_make_entry(f"tell me about {rt}", desc))
                entries.append(_make_entry(f"{rt}", desc))

    # 4) Amenities → timings/basic info
    ams = raw.get("amenities", [])
    if isinstance(ams, list):
        for a in ams:
            name = (a.get("amenity_name") or "").strip()
            desc = a.get("description") or ""
            rules = a.get("rules") or {}
            if name and desc:
                entries.append(_make_entry(f"{name}", desc))
                entries.append(_make_entry(f"tell me about {name}", desc))
            if isinstance(rules, dict) and rules.get("timings"):
                entries.append(_make_entry(f"{name} timings", str(rules["timings"])))

    # 5) Menus → lightweight coverage
    menus = raw.get("menus", {})
    if isinstance(menus, dict):
        for meal, items in menus.items():
            if isinstance(items, list):
                for it in items:
                    nm = (it.get("name") or "").strip()
                    dsc = it.get("description") or ""
                    if nm and dsc:
                        entries.append(_make_entry(f"what is in {nm}", dsc))
                        entries.append(_make_entry(f"{nm}", dsc))
                # Add a generic question per meal
                entries.append(_make_entry(f"what is in {meal} menu", f"{meal.title()} menu available."))

    # De-duplicate by (q, a)
    seen = set()
    deduped: List[Dict[str, str]] = []
    for e in entries:
        key = (e["q"], e["a"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    return deduped


class FAQRetriever:
    """
    Robust FAQ retriever:
      - Prefers BM25 if rank_bm25 is available
      - Else falls back to TF-IDF cosine (scikit-learn)
      - Else falls back to a tiny keyword/Jaccard scorer
    Works with your hospitality JSON (faq + optional sections).
    """

    def __init__(self, json_path: str | Path):
        self.path = Path(json_path)
        raw = _read_json(self.path)
        self.db_version = str(raw.get("db_version", "")).strip() or None

        self.entries: List[Dict[str, str]] = _build_entries_from_data(raw)
        self.questions: List[str] = [e["q"] for e in self.entries]
        self._backend = "keyword"

        # BM25
        if HAVE_BM25 and self.questions:
            self._bm25 = BM25Okapi([_tokens(q) for q in self.questions])
            self._backend = "bm25"
            return

        # TF-IDF
        if HAVE_SK and self.questions:
            self._vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self._tfidf = self._vec.fit_transform(self.questions)
            self._backend = "tfidf"
            return

        # Keyword/Jaccard fallback (no external deps)
        self._q_tokens = [set(_tokens(q)) for q in self.questions]
        self._backend = "keyword"

    # ---------- public helpers ----------
    @property
    def index_size(self) -> int:
        return len(self.entries)

    @property
    def sample_questions(self) -> List[str]:
        return [e["q"] for e in self.entries[: min(8, len(self.entries))]]

    @property
    def backend(self) -> str:
        return self._backend

    # ---------- retrieval ----------
    def _scores_bm25(self, query: str) -> Tuple[List[float], float]:
        scores = list(self._bm25.get_scores(_tokens(query)))  # type: ignore[attr-defined]
        # normalize per-query BM25 into [0,1] for a threshold that "feels" consistent
        mx, mn = max(scores) if scores else 1.0, min(scores) if scores else 0.0
        norm = 0.0 if mx == mn else (max(scores) - mn) / (mx - mn)
        return scores, float(norm)

    def _scores_tfidf(self, query: str) -> Tuple[List[float], float]:
        qv = self._vec.transform([query])  # type: ignore[attr-defined]
        sims = cosine_similarity(qv, self._tfidf).ravel().tolist()  # type: ignore[attr-defined]
        return sims, float(max(sims) if sims else 0.0)

    def _scores_keyword(self, query: str) -> Tuple[List[float], float]:
        # Jaccard on tokens as a last resort
        qset = set(_tokens(query))
        scores: List[float] = []
        for toks in self._q_tokens:
            inter = len(qset & toks)
            union = len(qset | toks) or 1
            scores.append(inter / union)
        return scores, float(max(scores) if scores else 0.0)

    def _rank(self, query: str) -> Tuple[int, float, List[Tuple[int, float]]]:
        if not query.strip() or not self.entries:
            return -1, 0.0, []

        if self._backend == "bm25":
            scores, best_norm = self._scores_bm25(query)
        elif self._backend == "tfidf":
            scores, best_norm = self._scores_tfidf(_normalize(query))
        else:
            scores, best_norm = self._scores_keyword(query)

        # rank indices by score desc
        ranked = sorted([(i, s) for i, s in enumerate(scores)], key=lambda x: x[1], reverse=True)
        best_idx = ranked[0][0] if ranked else -1
        return best_idx, best_norm, ranked

    def answer(self, query: str, threshold: float = 0.6, topk: int = 3) -> Dict[str, Any]:
        """
        Returns:
          {
            "found": bool,
            "score": float (0..1 approx),
            "question": str,
            "answer": str,
            "backend": "bm25"|"tfidf"|"keyword",
            "candidates": [{"question":..., "score":...}, ...]
          }
        """
        best_idx, best_norm, ranked = self._rank(query)
        if best_idx < 0:
            return {
                "found": False,
                "score": 0.0,
                "question": "",
                "answer": "",
                "backend": self._backend,
                "candidates": [],
            }

        found = best_norm >= threshold
        best_q = self.questions[best_idx]
        best_a = self.entries[best_idx]["a"] if found else ""

        # top-k suggestions (skip the top hit)
        cands = []
        for i, (idx, sc) in enumerate(ranked[1 : 1 + max(0, topk)]):
            cands.append({"question": self.questions[idx], "score": float(sc)})

        return {
            "found": found,
            "score": float(best_norm),
            "question": best_q,
            "answer": best_a,
            "backend": self._backend,
            "candidates": cands,
        }
