# core/rag/router.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

class HotelQARouter:
    """
    Minimal pass-through router so the app runs.
    Replace later with the rule-based router if you want.
    """
    def __init__(self, json_path: str, retriever: Any):
        self.path = Path(json_path)
        self.data: Dict = json.loads(self.path.read_text(encoding="utf-8"))
        self.retriever = retriever
        rooms = self.data.get("rooms", []) if isinstance(self.data, dict) else []
        self.room_types = [r.get("room_type") for r in rooms if r.get("room_type")]

    def answer(self, query: str, threshold_retriever: float = 0.6) -> Dict:
        res = self.retriever.answer(query, threshold=threshold_retriever)
        res["kind"] = "retrieval" if res.get("found") else "none"
        return res
