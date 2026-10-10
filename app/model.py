"""Model wrapper: loads the trained artifact and turns tickets into API-shaped predictions.

One artifact holds three linear classifiers that share the same feature matrix:
  category  (11 classes)            -> `category`, `confidence`
  urgency   (binary)                -> `is_urgent`
  secondary (none + 11 classes)     -> `secondary_category`
The consistency rules of the API contract are enforced here, in one place, so that
/predict, /predict/batch and the async jobs can never disagree.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np

from .features import normalize

CATEGORIES = [
    "payment_refund", "ride_trip_issue", "lost_item", "order_missing_wrong", "delivery_delay", "food_quality",
    "account_promo", "safety_conduct", "app_technical", "general_inquiry", "spam_irrelevant",
]
TEAM = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}
NONE = "none"


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def default_model_dir() -> Path:
    """MODEL_DIR env var, otherwise the single version folder listed in models/CURRENT."""
    env = os.environ.get("MODEL_DIR")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parent.parent / "models"
    return root / (root / "CURRENT").read_text(encoding="utf-8").strip()


class TicketModel:
    def __init__(self, model_dir: str | os.PathLike | None = None):
        self.dir = Path(model_dir) if model_dir else default_model_dir()
        art = joblib.load(self.dir / "model.joblib")
        self.featurizer = art["featurizer"]
        self.clf_category = art["clf_category"]
        self.clf_urgent = art["clf_urgent"]
        self.clf_secondary = art["clf_secondary"]
        self.temperature = float(art["temperature"])
        self.urgent_threshold = float(art["urgent_threshold"])
        self.secondary_margin = float(art["secondary_margin"])
        self.review_threshold = float(art["review_threshold"])
        self.version = str(art["version"])
        self.categories = list(self.clf_category.classes_)
        self.secondary_classes = list(self.clf_secondary.classes_)
        self._none_idx = self.secondary_classes.index(NONE)
        metrics_path = self.dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    # ------------------------------------------------------------------
    def _raw(self, tickets: list[dict]):
        texts = [normalize(t.get("subject"), t.get("text")) for t in tickets]
        X = self.featurizer.transform(texts)
        probs = softmax(self.clf_category.decision_function(X), self.temperature)
        urgent = self.clf_urgent.decision_function(X)
        secondary = self.clf_secondary.decision_function(X)
        return texts, probs, urgent, secondary

    def predict(self, tickets: list[dict]) -> list[dict]:
        """tickets: validated dicts with channel / subject / text (+ optional ticket_id). Order is preserved."""
        if not tickets:
            return []
        _, probs, urgent, secondary = self._raw(tickets)
        out = []
        for i, ticket in enumerate(tickets):
            k = int(np.argmax(probs[i]))
            category = self.categories[k]
            confidence = round(float(min(max(probs[i, k], 0.0), 1.0)), 4)

            # secondary: best non-"none" class that is not the primary, kept only if it beats "none" by a margin
            sec_scores = secondary[i].copy()
            none_score = sec_scores[self._none_idx]
            sec_scores[self._none_idx] = -np.inf
            if category in self.secondary_classes:
                sec_scores[self.secondary_classes.index(category)] = -np.inf
            j = int(np.argmax(sec_scores))
            secondary_category = self.secondary_classes[j] if sec_scores[j] - none_score > self.secondary_margin else None

            is_urgent = bool(urgent[i] > self.urgent_threshold)

            # contract rules: spam is never urgent and never has a secondary label
            if category == "spam_irrelevant":
                is_urgent = False
                secondary_category = None

            row = {}
            if isinstance(ticket.get("ticket_id"), str):
                row["ticket_id"] = ticket["ticket_id"]
            row.update(
                category=category,
                secondary_category=secondary_category,
                team=TEAM[category],
                is_urgent=is_urgent,
                confidence=confidence,
                model_version=self.version,
                needs_human_review=bool(confidence < self.review_threshold),
            )
            out.append(row)
        return out

    def explain(self, ticket: dict, top_k: int = 3) -> dict:
        """Extra detail for the demo UI only (never part of the scored API responses)."""
        texts, probs, urgent, _ = self._raw([ticket])
        order = np.argsort(-probs[0])[:top_k]
        return {
            "top_categories": [{"category": self.categories[k], "probability": round(float(probs[0, k]), 4)} for k in order],
            "urgency_score": round(float(urgent[0]), 4),
            "urgency_threshold": self.urgent_threshold,
            "concepts": self.featurizer.fired_concepts(texts[0]),
            "keywords": self.featurizer.matched_keywords(texts[0]),
        }
