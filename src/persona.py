"""Buyer persona recommendation rules."""

from __future__ import annotations

from typing import Dict

import pandas as pd


PERSONA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Student": {"Price": 0.30, "Battery": 0.25, "Performance": 0.20, "Durability": 0.10, "Design": 0.10, "General": 0.05},
    "Office User": {"Battery": 0.25, "Performance": 0.25, "Quality": 0.20, "Display": 0.15, "Support": 0.10, "General": 0.05},
    "Gamer": {"Performance": 0.35, "Display": 0.20, "Battery": 0.15, "Sound": 0.10, "Quality": 0.10, "General": 0.10},
    "Budget Buyer": {"Price": 0.40, "Quality": 0.20, "Battery": 0.15, "Support": 0.10, "Delivery": 0.10, "General": 0.05},
    "Gift Buyer": {"Design": 0.25, "Quality": 0.25, "Delivery": 0.20, "Support": 0.15, "General": 0.15},
}


def persona_score(aspect_summary: pd.DataFrame, persona: str) -> Dict[str, object]:
    """Calculate buyer suitability score from aspect sentiment summary."""
    weights = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["Student"])
    if aspect_summary.empty:
        return {"score": 0, "verdict": "Not enough data", "reason": "No aspect summary available."}

    summary = aspect_summary.set_index("aspect").to_dict(orient="index")
    weighted_score = 0.0
    available_weight = 0.0
    positives = []
    risks = []

    for aspect, weight in weights.items():
        if aspect in summary:
            pct = float(summary[aspect]["positive_pct"])
            weighted_score += pct * weight
            available_weight += weight
            if pct >= 65:
                positives.append(aspect)
            elif pct < 45:
                risks.append(aspect)

    if available_weight == 0:
        score = float(aspect_summary["positive_pct"].mean())
    else:
        score = weighted_score / available_weight

    if score >= 70:
        verdict = "Strong Match"
    elif score >= 55:
        verdict = "Good Match"
    elif score >= 40:
        verdict = "Average Match"
    else:
        verdict = "Weak Match"

    reason_parts = []
    if positives:
        reason_parts.append("Positive signals: " + ", ".join(positives[:4]))
    if risks:
        reason_parts.append("Risk areas: " + ", ".join(risks[:4]))
    if not reason_parts:
        reason_parts.append("The score is based on the weighted aspect sentiment for this buyer type.")

    return {"score": round(score, 1), "verdict": verdict, "reason": " | ".join(reason_parts)}
