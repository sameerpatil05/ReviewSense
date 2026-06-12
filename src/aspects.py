"""Aspect extraction and aspect-level sentiment summaries."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

import pandas as pd


ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Battery": ["battery", "backup", "charge", "charging", "charger", "power"],
    "Price": ["price", "cost", "cheap", "expensive", "value", "money", "budget", "worth"],
    "Quality": ["quality", "build", "material", "finish", "premium", "plastic"],
    "Durability": ["durable", "weak", "damage", "damaged", "broken", "long lasting", "lasting"],
    "Delivery": ["delivery", "packaging", "package", "courier", "shipping", "delivered", "box"],
    "Performance": ["performance", "fast", "slow", "smooth", "lag", "speed", "processor", "heating"],
    "Display": ["display", "screen", "brightness", "color", "resolution", "panel"],
    "Sound": ["sound", "speaker", "audio", "bass", "volume", "mic"],
    "Camera": ["camera", "photo", "video", "selfie", "lens"],
    "Design": ["design", "look", "looks", "style", "weight", "lightweight", "compact"],
    "Support": ["support", "warranty", "service", "return", "replacement", "refund"],
}


def extract_aspects(text: str) -> List[str]:
    """Return aspects mentioned in a review using transparent keyword rules."""
    text = f" {str(text).lower()} "
    aspects = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(f" {keyword} " in text or keyword in text for keyword in keywords):
            aspects.append(aspect)
    return aspects or ["General"]


def add_aspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add aspect list and aspect count columns."""
    result = df.copy()
    result["aspects"] = result["clean_review"].map(extract_aspects)
    result["aspect_count"] = result["aspects"].map(len)
    return result


def aspect_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create aspect-wise sentiment count and positive percentage table."""
    rows = []
    for _, row in df.iterrows():
        for aspect in row.get("aspects", ["General"]):
            rows.append(
                {
                    "aspect": aspect,
                    "sentiment": row.get("predicted_sentiment", row.get("sentiment_label", "Neutral")),
                    "rating": row.get("rating", 3),
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return pd.DataFrame(columns=["aspect", "total_mentions", "positive", "neutral", "negative", "positive_pct"])

    pivot = (
        expanded.pivot_table(index="aspect", columns="sentiment", values="rating", aggfunc="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in ["Positive", "Neutral", "Negative"]:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot["total_mentions"] = pivot[["Positive", "Neutral", "Negative"]].sum(axis=1)
    pivot["positive_pct"] = (pivot["Positive"] / pivot["total_mentions"] * 100).round(1)
    pivot = pivot.rename(columns={"Positive": "positive", "Neutral": "neutral", "Negative": "negative"})
    return pivot[["aspect", "total_mentions", "positive", "neutral", "negative", "positive_pct"]].sort_values(
        ["total_mentions", "positive_pct"], ascending=[False, False]
    )


def top_aspect_words(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Count the most detected aspects across reviews."""
    counter: Counter[str] = Counter()
    for aspects in df.get("aspects", []):
        counter.update(aspects)
    return pd.DataFrame(counter.most_common(top_n), columns=["aspect", "mentions"])
