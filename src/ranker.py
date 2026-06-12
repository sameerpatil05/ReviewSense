"""Helpful review ranking logic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_review_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    """Rank reviews using helpful votes, detail, aspects, and model confidence."""
    result = df.copy()
    helpful = result["helpful_votes"].fillna(0).astype(float)
    length = result.get("review_length", result["clean_review"].str.split().map(len)).fillna(0).astype(float)
    aspects = result.get("aspect_count", 1)
    confidence = result.get("sentiment_confidence", 0.65)

    helpful_norm = np.log1p(helpful) / max(np.log1p(helpful.max()), 1.0)
    length_norm = np.clip(length / 80, 0, 1)
    aspect_norm = np.clip(pd.Series(aspects).astype(float) / 4, 0, 1)
    confidence_norm = pd.Series(confidence).astype(float).clip(0, 1)

    # Balanced formula: useful + detailed + covers product aspects + model confidence.
    result["helpfulness_score"] = (
        0.40 * helpful_norm + 0.25 * length_norm + 0.20 * aspect_norm + 0.15 * confidence_norm
    ).round(4)
    return result.sort_values("helpfulness_score", ascending=False).reset_index(drop=True)
