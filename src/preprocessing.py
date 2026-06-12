"""Text cleaning and dataframe preparation utilities for ReviewSense."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["review_text", "rating"]
OPTIONAL_COLUMNS = [
    "helpful_votes",
    "product_id",
    "product_title",
    "category",
    "timestamp",
]

_COLUMN_ALIASES = {
    "text": "review_text",
    "review": "review_text",
    "review_body": "review_text",
    "content": "review_text",
    "body": "review_text",
    "stars": "rating",
    "score": "rating",
    "overall": "rating",
    "helpful": "helpful_votes",
    "votes": "helpful_votes",
    "asin": "product_id",
    "title": "product_title",
}


def clean_text(text: object) -> str:
    """Lowercase and normalize review text without removing useful words."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common review dataset column names to ReviewSense schema."""
    renamed = {}
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for alias, canonical in _COLUMN_ALIASES.items():
        if alias in lowered and canonical not in df.columns:
            renamed[lowered[alias]] = canonical
    return df.rename(columns=renamed)


def rating_to_sentiment(rating: object) -> str:
    """Convert 1-5 star rating into sentiment label."""
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return "Neutral"
    if value >= 4:
        return "Positive"
    if value <= 2:
        return "Negative"
    return "Neutral"


def validate_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and fill required columns for the Streamlit application."""
    df = normalize_columns(df.copy())

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Your CSV should contain review_text and rating columns."
        )

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            if column == "helpful_votes":
                df[column] = 0
            elif column == "product_title":
                df[column] = "Unknown Product"
            elif column == "category":
                df[column] = "General"
            elif column == "product_id":
                df[column] = "P-UNKNOWN"
            else:
                df[column] = ""

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).clip(1, 5)
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).clip(lower=0)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.strip().ne("")].reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset has no non-empty reviews after preprocessing.")

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with clean_text and sentiment_label columns."""
    df = validate_reviews_df(df)
    df["clean_review"] = df["review_text"].map(clean_text)
    df["sentiment_label"] = df["rating"].map(rating_to_sentiment)
    df["review_length"] = df["clean_review"].str.split().map(len)
    return df
