"""Sentiment model training and prediction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def build_sentiment_pipeline() -> Pipeline:
    """Build a lightweight TF-IDF + Logistic Regression classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and counts.min() >= 2


def train_sentiment_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, object]]:
    """Train model and return model plus evaluation metrics."""
    if df["sentiment_label"].nunique() < 2:
        raise ValueError("Need at least two sentiment classes to train the model.")

    X = df["clean_review"].astype(str)
    y = df["sentiment_label"].astype(str)
    model = build_sentiment_pipeline()

    metrics: Dict[str, object] = {}
    if len(df) >= 12 and _can_stratify(y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        labels = [label for label in LABEL_ORDER if label in sorted(y.unique())]
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "classification_report": classification_report(y_test, y_pred, labels=labels, zero_division=0),
            "test_size": int(len(y_test)),
        }
        model.fit(X, y)
    else:
        model.fit(X, y)
        metrics = {
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "classification_report": "Not enough balanced data for a reliable train/test split.",
            "test_size": 0,
        }

    return model, metrics


def predict_sentiment(model: Pipeline, texts: List[str] | pd.Series) -> pd.DataFrame:
    """Predict sentiment labels and confidence scores for review texts."""
    texts = pd.Series(texts).fillna("").astype(str)
    predictions = model.predict(texts)

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(texts)
        classes = list(model.named_steps["classifier"].classes_)
        confidence = probabilities.max(axis=1)
        probability_columns = {
            f"prob_{label.lower()}": probabilities[:, classes.index(label)] if label in classes else np.zeros(len(texts))
            for label in LABEL_ORDER
        }
    else:
        confidence = np.ones(len(texts))
        probability_columns = {f"prob_{label.lower()}": np.zeros(len(texts)) for label in LABEL_ORDER}

    result = pd.DataFrame(
        {
            "predicted_sentiment": predictions,
            "sentiment_confidence": np.round(confidence, 4),
        }
    )
    for column, values in probability_columns.items():
        result[column] = np.round(values, 4)
    return result
