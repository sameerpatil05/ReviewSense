"""Simple semantic-style review search using TF-IDF cosine similarity."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text


class ReviewSearchEngine:
    """Lightweight searchable index for product reviews."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
        self.matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame) -> "ReviewSearchEngine":
        self.df = df.reset_index(drop=True).copy()
        documents = self.df["clean_review"].fillna("").astype(str)
        self.matrix = self.vectorizer.fit_transform(documents)
        return self

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        if self.matrix is None or self.df is None:
            raise ValueError("Search engine is not fitted yet.")
        query_clean = clean_text(query)
        if not query_clean:
            return self.df.head(0).copy()
        query_vector = self.vectorizer.transform([query_clean])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results["search_score"] = scores[top_indices].round(4)
        return results[results["search_score"] > 0].reset_index(drop=True)
