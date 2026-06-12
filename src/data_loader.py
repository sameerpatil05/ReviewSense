"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .preprocessing import preprocess_dataframe


def load_csv(path_or_buffer: str | Path | BinaryIO) -> pd.DataFrame:
    """Load and preprocess a CSV dataset."""
    df = pd.read_csv(path_or_buffer)
    return preprocess_dataframe(df)


def load_sample_data() -> pd.DataFrame:
    """Load bundled sample reviews so the app runs immediately."""
    root = Path(__file__).resolve().parents[1]
    return load_csv(root / "data" / "sample_reviews.csv")
