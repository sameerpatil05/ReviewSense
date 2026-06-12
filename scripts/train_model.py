"""Train ReviewSense sentiment model from the command line.

Usage:
    python scripts/train_model.py --data data/sample_reviews.csv --output models/sentiment_model.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_csv  # noqa: E402
from src.model import train_sentiment_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReviewSense sentiment model")
    parser.add_argument("--data", default="data/sample_reviews.csv", help="Path to reviews CSV")
    parser.add_argument("--output", default="models/sentiment_model.joblib", help="Output model path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = ROOT / args.data
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path)
    model, metrics = train_sentiment_model(df)
    joblib.dump({"model": model, "metrics": metrics}, output_path)

    print(f"Saved model to {output_path}")
    print("Metrics:")
    for key, value in metrics.items():
        if key != "classification_report":
            print(f"  {key}: {value}")
    print("\nClassification Report:")
    print(metrics.get("classification_report", "N/A"))


if __name__ == "__main__":
    main()
