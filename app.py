"""ReviewSense Streamlit application.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.aspects import add_aspect_columns, aspect_sentiment_summary, top_aspect_words
from src.data_loader import load_csv, load_sample_data
from src.model import predict_sentiment, train_sentiment_model
from src.persona import PERSONA_WEIGHTS, persona_score
from src.ranker import add_review_rank_score
from src.search import ReviewSearchEngine


st.set_page_config(
    page_title="ReviewSense",
    page_icon="🛒",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_sample_cached() -> pd.DataFrame:
    return load_sample_data()


@st.cache_data(show_spinner=False)
def _read_uploaded_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    return load_csv(io.BytesIO(uploaded_bytes))


@st.cache_resource(show_spinner=False)
def _train_cached(csv_text: str):
    df = pd.read_csv(io.StringIO(csv_text))
    from src.preprocessing import preprocess_dataframe

    df = preprocess_dataframe(df)
    model, metrics = train_sentiment_model(df)
    return model, metrics


def prepare_dataset(df: pd.DataFrame):
    """Train model, add predictions, aspects, and ranking columns."""
    csv_text = df.to_csv(index=False)
    model, metrics = _train_cached(csv_text)
    predictions = predict_sentiment(model, df["clean_review"])
    enriched = pd.concat([df.reset_index(drop=True), predictions], axis=1)
    enriched = add_aspect_columns(enriched)
    enriched = add_review_rank_score(enriched)
    summary = aspect_sentiment_summary(enriched)
    search_engine = ReviewSearchEngine().fit(enriched)
    return enriched, summary, search_engine, metrics


def metric_value(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def show_review_cards(df: pd.DataFrame, limit: int = 5):
    for _, row in df.head(limit).iterrows():
        with st.container(border=True):
            st.markdown(f"**{row.get('product_title', 'Product')}**  ")
            st.caption(
                f"Rating: {row.get('rating', 'N/A')} | Sentiment: {row.get('predicted_sentiment', 'N/A')} | "
                f"Helpful votes: {int(row.get('helpful_votes', 0))} | Score: {row.get('helpfulness_score', 'N/A')}"
            )
            st.write(row.get("review_text", ""))
            st.caption("Aspects: " + ", ".join(row.get("aspects", ["General"])))


def main():
    st.title("🛒 ReviewSense")
    st.subheader("NLP-Based Product Review Intelligence and Search System")

    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader(
            "Upload Amazon-style reviews CSV",
            type=["csv"],
            help="Required columns: review_text, rating. Optional: helpful_votes, product_id, product_title, category, timestamp.",
        )
        st.info(
            "No dataset? The app uses a bundled sample dataset so you can run the project immediately."
        )
        st.markdown("### Project Components")
        st.write(" Sentiment Classification")
        st.write(" Aspect-Based Opinion Mining")
        st.write(" Review Search")
        st.write(" Helpful Review Ranking")
        st.write(" Buyer Persona Recommendation")

    try:
        if uploaded_file is not None:
            df = _read_uploaded_csv(uploaded_file.getvalue())
            data_source = "Uploaded CSV"
        else:
            df = _load_sample_cached()
            data_source = "Bundled sample dataset"

        enriched, aspect_summary, search_engine, metrics = prepare_dataset(df)
    except Exception as exc:  # pragma: no cover - user-facing error path
        st.error(f"Could not load or process dataset: {exc}")
        st.stop()

    st.caption(f"Data source: {data_source} | Reviews: {len(enriched):,}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{len(enriched):,}")
    c2.metric("Sentiment Accuracy", metric_value(metrics.get("accuracy")))
    c3.metric("Macro F1", metric_value(metrics.get("f1_macro")))
    c4.metric("Detected Aspects", f"{aspect_summary['aspect'].nunique() if not aspect_summary.empty else 0}")

    tabs = st.tabs(
        [
            " Overview",
            " Review Search",
            " Aspect Intelligence",
            " Helpful Reviews",
            " Buyer Persona",
            " Dataset & Metrics",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Sentiment Distribution")
            sentiment_counts = enriched["predicted_sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig = px.pie(sentiment_counts, names="sentiment", values="count", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("### Top Mentioned Aspects")
            aspect_counts = top_aspect_words(enriched)
            fig = px.bar(aspect_counts, x="aspect", y="mentions", text="mentions")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Sample Processed Reviews")
        st.dataframe(
            enriched[[
                "product_title",
                "rating",
                "predicted_sentiment",
                "sentiment_confidence",
                "aspects",
                "helpfulness_score",
                "review_text",
            ]].head(15),
            use_container_width=True,
        )

    with tabs[1]:
        st.markdown("### Search inside product reviews")
        query = st.text_input(
            "Search query",
            value="battery backup and performance",
            placeholder="Example: late delivery, battery issue, value for money",
        )
        top_k = st.slider("Number of results", min_value=3, max_value=20, value=8)
        results = search_engine.search(query, top_k=top_k)
        if results.empty:
            st.warning("No matching reviews found. Try a broader query.")
        else:
            st.dataframe(
                results[[
                    "search_score",
                    "product_title",
                    "rating",
                    "predicted_sentiment",
                    "aspects",
                    "review_text",
                ]],
                use_container_width=True,
            )
            show_review_cards(results, limit=3)

    with tabs[2]:
        st.markdown("### Aspect-wise sentiment summary")
        st.dataframe(aspect_summary, use_container_width=True)
        if not aspect_summary.empty:
            fig = px.bar(
                aspect_summary.sort_values("positive_pct", ascending=False),
                x="aspect",
                y="positive_pct",
                text="positive_pct",
                hover_data=["total_mentions", "positive", "neutral", "negative"],
            )
            fig.update_layout(yaxis_title="Positive sentiment %", xaxis_title="Aspect")
            st.plotly_chart(fig, use_container_width=True)

        selected_aspect = st.selectbox("Filter reviews by aspect", aspect_summary["aspect"].tolist() if not aspect_summary.empty else ["General"])
        aspect_reviews = enriched[enriched["aspects"].map(lambda xs: selected_aspect in xs)]
        show_review_cards(aspect_reviews.sort_values("helpfulness_score", ascending=False), limit=5)

    with tabs[3]:
        st.markdown("### Most helpful reviews")
        st.write("Ranking uses helpful votes, review detail, aspect coverage, and sentiment confidence.")
        ranked = enriched.sort_values("helpfulness_score", ascending=False)
        st.dataframe(
            ranked[[
                "helpfulness_score",
                "helpful_votes",
                "review_length",
                "aspect_count",
                "sentiment_confidence",
                "product_title",
                "rating",
                "predicted_sentiment",
                "review_text",
            ]].head(20),
            use_container_width=True,
        )
        show_review_cards(ranked, limit=5)

    with tabs[4]:
        st.markdown("### Buyer persona recommendation")
        persona = st.selectbox("Select buyer type", list(PERSONA_WEIGHTS.keys()))
        score = persona_score(aspect_summary, persona)
        c1, c2 = st.columns([1, 2])
        c1.metric("Suitability Score", f"{score['score']} / 100")
        c1.metric("Verdict", score["verdict"])
        c2.markdown("#### Reason")
        c2.write(score["reason"])
        st.markdown("#### Aspect weights used")
        weights_df = pd.DataFrame(
            [{"aspect": k, "weight": v} for k, v in PERSONA_WEIGHTS[persona].items()]
        ).sort_values("weight", ascending=False)
        st.dataframe(weights_df, use_container_width=True)

    with tabs[5]:
        st.markdown("### Model metrics")
        metric_df = pd.DataFrame(
            [
                {"metric": "Accuracy", "value": metrics.get("accuracy")},
                {"metric": "Macro Precision", "value": metrics.get("precision_macro")},
                {"metric": "Macro Recall", "value": metrics.get("recall_macro")},
                {"metric": "Macro F1", "value": metrics.get("f1_macro")},
                {"metric": "Test samples", "value": metrics.get("test_size")},
            ]
        )
        st.dataframe(metric_df, use_container_width=True)
        st.markdown("#### Classification report")
        st.code(metrics.get("classification_report", "Not available"))

        st.markdown("### Dataset preview")
        st.dataframe(enriched.head(50), use_container_width=True)

        csv = enriched.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download processed reviews CSV",
            csv,
            file_name="reviewsense_processed_reviews.csv",
            mime="text/csv",
        )

        st.markdown("### Expected CSV format")
        st.code("review_text,rating,helpful_votes,product_id,product_title,category,timestamp")


if __name__ == "__main__":
    main()
