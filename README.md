<<<<<<< HEAD
# ReviewSense: NLP-Based Product Review Intelligence and Search System
LIVE LINK - https://reviewsenseml.streamlit.app/
ReviewSense is a complete Streamlit + ML project for analyzing product reviews. It is designed for portfolios, internships, and applications such as Amazon ML Summer School.

Instead of doing only positive/negative sentiment analysis, ReviewSense adds:

- Sentiment classification
- Aspect-based opinion mining
- Product review search
- Helpful review ranking
- Buyer persona recommendation
- Streamlit dashboard
- Evaluation metrics

## Demo Preview
Run on browser:
https://reviewsenseml.streamlit.app/

Run locally:

```bash
streamlit run app.py
```

The app works immediately with the bundled sample dataset in `data/sample_reviews.csv`.

## Project Components

| Component | Description |
|---|---|
| Sentiment Classification | Converts review text into Positive, Neutral, or Negative sentiment |
| Aspect Extraction | Detects aspects such as Battery, Price, Delivery, Display, Performance, Quality, Support |
| Review Search | Searches reviews using TF-IDF cosine similarity |
| Helpful Review Ranking | Ranks useful reviews using votes, review length, aspects, and confidence |
| Buyer Persona | Gives recommendation score for Student, Gamer, Office User, Budget Buyer, Gift Buyer |
| Streamlit UI | Interactive dashboard for analysis and demo |

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- GitHub

## Folder Structure

```text
ReviewSense/
│
├── app.py
├── requirements.txt
├── README.md
├── FORM_ANSWERS.md
│
├── data/
│   └── sample_reviews.csv
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── aspects.py
│   ├── search.py
│   ├── ranker.py
│   ├── persona.py
│   └── data_loader.py
│
├── scripts/
│   └── train_model.py
│
├── notebooks/
│   └── README.md
│
└── .streamlit/
    └── config.toml
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ReviewSense.git
cd ReviewSense
```

### 2. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit app

```bash
streamlit run app.py
```

## Dataset Format

The app accepts a CSV file with these columns:

```text
review_text,rating,helpful_votes,product_id,product_title,category,timestamp
```

Required columns:

- `review_text`
- `rating`

Optional columns:

- `helpful_votes`
- `product_id`
- `product_title`
- `category`
- `timestamp`

## Recommended Real Dataset

Use a selected subset of the Amazon Reviews 2023 dataset, such as Electronics or Appliances.

For a student project, do not load the full dataset. Use a sample like:

- 50,000 to 100,000 reviews
- Around 50 MB to 150 MB after sampling/preprocessing

Fields useful for this project:

- Review text
- Rating
- Helpful votes
- Product ID
- Product title/category
- Timestamp

## Train Model from Command Line

```bash
python scripts/train_model.py --data data/sample_reviews.csv --output models/sentiment_model.joblib
```

## GitHub Push Commands

```bash
git init
git add .
git commit -m "Initial commit: ReviewSense NLP review intelligence app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ReviewSense.git
git push -u origin main
```

## Deploy on Streamlit Cloud

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Select your GitHub repository.
4. Set main file path as:

```text
app.py
```

5. Deploy.

## Resume Line

**ReviewSense — NLP-Based Product Review Intelligence and Search System**  
Built an NLP system to analyze product reviews using sentiment classification, aspect-based opinion mining, search, and helpful review ranking. Developed a Streamlit dashboard to summarize product pros, cons, buyer suitability, and persona-based recommendations.

## Interview Explanation

I built an NLP-based product review intelligence system that does not just classify reviews as positive or negative. It extracts feature-wise opinions such as battery, delivery, price, performance, and quality, enables search inside reviews, ranks helpful reviews, and gives buyer-persona recommendations.
=======
# ReviewSense
>>>>>>> 54d342b8d1f073651d323d4b3395234a733aab83
