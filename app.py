from flask import Flask, render_template, request
import joblib
import re
import requests
import feedparser
from datetime import datetime, timedelta
from collections import defaultdict

# ----------------------------
# Flask App Init
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ----------------------------
# Helper Functions
# ----------------------------
def clean_text(text):
    """Remove URLs, punctuation, lowercase."""
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# Trusted source scores
SOURCE_SCORES = {
    "The Washington Post": 95,
    "BBC News": 90,
    "CNN": 85,
    "GNews": 80,
    "GoogleNews": 80
}

NEWSAPI_KEY = "ec7945ef0b6b4a6d9b8738462c159fde"
GNEWS_KEY = "ddb8f6145d6e116da72ef6a9b0708318"

def recency_factor(published_at):
    """Boost score if recent."""
    try:
        if isinstance(published_at, str):
            published_date = datetime.fromisoformat(published_at[:-1])
        else:  # RSS feed
            published_date = datetime(*published_at[:6])
        delta = datetime.now() - published_date
        if delta < timedelta(days=1):
            return 1.05
        elif delta < timedelta(days=3):
            return 1.02
        else:
            return 0.95
    except:
        return 1.0

def compute_score(source_name, published_at):
    base = SOURCE_SCORES.get(source_name, 50)
    return min(int(base * recency_factor(published_at)), 100)

def verify_with_trusted_sources(news_text):
    """Check news_text against NewsAPI, GNews, and Google RSS."""
    headline_data = defaultdict(list)

    # NewsAPI
    try:
        url = f"https://newsapi.org/v2/everything?q={news_text[:50]}&language=en&apiKey={NEWSAPI_KEY}"
        newsapi_articles = requests.get(url).json().get('articles', [])
        for art in newsapi_articles:
            source_name = "The Washington Post" if "washingtonpost" in art['url'] else "NewsAPI"
            headline_data[art['title']].append((source_name, art['publishedAt']))
    except:
        pass

    # GNews
    try:
        url = f"https://gnews.io/api/v4/top-headlines?q={news_text[:50]}&token={GNEWS_KEY}"
        gnews_articles = requests.get(url).json().get('articles', [])
        for art in gnews_articles:
            headline_data[art['title']].append(("GNews", art['publishedAt']))
    except:
        pass

    # Google RSS
    try:
        rss_url = f"https://news.google.com/rss/search?q={news_text[:50]}&hl=en-US&gl=US&ceid=US:en"
        rss_feed = feedparser.parse(rss_url)
        for art in rss_feed.entries:
            headline_data[art.title].append(("GoogleNews", art.published_parsed))
    except:
        pass

    results = []
    for title, sources in headline_data.items():
        if news_text.lower() in title.lower():
            scores = [compute_score(src, date) for src, date in sources]
            combined_score = 1.0
            for s in scores:
                combined_score *= (1 - s / 100)
            combined_score = 100 * (1 - combined_score)
            results.append({
                "title": title,
                "sources": [src for src, _ in sources],
                "trust_score": int(combined_score)
            })
    return results

# ----------------------------
# Flask Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "")
    if not news_text.strip():
        return render_template("index.html", result="Please enter news text to analyze.")

    # ML Prediction
    cleaned = clean_text(news_text)
    text_vec = vectorizer.transform([cleaned])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0][prediction] * 100
    label = f"FAKE NEWS ({probability:.2f}% confidence)" if prediction == 1 else f"REAL NEWS ({probability:.2f}% confidence)"

    # Trusted source verification
    trusted_results = verify_with_trusted_sources(news_text)
    return render_template(
        "index.html",
        result=label,
        verified=bool(trusted_results),
        sources=[f"{r['title']} ({r['trust_score']}%)" for r in trusted_results]
    )

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
