import requests
import feedparser
from datetime import datetime, timedelta
from collections import defaultdict
 
SOURCE_SCORES = {
    "The Washington Post": 95,
    "BBC News": 90,
    "CNN": 85,
    "GNews": 80,
    "GoogleNews": 80
}

# ----------------------------
# 2️⃣ Fetch NewsAPI headlines
# ----------------------------
NEWSAPI_KEY = "ec7945ef0b6b4a6d9b8738462c159fde"
newsapi_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWSAPI_KEY}"
newsapi_response = requests.get(newsapi_url).json()
newsapi_articles = newsapi_response.get('articles', [])

# ----------------------------
# 3️⃣ Fetch GNews headlines
# ----------------------------
GNEWS_KEY = "ddb8f6145d6e116da72ef6a9b0708318"
gnews_url = f"https://gnews.io/api/v4/top-headlines?country=us&token={GNEWS_KEY}"
gnews_response = requests.get(gnews_url).json()
gnews_articles = gnews_response.get('articles', [])

# ----------------------------
# 4️⃣ Fetch Google News RSS headlines
# ----------------------------
rss_url = "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
rss_feed = feedparser.parse(rss_url)
rss_articles = rss_feed.entries

# ----------------------------
# 5️⃣ Utility functions
# ----------------------------
def recency_factor(published_at):
    """Boost score if article is recent."""
    try:
        if isinstance(published_at, str):
            published_date = datetime.fromisoformat(published_at[:-1])
        else:  # RSS feed gives time_struct
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

# ----------------------------
# 6️⃣ Aggregate headlines & compute trust
# ----------------------------
# Using a dict to merge repeated headlines across sources
headline_data = defaultdict(list)

# NewsAPI
for art in newsapi_articles:
    headline_data[art['title']].append(("The Washington Post" if "washingtonpost" in art['url'] else "NewsAPI", art['publishedAt']))

# GNews
for art in gnews_articles:
    headline_data[art['title']].append(("GNews", art['publishedAt']))

# Google RSS
for art in rss_articles:
    headline_data[art['title']].append(("GoogleNews", art.published_parsed))

# Compute final trust score
for title, sources in headline_data.items():
    scores = [compute_score(src, date) for src, date in sources]
    # Combine scores to boost confidence if multiple sources report the same headline
    
    combined_score = 1.0
    for s in scores:
        combined_score *= (1 - s / 100)
        combined_score = 100 * (1 - combined_score)

    print(f"Headline: {title}")
    print(f"Reported by: {', '.join([src for src, _ in sources])}")
    print(f"Trust Score: {int(combined_score)}%\n")
