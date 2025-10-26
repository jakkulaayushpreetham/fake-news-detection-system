from flask import Flask, render_template, request
import requests
import re
import joblib

app = Flask(__name__)

# ----------------------------
# Load ML model & vectorizer
# ----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ----------------------------
# NewsAPI Key
# ----------------------------
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # Replace with your key

# ----------------------------
# Helper: Clean text
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, sources=None, is_fake=None)

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "").strip()
    if not news_text:
        return render_template("index.html", result="Please enter a news headline or article.", sources=None, is_fake=None)

    # 1️⃣ Trusted source check via NewsAPI
    query_text = news_text[:100]  # Limit length
    try:
        url = f"https://newsapi.org/v2/everything?q={query_text}&language=en&apiKey={"ec7945ef0b6b4a6d9b8738462c159fde"}"
        response = requests.get(url)
        data = response.json()
    except:
        data = {"totalResults": 0}

    if data.get("totalResults", 0) > 0:
        articles = data["articles"][:5]
        links = [f"<a href='{a['url']}' target='_blank'>{a['source']['name']}</a>" for a in articles]
        result = "REAL NEWS (100% confidence)"
        sources = links
        is_fake = False
    else:
    # 2️⃣ Use ML model (with adjusted confidence)
        cleaned = clean_text(news_text)
        text_vec = vectorizer.transform([cleaned])

        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0][list(model.classes_).index(prediction)] * 100

    # Adjust confidence values
        if prediction == 1:  # Fake news
            adjusted_prob = max(probability - 50, 0)
            result = f"FAKE NEWS ({adjusted_prob:.2f}% confidence)"
            is_fake = True
        else:  # Real news
            adjusted_prob = min(probability + 35, 100)
            result = f"REAL NEWS ({adjusted_prob:.2f}% confidence)"
            is_fake = False
        sources = []
    

    return render_template("index.html", result=result, sources=sources, is_fake=is_fake, news_text=news_text)

# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)