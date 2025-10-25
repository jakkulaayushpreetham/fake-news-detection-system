# -----------------------------------------------------------
# 🧠 Intelligent Fake News Detection System (Flask Version)
# Author: Jakkula Ayush Preetham
# Dataset: Fake.csv & True.csv (Kaggle)
# -----------------------------------------------------------

from flask import Flask, render_template, request
import joblib
import re
import pandas as pd
import os

# -----------------------------------------------------------
# 1️⃣ Initialize Flask App
# -----------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------
# 2️⃣ Paths and Model Loading
# -----------------------------------------------------------
FAKE_PATH = "C:/Havkathon/fake-news-detection-system/dataset/Fake.csv"
TRUE_PATH = "C:/Havkathon/fake-news-detection-system/dataset/True.csv"
MERGED_PATH = "fake_news_dataset.csv"

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print(" Model and vectorizer loaded successfully.")

# -----------------------------------------------------------
# 3️⃣ Merge Fake & True datasets if not already combined
# -----------------------------------------------------------
if not os.path.exists(MERGED_PATH):
    if os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH):
        print("🔹 Merging Fake.csv and True.csv ...")
        fake_df = pd.read_csv(FAKE_PATH)
        true_df = pd.read_csv(TRUE_PATH)

        fake_df["label"] = 1  # 1 = Fake
        true_df["label"] = 0  # 0 = Real

        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_df.to_csv(MERGED_PATH, index=False)
        print(f" Merged dataset saved as '{MERGED_PATH}' with {len(combined_df)} rows.")
    else:
        print(" Warning: Fake.csv or True.csv not found. Skipping dataset creation.")
else:
    print(f" Using existing dataset: {MERGED_PATH}")

# -----------------------------------------------------------
# 4️⃣ Helper Function — Clean Input Text
# -----------------------------------------------------------
def clean_text(text):
    """Remove URLs, punctuation, and lowercase the text."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

# -----------------------------------------------------------
# 5️⃣ Home Page Route
# -----------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

# -----------------------------------------------------------
# 6️⃣ Prediction Route
# -----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "")
    if not news_text.strip():
        return render_template("index.html", result=" Please enter a news text to analyze.")

    # Clean and vectorize
    cleaned = clean_text(news_text)
    text_vec = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0][prediction] * 100

    if prediction == 1:
        result = f" FAKE NEWS ({probability:.2f}% confidence)"
    else:
        result = f" REAL NEWS ({probability:.2f}% confidence)"

    return render_template("index.html", result=result)

# -----------------------------------------------------------
# 7️⃣ Run Flask App
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
