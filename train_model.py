import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re

# -----------------------------
# Helper: Clean Text
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # remove punctuation/numbers
    return text.lower().strip()

# -----------------------------
# Load datasets
# -----------------------------
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 1
true["label"] = 0

data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1, random_state=42)

# Clean text
data['text'] = data['text'].apply(clean_text)

# -----------------------------
# Vectorize
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(X, y)

# -----------------------------
# Save model & vectorizer
# -----------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")