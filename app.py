"""
app.py  ─  Flask web server for Naive Bayes Spam Detector
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import sys
import pickle
from flask import Flask, request, jsonify, render_template_string

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from utils.text_processing import clean_text
from model.train import MultinomialNaiveBayes, CountVectorizer  # noqa: F401

MODEL_PATH = os.path.join(ROOT, "model", "naive_bayes.pkl")
VEC_PATH   = os.path.join(ROOT, "model", "vectorizer.pkl")

app = Flask(__name__)

# ── Load model once at startup ───────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run  python model/train.py  first."
    )

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(VEC_PATH, "rb") as f:
    VECTORIZER = pickle.load(f)

print("✅  Model loaded successfully.")

# ── HTML template (single-file, no templates/ folder needed) ────────────────
HTML = open(os.path.join(ROOT, "templates", "index.html")).read()


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    cleaned = clean_text(text)
    X       = VECTORIZER.transform([cleaned])
    proba   = MODEL.predict_proba(X)[0]

    spam_pct = round(proba.get("spam", 0) * 100, 1)
    ham_pct  = round(proba.get("ham",  0) * 100, 1)
    label    = "spam" if spam_pct > ham_pct else "ham"

    # Top words that pushed towards spam
    top_spam_words = _top_words(cleaned, "spam", n=5)
    top_ham_words  = _top_words(cleaned, "ham",  n=5)

    return jsonify({
        "label":          label,
        "spam_pct":       spam_pct,
        "ham_pct":        ham_pct,
        "cleaned_text":   cleaned,
        "top_spam_words": top_spam_words,
        "top_ham_words":  top_ham_words,
        "word_count":     len(text.split()),
        "char_count":     len(text),
    })


def _top_words(cleaned_text: str, cls: str, n: int = 5) -> list[str]:
    """Return the n words in the text with the highest log-likelihood for cls."""
    import math
    words = cleaned_text.split()
    vocab = VECTORIZER.vocabulary_
    ll    = MODEL.log_likelihood_.get(cls, [])
    scored = []
    for w in set(words):
        if w in vocab:
            scored.append((w, ll[vocab[w]]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:n]]


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "templates"), exist_ok=True)
    app.run(debug=True, port=5000)
