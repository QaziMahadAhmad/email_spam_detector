import os
import sys
import pickle
from flask import Flask, request, jsonify, render_template_string

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "model", "naive_bayes.pkl")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run python model/train.py first.")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

print("Model loaded successfully.")

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

    proba   = MODEL.predict_proba([text])[0]
    classes = list(MODEL.classes_)
    spam_pct = round(proba[classes.index("spam")] * 100, 1)
    ham_pct  = round(proba[classes.index("ham")]  * 100, 1)
    label    = "spam" if spam_pct > ham_pct else "ham"

    return jsonify({
        "label":          label,
        "spam_pct":       spam_pct,
        "ham_pct":        ham_pct,
        "cleaned_text":   text,
        "top_spam_words": [],
        "top_ham_words":  [],
        "word_count":     len(text.split()),
        "char_count":     len(text),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)