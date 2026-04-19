"""
predict.py  ─  main entry point
Loads the trained model and lets the user check any email or sentence.

Usage:
    python predict.py                        # interactive mode
    python predict.py "Win a free iPhone!"  # single check from command line
"""

import os
import sys
import pickle

# Allow sibling imports
sys.path.insert(0, os.path.dirname(__file__))

from utils.text_processing import clean_text
# Must import these so pickle can deserialise the saved model objects
from model.train import MultinomialNaiveBayes, CountVectorizer  # noqa: F401

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "naive_bayes.pkl")
VEC_PATH   = os.path.join(os.path.dirname(__file__), "model", "vectorizer.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        print("\n[!] Model not found. Please train first:")
        print("      python model/train.py\n")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty input"}), 400

    proba   = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    spam_pct = round(proba[list(classes).index("spam")] * 100, 1)
    ham_pct  = round(proba[list(classes).index("ham")]  * 100, 1)
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

def print_result(text: str, result: dict):
    bar_len = 30
    spam_bars = int(result["spam_pct"] / 100 * bar_len)
    ham_bars  = bar_len - spam_bars

    verdict = "🚨 SPAM" if result["label"] == "spam" else "✅ NOT SPAM"

    print("\n" + "─" * 52)
    print(f"  Input   : {text[:60]}{'…' if len(text) > 60 else ''}")
    print(f"  Cleaned : {result['cleaned_text'][:60]}")
    print(f"  Verdict : {verdict}")
    print(f"  Spam    : {'█' * spam_bars}{'░' * (bar_len - spam_bars)}  {result['spam_pct']:.1f}%")
    print(f"  Ham     : {'█' * ham_bars}{'░' * (bar_len - ham_bars)}  {result['ham_pct']:.1f}%")
    print("─" * 52)


def interactive_mode(model, vectorizer):
    print("\n╔══════════════════════════════════════════╗")
    print("║      Naive Bayes Spam Detector  v1.0     ║")
    print("╚══════════════════════════════════════════╝")
    print("  Type an email/sentence and press Enter.")
    print("  Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            text = input("  >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if not text:
            print("  (empty input, try again)")
            continue

        result = predict(text, model, vectorizer)
        print_result(text, result)


def main():
    model, vectorizer = load_model()

    # Command-line single check
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict(text, model, vectorizer)
        print_result(text, result)
    else:
        interactive_mode(model, vectorizer)


if __name__ == "__main__":
    main()
