import os
import sys
import csv
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive_bayes.pkl")
VEC_PATH   = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")


def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["label"].strip().lower())
            texts.append(row["message"])
    return texts, labels


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    print("Loading data ...")
    texts, labels = load_data(os.path.abspath(DATA_PATH))
    print(f"  Total: {len(texts)} | Spam: {labels.count('spam')} | Ham: {labels.count('ham')}")

    # Add extra work-from-home spam examples
    extra_spam = [
        "Work from home and earn 5000 weekly. No experience needed. Sign up now!",
        "Earn money from home today. Limited slots available. Join now instantly.",
        "Make money online working from home. Weekly payments guaranteed. Start now!",
        "Work at home opportunity. Earn 500 daily. No experience required. Apply now!",
        "Start earning from home today. Sign up and make money instantly. Free to join!",
        "Home based job. Earn weekly. No experience needed. Limited slots. Apply today!",
        "Get paid weekly working from home. Earn 1000 per day. Sign up free now!",
        "Make extra income from home. Weekly salary. No skills needed. Join today free!",
        "Work from home earn 5000 weekly no experience needed start today limited slots",
        "Earn 5000 per week from home. Begin making money instantly. No experience needed.",
    ]
    texts  += extra_spam
    labels += ["spam"] * len(extra_spam)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("nb", ComplementNB(alpha=0.05)),
    ])

    print("Training ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()