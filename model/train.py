"""
model/train.py
Trains a Multinomial Naive Bayes spam classifier and saves it to disk.

Usage:
    python train.py
"""

import os
import sys
import csv
import pickle

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.text_processing import clean_text

# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive_bayes.pkl")
VEC_PATH   = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")


# ── minimal CountVectorizer ─────────────────────────────────────────────────
class CountVectorizer:
    """
    Converts a list of cleaned text strings into a bag-of-words
    feature matrix (list of dicts) without any external dependency.
    """

    def __init__(self, max_features: int = 3000):
        self.max_features = max_features
        self.vocabulary_: dict[str, int] = {}

    def fit(self, texts: list[str]) -> "CountVectorizer":
        counts: dict[str, int] = {}
        for text in texts:
            for word in text.split():
                counts[word] = counts.get(word, 0) + 1
        top = sorted(counts, key=counts.get, reverse=True)[: self.max_features]
        self.vocabulary_ = {word: i for i, word in enumerate(top)}
        return self

    def transform(self, texts: list[str]) -> list[list[int]]:
        n = len(self.vocabulary_)
        matrix = []
        for text in texts:
            row = [0] * n
            for word in text.split():
                if word in self.vocabulary_:
                    row[self.vocabulary_[word]] += 1
            matrix.append(row)
        return matrix

    def fit_transform(self, texts: list[str]) -> list[list[int]]:
        self.fit(texts)
        return self.transform(texts)


# ── Naive Bayes classifier ──────────────────────────────────────────────────
class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes with Laplace smoothing.

    Stores log-probabilities to avoid underflow when multiplying
    many small per-word probabilities together.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: list = []
        self.log_class_prior_: dict = {}   # log P(class)
        self.log_likelihood_: dict = {}    # log P(word | class)

    def fit(self, X: list[list[int]], y: list[str]) -> "MultinomialNaiveBayes":
        import math

        self.classes_ = list(set(y))
        n_samples = len(y)
        n_features = len(X[0]) if X else 0

        for cls in self.classes_:
            # Indices belonging to this class
            cls_indices = [i for i, label in enumerate(y) if label == cls]
            n_cls = len(cls_indices)

            # ── Prior: P(class) ───────────────────────────────────────────
            self.log_class_prior_[cls] = math.log(n_cls / n_samples)

            # ── Word counts for this class ────────────────────────────────
            word_counts = [0] * n_features
            for i in cls_indices:
                for j, count in enumerate(X[i]):
                    word_counts[j] += count

            total_words = sum(word_counts)

            # ── Likelihood: P(word | class) with Laplace smoothing ────────
            self.log_likelihood_[cls] = [
                math.log((word_counts[j] + self.alpha) /
                          (total_words + self.alpha * n_features))
                for j in range(n_features)
            ]

        return self

    def predict_proba(self, X: list[list[int]]) -> list[dict]:
        """Return list of {class: probability} dicts, one per sample."""
        import math

        results = []
        for sample in X:
            log_scores = {}
            for cls in self.classes_:
                score = self.log_class_prior_[cls]
                for j, count in enumerate(sample):
                    if count > 0:
                        score += count * self.log_likelihood_[cls][j]
                log_scores[cls] = score

            # Convert log-scores to probabilities via softmax
            max_score = max(log_scores.values())
            exp_scores = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
            total = sum(exp_scores.values())
            results.append({cls: exp_scores[cls] / total for cls in self.classes_})

        return results

    def predict(self, X: list[list[int]]) -> list[str]:
        return [max(proba, key=proba.get) for proba in self.predict_proba(X)]


# ── training routine ────────────────────────────────────────────────────────
def load_data(path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["label"].strip().lower())
            texts.append(clean_text(row["message"]))
    return texts, labels


def train_test_split(texts, labels, test_ratio=0.2, seed=42):
    import random
    random.seed(seed)
    data = list(zip(texts, labels))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train, test = data[:split], data[split:]
    X_train, y_train = zip(*train) if train else ([], [])
    X_test,  y_test  = zip(*test)  if test  else ([], [])
    return list(X_train), list(y_train), list(X_test), list(y_test)


def evaluate(y_true, y_pred):
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    total   = len(y_true)
    accuracy = correct / total if total else 0

    # Per-class metrics
    classes = list(set(y_true))
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 42)
    for cls in sorted(classes):
        tp = sum(a == cls and b == cls for a, b in zip(y_true, y_pred))
        fp = sum(a != cls and b == cls for a, b in zip(y_true, y_pred))
        fn = sum(a == cls and b != cls for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall    = tp / (tp + fn) if (tp + fn) else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0)
        print(f"{cls:<10} {precision:>10.2%} {recall:>10.2%} {f1:>8.2%}")
    print(f"\nOverall accuracy: {accuracy:.2%}  ({correct}/{total})")


def main():
    # 1. Load data
    data_path = os.path.abspath(DATA_PATH)
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}.")
        print("Run  python data/prepare_data.py  first.\n")
        sys.exit(1)

    print("Loading and cleaning data …")
    texts, labels = load_data(data_path)
    print(f"  Total samples: {len(texts)}")
    print(f"  Spam: {labels.count('spam')}  |  Ham: {labels.count('ham')}")

    # 2. Train/test split
    X_train_raw, y_train, X_test_raw, y_test = train_test_split(
        texts, labels, test_ratio=0.2
    )

    # 3. Vectorise
    print("\nBuilding vocabulary and vectorising …")
    vectorizer = CountVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 4. Train
    print("\nTraining Multinomial Naive Bayes …")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("\n── Evaluation on test set ──")
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)

    # 6. Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nModel saved   → {os.path.abspath(MODEL_PATH)}")
    print(f"Vectorizer    → {os.path.abspath(VEC_PATH)}")


if __name__ == "__main__":
    main()
