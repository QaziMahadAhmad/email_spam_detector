# Spam Detector — Naive Bayes (Pure Python)

A complete spam/ham email classifier built from scratch using
Multinomial Naive Bayes. **No scikit-learn, no NLTK — only the
Python standard library.**

---

## Project Structure

```
spam_detector/
├── main.py               ← one-click setup + run
├── predict.py            ← interactive predictor (use after training)
│
├── data/
│   ├── prepare_data.py   ← downloads dataset (or generates sample)
│   └── spam.csv          ← created by prepare_data.py
│
├── model/
│   ├── train.py          ← training script + CountVectorizer + NB
│   ├── naive_bayes.pkl   ← saved model (created after training)
│   └── vectorizer.pkl    ← saved vectorizer (created after training)
│
└── utils/
    └── text_processing.py ← cleaning, tokenisation, stemming
```

---

## Quick Start

### 1. Open in PyCharm
File → Open → select the `spam_detector/` folder.

### 2. Run everything at once
```
python main.py
```
This will:
1. Download / generate the dataset
2. Train the model and print evaluation metrics
3. Launch the interactive predictor

### 3. Or run steps individually
```bash
# Step 1 — prepare data
python data/prepare_data.py

# Step 2 — train
python model/train.py

# Step 3 — predict interactively
python predict.py

# Step 3b — single command-line check
python predict.py "Congratulations, you won a free prize!"
```

---

## How Naive Bayes Works Here

```
P(spam | email) ∝ P(spam) × ∏ P(word | spam)
P(ham  | email) ∝ P(ham)  × ∏ P(word | ham)
```

1. **Training** — count how often each word appears in spam vs ham
   messages and compute log-probabilities (Laplace-smoothed).
2. **Prediction** — for a new email, multiply its word probabilities
   under each class and pick the winner (argmax).
3. **Log-space** — all multiplications are done in log-space to
   prevent floating-point underflow.

---

## Getting a Larger Dataset

The built-in sample (20 emails) is enough to verify the code.
For real accuracy (~97 %+) use the full SMS Spam Collection:

1. Download from https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Extract `SMSSpamCollection` into `data/`
3. Re-run `python data/prepare_data.py` (it will convert TSV → CSV)
4. Re-train with `python model/train.py`

---

## Key Files Explained

| File | What it does |
|------|-------------|
| `utils/text_processing.py` | lowercase → remove URLs/punctuation → remove stop words → simple stemmer |
| `model/train.py` | `CountVectorizer` (bag-of-words), `MultinomialNaiveBayes`, train/test split, evaluation |
| `predict.py` | loads saved model, cleans input, prints probability bar chart |
