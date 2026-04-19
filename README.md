# 📧 Email Spam Detector

A machine learning web app that detects spam emails using a **Multinomial Naive Bayes** classifier built entirely from scratch — no scikit-learn, no NLTK, only pure Python.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-97.31%25-brightgreen?style=flat)

---

## 🚀 Live Demo

👉 **[Try it here](https://your-app.streamlit.app)** ← replace with your Streamlit link

---

## ✨ Features

- ✅ Classifies any email or sentence as **spam** or **not spam**
- ✅ Shows **spam vs ham probability** with visual bars
- ✅ Highlights the **key words** that triggered the decision
- ✅ Built from scratch — every algorithm written in pure Python
- ✅ Trained on the **UCI SMS Spam Collection** (5,574 real emails)
- ✅ **97.31% accuracy** on the test set

---

## 🧠 How It Works

Naive Bayes asks: *given the words in this email, which is more likely — spam or ham?*

```
P(spam | email) ∝ P(spam) × P(w₁|spam) × P(w₂|spam) × ... × P(wₙ|spam)
P(ham  | email) ∝ P(ham)  × P(w₁|ham)  × P(w₂|ham)  × ... × P(wₙ|ham)
```

The class with the higher probability wins. To avoid floating-point underflow from multiplying thousands of tiny numbers, all calculations are done in **log-space**.

**Pipeline:**

```
Raw email → Lowercase → Remove URLs & punctuation → Tokenise
         → Remove stop words → Stem → Vectorise → Naive Bayes → Verdict
```

---

## 📁 Project Structure

```
email-spam-detector/
│
├── streamlit_app.py        ← Streamlit web app (deployed)
├── app.py                  ← Flask web app (run locally)
├── main.py                 ← One-click local setup + run
├── predict.py              ← CLI predictor
├── requirements.txt
│
├── data/
│   ├── prepare_data.py     ← Downloads & prepares the dataset
│   └── spam.csv            ← Generated after running prepare_data.py
│
├── model/
│   ├── train.py            ← CountVectorizer + Naive Bayes from scratch
│   ├── naive_bayes.pkl     ← Saved model (generated after training)
│   └── vectorizer.pkl      ← Saved vectorizer (generated after training)
│
├── utils/
│   └── text_processing.py  ← Cleaning, tokenisation, stemming
│
└── templates/
    └── index.html          ← Flask frontend
```

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/QaziMahadAhmad/email-spam-detector.git
cd email-spam-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run everything at once**
```bash
python main.py
```

This will automatically download the dataset, train the model, and launch the web app at `http://127.0.0.1:5000`.

**Or run steps individually:**
```bash
python data/prepare_data.py   # download & prepare dataset
python model/train.py         # train the model
python app.py                 # start Flask web server
```

---

## 📊 Model Performance

Trained on 5,574 emails from the UCI SMS Spam Collection dataset.

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Ham   | 98.63%    | 98.22% | 98.43%   |
| Spam  | 89.57%    | 91.82% | 90.68%   |
| **Overall** | — | — | **97.31%** |

---

## 🛠️ Built With

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Web framework | Flask + Streamlit |
| ML algorithm | Multinomial Naive Bayes (from scratch) |
| Vectoriser | Bag-of-Words CountVectorizer (from scratch) |
| Dataset | UCI SMS Spam Collection (5,574 emails) |
| Deployment | Streamlit Cloud |

---

## 👤 Author

**Mahad Ahmad**
- GitHub: [@QaziMahadAhmad](https://github.com/QaziMahadAhmad)
