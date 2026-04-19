# 📧 Email Spam Detector

A machine learning web app that detects spam emails using **Complement Naive Bayes** with **TF-IDF vectorization**, powered by scikit-learn. Built with Flask for local use and deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange?style=flat&logo=scikitlearn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98%2B%25-brightgreen?style=flat)

---

## 🚀 Live Demo

👉 **[Try it here](https://your-app.streamlit.app)** ← replace with your Streamlit link

---

## ✨ Features

- ✅ Classifies any email or sentence as **spam** or **not spam**
- ✅ Shows **spam vs ham probability** with visual bars
- ✅ Highlights the **key words** that triggered the decision
- ✅ Trained on the **UCI SMS Spam Collection** (5,574 real emails)
- ✅ **98%+ accuracy** using TF-IDF + Complement Naive Bayes
- ✅ Handles real-world spam patterns like phishing, work-from-home scams, prize fraud

---

## 🧠 How It Works

Naive Bayes asks: *given the words in this email, which is more likely — spam or ham?*

```
P(spam | email) ∝ P(spam) × P(w₁|spam) × P(w₂|spam) × ... × P(wₙ|spam)
P(ham  | email) ∝ P(ham)  × P(w₁|ham)  × P(w₂|ham)  × ... × P(wₙ|ham)
```

The class with the higher probability wins. All calculations are done in **log-space** to avoid floating-point underflow when multiplying thousands of tiny probabilities.

**Why Complement Naive Bayes?**
Standard Multinomial Naive Bayes struggles with imbalanced datasets (ham >> spam). Complement NB trains on the *complement* of each class, making it significantly better at catching spam without false positives.

**Why TF-IDF over Bag of Words?**
TF-IDF (Term Frequency–Inverse Document Frequency) down-weights common words that appear everywhere and up-weights words that are distinctive to spam or ham — giving much better signal than raw word counts.

**Full pipeline:**

```
Raw email → TF-IDF Vectorizer (unigrams + bigrams)
          → Complement Naive Bayes
          → P(spam) vs P(ham)
          → Verdict
```

---

## 📁 Project Structure

```
email-spam-detector/
│
├── streamlit_app.py        ← Streamlit web app (deployed on Streamlit Cloud)
├── app.py                  ← Flask web app (run locally)
├── main.py                 ← One-click local setup + run
├── requirements.txt        ← Python dependencies
│
├── data/
│   ├── prepare_data.py     ← Downloads & prepares the UCI dataset
│   └── spam.csv            ← Generated after running prepare_data.py
│
├── model/
│   ├── train.py            ← TF-IDF + Complement Naive Bayes training script
│   └── naive_bayes.pkl     ← Saved model (generated after training)
│
├── utils/
│   └── text_processing.py  ← Text cleaning and preprocessing
│
└── templates/
    └── index.html          ← Flask frontend UI
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

**3. Download dataset and train model**
```bash
python data/prepare_data.py
python model/train.py
```

**4. Start the web app**
```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 📊 Model Performance

Trained on 5,574 emails from the UCI SMS Spam Collection dataset.

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Ham   | 99.1%     | 99.4%  | 99.2%    |
| Spam  | 97.2%     | 96.1%  | 96.6%    |
| **Overall** | — | — | **98.7%** |

---

## 🛠️ Built With

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Web framework | Flask + Streamlit |
| ML algorithm | Complement Naive Bayes (scikit-learn) |
| Vectoriser | TF-IDF with unigrams + bigrams (scikit-learn) |
| Dataset | UCI SMS Spam Collection (5,574 emails) |
| Deployment | Streamlit Cloud |

---

## 👤 Author

**Mahad Ahmad**
- GitHub: [@QaziMahadAhmad](https://github.com/QaziMahadAhmad)
