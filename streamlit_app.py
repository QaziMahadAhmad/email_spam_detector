import streamlit as st
import pickle
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="SpamScan • Spam Detector",
    page_icon="📧",
    layout="centered",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>

<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 760px !important;
}
.stApp {
    background-color: #0a0a0f !important;
    font-family: 'DM Mono', monospace !important;
    color: #e8e8f0 !important;
}
.headline {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 7vw, 56px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 14px;
    color: #fff;
}
.headline em { color: #c8ff00; }
.subline {
    font-size: 13px;
    color: #6b6b80;
    margin-bottom: 36px;
}
.input-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px;
}
.stTextArea textarea {
    background: transparent !important;
    border: none !important;
    color: #e8e8f0 !important;
}
.stButton > button {
    background: #c8ff00 !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model", "naive_bayes.pkl")
    data_path = os.path.join(base, "data", "spam.csv")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    texts, labels = [], []
    with open(data_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["label"].strip().lower())
            texts.append(row["message"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", ComplementNB()),
    ])
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


model = load_or_train_model()

# Header
st.markdown("""
<div class="headline">Is this email <em>spam?</em></div>
<div class="subline">Paste an email and get instant results powered by machine learning.</div>
""", unsafe_allow_html=True)

# Input
st.markdown('<div class="input-card">', unsafe_allow_html=True)

email_input = st.text_area(
    "",
    height=150,
    placeholder='Paste your email content here...\n\nExample: "Congratulations! You won a free iPhone!"',
)

st.markdown('</div>', unsafe_allow_html=True)

# Single Button
scan = st.button("Scan Email")

# Prediction
if scan:
    if not email_input.strip():
        st.error("Please enter some text first.")
    else:
        result = model.predict([email_input])[0]
        proba = model.predict_proba([email_input])[0]

        if result == "spam":
            st.error(f"🚨 SPAM detected ({max(proba)*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam ({max(proba)*100:.2f}%)")