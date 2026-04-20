import streamlit as st
import pickle
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="SpamScan — Spam Detector",
    page_icon="📧",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
.title {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.1;
    margin-bottom: 6px;
}
.title span { color: #c8ff00; }
.subtitle { color: #888; font-size: 13px; margin-bottom: 32px; }
.verdict-spam {
    background: rgba(255,77,109,0.12);
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
}
.verdict-ham {
    background: rgba(200,255,0,0.08);
    border: 1px solid rgba(200,255,0,0.3);
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
}
.verdict-title-spam {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #ff4d6d;
    letter-spacing: -1px;
}
.verdict-title-ham {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #c8ff00;
    letter-spacing: -1px;
}
.verdict-sub { color: #888; font-size: 13px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    base       = os.path.dirname(__file__)
    model_path = os.path.join(base, "model", "naive_bayes.pkl")
    data_path  = os.path.join(base, "data", "spam.csv")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # Train fresh if no model found
    texts, labels = [], []
    with open(data_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["label"].strip().lower())
            texts.append(row["message"])

    extra_spam = [
        "Work from home and earn 5000 weekly. No experience needed. Sign up now!",
        "Earn money from home today. Limited slots available. Join now instantly.",
        "Make money online working from home. Weekly payments guaranteed. Start now!",
        "Work at home opportunity. Earn 500 daily. No experience required. Apply now!",
        "Start earning from home today. Sign up and make money instantly. Free to join!",
        "Get paid weekly working from home. Earn 1000 per day. Sign up free now!",
    ]
    texts  += extra_spam
    labels += ["spam"] * len(extra_spam)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
        ("nb",    ComplementNB(alpha=0.05)),
    ])
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


model = load_or_train_model()

# UI
st.markdown('<div class="title">Is this email <span>spam?</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Drop in any email and we will tell you instantly — spam or not. Powered by machine learning trained on thousands of real messages.</div>', unsafe_allow_html=True)

email_input = st.text_area(
    "Email / Message",
    height=160,
    placeholder="Paste your email here…",
    label_visibility="collapsed",
)

scan = st.button("🔍  Scan Email", use_container_width=True, type="primary")

if scan:
    if not email_input.strip():
        st.warning("Please enter some text first.")
    else:
        proba   = model.predict_proba([email_input])[0]
        classes = list(model.classes_)
        spam_pct = round(proba[classes.index("spam")] * 100, 1)
        ham_pct  = round(proba[classes.index("ham")]  * 100, 1)
        label    = "spam" if spam_pct > ham_pct else "ham"

        st.markdown("---")

        if label == "spam":
            st.markdown(f"""
            <div class="verdict-spam">
                <div class="verdict-title-spam">🚨 SPAM DETECTED</div>
                <div class="verdict-sub">This message shows strong spam characteristics.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-ham">
                <div class="verdict-title-ham">✅ NOT SPAM</div>
                <div class="verdict-sub">This message appears to be a legitimate email.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spam probability**")
            st.progress(spam_pct / 100)
            st.markdown(f"**{spam_pct}%**")
        with col2:
            st.markdown("**Ham probability**")
            st.progress(ham_pct / 100)
            st.markdown(f"**{ham_pct}%**")

        c1, c2 = st.columns(2)
        c1.metric("Words",      len(email_input.split()))
        c2.metric("Characters", len(email_input))