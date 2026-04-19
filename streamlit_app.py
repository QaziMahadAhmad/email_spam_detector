import streamlit as st
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.text_processing import clean_text
from model.train import MultinomialNaiveBayes, CountVectorizer  # noqa: F401

st.set_page_config(
    page_title="SpamScan — Naive Bayes",
    page_icon="📧",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.title {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.1;
    margin-bottom: 6px;
}
.title span { color: #c8ff00; }

.subtitle {
    color: #888;
    font-size: 13px;
    margin-bottom: 32px;
}

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

.word-spam {
    display: inline-block;
    background: rgba(255,77,109,0.12);
    border: 1px solid rgba(255,77,109,0.3);
    color: #ff4d6d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    margin: 3px;
}
.word-ham {
    display: inline-block;
    background: rgba(200,255,0,0.08);
    border: 1px solid rgba(200,255,0,0.25);
    color: #c8ff00;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    margin: 3px;
}
.section-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 8px;
}
.cleaned-box {
    background: #111118;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 12px;
    color: #555;
    word-break: break-word;
    line-height: 1.8;
}
stButton > button {
    background: #c8ff00 !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "model", "naive_bayes.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "model", "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def top_words(cleaned_text, cls, model, vectorizer, n=5):
    words = cleaned_text.split()
    vocab = vectorizer.vocabulary_
    ll    = model.log_likelihood_.get(cls, [])
    scored = [(w, ll[vocab[w]]) for w in set(words) if w in vocab]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:n]]


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="title">Is this email <span>spam?</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multinomial Naive Bayes · Pure Python · No external ML libraries</div>', unsafe_allow_html=True)


# ── Text input ───────────────────────────────────────────────────────────────
email_input = st.text_area(
    "Email / Message",
    value="",
    height=160,
    placeholder='Paste your email here…\n\ne.g. "Win a free vacation! Click now to claim your reward."',
    label_visibility="collapsed",
)

scan = st.button("🔍  Scan Email", use_container_width=True, type="primary")

# ── Run prediction ───────────────────────────────────────────────────────────
if scan:
    if not email_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing…"):
            try:
                model, vectorizer = load_model()
            except FileNotFoundError:
                st.error("Model not found. Run `python model/train.py` first.")
                st.stop()

            cleaned  = clean_text(email_input)
            X        = vectorizer.transform([cleaned])
            proba    = model.predict_proba(X)[0]
            spam_pct = round(proba.get("spam", 0) * 100, 1)
            ham_pct  = round(proba.get("ham",  0) * 100, 1)
            label    = "spam" if spam_pct > ham_pct else "ham"

        st.markdown("---")

        # Verdict banner
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

        # Probability bars
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Spam probability</div>', unsafe_allow_html=True)
            st.progress(spam_pct / 100)
            st.markdown(f"**{spam_pct}%**")
        with col2:
            st.markdown('<div class="section-label">Ham probability</div>', unsafe_allow_html=True)
            st.progress(ham_pct / 100)
            st.markdown(f"**{ham_pct}%**")

        st.markdown("<br/>", unsafe_allow_html=True)

        # Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Words",      len(email_input.split()))
        c2.metric("Characters", len(email_input))
        c3.metric("Features",   len(cleaned.split()))

        st.markdown("<br/>", unsafe_allow_html=True)

        # Key words
        spam_words = top_words(cleaned, "spam", model, vectorizer)
        ham_words  = top_words(cleaned, "ham",  model, vectorizer)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">🚨 Top spam signals</div>', unsafe_allow_html=True)
            if spam_words:
                pills = "".join(f'<span class="word-spam">{w}</span>' for w in spam_words)
            else:
                pills = "<span style='color:#555;font-size:12px'>none found</span>"
            st.markdown(pills, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-label">✅ Top ham signals</div>', unsafe_allow_html=True)
            if ham_words:
                pills = "".join(f'<span class="word-ham">{w}</span>' for w in ham_words)
            else:
                pills = "<span style='color:#555;font-size:12px'>none found</span>"
            st.markdown(pills, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Cleaned text
        st.markdown('<div class="section-label">Processed text (what the model sees)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cleaned-box">{cleaned or "(no recognisable tokens)"}</div>', unsafe_allow_html=True)