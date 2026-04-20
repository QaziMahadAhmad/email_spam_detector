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
:root {
    --bg:       #0a0a0f;
    --surface:  #111118;
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.13);
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --accent:   #c8ff00;
    --spam-clr: #ff4d6d;
    --ham-clr:  #c8ff00;
}
.stApp {
    background-color: #0a0a0f !important;
    font-family: 'DM Mono', monospace !important;
    color: #e8e8f0 !important;
}
.spamscan-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 36px 0 44px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 44px;
}
.logo { display: flex; align-items: center; gap: 12px; }
.logo-icon {
    width: 38px; height: 38px;
    background: #c8ff00;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
}
.logo-text span { color: #c8ff00; }
.badge {
    font-size: 10px;
    font-weight: 500;
    color: #6b6b80;
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 30px;
    padding: 5px 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
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
.headline em { font-style: normal; color: #c8ff00; }
.subline {
    font-size: 13px;
    color: #6b6b80;
    line-height: 1.7;
    margin-bottom: 36px;
    max-width: 480px;
    font-family: 'DM Mono', monospace;
}
.input-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px 28px 20px;
    margin-bottom: 8px;
}
.stTextArea textarea {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
    caret-color: #c8ff00 !important;
    box-shadow: none !important;
    padding: 0 !important;
    resize: none !important;
}
.stTextArea textarea::placeholder { color: #6b6b80 !important; }
.stTextArea textarea:focus { box-shadow: none !important; border: none !important; }
.stTextArea > div > div { background: transparent !important; border: none !important; }
.stTextArea { margin-bottom: 0 !important; }

/* Scan Email button - green */
div[data-testid="column"]:first-child .stButton > button {
    background: #c8ff00 !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    margin-top: 8px !important;
}
div[data-testid="column"]:first-child .stButton > button:hover { opacity: 0.88 !important; }

/* Check Another Email button - outlined */
div[data-testid="column"]:last-child .stButton > button {
    background: transparent !important;
    color: #6b6b80 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    margin-top: 8px !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
    color: #e8e8f0 !important;
    border-color: rgba(255,255,255,0.25) !important;
}

.verdict-spam {
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.2);
    border-radius: 14px;
    padding: 28px 32px;
    margin: 20px 0 16px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.verdict-ham {
    background: rgba(200,255,0,0.06);
    border: 1px solid rgba(200,255,0,0.18);
    border-radius: 14px;
    padding: 28px 32px;
    margin: 20px 0 16px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.verdict-icon { font-size: 40px; line-height: 1; flex-shrink: 0; }
.verdict-info { flex: 1; }
.verdict-title-spam {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #ff4d6d;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 4px;
}
.verdict-title-ham {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #c8ff00;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 4px;
}
.verdict-sub { color: #6b6b80; font-size: 12px; font-family: 'DM Mono', monospace; }
.verdict-pct-spam {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #ff4d6d;
    letter-spacing: -1px;
    flex-shrink: 0;
}
.verdict-pct-ham {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #c8ff00;
    letter-spacing: -1px;
    flex-shrink: 0;
}
.prob-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.prob-box {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 18px 20px;
}
.prob-label { font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase; color: #6b6b80; margin-bottom: 10px; font-family: 'DM Mono', monospace; }
.prob-bar-bg { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; margin-bottom: 10px; }
.prob-bar-spam { height: 100%; background: #ff4d6d; border-radius: 3px; }
.prob-bar-ham  { height: 100%; background: #c8ff00; border-radius: 3px; }
.prob-pct-spam { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 700; color: #ff4d6d; letter-spacing: -1px; }
.prob-pct-ham  { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 700; color: #c8ff00; letter-spacing: -1px; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.stat-box {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 16px 18px;
}
.stat-label { font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: #6b6b80; margin-bottom: 8px; font-family: 'DM Mono', monospace; }
.stat-val { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #fff; }
.spamscan-footer {
    text-align: center;
    font-size: 11px;
    color: #6b6b80;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-family: 'DM Mono', monospace;
}
[data-testid="stMetric"] { display: none; }
div[data-testid="stProgress"] { display: none; }
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

# Header
st.markdown("""
<div class="spamscan-header">
  <div class="logo">
    <div class="logo-icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
        <polyline points="22,6 12,13 2,6"/>
      </svg>
    </div>
    <span class="logo-text">Spam<span>Scan</span></span>
  </div>
  <span class="badge">TF-IDF · Complement NB · v2.0</span>
</div>
""", unsafe_allow_html=True)

# Headline
st.markdown("""
<div class="headline">Is this email<br/><em>spam?</em></div>
<div class="subline">Drop in any email and we'll tell you instantly — spam or not. Powered by machine learning trained on thousands of real messages.</div>
""", unsafe_allow_html=True)

# Input card — no label
st.markdown('<div class="input-card">', unsafe_allow_html=True)
email_input = st.text_area(
    "",
    height=10,
    placeholder='Paste your email content here?\n\ne.g. "Congratulations! You have won a free iPhone. Click now to claim."',
)
st.markdown('</div>', unsafe_allow_html=True)

# Buttons side by side
scan = st.button("?  Scan Email", use_container_width=True)
if clear:
    st.rerun()

# Result
if scan:
    if not email_input.strip():
        st.markdown('<div style="color:#ff4d6d;font-size:13px;font-family:DM Mono,monospace;margin-top:8px">⚠ Please enter some text first.</div>', unsafe_allow_html=True)
    else:
        proba    = model.predict_proba([email_input])[0]
        classes  = list(model.classes_)
        spam_pct = round(proba[classes.index("spam")] * 100, 1)
        ham_pct  = round(proba[classes.index("ham")]  * 100, 1)
        label    = "spam" if spam_pct > ham_pct else "ham"

        if label == "spam":
            st.markdown(f"""
            <div class="verdict-spam">
              <div class="verdict-icon">🚨</div>
              <div class="verdict-info">
                <div class="verdict-title-spam">SPAM DETECTED</div>
                <div class="verdict-sub">This message shows strong spam characteristics.</div>
              </div>
              <div class="verdict-pct-spam">{spam_pct}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-ham">
              <div class="verdict-icon">✅</div>
              <div class="verdict-info">
                <div class="verdict-title-ham">NOT SPAM</div>
                <div class="verdict-sub">This message appears to be a legitimate email.</div>
              </div>
              <div class="verdict-pct-ham">{ham_pct}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-grid">
          <div class="prob-box">
            <div class="prob-label">Spam probability</div>
            <div class="prob-bar-bg"><div class="prob-bar-spam" style="width:{spam_pct}%"></div></div>
            <div class="prob-pct-spam">{spam_pct}%</div>
          </div>
          <div class="prob-box">
            <div class="prob-label">Ham probability</div>
            <div class="prob-bar-bg"><div class="prob-bar-ham" style="width:{ham_pct}%"></div></div>
            <div class="prob-pct-ham">{ham_pct}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

        word_count = len(email_input.split())
        char_count = len(email_input)
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-label">Words</div>
            <div class="stat-val">{word_count}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Characters</div>
            <div class="stat-val">{char_count}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Confidence</div>
            <div class="stat-val">{max(spam_pct, ham_pct)}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="spamscan-footer">
  Built with <strong>Complement Naive Bayes</strong> + TF-IDF + Streamlit &nbsp;·&nbsp; scikit-learn
</div>
""", unsafe_allow_html=True)