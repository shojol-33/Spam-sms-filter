import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam SMS Filter",
    page_icon="📱",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .spam-badge {
        background: #ff4444;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .ham-badge {
        background: #00c853;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .result-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 12px;
        border-left: 4px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on SMS dataset…")
def load_models():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X, y = df['text'], df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    nb = MultinomialNB()
    lr = LogisticRegression(solver='liblinear')
    nb.fit(X_train_vec, y_train)
    lr.fit(X_train_vec, y_train)

    nb_preds = nb.predict(X_test_vec)
    lr_preds = lr.predict(X_test_vec)

    metrics = {
        "nb_acc": accuracy_score(y_test, nb_preds),
        "lr_acc": accuracy_score(y_test, lr_preds),
        "nb_report": classification_report(y_test, nb_preds, target_names=["Ham", "Spam"], output_dict=True),
        "lr_report": classification_report(y_test, lr_preds, target_names=["Ham", "Spam"], output_dict=True),
        "vocab_size": X_train_vec.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return vectorizer, nb, lr, metrics


vectorizer, nb_model, lr_model, metrics = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📱 Spam SMS Filter")
st.markdown("Classify any SMS message as **Ham** (legitimate) or **Spam** using ML.")
st.divider()

# ── Prediction section ────────────────────────────────────────────────────────
st.subheader("🔍 Classify a Message")

user_input = st.text_area(
    "Enter an SMS message:",
    placeholder="e.g. URGENT! You've won a £1,000 prize. Call now to claim!",
    height=120,
)

model_choice = st.radio(
    "Choose classifier:",
    ["Both", "Naive Bayes", "Logistic Regression"],
    horizontal=True,
)

if st.button("Classify", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        vec = vectorizer.transform([user_input])

        nb_pred = nb_model.predict(vec)[0]
        lr_pred = lr_model.predict(vec)[0]

        nb_proba = nb_model.predict_proba(vec)[0]
        lr_proba = lr_model.predict_proba(vec)[0]

        def render_result(name, pred, proba):
            label    = "🚨 SPAM"   if pred == 1 else "🟢 HAM"
            badge    = "spam-badge" if pred == 1 else "ham-badge"
            conf     = proba[pred] * 100
            st.markdown(f"""
            <div class="result-card">
                <strong>{name}</strong><br>
                <span class="{badge}">{label}</span>
                &nbsp;&nbsp;<small style="color:#666;">Confidence: {conf:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Results")
        if model_choice in ("Both", "Naive Bayes"):
            render_result("Naive Bayes", nb_pred, nb_proba)
        if model_choice in ("Both", "Logistic Regression"):
            render_result("Logistic Regression", lr_pred, lr_proba)

st.divider()

# ── Model metrics ─────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Training Samples", f"{metrics['train_size']:,}")
col2.metric("Test Samples",     f"{metrics['test_size']:,}")
col3.metric("Vocabulary Size",  f"{metrics['vocab_size']:,}")

tab1, tab2 = st.tabs(["Naive Bayes", "Logistic Regression"])

def show_metrics(report, acc):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{acc*100:.2f}%")
    c2.metric("Spam Precision", f"{report['Spam']['precision']*100:.2f}%")
    c3.metric("Spam Recall",    f"{report['Spam']['recall']*100:.2f}%")

    df_report = pd.DataFrame(report).transpose().round(3)
    df_report = df_report.drop(index=["accuracy"], errors="ignore")
    st.dataframe(df_report[["precision", "recall", "f1-score", "support"]], use_container_width=True)

with tab1:
    show_metrics(metrics["nb_report"], metrics["nb_acc"])

with tab2:
    show_metrics(metrics["lr_report"], metrics["lr_acc"])

st.divider()

# ── Sample messages ───────────────────────────────────────────────────────────
st.subheader("💬 Try Sample Messages")

samples = {
    "Ham — Casual":   "Hey, are we still meeting up for lunch at 1 PM today?",
    "Ham — Personal": "Mum says your dinner is in the fridge. Don't forget to call her!",
    "Spam — Prize":   "URGENT! Your mobile number has won a £2,000 prize! Call 09051234567 now.",
    "Spam — Link":    "Free entry in 2 a weekly competition to win FA Cup final tkts! Txt FA to 87121.",
}

chosen = st.selectbox("Pick an example:", list(samples.keys()))
st.code(samples[chosen], language=None)

if st.button("Classify Sample", use_container_width=True):
    vec = vectorizer.transform([samples[chosen]])
    nb_pred = nb_model.predict(vec)[0]
    lr_pred = lr_model.predict(vec)[0]
    nb_conf = nb_model.predict_proba(vec)[0][nb_pred] * 100
    lr_conf = lr_model.predict_proba(vec)[0][lr_pred] * 100

    label_map = {0: "🟢 HAM (Legit)", 1: "🚨 SPAM"}
    c1, c2 = st.columns(2)
    c1.success(f"**Naive Bayes:** {label_map[nb_pred]}  \nConfidence: {nb_conf:.1f}%")
    c2.info(   f"**Logistic Reg:** {label_map[lr_pred]}  \nConfidence: {lr_conf:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
---
<center><small>Built with Streamlit · TF-IDF · Scikit-learn · SMS Spam Collection Dataset</small></center>
""", unsafe_allow_html=True)
