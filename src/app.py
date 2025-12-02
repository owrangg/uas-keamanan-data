import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------
# 1. Load Model & Vectorizer (HARUS kamu simpan dulu)
# -------------------------------------------------------

@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    
    tfidf = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    model = joblib.load(os.path.join(models_dir, "rf.pkl"))   # ganti jika ingin pakai XGBoost/LGBM/CNN
    return tfidf, scaler, model

tfidf, scaler, model = load_models()

# -------------------------------------------------------
# 2. PREPROCESSING
# -------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = [w for w in text.split() if w not in stop_words]
    words = [lemm.lemmatize(w) for w in words]

    return " ".join(words)

def extract_features(text):
    cleaned = clean_text(text)

    # TF-IDF
    tfidf_vec = tfidf.transform([cleaned]).toarray()

    # Length scaling
    length = len(text)
    length_scaled = scaler.transform([[length]])

    final_features = np.hstack([tfidf_vec, length_scaled])
    return final_features

# -------------------------------------------------------
# 3. STREAMLIT UI
# -------------------------------------------------------

st.title("📩 Spam Message Detector")
st.write("Masukkan pesan SMS untuk diprediksi sebagai *spam* atau *ham*.")

input_text = st.text_area("Masukkan teks pesan:")

if st.button("Prediksi"):
    if len(input_text.strip()) == 0:
        st.warning("Isi pesan dulu ya.")
    else:
        features = extract_features(input_text)
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if pred == 1:
            st.error(f"🔴 **SPAM** — Probabilitas: {proba:.4f}")
        else:
            st.success(f"🟢 **HAM** — Probabilitas spam: {proba:.4f}")
