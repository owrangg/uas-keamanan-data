import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Config ---
st.set_page_config(page_title="Spam Detection App", layout="wide")

# --- NLTK Setup ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

# --- Functions ---

@st.cache_data
def load_data():
    try:
        # Construct absolute path to data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'spam.csv')
        
        df = pd.read_csv(data_path, encoding='latin-1')
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        df['length'] = df['message'].apply(len)
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except FileNotFoundError:
        st.error(f"File not found. Please make sure it is in the data directory.")
        return None

@st.cache_resource
def load_artifacts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    
    artifacts = {}
    try:
        artifacts['tfidf'] = joblib.load(os.path.join(models_dir, 'tfidf.pkl'))
        artifacts['scaler'] = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        artifacts['tokenizer'] = joblib.load(os.path.join(models_dir, 'tokenizer.pkl'))
        
        # Load Models
        artifacts['rf'] = joblib.load(os.path.join(models_dir, 'rf.pkl'))
        artifacts['lr'] = joblib.load(os.path.join(models_dir, 'lr.pkl'))
        artifacts['xgb'] = joblib.load(os.path.join(models_dir, 'xgb.pkl'))
        artifacts['lgbm'] = joblib.load(os.path.join(models_dir, 'lgbm.pkl'))
        artifacts['isoforest'] = joblib.load(os.path.join(models_dir, 'isoforest.pkl'))
        artifacts['cnn'] = load_model(os.path.join(models_dir, 'cnn.h5'))
        
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        return None
        
    return artifacts

def count_uppercase_ratio(text):
    total_chars = len(text)
    if total_chars == 0:
        return 0
    uppercase_count = sum(c.isupper() for c in text)
    return uppercase_count / total_chars

def remove_punctuation_numbers(text):
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@st.cache_data
def preprocess_dataframe(df):
    df = df.copy()
    df['uppercase_ratio'] = df['message'].apply(count_uppercase_ratio)
    df['lower_case'] = df['message'].str.lower()
    df['remove_punctuation'] = df['lower_case'].apply(remove_punctuation_numbers)
    df['stopwords'] = df['remove_punctuation'].apply(remove_stopwords)
    df['lemmatization'] = df['stopwords'].apply(apply_lemmatization)
    return df

# --- Main App ---

st.title("🛡️ Spam Detection Dashboard")

df_raw = load_data()
artifacts = load_artifacts()

if df_raw is not None and artifacts is not None:
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["EDA", "Model Info & Evaluation", "Prediction"])

    # Preprocessing (Cached)
    with st.spinner('Preprocessing data...'):
        df = preprocess_dataframe(df_raw)

    if page == "EDA":
        st.header("Exploratory Data Analysis")

        # 1. Class Distribution
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df['label'].value_counts())
        with col2:
            fig, ax = plt.subplots()
            sns.countplot(x='label', data=df, hue='label', palette=['skyblue', 'pink'], legend=False, ax=ax)
            st.pyplot(fig)

        # 2. Message Length
        st.subheader("Message Length Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        df[df['label'] == 'ham']['length'].hist(bins=50, alpha=0.7, label='Ham', color='skyblue', ax=ax)
        df[df['label'] == 'spam']['length'].hist(bins=50, alpha=0.7, label='Spam', color='salmon', ax=ax)
        ax.legend()
        st.pyplot(fig)

        # 3. Word Clouds
        st.subheader("Word Clouds")
        col_spam, col_ham = st.columns(2)
        
        with col_spam:
            st.markdown("**Spam Messages**")
            spam_words = ' '.join(list(df[df['label'] == 'spam']['message']))
            spam_wc = WordCloud(width=400, height=300, background_color='white', max_words=100, colormap='Reds').generate(spam_words)
            fig, ax = plt.subplots()
            ax.imshow(spam_wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        with col_ham:
            st.markdown("**Ham Messages**")
            ham_words = ' '.join(list(df[df['label'] == 'ham']['message']))
            ham_wc = WordCloud(width=400, height=300, background_color='white', max_words=100, colormap='Blues').generate(ham_words)
            fig, ax = plt.subplots()
            ax.imshow(ham_wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        # 4. Uppercase Ratio
        st.subheader("Uppercase Ratio")
        fig, ax = plt.subplots()
        sns.boxplot(x='label', y='uppercase_ratio', data=df, hue='label', palette=['skyblue', 'salmon'], legend=False, ax=ax)
        st.pyplot(fig)

    elif page == "Model Info & Evaluation":
        st.header("Model Performance (Pre-trained)")
        
        model_option = st.selectbox("Select Model to View Performance", [
            "Random Forest", 
            "Logistic Regression", 
            "XGBoost", 
            "LightGBM", 
            "Isolation Forest (Anomaly Detection)",
            "CNN (Deep Learning)"
        ])
        
        # Map option to artifact key
        model_map = {
            "Random Forest": "rf",
            "Logistic Regression": "lr",
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "Isolation Forest (Anomaly Detection)": "isoforest",
            "CNN (Deep Learning)": "cnn"
        }
        
        model_key = model_map[model_option]
        model = artifacts[model_key]
        
        # Prepare Data for Evaluation (Using entire dataset for demo purposes, or split if preferred)
        # For a true evaluation, we should use a hold-out set. Here we'll split on the fly to show metrics.
        from sklearn.model_selection import train_test_split
        
        # Feature Extraction
        tfidf_vectorizer = artifacts['tfidf']
        scaler = artifacts['scaler']
        
        X_tfidf = pd.DataFrame(tfidf_vectorizer.transform(df['lemmatization']).toarray())
        X_length_scaled = pd.DataFrame(scaler.transform(df[['length']]), columns=['length_scaled'])
        X_final = pd.concat([X_tfidf, X_length_scaled], axis=1)
        Y = df['label_num']
        
        # Split (same seed as training to try and approximate validation set)
        _, X_val, _, Y_val = train_test_split(X_final, Y, test_size=0.3, random_state=42, stratify=Y)
        # Note: The original script split 70/15/15. This is just a rough approximation for visualization.
        
        st.info(f"Evaluating {model_option} on Validation Data...")
        
        Y_pred = None
        Y_pred_proba = None
        
        if model_key == 'cnn':
            tokenizer = artifacts['tokenizer']
            sequences = tokenizer.texts_to_sequences(df['lemmatization'])
            padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
            # Split padded data
            _, X_val_pad, _, Y_val = train_test_split(padded, Y, test_size=0.3, random_state=42, stratify=Y)
            
            Y_prob = model.predict(X_val_pad, verbose=0)
            Y_pred = (Y_prob > 0.5).astype("int32")
            Y_pred_proba = Y_prob
            
        elif model_key == 'isoforest':
            # IF predicts -1 for anomaly (spam) and 1 for normal (ham)
            if_pred = model.predict(X_val.values)
            Y_pred = [1 if x == -1 else 0 for x in if_pred]
            Y_pred_proba = -model.decision_function(X_val.values)
            
        else:
            Y_pred = model.predict(X_val.values)
            Y_pred_proba = model.predict_proba(X_val.values)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.write("Confusion Matrix")
            cm = confusion_matrix(Y_val, Y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("ROC Curve")
            fpr, tpr, _ = roc_curve(Y_val, Y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.legend()
            st.pyplot(fig)

        st.text("Classification Report")
        st.code(classification_report(Y_val, Y_pred))

    elif page == "Prediction":
        st.header("Spam Prediction")
        
        model_option = st.selectbox("Select Model", [
            "Random Forest", 
            "Logistic Regression", 
            "XGBoost", 
            "LightGBM", 
            "Isolation Forest (Anomaly Detection)",
            "CNN (Deep Learning)"
        ])
        
        model_map = {
            "Random Forest": "rf",
            "Logistic Regression": "lr",
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "Isolation Forest (Anomaly Detection)": "isoforest",
            "CNN (Deep Learning)": "cnn"
        }
        
        user_input = st.text_area("Enter message to classify:")
        
        if st.button("Predict"):
            if not user_input:
                st.error("Please enter some text.")
            else:
                # Preprocess Input
                processed_text = user_input.lower()
                processed_text = remove_punctuation_numbers(processed_text)
                processed_text = remove_stopwords(processed_text)
                processed_text = apply_lemmatization(processed_text)
                
                st.info(f"Processed Text: {processed_text}")
                
                model_key = model_map[model_option]
                model = artifacts[model_key]
                
                prediction = None
                probability = None
                
                if model_key == 'cnn':
                    tokenizer = artifacts['tokenizer']
                    seq = tokenizer.texts_to_sequences([processed_text])
                    pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
                    prob = model.predict(pad, verbose=0)[0][0]
                    prediction = 1 if prob > 0.5 else 0
                    probability = prob
                    
                else:
                    # Feature Extraction
                    tfidf_vectorizer = artifacts['tfidf']
                    scaler = artifacts['scaler']
                    
                    input_tfidf = tfidf_vectorizer.transform([processed_text])
                    input_len = len(user_input)
                    input_len_scaled = scaler.transform([[input_len]])
                    
                    input_final = pd.concat([
                        pd.DataFrame(input_tfidf.toarray()),
                        pd.DataFrame(input_len_scaled, columns=['length_scaled'])
                    ], axis=1)
                    
                    if model_key == 'isoforest':
                        pred_raw = model.predict(input_final)[0]
                        prediction = 1 if pred_raw == -1 else 0
                        probability = -model.decision_function(input_final)[0]
                    else:
                        prediction = model.predict(input_final)[0]
                        probability = model.predict_proba(input_final)[0][1]

                # Display Result
                st.markdown("### Result:")
                if prediction == 1:
                    st.error(f"🚨 SPAM Detected! (Confidence: {probability:.2f})")
                else:
                    st.success(f"✅ HAM (Not Spam). (Confidence: {1-probability if model_key != 'isoforest' else 'N/A'}:.2f)")

