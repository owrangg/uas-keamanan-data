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
from sklearn.model_selection import train_test_split
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
        st.error(f"Error loading artifacts: {e}. Pastikan file model Anda ada.")
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


def perform_data_split(df, tfidf_vectorizer, scaler, tokenizer, maxlen=200):
    """Melakukan split data 70% train, 15% validation, 15% test dan pra-pemrosesan fitur."""
    
    Y = df['label_num']
    
    # Split 1: 70% Train, 30% Temp (Val + Test)
    # Gunakan df.index sebagai Y agar Y_train/Y_temp adalah index,
    # tetapi karena kita menggunakan stratify=Y, kita harus menggunakan label aslinya.
    # Namun, kita pastikan X_val_df dan Y_val diproses secara independen.
    X_train_df, X_temp_df, Y_train, Y_temp = train_test_split(
        df, Y, test_size=0.3, random_state=42, stratify=Y
    )
    
    # Split 2: 50% Validation, 50% Test (dari 30% Temp, jadi 15% Val, 15% Test)
    # Kita hanya membutuhkan X_val_df dan Y_val
    X_val_df, X_test_df, Y_val, Y_test = train_test_split(
        X_temp_df, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
    )
    
    # --- Validation Data Feature Extraction ---
    
    # 1. TFIDF Features - Hanya menggunakan X_val_df yang 15%
    X_val_tfidf_array = tfidf_vectorizer.transform(X_val_df['lemmatization']).toarray()
    X_val_tfidf = pd.DataFrame(X_val_tfidf_array)
    
    # 2. Length Scaled Feature - Hanya menggunakan X_val_df yang 15%
    X_val_length_scaled_array = scaler.transform(X_val_df[['length']])
    X_val_length_scaled = pd.DataFrame(X_val_length_scaled_array, columns=['length_scaled'])
    
    # 3. Concatenate (aman karena keduanya dibuat dari array)
    X_val_final = pd.concat([X_val_tfidf, X_val_length_scaled], axis=1)
    
    # 4. CNN data 
    sequences_val = tokenizer.texts_to_sequences(X_val_df['lemmatization'])
    X_val_pad = pad_sequences(sequences_val, maxlen=maxlen, padding='post', truncating='post')
    
    # 5. Y_val label (PASTIKAN index-nya di-reset agar cocok)
    # Gunakan .copy() untuk menjamin Y_val adalah objek baru dan konsisten.
    Y_val_clean = Y_val.copy().reset_index(drop=True)
    
    # --- Test Data Feature Extraction ---
    
    # 1. TFIDF Features - Test
    X_test_tfidf_array = tfidf_vectorizer.transform(X_test_df['lemmatization']).toarray()
    X_test_tfidf = pd.DataFrame(X_test_tfidf_array)
    
    # 2. Length Scaled Feature - Test
    X_test_length_scaled_array = scaler.transform(X_test_df[['length']])
    X_test_length_scaled = pd.DataFrame(X_test_length_scaled_array, columns=['length_scaled'])
    
    # 3. Concatenate - Test
    X_test_final = pd.concat([X_test_tfidf, X_test_length_scaled], axis=1)
    
    # 4. CNN data - Test
    sequences_test = tokenizer.texts_to_sequences(X_test_df['lemmatization'])
    X_test_pad = pad_sequences(sequences_test, maxlen=maxlen, padding='post', truncating='post')
    
    # 5. Y_test label
    Y_test_clean = Y_test.copy().reset_index(drop=True)
    
    return X_val_final, Y_val_clean, X_val_pad, X_test_final, Y_test_clean, X_test_pad
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
        
    # Data Split & Feature Extraction (Cached)
    # Ini harus menghasilkan sekitar 836 baris (15% dari 5572) untuk Val dan Test
    X_val_final, Y_val, X_val_pad, X_test_final, Y_test, X_test_pad = perform_data_split(
        df, artifacts['tfidf'], artifacts['scaler'], artifacts['tokenizer']
    )

    if page == "EDA":
        # ... (Kode EDA Anda tetap sama) ...
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
        
        # ---
        
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
        
        st.info(f"Evaluating **{model_option}** on **Validation Data (15%, {len(Y_val)} samples)**.")
        
        Y_pred = None
        Y_pred_proba = None
        
        if model_key == 'cnn':
            # Menggunakan data yang sudah dipad
            X_eval = X_val_pad
            Y_prob = model.predict(X_eval, verbose=0)
            # Pastikan output prediksi 1D
            Y_pred = (Y_prob > 0.5).astype("int32").flatten()
            Y_pred_proba = Y_prob.flatten()
            
        elif model_key == 'isoforest':
            # Menggunakan data TFIDF + Length
            X_eval = X_val_final
            if_pred = model.predict(X_eval.values)
            # IF predicts -1 (spam) and 1 (ham) -> convert to 1 (spam) and 0 (ham)
            Y_pred = [1 if x == -1 else 0 for x in if_pred]
            Y_pred_proba = -model.decision_function(X_eval.values) # Jarak ke anomali, lebih besar = lebih mungkin spam
            
        else:
            # Menggunakan data TFIDF + Length
            X_eval = X_val_final
            Y_pred = model.predict(X_eval.values)
            Y_pred_proba = model.predict_proba(X_eval.values)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.write("Confusion Matrix")
            cm = confusion_matrix(Y_val, Y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)
        
        with col2:
            st.write("ROC Curve")
            # Hanya tampilkan ROC jika model memiliki fungsi predict_proba standar
            if model_key != 'isoforest':
                fpr, tpr, _ = roc_curve(Y_val, Y_pred_proba)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
                ax.plot([0, 1], [0, 1], 'k--', label='Chance')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                 st.info("ROC Curve for Isolation Forest is typically calculated differently (e.g., based on decision function/anomaly score).")

        st.text("Classification Report")
        st.code(classification_report(Y_val, Y_pred, target_names=['ham', 'spam']))

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
                    # Feature Extraction (TFIDF + Length)
                    tfidf_vectorizer = artifacts['tfidf']
                    scaler = artifacts['scaler']
                    
                    input_tfidf = tfidf_vectorizer.transform([processed_text])
                    input_len = len(user_input)
                    # NOTE: scaler.transform expects a 2D array
                    input_len_scaled = scaler.transform([[input_len]])
                    
                    input_final = pd.concat([
                        pd.DataFrame(input_tfidf.toarray()),
                        pd.DataFrame(input_len_scaled, columns=['length_scaled'])
                    ], axis=1)
                    
                    if model_key == 'isoforest':
                        pred_raw = model.predict(input_final.values)[0]
                        prediction = 1 if pred_raw == -1 else 0
                        probability = -model.decision_function(input_final.values)[0] # Anomaly score
                    else:
                        prediction = model.predict(input_final.values)[0]
                        probability = model.predict_proba(input_final.values)[0][1]

                # Display Result
                st.markdown("### Result:")
                if prediction == 1:
                    st.error(f"🚨 **SPAM Detected!** (Confidence: {probability:.2f})")
                else:
                    # Confidence untuk HAM
                    confidence_display = f"Confidence: {1 - probability:.2f}" if model_key != 'isoforest' else 'Anomaly Score: N/A'
                    st.success(f"✅ **HAM (Not Spam).** ({confidence_display})")