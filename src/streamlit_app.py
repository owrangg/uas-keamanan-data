import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
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
        st.error(f"File not found at {data_path}. Please make sure it is in the data directory.")
        return None

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

st.title(" Spam Detection Dashboard")

df_raw = load_data()

if df_raw is not None:
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["EDA", "Model Training", "Prediction"])

    # Preprocessing
    with st.spinner('Preprocessing data...'):
        df = preprocess_dataframe(df_raw)

    # Prepare Features
    # We use session state to ensure we use the SAME vectorizer/scaler for prediction as we did for training
    if 'vectorizer' not in st.session_state:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        # Fit on the entire dataset for the vocabulary (mimicking original script)
        # Ideally we should fit only on Train, but following original logic:
        X_tfidf_sparse = tfidf_vectorizer.fit_transform(df['lemmatization'])
        st.session_state['vectorizer'] = tfidf_vectorizer
        st.session_state['X_tfidf_sparse'] = X_tfidf_sparse
    else:
        tfidf_vectorizer = st.session_state['vectorizer']
        X_tfidf_sparse = st.session_state['X_tfidf_sparse']

    X_tfidf = pd.DataFrame(X_tfidf_sparse.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    if 'scaler' not in st.session_state:
        scaler = StandardScaler()
        X_length_scaled = scaler.fit_transform(df[['length']])
        st.session_state['scaler'] = scaler
        st.session_state['X_length_scaled'] = X_length_scaled
    else:
        scaler = st.session_state['scaler']
        X_length_scaled = st.session_state['X_length_scaled']
    
    X_length_scaled_df = pd.DataFrame(X_length_scaled, columns=['length_scaled'])
    
    X_final = pd.concat([X_tfidf, X_length_scaled_df], axis=1)
    Y = df['label_num']

    # Splitting
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_final, Y, test_size=0.3, random_state=42, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

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

    elif page == "Model Training":
        st.header("Model Training & Evaluation")
        
        model_option = st.selectbox("Select Model", [
            "Random Forest", 
            "Logistic Regression", 
            "XGBoost", 
            "LightGBM", 
            "Isolation Forest (Anomaly Detection)",
            "CNN (Deep Learning)"
        ])

        if st.button("Train Model"):
            with st.spinner(f"Training {model_option}..."):
                
                if model_option == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_val)
                    Y_pred_proba = model.predict_proba(X_val)[:, 1]
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = 'sklearn'

                elif model_option == "Logistic Regression":
                    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_val)
                    Y_pred_proba = model.predict_proba(X_val)[:, 1]
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = 'sklearn'

                elif model_option == "XGBoost":
                    scale_pos_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_val)
                    Y_pred_proba = model.predict_proba(X_val)[:, 1]
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = 'sklearn'

                elif model_option == "LightGBM":
                    scale_pos_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]
                    model = LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_val)
                    Y_pred_proba = model.predict_proba(X_val)[:, 1]
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = 'sklearn'

                elif model_option == "Isolation Forest (Anomaly Detection)":
                    spam_ratio = Y.sum() / Y.shape[0]
                    model = IsolationForest(contamination=spam_ratio, random_state=42, n_jobs=-1)
                    model.fit(X_final) # IF is unsupervised usually, but here we use it on full data or train data
                    # For consistency with other flows, let's predict on Val set
                    # Note: IF returns -1 for anomaly (spam) and 1 for normal (ham)
                    if_pred = model.predict(X_val)
                    Y_pred = [1 if x == -1 else 0 for x in if_pred]
                    Y_pred_proba = -model.decision_function(X_val) # Score
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = 'isolation_forest'

                elif model_option == "CNN (Deep Learning)":
                    MAX_WORDS = 10000
                    MAX_LEN = 200
                    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
                    tokenizer.fit_on_texts(df['lemmatization'])
                    
                    sequences = tokenizer.texts_to_sequences(df['lemmatization'])
                    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
                    
                    X_train_pad = padded_sequences[X_train.index]
                    X_val_pad = padded_sequences[X_val.index]
                    
                    model = Sequential([
                        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
                        Conv1D(filters=128, kernel_size=5, activation='relu'),
                        GlobalMaxPooling1D(),
                        Dense(10, activation='relu'),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid')
                    ])
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    model.fit(X_train_pad, Y_train, epochs=5, batch_size=32, verbose=0) # Reduced epochs for demo
                    
                    Y_prob = model.predict(X_val_pad)
                    Y_pred = (Y_prob > 0.5).astype("int32")
                    Y_pred_proba = Y_prob
                    
                    st.session_state['trained_model'] = model
                    st.session_state['model_type'] = 'cnn'
                    st.session_state['tokenizer'] = tokenizer # Save tokenizer for prediction

                # Evaluation Metrics
                st.success(f"{model_option} Trained Successfully!")
                
                st.subheader("Evaluation Results (Validation Set)")
                
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
        
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' tab.")
        else:
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
                    
                    st.info(f"Processed Text: {processed_text}") # Debug info
                    
                    model = st.session_state['trained_model']
                    model_type = st.session_state['model_type']
                    
                    prediction = None
                    probability = None

                    if model_type == 'sklearn':
                        # Retrieve the EXACT vectorizer and scaler used for training
                        tfidf_vectorizer = st.session_state['vectorizer']
                        scaler = st.session_state['scaler']
                        
                        input_tfidf = tfidf_vectorizer.transform([processed_text])
                        input_len = len(user_input)
                        input_len_scaled = scaler.transform([[input_len]])
                        
                        input_final = pd.concat([
                            pd.DataFrame(input_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()),
                            pd.DataFrame(input_len_scaled, columns=['length_scaled'])
                        ], axis=1)
                        
                        prediction = model.predict(input_final)[0]
                        probability = model.predict_proba(input_final)[0][1]
                        
                    elif model_type == 'isolation_forest':
                        tfidf_vectorizer = st.session_state['vectorizer']
                        scaler = st.session_state['scaler']
                        
                        input_tfidf = tfidf_vectorizer.transform([processed_text])
                        input_len = len(user_input)
                        input_len_scaled = scaler.transform([[input_len]])
                        input_final = pd.concat([
                            pd.DataFrame(input_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()),
                            pd.DataFrame(input_len_scaled, columns=['length_scaled'])
                        ], axis=1)
                        
                        pred_raw = model.predict(input_final)[0]
                        prediction = 1 if pred_raw == -1 else 0
                        probability = -model.decision_function(input_final)[0]
                        
                    elif model_type == 'cnn':
                        tokenizer = st.session_state['tokenizer']
                        seq = tokenizer.texts_to_sequences([processed_text])
                        pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
                        prob = model.predict(pad)[0][0]
                        prediction = 1 if prob > 0.5 else 0
                        probability = prob

                    # Display Result
                    st.markdown("### Result:")
                    if prediction == 1:
                        st.error(f"🚨 SPAM Detected! (Confidence: {probability:.2f})")
                    else:
                        st.success(f"✅ HAM (Not Spam). (Confidence: {1-probability if model_type != 'isolation_forest' else 'N/A'}:.2f)")

