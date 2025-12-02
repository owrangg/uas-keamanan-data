import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Install/download NLTK resources (hanya perlu sekali)
# try:
#    nltk.data.find('corpora/stopwords')
#except nltk.downloader.DownloadError:
#    nltk.download('stopwords')
#try:
#    nltk.data.find('corpora/wordnet')
#except nltk.downloader.DownloadError:
#    nltk.download('wordnet')

# --- Fungsi Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_punctuation_numbers(text):
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def apply_lemmatization(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation_numbers(text)
    text = remove_stopwords(text)
    text = apply_lemmatization(text)
    return text

# --- Konfigurasi Keras/TensorFlow (untuk CNN) ---
# Menggunakan dummy Keras/TF, karena model Keras tidak dapat
# diserialisasi dengan @st.cache_data secara mudah di lingkungan Streamlit
# Untuk deployment nyata, model harus dilatih di luar cache dan dimuat
# menggunakan tf.keras.models.load_model('model_cnn.h5')
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    KERAS_AVAILABLE = True
except ImportError:
    st.warning("TensorFlow/Keras tidak ditemukan. Model CNN akan dilewati.")
    KERAS_AVAILABLE = False

# Parameter CNN
MAX_WORDS = 10000
MAX_LEN = 200

# --- Fungsi Utama untuk Memuat dan Melatih/Mempersiapkan Data dan Model ---
@st.cache_data
def load_and_prepare_data():
    """Memuat data, melakukan preprocessing, feature engineering, dan splitting."""
    
    # Ganti dengan path file yang sesuai jika bukan 'spam.csv'
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'spam.csv')
        df = pd.read_csv(data_path, encoding='latin-1')
    except FileNotFoundError:
        st.error("File 'spam.csv' tidak ditemukan. Pastikan file ada di direktori data.")
        return None, None, None, None, None, None, None

    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['length'] = df['message'].apply(len)
    
    # Preprocessing
    df['lemmatization'] = df['message'].apply(preprocess_text)
    
    # Feature Extraction (TF-IDF + Scaling Length)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf_sparse = tfidf_vectorizer.fit_transform(df['lemmatization'])
    X_tfidf = pd.DataFrame(X_tfidf_sparse.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    scaler = StandardScaler()
    X_length_scaled = scaler.fit_transform(df[['length']])
    X_length_scaled_df = pd.DataFrame(X_length_scaled, columns=['length_scaled'])
    
    X_final = pd.concat([X_tfidf, X_length_scaled_df], axis=1)
    Y = df['label_num']
    
    # Splitting dataset (70% Train, 15% Val, 15% Test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_final, Y, test_size=0.3, random_state=42, stratify=Y
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
    )
    
    # Data untuk CNN (Membutuhkan Tokenizer/Padding)
    if KERAS_AVAILABLE:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
        tokenizer.fit_on_texts(df['lemmatization'])
        sequences = tokenizer.texts_to_sequences(df['lemmatization'])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        X_train_pad = padded_sequences[X_train.index]
        X_val_pad = padded_sequences[X_val.index]
        X_test_pad = padded_sequences[X_test.index]
        
        return X_train, X_val, Y_train, Y_val, X_train_pad, X_val_pad, df
    
    return X_train, X_val, Y_train, Y_val, None, None, df

@st.cache_resource
def train_models(X_train, X_val, Y_train, Y_val, df):
    """Melatih semua model dan mengembalikan prediksi, probabilitas, dan skor."""
    
    scale_pos_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        'LightGBM': LGBMClassifier(objective='binary', n_estimators=100, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    # Train Traditional Models
    for name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val)
        Y_proba = model.predict_proba(X_val)[:, 1]
        
        results[name] = {
            'model': model,
            'Y_pred': Y_pred,
            'Y_proba': Y_proba,
            'report': classification_report(Y_val, Y_pred, target_names=['Ham (0)', 'Spam (1)'], output_dict=True)
        }
        
    # Isolation Forest (Anomaly Detection, trained on ALL data for anomaly score)
    spam_ratio = df['label_num'].sum() / df.shape[0]
    if_model = IsolationForest(
        n_estimators=100, contamination=spam_ratio, random_state=42, n_jobs=-1
    )
    if_model.fit(df.drop(columns=['label', 'message', 'label_num', 'length', 'lemmatization'], errors='ignore'))
    
    # Prediksi untuk IF adalah anomaly score, bukan proba.
    # Skor anomali *negatif* digunakan sebagai "skor spam" (lebih tinggi = lebih spam)
    Y_pred_if_score = -if_model.decision_function(X_val)
    Y_pred_if = if_model.predict(X_val)
    # Konversi IF prediction (1=normal, -1=anomaly) ke binary (0=Ham, 1=Spam)
    Y_pred_if_binary = np.where(Y_pred_if == 1, 0, 1)
    
    results['Isolation Forest'] = {
        'model': if_model,
        'Y_pred': Y_pred_if_binary,
        'Y_proba': Y_pred_if_score, # Menggunakan skor anomali negatif sebagai "proba" untuk ROC
        'report': classification_report(Y_val, Y_pred_if_binary, target_names=['Ham (0)', 'Spam (1)'], output_dict=True)
    }

    return results

@st.cache_resource
def train_cnn_model(X_train_pad, X_val_pad, Y_train, Y_val):
    """Melatih model CNN (jika Keras tersedia)."""
    if not KERAS_AVAILABLE:
        return None
        
    # Parameter CNN
    EMBEDDING_DIM = 128
    
    cnn_model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Pelatihan dengan jumlah epoch yang lebih sedikit untuk Streamlit
    history = cnn_model.fit(
        X_train_pad, Y_train,
        epochs=5,  # Mengurangi dari 20 menjadi 5 untuk kecepatan
        batch_size=32,
        validation_data=(X_val_pad, Y_val),
        verbose=0
    )
    
    Y_prob_cnn = cnn_model.predict(X_val_pad, verbose=0)
    Y_pred_cnn = (Y_prob_cnn > 0.5).astype("int32").flatten()
    
    return {
        'model': cnn_model,
        'Y_pred': Y_pred_cnn,
        'Y_proba': Y_prob_cnn.flatten(),
        'report': classification_report(Y_val, Y_pred_cnn, target_names=['Ham (0)', 'Spam (1)'], output_dict=True)
    }


# --- Streamlit UI ---

st.set_page_config(
    page_title="Perbandingan Model Spam/Ham SMS",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📧 Perbandingan Model Klasifikasi Spam SMS")
st.markdown("Aplikasi ini membandingkan kinerja **Random Forest, Logistic Regression, XGBoost, LightGBM, Isolation Forest, dan CNN** dalam mendeteksi spam SMS (menggunakan Validation Set).")
st.divider()

# 1. Memuat Data dan Melatih Model
X_train, X_val, Y_train, Y_val, X_train_pad, X_val_pad, df = load_and_prepare_data()

if df is not None:
    results = train_models(X_train, X_val, Y_train, Y_val, df)
    
    if KERAS_AVAILABLE and X_val_pad is not None:
        with st.spinner("Melatih Model CNN..."):
            cnn_result = train_cnn_model(X_train_pad, X_val_pad, Y_train, Y_val)
            results['CNN'] = cnn_result

    
    # 2. Ringkasan Performa
    st.header("📊 Ringkasan Performa Model (Validation Set)")

    metrics_list = []
    
    for name, res in results.items():
        report = res['report']
        roc_auc = roc_auc_score(Y_val, res['Y_proba'])

        metrics_list.append({
            'Model': name,
            'Accuracy': report['accuracy'],
            'Ham Precision': report['Ham (0)']['precision'],
            'Spam Recall': report['Spam (1)']['recall'],
            'Spam F1-Score': report['Spam (1)']['f1-score'],
            'AUC Score': roc_auc
        })

    df_metrics = pd.DataFrame(metrics_list).set_index('Model').sort_values(by='Spam F1-Score', ascending=False)
    
    # Highlighting the best score for each column
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    st.dataframe(
        df_metrics.style.format(
            "{:.4f}"
        ).apply(
            highlight_max, subset=pd.IndexSlice[:, ['Accuracy', 'Spam Recall', 'Spam F1-Score', 'AUC Score']]
        ),
        use_container_width=True
    )
    
    st.caption("*F1-Score untuk 'Spam' (Kelas 1) seringkali merupakan metrik terbaik dalam kasus ketidakseimbangan kelas untuk mengukur keseimbangan Presisi dan Recall dalam mendeteksi spam.*")
    st.divider()

    # 3. Visualisasi (Confusion Matrix & ROC)
    st.header("📈 Detail Visualisasi per Model")
    
    col_select, col_chart = st.columns([1, 2])

    with col_select:
        model_name = st.selectbox(
            "Pilih Model untuk Detail Visualisasi:",
            list(results.keys()),
            key='model_selector'
        )

    # Dapatkan hasil yang dipilih
    selected_result = results[model_name]
    Y_pred = selected_result['Y_pred']
    Y_proba = selected_result['Y_proba']
    cm = confusion_matrix(Y_val, Y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    # Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham (0)', 'Spam (1)'],
        yticklabels=['Ham (0)', 'Spam (1)'],
        ax=ax_cm
    )
    ax_cm.set_title(f'Confusion Matrix: {model_name}')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    
    ax_cm.text(
        0.5,
        -0.15,
        f'TN={TN}, FP={FP}\nFN={FN}, TP={TP}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax_cm.transAxes,
        fontsize=10,
        color='red'
    )
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(Y_val, Y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'Kurva ROC (AUC = {roc_auc:.4f})'
    )
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Performance')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR) / Recall')
    ax_roc.set_title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)

    
    with col_chart:
        tab_cm, tab_roc, tab_report = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report"])

        with tab_cm:
            st.pyplot(fig_cm)
            st.markdown(f"""
            * **False Positives (FP):** {FP} (Ham diklasifikasikan sebagai Spam)
            * **False Negatives (FN):** {FN} (Spam diklasifikasikan sebagai Ham)
            """)

        with tab_roc:
            st.pyplot(fig_roc)
            st.markdown(f"**Area Under the Curve (AUC):** **{roc_auc:.4f}**")
            
        with tab_report:
            st.code(
                classification_report(Y_val, Y_pred, target_names=['Ham (0)', 'Spam (1)']),
                language='text'
            )

    st.divider()
    
    # 4. Fitur Penting
    st.header("🔑 Fitur Paling Penting (Top 10)")
    
    # Fungsi untuk mendapatkan Feature Importance (disesuaikan per model)
    def get_feature_importance(model_name, model, X_train, df):
        if model_name == 'Random Forest':
            importances = model.feature_importances_
            feature_names = X_train.columns
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df_imp = df_imp.sort_values(by='importance', ascending=False).head(10)
            return df_imp
        
        elif model_name == 'Logistic Regression':
            coefficients = model.coef_[0]
            importance_abs = np.abs(coefficients)
            feature_names = X_train.columns
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance_abs})
            df_imp = df_imp.sort_values(by='importance', ascending=False).head(10)
            return df_imp
            
        elif model_name in ['XGBoost', 'LightGBM']:
            importances = model.feature_importances_
            feature_names = X_train.columns
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df_imp = df_imp.sort_values(by='importance', ascending=False).head(10)
            return df_imp

        elif model_name == 'Isolation Forest':
            # Menggunakan korelasi absolut dengan anomaly score (seperti di skrip asli)
            df_corr = X_val.copy()
            df_corr['anomaly_score'] = df['anomaly_score'] # anomaly_score diambil dari data frame asli
            correlation_with_score = df_corr.corr()['anomaly_score'].drop('anomaly_score', errors='ignore')
            importance_abs_corr = np.abs(correlation_with_score)
            
            feature_names = importance_abs_corr.index
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance_abs_corr})
            df_imp = df_imp.sort_values(by='importance', ascending=False).head(10)
            return df_imp

        # CNN Feature Importance (Tidak bisa dilakukan dengan mudah, lewati)
        return None

    
    imp_data = get_feature_importance(model_name, selected_result['model'], X_train, df)
    
    if imp_data is not None:
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x='importance',
            y='feature',
            data=imp_data,
            color='skyblue',
            ax=ax_imp
        )
        ax_imp.set_title(f'Feature Importance: {model_name}')
        ax_imp.set_xlabel('Importance Score')
        ax_imp.set_ylabel('Feature (Keyword/Length)')
        st.pyplot(fig_imp)
        st.dataframe(imp_data, use_container_width=True)
    else:
        st.info("Fitur Penting untuk model ini tidak mudah dihitung/divisualisasikan dalam konteks Streamlit.")


# --- Spam Detector Interaktif (Contoh Penggunaan) ---
st.divider()
st.header("🔬 Coba Deteksi Spam Interaktif")

test_message = st.text_area(
    "Masukkan pesan SMS untuk diuji:",
    "WINNER! U R awarded a Prize of $1000 cash! Call 09061744154 now or send STOP to 87239"
)

if st.button("Deteksi Spam"):
    if not test_message:
        st.warning("Mohon masukkan teks pesan.")
    else:
        with st.spinner("Memproses..."):
            # 1. Preprocessing
            processed_text = preprocess_text(test_message)

            # 2. Feature Extraction (TF-IDF + Length)
            test_length = len(test_message)
            
            # Re-initialize vectorizer/scaler (TIDAK IDEAL, HARUSNYA DISIMPAN)
            # Karena di-cache, ini seharusnya menggunakan yang sudah fit dari load_and_prepare_data
            tfidf_vectorizer_temp = TfidfVectorizer(max_features=5000)
            tfidf_vectorizer_temp.fit_transform(df['lemmatization']) # Re-fit hanya untuk mendapatkan object
            
            scaler_temp = StandardScaler()
            scaler_temp.fit(df[['length']]) # Re-fit hanya untuk mendapatkan object
            
            
            test_tfidf_sparse = tfidf_vectorizer_temp.transform([processed_text])
            test_tfidf = pd.DataFrame(test_tfidf_sparse.toarray(), columns=tfidf_vectorizer_temp.get_feature_names_out())
            
            test_length_scaled = scaler_temp.transform([[test_length]])
            test_length_scaled_df = pd.DataFrame(test_length_scaled, columns=['length_scaled'])
            
            # Pastikan fitur TF-IDF memiliki kolom yang sama dengan X_train/X_val
            missing_cols = set(X_train.columns) - set(test_tfidf.columns)
            for c in missing_cols:
                test_tfidf[c] = 0
            
            X_test_final = pd.concat([test_tfidf[X_train.columns.drop('length_scaled')], test_length_scaled_df], axis=1)

            
            st.subheader("Hasil Prediksi")
            
            for name, res in results.items():
                model = res['model']
                
                # Handling untuk CNN
                if name == 'CNN' and KERAS_AVAILABLE:
                    # Tokenizer/Padding untuk CNN
                    tokenizer_cnn = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
                    tokenizer_cnn.fit_on_texts(df['lemmatization'])
                    
                    test_sequence = tokenizer_cnn.texts_to_sequences([processed_text])
                    test_pad = pad_sequences(test_sequence, maxlen=MAX_LEN, padding='post', truncating='post')
                    
                    proba = model.predict(test_pad, verbose=0)[0][0]
                    prediction = 'SPAM' if proba > 0.5 else 'HAM'
                    st.info(f"**{name}:** **{prediction}** (Probabilitas Spam: **{proba:.4f}**)")
                    continue

                # Isolation Forest menggunakan decision_function
                if name == 'Isolation Forest':
                    score = model.decision_function(X_test_final)[0]
                    prediction = 'SPAM' if score < 0 else 'HAM' # score < 0 = Anomaly/Spam
                    st.info(f"**{name}:** **{prediction}** (Anomaly Score: **{score:.4f}**)")
                    continue
                
                # Model Klasifikasi Biner Lainnya
                proba = model.predict_proba(X_test_final)[0][1]
                prediction = 'SPAM' if proba > 0.5 else 'HAM'
                st.info(f"**{name}:** **{prediction}** (Probabilitas Spam: **{proba:.4f}**)")