import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest # IF Added
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tensorflow as tf # TensorFlow Added
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONSTANTS for CNN (from user's script) ---
MAX_WORDS = 10000
EMBEDDING_DIM = 128
MAX_LEN = 200

# --- 1. SETUP & LOAD DATA ---

@st.cache_data
def load_data_and_setup():
    """Memuat data dan melakukan setup NLTK."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        st.error(f"Gagal mengunduh data NLTK: {e}")

    # Memuat Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'spam.csv')
    df = pd.read_csv(data_path, encoding='latin-1')
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df['length'] = df['message'].apply(len)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

# --- 2. PREPROCESSING FUNCTIONS ---

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Mengaplikasikan seluruh langkah preprocessing (Case Folding, Punctuation/Number Removal, Stop Word Removal, Lemmatization)."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# --- 3. TRAINING DAN FEATURE ENGINEERING (cached) ---

# Menggunakan st.cache_resource agar pelatihan model hanya dilakukan sekali
@st.cache_resource
def train_all_components(df):
    """Melakukan preprocessing seluruh data dan melatih SEMUA model."""
    df['lemmatization'] = df['message'].apply(preprocess_text)
    
    # Feature Engineering (TF-IDF + Length Scaling) untuk Model ML Klasik
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf_sparse = tfidf_vectorizer.fit_transform(df['lemmatization'])
    X_tfidf = pd.DataFrame(X_tfidf_sparse.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    scaler = StandardScaler()
    X_length_scaled = scaler.fit_transform(df[['length']])
    X_length_scaled_df = pd.DataFrame(X_length_scaled, columns=['length_scaled'])

    X_final = pd.concat([X_tfidf, X_length_scaled_df], axis=1)
    Y = df['label_num']

    # Splitting data (untuk evaluasi 70/15/15)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_final, Y, test_size=0.3, random_state=42, stratify=Y
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
    )
    
    models = {}
    
    # 1. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, Y_train)
    models['Random Forest'] = rf_model

    # 2. Logistic Regression
    lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    lr_model.fit(X_train, Y_train)
    models['Logistic Regression'] = lr_model

    # 3. XGBoost
    scale_pos_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]
    xgb_model = XGBClassifier(
        objective='binary:logistic', n_estimators=100, learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, 
        eval_metric='logloss', n_jobs=-1
    )
    xgb_model.fit(X_train, Y_train)
    models['XGBoost'] = xgb_model

    # 4. LightGBM
    scale_pos_weight_lgbm = Y_train.value_counts()[0] / Y_train.value_counts()[1]
    lgbm_model = LGBMClassifier(
        objective='binary', n_estimators=100, learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight_lgbm, random_state=42, n_jobs=-1
    )
    lgbm_model.fit(X_train, Y_train)
    models['LightGBM'] = lgbm_model

    # 5. Isolation Forest (IF) - Dilatih pada SELURUH data (X_final)
    spam_ratio = Y.sum() / Y.shape[0]
    if_model = IsolationForest(n_estimators=100, contamination=spam_ratio, random_state=42, n_jobs=-1)
    if_model.fit(X_final)
    models['Isolation Forest'] = if_model

    # --- 6. CNN SETUP dan PELATIHAN ---
    cnn_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    cnn_tokenizer.fit_on_texts(df['lemmatization'])

    sequences = cnn_tokenizer.texts_to_sequences(df['lemmatization'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    X_train_pad = padded_sequences[X_train.index]
    X_val_pad = padded_sequences[X_val.index]

    cnn_model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Pelatihan CNN (menggunakan 10 epochs, bukan 20, untuk kecepatan caching awal)
    cnn_model.fit(
        X_train_pad, Y_train,
        epochs=10, 
        batch_size=32,
        validation_data=(X_val_pad, Y_val),
        verbose=0 # Sembunyikan output pelatihan
    )
    models['CNN'] = cnn_model

    return models, tfidf_vectorizer, scaler, cnn_tokenizer, X_val, Y_val, X_val_pad, X_final, Y, df

# --- 4. PREDICTION FUNCTIONS ---

def predict_message(message, model, vectorizer, scaler):
    """Prediksi untuk model ML Klasik (RF, LR, XGB, LGBM)."""
    # 1. Preprocessing Teks
    processed_text = preprocess_text(message)
    
    # 2. Feature Engineering
    X_tfidf_sparse = vectorizer.transform([processed_text])
    X_tfidf_df = pd.DataFrame(X_tfidf_sparse.toarray(), columns=vectorizer.get_feature_names_out())
    message_length = len(message)
    X_length_scaled = scaler.transform(np.array([[message_length]]))
    X_length_scaled_df = pd.DataFrame(X_length_scaled, columns=['length_scaled'])
    X_final_input = pd.concat([X_tfidf_df, X_length_scaled_df], axis=1)
    
    # 3. Prediksi
    prediction = model.predict(X_final_input)[0]
    prediction_proba = model.predict_proba(X_final_input)[0][1] # Probabilitas Spam (kelas 1)
    
    return prediction, prediction_proba, processed_text

def predict_message_cnn(message, model, cnn_tokenizer):
    """Prediksi untuk model CNN."""
    processed_text = preprocess_text(message)
    sequences = cnn_tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict probability (Keras/TensorFlow)
    probability = model.predict(padded, verbose=0)[0][0]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability, processed_text

def predict_message_if(message, model, vectorizer, scaler):
    """Prediksi untuk model Isolation Forest."""
    # 1. Preprocessing & Feature Engineering (sama seperti ML Klasik)
    processed_text = preprocess_text(message)
    X_tfidf_sparse = vectorizer.transform([processed_text])
    X_tfidf_df = pd.DataFrame(X_tfidf_sparse.toarray(), columns=vectorizer.get_feature_names_out())
    message_length = len(message)
    X_length_scaled = scaler.transform(np.array([[message_length]]))
    X_length_scaled_df = pd.DataFrame(X_length_scaled, columns=['length_scaled'])
    X_final_input = pd.concat([X_tfidf_df, X_length_scaled_df], axis=1)
    
    # 2. Prediction
    # IF returns -1 (anomaly/spam) or 1 (normal/ham)
    prediction_if = model.predict(X_final_input)[0]
    prediction = 1 if prediction_if == -1 else 0
    
    # Anomaly Score: Lebih rendah = lebih anomali (risiko spam lebih tinggi)
    anomaly_score = model.decision_function(X_final_input)[0]
    
    return prediction, anomaly_score, processed_text

# --- 5. VISUALISASI STREAMLIT (Tetap sama) ---

# ... (plot_wordcloud, plot_confusion_matrix, plot_roc_curve) - Menggunakan fungsi yang sama dari respons sebelumnya.

def plot_wordcloud(text, title, colormap):
    """Menampilkan Word Cloud."""
    wc = WordCloud(width=800, height=500, background_color='white', max_words=100, colormap=colormap).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

def plot_confusion_matrix(cm, model_name):
    """Menampilkan Confusion Matrix."""
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham (0)', 'Spam (1)'],
        yticklabels=['Ham (0)', 'Spam (1)']
    )
    ax.set_title(f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Prediksi Label')
    ax.set_ylabel('Aktual Label')

    TN, FP, FN, TP = cm.ravel()
    plt.text(
        0.5,
        -0.15,
        f'TN={TN}, FP={FP}\nFN={FN}, TP={TP}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=10,
        color='red'
    )
    st.pyplot(plt)

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """Menampilkan Kurva ROC."""
    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'Kurva ROC {model_name} (AUC = {roc_auc:.4f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Kinerja Acak')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title(f'Kurva ROC: {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(
        page_title="Aplikasi Deteksi SPAM SMS",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📧 Aplikasi Deteksi SPAM SMS (Keamanan Data)")
    st.markdown("Aplikasi interaktif untuk analisis dan prediksi *spam* menggunakan model *Machine Learning* dan *Deep Learning*.")
    st.warning("PENTING: Model dilatih saat aplikasi dijalankan pertama kali. Tunggu beberapa saat untuk memuat semua model, termasuk CNN dan Isolation Forest.")

    # Memuat data & melatih model
    df = load_data_and_setup()
    models, vectorizer, scaler, cnn_tokenizer, X_val, Y_val, X_val_pad, X_final, Y_full, df_full = train_all_components(df)
    
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio(
        "Pilih Halaman",
        ["Dashboard EDA", "Prediksi Real-Time", "Evaluasi Model"]
    )

    if menu == "Dashboard EDA":
        # ... (Bagian EDA tetap sama)
        st.header("📊 Dashboard Exploratory Data Analysis (EDA)")
        st.subheader("Distribusi Kelas ('Ham' vs 'Spam')")
        label_counts = df['label'].value_counts()
        persentase = (label_counts / len(df)) * 100
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(pd.DataFrame({
                'Jumlah': label_counts.values,
                'Persentase': [f'{p:.2f}%' for p in persentase.values]
            }, index=label_counts.index))
            st.warning(f"Data mengalami **Class Imbalance** dengan pesan Spam hanya **{persentase['spam']:.2f}%**.")

        with col2:
            plt.figure(figsize=(8, 6))
            sns.barplot(x=label_counts.index, y=label_counts.values, palette=['skyblue', 'pink'])
            plt.title('Distribusi Kelas "ham" vs "spam"')
            plt.xlabel('Label Pesan')
            plt.ylabel('Jumlah Pesan')
            st.pyplot(plt)

        st.markdown("---")

        st.subheader("Analisis Panjang Pesan (Message Length)")
        st.text("Statistik Deskriptif:")
        st.dataframe(df.groupby('label')['length'].describe())
        
        plt.figure(figsize=(12, 5))
        df[df['label'] == 'ham']['length'].hist(bins=50, alpha=0.7, label='Ham', color='skyblue')
        df[df['label'] == 'spam']['length'].hist(bins=50, alpha=0.7, label='Spam', color='salmon')
        plt.legend()
        plt.title('Perbandingan Distribusi Panjang Pesan (Ham vs. Spam)')
        plt.xlabel('Panjang Pesan (Karakter)')
        plt.ylabel('Frekuensi')
        st.pyplot(plt)
        
        st.markdown("---")

        st.subheader("Visualisasi Kata Kunci (Word Cloud)")
        spam_words = ' '.join(list(df[df['label'] == 'spam']['message']))
        ham_words = ' '.join(list(df[df['label'] == 'ham']['message']))

        col_spam, col_ham = st.columns(2)
        with col_spam:
            plot_wordcloud(spam_words, 'Word Cloud: Pesan SPAM', 'Reds')
        with col_ham:
            plot_wordcloud(ham_words, 'Word Cloud: Pesan HAM', 'Blues')

    elif menu == "Prediksi Real-Time":
        # --- Bagian 2: PREDIKSI REAL-TIME ---
        st.header("🎯 Prediksi Deteksi SPAM Real-Time")
        
        # Pilihan Model
        model_name = st.selectbox(
            "Pilih Model untuk Prediksi:",
            list(models.keys())
        )
        selected_model = models[model_name]
        
        # Input Teks Pesan (Dikosongkan sesuai permintaan)
        user_input = st.text_area("Masukkan Pesan SMS untuk Diperiksa:", 
                                  value="", 
                                  height=150)

        if st.button("Prediksi"):
            if user_input:
                st.info(f"Menggunakan Model: **{model_name}**")
                
                # Logic untuk 6 model yang berbeda
                if model_name in ['Random Forest', 'Logistic Regression', 'XGBoost', 'LightGBM']:
                    prediction, probability, processed_text = predict_message(user_input, selected_model, vectorizer, scaler)
                    proba_display = f"{probability:.4f}"
                    proba_delta = f"{(probability * 100):.2f}%"
                    proba_label = "Probabilitas SPAM"
                    result_text = f"**{model_name}**"

                elif model_name == 'CNN':
                    prediction, probability, processed_text = predict_message_cnn(user_input, selected_model, cnn_tokenizer)
                    proba_display = f"{probability:.4f}"
                    proba_delta = f"{(probability * 100):.2f}%"
                    proba_label = "Probabilitas SPAM"
                    result_text = f"**{model_name}**"

                elif model_name == 'Isolation Forest':
                    prediction, anomaly_score, processed_text = predict_message_if(user_input, selected_model, vectorizer, scaler)
                    
                    # Tampilan untuk Isolation Forest
                    proba_display = f"{anomaly_score:.4f}"
                    proba_delta = "N/A"
                    proba_label = "Anomaly Score (Lebih rendah = Lebih Anomali)"
                    result_text = f"**{model_name}** (Anomaly Detection)"

                col_res, col_score = st.columns(2)
                
                with col_res:
                    if prediction == 1:
                        st.error(f"🚨 HASIL PREDIKSI {result_text}: **SPAM**")
                        st.balloons()
                    else:
                        st.success(f"✅ HASIL PREDIKSI {result_text}: **HAM** (Bukan Spam)")

                with col_score:
                    st.metric(
                        label=proba_label,
                        value=proba_display,
                        delta=proba_delta if model_name != 'Isolation Forest' else None # Tidak ada delta untuk Anomaly Score
                    )
                
                st.markdown("---")
                st.subheader("Detail Pesan dan Preprocessing")
                st.text(f"Panjang Pesan: {len(user_input)} karakter")
                st.text(f"Pesan Setelah Preprocessing: '{processed_text}'")
                
            else:
                st.warning("Mohon masukkan pesan teks untuk memulai prediksi.")

    elif menu == "Evaluasi Model":
        # --- Bagian 3: EVALUASI MODEL ---
        st.header("📈 Evaluasi Kinerja Model (Validation Set)")

        model_name_eval = st.selectbox(
            "Pilih Model untuk Evaluasi:",
            list(models.keys())
        )
        selected_model_eval = models[model_name_eval]
        
        # --- Logic Evaluasi berdasarkan Tipe Model ---
        
        if model_name_eval == 'Isolation Forest':
            st.warning("Isolation Forest dievaluasi menggunakan **Full Dataset** (Bukan Validation Set) karena ini adalah model Deteksi Anomali yang dilatih pada seluruh data (Unsupervised).")
            # Prediksi pada Full Data
            Y_pred_if = selected_model_eval.predict(X_final)
            Y_pred_if_binary = np.where(Y_pred_if == -1, 1, 0) # Mapping -1(Spam)/1(Ham) ke 1/0
            Y_val_eval = Y_full
            Y_pred_eval = Y_pred_if_binary
            Y_pred_proba_eval = -selected_model_eval.decision_function(X_final) # Anomaly score sebagai proxy proba
            
        elif model_name_eval == 'CNN':
            # Prediksi pada Validation Set (Padded)
            Y_pred_proba_cnn = selected_model_eval.predict(X_val_pad, verbose=0)
            Y_pred_cnn = (Y_pred_proba_cnn > 0.5).astype("int32")
            Y_val_eval = Y_val
            Y_pred_eval = Y_pred_cnn
            Y_pred_proba_eval = Y_pred_proba_cnn
            
        else: # ML Klasik (RF, LR, XGB, LGBM)
            # Prediksi pada Validation Set (TF-IDF + Length)
            Y_pred_eval = selected_model_eval.predict(X_val)
            Y_pred_proba_eval = selected_model_eval.predict_proba(X_val)[:, 1]
            Y_val_eval = Y_val
        
        # --- Perhitungan Metrik ---
        cm = confusion_matrix(Y_val_eval, Y_pred_eval)
        cr = classification_report(Y_val_eval, Y_pred_eval, target_names=['Ham (0)', 'Spam (1)'], output_dict=True, zero_division=0)
        
        # ROC AUC hanya untuk model yang memiliki probabilitas (atau score yang bisa diinterpretasikan)
        if model_name_eval != 'Isolation Forest' or (model_name_eval == 'Isolation Forest'):
             roc_auc = roc_auc_score(Y_val_eval, Y_pred_proba_eval)
        
        st.subheader(f"Metrik Kinerja: {model_name_eval}")
        
        # Menampilkan Metrik Utama
        col_acc, col_prec, col_recall, col_f1, col_auc = st.columns(5)
        
        with col_acc: st.metric("Accuracy", f"{cr['accuracy']:.4f}")
        with col_prec: st.metric("Precision (Spam)", f"{cr['Spam (1)']['precision']:.4f}")
        with col_recall: st.metric("Recall (Spam)", f"{cr['Spam (1)']['recall']:.4f}")
        with col_f1: st.metric("F1-Score (Spam)", f"{cr['Spam (1)']['f1-score']:.4f}")
        with col_auc: st.metric("ROC AUC", f"{roc_auc:.4f}")

        # Visualisasi
        st.markdown("---")
        
        col_cm, col_roc = st.columns(2)
        
        with col_cm:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(cm, model_name_eval)
        
        with col_roc:
            st.subheader("ROC Curve")
            # Pastikan Y_pred_proba_eval di-flatten jika dari CNN
            if Y_pred_proba_eval.ndim > 1:
                Y_pred_proba_eval = Y_pred_proba_eval.flatten()
                
            fpr, tpr, _ = roc_curve(Y_val_eval, Y_pred_proba_eval)
            plot_roc_curve(fpr, tpr, roc_auc, model_name_eval)
            
        st.markdown("---")
        
        # Feature Importance (Tidak ditampilkan untuk CNN, karena lebih kompleks)
        if model_name_eval in ['Random Forest', 'XGBoost', 'LightGBM']:
            st.subheader("Feature Importance")
            if model_name_eval == 'Random Forest':
                importances = selected_model_eval.feature_importances_
            elif model_name_eval == 'XGBoost' or model_name_eval == 'LightGBM':
                importances = selected_model_eval.feature_importances_
            
            feature_names = X_val.columns
            df_importance = pd.DataFrame({'feature': feature_names,'importance': importances})
            df_importance = df_importance.sort_values(by='importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=df_importance, color='skyblue')
            plt.title(f'Top 10 Feature Importance ({model_name_eval})')
            plt.xlabel('Importance Score')
            plt.ylabel('Fitur')
            st.pyplot(plt)
        
        elif model_name_eval == 'Logistic Regression':
            st.subheader("Feature Importance (Abs. Coefficient)")
            coefficients = selected_model_eval.coef_[0]
            importance_abs = np.abs(coefficients)
            
            feature_names = X_val.columns
            df_importance = pd.DataFrame({'feature': feature_names, 'importance': importance_abs})
            df_importance = df_importance.sort_values(by='importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=df_importance, color='skyblue')
            plt.title(f'Top 10 Feature Importance ({model_name_eval})')
            plt.xlabel('Nilai Absolut Koefisien')
            plt.ylabel('Fitur')
            st.pyplot(plt)
        
        elif model_name_eval == 'Isolation Forest':
            st.subheader("Feature Importance (Korelasi Absolut dengan Anomaly Score)")
            df_corr = X_final.copy()
            df_corr['anomaly_score'] = selected_model_eval.decision_function(X_final)
            correlation_with_score = df_corr.corr()['anomaly_score'].drop('anomaly_score')
            importance_abs_corr = np.abs(correlation_with_score)

            df_importance_if = pd.DataFrame({'feature': importance_abs_corr.index, 'importance': importance_abs_corr})
            df_importance_if = df_importance_if.sort_values(by='importance', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=df_importance_if, color='skyblue')
            plt.title(f'Top 10 Feature Importance ({model_name_eval})')
            plt.xlabel('Nilai Absolut Korelasi')
            plt.ylabel('Fitur')
            st.pyplot(plt)


if __name__ == '__main__':
    main()