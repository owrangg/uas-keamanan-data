import pandas as pd
import numpy as np
import re, string, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df = pd.read_csv("../data/spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

df["clean"] = df["message"].apply(clean_text)
df["length"] = df["message"].apply(len)

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean']).toarray()

scaler = StandardScaler()
length_scaled = scaler.fit_transform(df[['length']])

# final features
import pandas as pd
X = pd.concat([
    pd.DataFrame(X_tfidf),
    pd.DataFrame(length_scaled, columns=["length_scaled"])
], axis=1)

y = df["label_num"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "rf": RandomForestClassifier(class_weight='balanced'),
    "lr": LogisticRegression(class_weight='balanced'),
    "xgb": XGBClassifier(eval_metric='logloss'),
    "lgbm": LGBMClassifier()
}

trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained[name] = model
    print(f"\n=== {name.upper()} ===")
    preds = model.predict(X_val)
    print(confusion_matrix(y_val, preds))
    print(classification_report(y_val, preds))

iso = IsolationForest(contamination=y.mean())
iso.fit(X_train)
trained["iso"] = iso

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
tokenizer.fit_on_texts(df["clean"])

seq = tokenizer.texts_to_sequences(df["clean"])
pad = pad_sequences(seq, maxlen=MAX_LEN)

X_train_pad = pad[X_train.index]
X_val_pad = pad[X_val.index]

cnn = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(10, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
cnn.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_val_pad, y_val), verbose=1)

# folder
# folder
import os
os.makedirs("../models", exist_ok=True)

# Traditional ML
joblib.dump(trained["rf"], "../models/rf.pkl")
joblib.dump(trained["lr"], "../models/lr.pkl")
joblib.dump(trained["xgb"], "../models/xgb.pkl")
joblib.dump(trained["lgbm"], "../models/lgbm.pkl")
joblib.dump(trained["iso"], "../models/isoforest.pkl")

# TF-IDF + Scaler
joblib.dump(tfidf, "../models/tfidf.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

# CNN Model
cnn.save("../models/cnn.h5")

# Tokenizer
joblib.dump(tokenizer, "../models/tokenizer.pkl")

print("ALL ARTIFACTS SAVED SUCCESSFULLY 🎉")
