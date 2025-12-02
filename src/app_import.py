import joblib
from tensorflow.keras.models import save_model

# Misal var di notebook: tfidf_vectorizer, scaler, rf_model, lr_model, xgb_model, lgbm_model, if_model, cnn_model
joblib.dump(tfidf_vectorizer, "../models/tfidf.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(rf_model, "../models/rf.pkl")
joblib.dump(lr_model, "../models/lr.pkl")
joblib.dump(xgb_model, "../models/xgb.pkl")
joblib.dump(lgbm_model, "../models/lgbm.pkl")
joblib.dump(if_model, "../models/isoforest.pkl")

# Untuk CNN (keras)
cnn_model.save("../models/cnn.h5")

# Optional: simpan juga dataset preprocessing agar EDA identik
df.to_parquet("../data/df_processed.parquet", index=False)