# 🛡️ Spam Detection Dashboard

A comprehensive Streamlit web application for analyzing and detecting spam SMS messages using various Machine Learning and Deep Learning algorithms. This project was developed for the Data Security (Keamanan Data) Final Exam.

## 🚀 Features

### 1. 📊 Exploratory Data Analysis (EDA)
Visualize and understand the dataset with interactive charts:
- **Class Distribution**: See the balance between Ham (normal) and Spam messages.
- **Message Length Analysis**: Compare the length of spam vs. ham messages.
- **Word Clouds**: Visualize the most frequent words in both categories.
- **Uppercase Ratio**: Analyze the usage of uppercase letters in spam messages.

### 2. 🤖 Model Training & Evaluation
Train and evaluate multiple models directly within the app:
- **Classic ML**: Random Forest, Logistic Regression, XGBoost, LightGBM.
- **Anomaly Detection**: Isolation Forest.
- **Deep Learning**: CNN (Convolutional Neural Network) with TensorFlow/Keras.

**Metrics Provided:**
- Confusion Matrix
- ROC Curve & AUC Score
- Classification Report (Precision, Recall, F1-Score)

### 3. 🎯 Real-time Prediction
- Test the trained models with your own custom text input.
- Get immediate feedback on whether a message is **Spam** or **Ham**.
- View confidence scores or anomaly scores.

## 🛠️ Tech Stack

- **Language**: Python
- **Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow (Keras)
- **NLP**: NLTK (Natural Language Toolkit)

## 📂 Project Structure

```
├── data/
│   └── spam.csv           # Dataset file
├── src/
│   └── streamlit_app.py   # Main application file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8 or higher

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/owrangg/uas-keamanan-data.git
   cd uas-keamanan-data
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run src/streamlit_app.py
   ```

4. **Access the App**
   Open your browser and go to `http://localhost:8501`.

## 📝 Dataset
The dataset used is the **SMS Spam Collection Dataset**, containing 5,574 SMS messages tagged as `ham` (legitimate) or `spam`.

## 👤 Authors
**Chellynia Cristabel**

**Denis Winata**

**Meyry Ardianti**

**Refanthi Eunike Siregar**
