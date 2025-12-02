🛡️ Email Spam Classification Project

Course: Keamanan Data
Lecturer: Puguh Hiskiawan, S.Si., M.Si., Ph.D.
Contributors:
Chellynia Cristabel
Denis Winata
Meyry Ardianti
Refanthi Eunike Siregar

This project presents a comprehensive comparative analysis of various machine learning (ML) and deep learning algorithms for the task of classifying text messages as spam or ham. The primary objective is to identify the most optimal model capable of balancing two critical risks: minimizing security risk (False Negatives) and maximizing user experience (minimizing False Positives).

🚀 Methodology and Key Findings
Six main models were tested, including Logistic Regression, Random Forest, XGBoost, LightGBM, CNN, and Isolation Forest.

🚀 Features
The database used in this analysis is a collection of email texts classified into two binary categories: 'spam' and 'ham'.
EDA: Exploratory Data Analysis covered class distribution, text length, and word clouds.
Preprocessing: Text cleaning, tokenization, stopword removal, and stemming were executed.
Balancing: SMOTE (Synthetic Minority Over-sampling Technique) was applied to handle data imbalance.
Modeling: Multiple algorithms were trained, including Random Forest, Logistic Regression, Isolation Forest, LightGBM, XGBoost, and CNN.
Evaluation: Comprehensive metrics like Confusion Matrix, ROC curve, and F1-Score were used.
Selection: LightGBM was chosen as the final model for its optimal balance of low FN and low FP.
Deployment: An interactive streamlit dashboard was developed for real-time spam detection and model comparison.

💻 Interactive Implementation (Streamlit)
The entire Playbook and modeling results are implemented in an interactive web application using Streamlit, which serves as:
Model Evaluation Dashboard: Users can dynamically select and visualize the performance metrics (Confusion Matrix, ROC Curve, Classification Report) for every tested algorithm.
Real-time Prediction: Users can input any test message, which is processed by the LightGBM model to instantly verify its classification as Spam or Ham.

⚙️ Installation and Usage
These instructions assume that all project dependencies (Streamlit, LightGBM, scikit-learn, etc.) are already installed in your Python environment and that you have navigated to the correct project folder.

Prerequisites
Python: Ensure Python (version 3.8 or higher) is installed.
Dependencies: All required libraries must be pre-installed.

Steps for Local Execution
Download Project Files: Ensure the application file (app.py) are saved in your current working directory.
Run the Streamlit Application: Open your Terminal or VS Code Integrated Terminal, navigate to the project folder, and execute the main application file:
streamlit run app.py
The interactive application will automatically launch and open in your default web browser.