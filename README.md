📄 RESUME CATEGORY PREDICTION USING PERCEPTRON & NLP
An NLP-Powered Machine Learning Project
🚀 Project Overview

This project focuses on building a Resume Category Prediction System using Natural Language Processing (NLP) and the Perceptron machine learning algorithm.
The system automatically analyzes resume content and classifies it into predefined job categories, helping automate the resume screening process.

The project demonstrates the complete machine learning lifecycle, including data preprocessing, feature extraction, model training, evaluation, and deployment using Python.

🎯 Problem Statement

Manual resume screening is time-consuming, repetitive, and inefficient.
Recruiters often spend significant time reviewing resumes to identify suitable job roles.

This project aims to automate resume screening by predicting the most suitable job category for a given resume using machine learning techniques.

🧠 Solution Approach

The solution follows a structured NLP and Machine Learning pipeline:

📥 1. Data Loading

Resume dataset containing resume text and job categories

🧹 2. Text Preprocessing

Removal of HTML tags

Conversion of text to lowercase

Removal of special characters and stopwords

Lemmatization for word normalization

🔎 3. Feature Extraction

TF-IDF (Term Frequency – Inverse Document Frequency)

Unigrams and Bigrams

Feature size optimization for efficiency

🤖 4. Model Training

Perceptron (Linear Classifier)

Balanced class weights

Regularization to reduce overfitting

📊 5. Model Evaluation

Test accuracy

Precision, Recall, and F1-score

5-fold Cross-Validation

Confusion Matrix analysis

🌐 6. Deployment

Interactive Streamlit Web Application

Real-time resume category prediction

🗂️ Dataset Information

The dataset contains resumes labeled with job categories.

Key columns:

Resume_str – Resume text

Category – Job category label

Sample Categories:

HR

Data Science

Software Developer

Web Developer

DevOps

Testing / QA

⚙️ Tech Stack

Language: Python

Libraries:

Pandas

NumPy

Scikit-Learn

NLTK

Matplotlib & Seaborn

Streamlit

ML Algorithm: Perceptron

Feature Engineering: TF-IDF Vectorization

📈 Model Performance

Test Accuracy: ~88% – 92%

5-Fold Cross-Validation Accuracy: ~65% – 72%

Cross-validation provides a more realistic evaluation of the Perceptron model on multi-class resume data.

📊 Visualizations Included

Resume category distribution

Resume length analysis

Top TF-IDF keywords

Confusion matrix

Model accuracy insights

🌐 Streamlit Dashboard

An interactive Streamlit dashboard is implemented to:

Paste resume text

Predict job category instantly

Display results in a clean web interface

▶️ Run the application:
streamlit run app.py

📁 Project Structure
├── app.py                  # Streamlit dashboard
├── Resume.csv              # Dataset
├── resume_model.ipynb      # Model training & analysis
├── README.md               # Project documentation
└── requirements.txt        # Dependencies

🧪 Sample Prediction

Input:

Experienced Python developer with knowledge of machine learning,
data analysis, pandas, numpy, and scikit-learn.


Output:

Predicted Category: Data Science

🧠 Key Learnings

Practical implementation of NLP preprocessing

Importance of feature engineering in text classification

Understanding limitations of linear models like Perceptron

Model evaluation using cross-validation

Deploying ML models as web applications

🚀 Future Enhancements

Compare with Logistic Regression and Linear SVM

Add prediction confidence score

Enable resume upload (PDF format)

Deploy on Streamlit Cloud

Improve performance using advanced models

👨‍💻 Author

Sudhanshu Gocher
Machine Learning & Data Science Enthusiast

⭐ If you find this project helpful, consider giving it a star!
