📄 Resume Category Prediction using Perceptron & NLP
📌 Project Overview

This project focuses on building a Resume Category Prediction System using Natural Language Processing (NLP) and the Perceptron machine learning algorithm.
The system automatically classifies resumes into predefined job categories based on their textual content.

The goal of this project is to demonstrate text preprocessing, feature extraction, model training, evaluation, and deployment using Python and machine learning techniques.

🎯 Problem Statement

Manual screening of resumes is time-consuming and inefficient.
This project aims to automate resume screening by predicting the most suitable job category for a resume using machine learning.

🧠 Solution Approach

The solution follows a structured NLP and ML pipeline:

Data Loading

Resume dataset containing resume text and job categories

Text Preprocessing

Removal of HTML tags

Lowercasing text

Removing special characters and stopwords

Lemmatization for word normalization

Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency)

Unigrams and bigrams

Dimensionality control using max_features

Model Training

Perceptron (linear classifier)

Balanced class weights

Regularization to avoid overfitting

Model Evaluation

Test accuracy

Classification report

5-fold cross-validation

Confusion matrix

Deployment

Interactive Streamlit dashboard for real-time resume classification

🗂️ Dataset

Resume dataset containing:

Resume_str → Resume text

Category → Job category label

Common categories include:

HR

Data Science

Software Developer

Web Developer

DevOps

Testing / QA

⚙️ Technologies Used

Programming Language: Python

Libraries:

Pandas

NumPy

Scikit-Learn

NLTK

Matplotlib / Seaborn

Streamlit

Machine Learning Algorithm: Perceptron

Feature Engineering: TF-IDF Vectorization

📊 Model Performance

Test Accuracy: ~88–92% (single split)

5-Fold Cross-Validation Accuracy: ~65–72%

Cross-validation accuracy is lower due to the simplicity of the Perceptron model and the complexity of multi-class text data.

Note: Cross-validation gives a more realistic estimate of model performance.

📈 Visualizations Included

Resume category distribution

Resume length distribution

Top TF-IDF keywords

Confusion matrix

Model accuracy analysis

🌐 Streamlit Web Application

An interactive Streamlit dashboard is built to:

Paste resume text

Predict job category instantly

Display results in a user-friendly interface

Run the app locally:
streamlit run app.py

📁 Project Structure
├── app.py                  # Streamlit dashboard
├── Resume.csv              # Dataset
├── resume_model.ipynb      # Jupyter Notebook (training & analysis)
├── README.md               # Project documentation
└── requirements.txt        # Dependencies

🧪 Sample Prediction

Input:

Experienced Python developer with knowledge of machine learning,
data analysis, pandas, numpy, and scikit-learn.


Output:

Predicted Category: Data Science

📝 Key Learnings

Practical application of NLP preprocessing techniques

Importance of feature engineering in text classification

Understanding limitations of linear models like Perceptron

Using cross-validation for reliable evaluation

Deploying ML models using Streamlit

🚀 Future Enhancements

Compare Perceptron with Logistic Regression and Linear SVM

Add confidence scores for predictions

Enable resume upload in PDF format

Deploy the app on Streamlit Cloud

Improve accuracy using advanced models

👨‍💻 Author

Sudhanshu Gocher
Machine Learning & Data Science Enthusiast

⭐ Acknowledgements

Scikit-Learn documentation

NLTK library

Open-source resume datasets
