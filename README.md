# 📌 Fake Message Detection using NLP

## Overview

This project implements a supervised machine learning system to classify SMS messages as **Spam** or **Not Spam (Ham)**.

The goal was to build the complete pipeline end-to-end — from raw text preprocessing to feature engineering, model training, validation, and deployment-ready prediction.

This is a **binary supervised classification problem**.

---

## Problem Statement

Spam messages often contain identifiable patterns such as promotional keywords, call-to-action phrases, and URLs.

The objective of this project is to:

- Convert raw text into numerical features  
- Train a classification model  
- Evaluate its performance properly  
- Enable real-time prediction on unseen messages  

---

## Approach

### 1️⃣ Text Preprocessing

Custom preprocessing pipeline including:

- Lowercasing  
- URL handling  
- Punctuation removal  
- Number removal  
- Tokenization  

Implemented in:
preprocessing.py


---

### 2️⃣ Feature Engineering

Used **TF-IDF (Term Frequency – Inverse Document Frequency)** with:

- `ngram_range = (1, 2)` → unigrams + bigrams  
- `min_df = 2`  
- `max_df = 0.9`  

This converts text into high-dimensional sparse vectors.

---

### 3️⃣ Model Training & Comparison

Models evaluated:

- Logistic Regression  
- Multinomial Naive Bayes  

Final selected model:

**Logistic Regression (with class_weight="balanced")**

Reason for selection:

- Higher spam recall  
- Better F1-score  
- More stable across splits  

Training file:
features_and_model.py


---

### 4️⃣ Model Validation

Performed **5-fold cross-validation** to ensure robustness.

Cross-validated F1 scores:
[0.9498, 0.9090, 0.9300, 0.9415, 0.9235]


Average F1 score ≈ **0.93**

This confirms stable generalization and low overfitting.

---

### 5️⃣ Deployment-Ready Prediction

The trained model and TF-IDF vectorizer are serialized using `joblib`.

Generated files:

- `logistic_model.pkl`
- `tfidf_vectorizer.pkl`

A real-time prediction interface is implemented in:
predict.py


---

## 📊 Final Performance

- **Accuracy:** ~0.98  
- **Spam Recall:** ~0.91  
- **Cross-Validated F1 Score:** ~0.93  

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Palak1725/Fake-Message-Detection-NLP.git
cd Fake-Message-Detection-NLP

## 2️⃣ Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn joblib

## 3️⃣ Train the Model (Optional)

If you want to retrain the model:

```bash
python features_and_model.py

This will:

- Preprocess the dataset

- Train the model

- Perform evaluation

- Save model files (.pkl)

## 4️⃣ Run Real-Time Prediction

'''bash
python predict.py

Enter any message to classify it.

Type exit to stop the program.

## 📂 Project Structure

Fake-Message-Detection-NLP/
│
├── preprocessing.py
├── models.py
├── features_and_model.py
├── predict.py
├── logistic_model.pkl
├── tfidf_vectorizer.pkl
├── cleaned_data/
└── README.md

## 🧠 Concepts Demonstrated

- Supervised Learning

-Binary Classification

-TF-IDF Feature Engineering

-Handling Class Imbalance

-Model Comparison

-Cross-Validation

-Model Serialization

-Deployment-Ready Inference Pipeline