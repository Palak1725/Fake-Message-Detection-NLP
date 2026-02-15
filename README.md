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

# Fake Message Detection System

An NLP-based machine learning system to detect fake or spam messages using binary classification techniques.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Palak1725/Fake-Message-Detection-NLP.git
cd Fake-Message-Detection-NLP
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

If you prefer using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn joblib
```

### 3. Train the Model (Optional)

If you want to retrain the model from scratch:

```bash
python features_and_model.py
```

This script will:
- Preprocess the dataset
- Extract features using TF-IDF vectorization
- Train multiple models and select the best performer
- Perform model evaluation with metrics
- Handle class imbalance issues
- Save the trained model and vectorizer as `.pkl` files

### 4. Run Real-Time Prediction

```bash
python predict.py
```

Once running:
- Enter any message to classify it as **Fake** or **Real**
- Type `exit` to stop the program

Example:
```
Enter a message: Congratulations! You've won a free iPhone. Click here to claim.
Prediction: Fake Message

Enter a message: Hey, are we still meeting for coffee at 5?
Prediction: Real Message
```

## 📂 Project Structure

```
Fake-Message-Detection-NLP/
│
├── preprocessing.py          # Text cleaning and preprocessing functions
├── models.py                  # Model definitions and training logic
├── features_and_model.py      # Feature extraction and model pipeline
├── predict.py                 # Real-time prediction interface
├── logistic_model.pkl         # Trained logistic regression model
├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
├── cleaned_data/              # Preprocessed dataset directory
└── README.md                  # Project documentation
```

## 🧠 Concepts Demonstrated

This project showcases the following machine learning and NLP concepts:

- **Supervised Learning** - Training models on labeled message data
- **Binary Classification** - Distinguishing between fake and real messages
- **TF-IDF Feature Engineering** - Converting text to numerical features
- **Handling Class Imbalance** - Techniques to manage uneven class distribution
- **Model Comparison** - Evaluating multiple algorithms to select the best
- **Cross-Validation** - Robust model validation techniques
- **Model Serialization** - Saving and loading trained models with joblib
- **Deployment-Ready Inference** - Real-time prediction pipeline

## 📊 Model Performance

The system uses Logistic Regression as the primary classifier, achieving high accuracy in distinguishing between fake and legitimate messages through careful feature engineering and model tuning.

## 🛠️ Technologies Used

- **Python** - Core programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning algorithms and utilities
- **joblib** - Model serialization

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Palak1725/Fake-Message-Detection-NLP/issues).

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---
```