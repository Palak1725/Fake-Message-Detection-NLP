import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer  #words->nos. on the basis of tf-idf
from sklearn.model_selection import train_test_split    #shuffles data, split safely, labels aligned
from models import train_logistic, train_naive_bayes

if __name__ == "__main__":
    df = pd.read_csv("cleaned_data/cleaned_sms.csv")
    df["processed_text"] = df["text"].apply(preprocess_text)    #human-ready text to model ready processed text
    X = df["processed_text"]    
    Y = df["label"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)  #ignore words appearing in less than 2 documents(rows in nlp) or more than 90% of documents
    X_tfidf = vectorizer.fit_transform(X)   #fit: scans all text, transform: converts each text to a numerical vector
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.2, random_state=42, stratify = Y)   #random_state = 42(can be any no just to fix the shuffle), start random state from 42, stratify- preserves label ratio across train and test sets

    #print(df.head())
    #print(X_tfidf.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    log_model = train_logistic(X_train, Y_train, X_test, Y_test)
    
    feature_names = vectorizer.get_feature_names_out()  #get the feature names (words) from the vectorizer
    coefficients = log_model.coef_[0]  #get the coefficients of the logistic regression model, it is a numpy array which stores the weights for every feature(every word). shape looks like (1, no_of_features). (1,...) because it's a binary classification problem, only one row of weights is needed. If it were multi-class, it would be (no_of_classes, no_of_features). a class is a possibe output label, here 0->not spam, 1->spam. [0] is extracting the first row. higher the positive coefficient, higher the probability of it being a spam and vice-versa.
    top_spam_indices = np.argsort(coefficients)[-20:] #argsort returns the indexes that would sort the array like [4,3,2,1,0]. [-20:] is taking 20 words from last of the array,i.e.,largest weights.
    print("\nTop 20 Spam-Indicating Words:")
    for idx in top_spam_indices[::-1]:
        print(feature_names[idx])

    top_ham_indices = np.argsort(coefficients)[:20]
    print("\nTop 20 Ham-Indicating Words:")
    for idx in top_ham_indices:
        print(feature_names[idx])

    nb_model = train_naive_bayes(X_train, Y_train, X_test, Y_test)
    joblib.dump(log_model, "logistic_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")