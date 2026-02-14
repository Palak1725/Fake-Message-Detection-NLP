import pandas as pd
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
    nb_model = train_naive_bayes(X_train, Y_train, X_test, Y_test)