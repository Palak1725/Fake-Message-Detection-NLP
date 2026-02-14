import joblib
from preprocessing import preprocess_text

model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
while True:
    text = input("\nEnter a message (or type 'exit'): ")
    if (text.lower() == "exit"):
        break

    processed = preprocess_text(text)
    vector = vectorizer.transform([processed])  #transform the processed text to a vector using the same vectorizer used during training
    prediction = model.predict(vector)[0]  #predict the label using the loaded model, [0] is to get the single prediction from the array

    if (prediction == 1):
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")