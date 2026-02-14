from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def train_logistic(X_train,  Y_train, X_test, Y_test):
    model = LogisticRegression(max_iter=1000, class_weight = "balanced")   #creates an empty model without our data fed in. max_iter =1000 to ensure convergence and make lregression learn by fine tuning it's steps in 100o iterations. smaller value model is forced to stop early, thus, stopped in mid-learning and half-baked model. higher value wastes computation- not needed. 
    model.fit(X_train, Y_train) #training on x,y train data
    Y_pred = model.predict(X_test)  #model's guesses and predictions on test messages saved in y_pred
    print("Logistic Regression resuts: ")
    print(classification_report(Y_test, Y_pred))  #compare actual labels vs predicted labels from model comparison on model's guessed output vs the actual test output
    return model

def train_naive_bayes(X_train, Y_train, X_test, Y_test):
    model = MultinomialNB()  #multinomial naive bayes- designed for text classification and works well with tf-idf features
    model.fit(X_train, Y_train) #training on x,y train data
    Y_pred = model.predict(X_test)  #model's guesses and predictions on test messages saved in y_pred
    print("Naive Bayes resuts: ")
    print(classification_report(Y_test, Y_pred))  #compare actual labels vs predicted labels from model comparison on model's guessed output vs the actual test output
    return model