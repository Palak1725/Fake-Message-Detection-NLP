import pandas as pd
nigerian_fraud_df = pd.read_csv("data/raw/archive/Nigerian_Fraud.csv")
nigerian_fraud_df["text"] = nigerian_fraud_df["sender"] + " " + nigerian_fraud_df["subject"] + " " + nigerian_fraud_df["body"]
nigerian_fraud_df = nigerian_fraud_df[["text", "label"]]
nigerian_fraud_df.to_csv("cleaned_Nigerian_Fraud.csv", index = False)
#print(nigerian_fraud_df.head())