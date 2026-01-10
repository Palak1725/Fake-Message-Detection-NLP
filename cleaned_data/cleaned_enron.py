import pandas as pd
enron_df = pd.read_csv("data/raw/archive/Enron.csv")
enron_df["text"] = enron_df["subject"] + " " + enron_df["body"]
enron_df = enron_df[["text", "label"]]
enron_df.to_csv("cleaned_enron.csv", index = False)
#print(enron_df.head())
#print(enron_df.shape)
#print(enron_df.columns)
#print(enron_df["label"].value_counts())
#print(enron_df.sample(2))