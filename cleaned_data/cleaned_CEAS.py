import pandas as pd
ceas_df = pd.read_csv("data/raw/archive/CEAS_08.csv")
ceas_df["text"] = ceas_df["sender"] + " " + ceas_df["subject"] + " " + ceas_df["body"]
ceas_df = ceas_df[["text", "label"]]
ceas_df.to_csv("cleaned_CEAS.csv", index = False)
#print(ceas_df.head())