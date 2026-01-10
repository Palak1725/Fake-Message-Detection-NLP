import pandas as pd
nazario_df = pd.read_csv("data/raw/archive/Nazario.csv")
nazario_df["text"] = nazario_df["sender"] + " " + nazario_df["subject"] + " " + nazario_df["body"]
nazario_df = nazario_df[["text", "label"]]
nazario_df.to_csv("cleaned_Nazario.csv", index = False)
#print(nazario_df.head())