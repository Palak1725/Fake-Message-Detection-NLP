import pandas as pd
spamAssasin_df = pd.read_csv("data/raw/archive/SpamAssasin.csv")
spamAssasin_df["text"] = spamAssasin_df["sender"] + " " + spamAssasin_df["subject"] + " " + spamAssasin_df["body"]
spamAssasin_df = spamAssasin_df[["text", "label"]]
spamAssasin_df.to_csv("cleaned_SpamAssasin.csv", index = False)
#print(spamAssasin_df.head())