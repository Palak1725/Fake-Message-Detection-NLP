import pandas as pd
ling_df = pd.read_csv("data/raw/archive/Ling.csv")
#remove nan by ""
ling_df["text"] = ling_df["subject"].fillna("") + " " + ling_df["body"].fillna("") 
ling_df = ling_df[["text", "label"]]
ling_df.to_csv("cleaned_ling.csv", index = False)
#print(ling_df.head().to_string(index =  False))