import pandas as pd
phishing_df = pd.read_csv("data/raw/archive/phishing_email.csv")
phishing_df["text"] = phishing_df["text_combined"]
phishing_df = phishing_df[["text", "label"]]
phishing_df.to_csv("cleaned_phishing_email.csv", index = False)
#print(phishing_df.head())