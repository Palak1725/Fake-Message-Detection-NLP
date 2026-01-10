import pandas as pd
sms_df = pd.read_csv(
    "data/raw/sms_spam.csv",
    sep="\t",
    encoding="latin-1"
)
sms_df.columns = ["label", "text"]
sms_df["label"] = sms_df["label"].map({"ham" : 0, "spam": 1})
sms_df.to_csv("cleaned_sms.csv", index = False)
#print(sms_df.head())
#print(sms_df["label"].value_counts())