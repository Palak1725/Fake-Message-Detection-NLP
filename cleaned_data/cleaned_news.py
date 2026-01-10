import pandas as pd
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")
true_df["label"] = 0
fake_df["label"] = 1
true_df["text"] = true_df["title"] + " " + true_df["text"]
fake_df["text"] = fake_df["title"] + " " + fake_df["text"]
news_df = pd.concat([true_df[["text", "label"]], fake_df[["text", "label"]]])
news_df["source"] = "news"
news_df.to_csv("news_clean.csv", index = False)
#print(news_df.head(3))
#print(news_df["label"])