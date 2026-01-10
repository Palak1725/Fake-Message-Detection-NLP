import pandas as pd

df = pd.read_csv("all_messages.csv")
print(df.head())
print(df['label'].value_counts())