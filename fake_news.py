import pandas as pd
true_df = pd.read_csv("C:\\Users\\palak\\.vscode\\Fake-Message-Detetction-NLP\\News _dataset\\True.csv")
fake_df = pd.read_csv("C:\\Users\\palak\\.vscode\\Fake-Message-Detetction-NLP\\News _dataset\\Fake.csv")
print(true_df.shape)
print(fake_df.shape)
print(true_df.head())
