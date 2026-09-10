import pandas as pd

df = pd.read_csv("outputs/processed_data/features.csv")

print(df["label"].value_counts())