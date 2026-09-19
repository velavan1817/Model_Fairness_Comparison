import pandas as pd

df = pd.read_csv("data/loan_data.csv")

print("First 5 rows:")
print(df.head())

print("Shape:", df.shape)

print("Columns:")
print(df.columns)