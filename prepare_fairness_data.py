import pandas as pd
import numpy as np

df = pd.read_csv("data/loan_data.csv")

np.random.seed(42)

df["gender"] = np.random.choice(
    ["Male", "Female"],
    size=len(df)
)

df.to_csv(
    "data/loan_data_fairness_demo.csv",
    index=False
)

print("Fairness demo dataset created successfully.")

print("\nGender distribution:")
print(df["gender"].value_counts())

print("\nColumns:")
print(df.columns.tolist())

print("\nSaved to:")
print("data/loan_data_fairness_demo.csv")