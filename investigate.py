import pandas as pd

df = pd.read_csv("data/loan_data.csv")

print("Loan approval counts:")
print(df["loan_approved"].value_counts())

print("\nPoints by loan approval:")
print(df.groupby("loan_approved")["points"].describe())

print("\nCredit score by loan approval:")
print(df.groupby("loan_approved")["credit_score"].describe())

print("\nIncome by loan approval:")
print(df.groupby("loan_approved")["income"].describe())