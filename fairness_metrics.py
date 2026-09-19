import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/loan_data_fairness_demo.csv")

X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "years_employed"
    ]
]

y = df["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

threshold = 0.5

predictions = (probabilities >= threshold).astype(int)

results = X_test.copy()

results["actual"] = y_test
results["prediction"] = predictions

results["gender"] = df.loc[X_test.index, "gender"]

print("\nFAIRNESS ANALYSIS")
print("=" * 60)

group_results = []

for gender in results["gender"].unique():

    group = results[results["gender"] == gender]

    positive_rate = group["prediction"].mean()

    group_results.append({
        "gender": gender,
        "samples": len(group),
        "positive_predictions": group["prediction"].sum(),
        "positive_rate": round(positive_rate, 4)
    })

fairness_df = pd.DataFrame(group_results)

print("\nDemographic Parity:")
print(fairness_df.to_string(index=False))


# Calculate Disparate Impact
male_rate = fairness_df.loc[
    fairness_df["gender"] == "Male",
    "positive_rate"
].iloc[0]

female_rate = fairness_df.loc[
    fairness_df["gender"] == "Female",
    "positive_rate"
].iloc[0]

if male_rate != 0:
    disparate_impact = female_rate / male_rate
else:
    disparate_impact = 0

print("\nDisparate Impact Ratio:")
print(round(disparate_impact, 4))

if 0.8 <= disparate_impact <= 1.25:
    print("Interpretation: Groups have relatively similar positive prediction rates.")
else:
    print("Interpretation: There is a noticeable difference in positive prediction rates.")


# Save results
import os

os.makedirs("results/fairness", exist_ok=True)

fairness_df.to_csv(
    "results/fairness/demographic_parity.csv",
    index=False
)

with open(
    "results/fairness/disparate_impact.txt",
    "w"
) as file:
    file.write(
        f"Disparate Impact Ratio: {round(disparate_impact, 4)}\n"
    )

print("\nSaved to:")
print("results/fairness/demographic_parity.csv")
print("results/fairness/disparate_impact.txt")