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

results["actual"] = y_test.values
results["prediction"] = predictions
results["gender"] = df.loc[X_test.index, "gender"].values


print("\nFAIRNESS ANALYSIS")
print("=" * 70)

group_results = []

for gender in results["gender"].unique():

    group = results[results["gender"] == gender]

    true_positive = (
        (group["actual"] == 1) &
        (group["prediction"] == 1)
    ).sum()

    actual_positive = (
        group["actual"] == 1
    ).sum()

    false_positive = (
        (group["actual"] == 0) &
        (group["prediction"] == 1)
    ).sum()

    actual_negative = (
        group["actual"] == 0
    ).sum()

    true_positive_rate = (
        true_positive / actual_positive
        if actual_positive > 0 else 0
    )

    false_positive_rate = (
        false_positive / actual_negative
        if actual_negative > 0 else 0
    )

    group_results.append({
        "gender": gender,
        "true_positive_rate": round(true_positive_rate, 4),
        "false_positive_rate": round(false_positive_rate, 4)
    })


fairness_df = pd.DataFrame(group_results)

print("\nEqual Opportunity:")
print(fairness_df.to_string(index=False))


# Equal Opportunity Difference
tpr_difference = (
    fairness_df["true_positive_rate"].max()
    - fairness_df["true_positive_rate"].min()
)

print("\nEqual Opportunity Difference:")
print(round(tpr_difference, 4))


# Equalized Odds
fpr_difference = (
    fairness_df["false_positive_rate"].max()
    - fairness_df["false_positive_rate"].min()
)

print("\nFalse Positive Rate Difference:")
print(round(fpr_difference, 4))

print("\nEqualized Odds:")
print(
    "TPR Difference =",
    round(tpr_difference, 4)
)

print(
    "FPR Difference =",
    round(fpr_difference, 4)
)


# Save results
import os

os.makedirs("results/fairness", exist_ok=True)

fairness_df.to_csv(
    "results/fairness/equal_opportunity.csv",
    index=False
)

with open(
    "results/fairness/equalized_odds.txt",
    "w"
) as file:

    file.write(
        f"TPR Difference: {round(tpr_difference, 4)}\n"
    )

    file.write(
        f"FPR Difference: {round(fpr_difference, 4)}\n"
    )

print("\nSaved to:")
print("results/fairness/equal_opportunity.csv")
print("results/fairness/equalized_odds.txt")