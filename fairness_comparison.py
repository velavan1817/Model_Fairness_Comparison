import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

thresholds = [0.3, 0.5, 0.7]

results = []


for model_name, model in models.items():

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        test_results = pd.DataFrame({
            "actual": y_test.values,
            "prediction": predictions,
            "gender": df.loc[X_test.index, "gender"].values
        })

        groups = test_results["gender"].unique()

        group_data = {}

        for group_name in groups:

            group = test_results[
                test_results["gender"] == group_name
            ]

            positive_rate = group["prediction"].mean()

            actual_positive = (
                group["actual"] == 1
            ).sum()

            actual_negative = (
                group["actual"] == 0
            ).sum()

            true_positive = (
                (group["actual"] == 1) &
                (group["prediction"] == 1)
            ).sum()

            false_positive = (
                (group["actual"] == 0) &
                (group["prediction"] == 1)
            ).sum()

            tpr = (
                true_positive / actual_positive
                if actual_positive > 0 else 0
            )

            fpr = (
                false_positive / actual_negative
                if actual_negative > 0 else 0
            )

            group_data[group_name] = {
                "positive_rate": positive_rate,
                "tpr": tpr,
                "fpr": fpr
            }


        group1 = groups[0]
        group2 = groups[1]

        rate1 = group_data[group1]["positive_rate"]
        rate2 = group_data[group2]["positive_rate"]

        tpr1 = group_data[group1]["tpr"]
        tpr2 = group_data[group2]["tpr"]

        fpr1 = group_data[group1]["fpr"]
        fpr2 = group_data[group2]["fpr"]


        # Demographic Parity Difference
        demographic_parity_difference = abs(
            rate1 - rate2
        )

        # Disparate Impact
        if rate2 != 0:
            disparate_impact = rate1 / rate2
        else:
            disparate_impact = 0

        # Equal Opportunity Difference
        equal_opportunity_difference = abs(
            tpr1 - tpr2
        )

        # Equalized Odds
        tpr_difference = abs(tpr1 - tpr2)
        fpr_difference = abs(fpr1 - fpr2)


        results.append({
            "model": model_name,
            "threshold": threshold,
            "demographic_parity_difference":
                round(demographic_parity_difference, 4),
            "disparate_impact":
                round(disparate_impact, 4),
            "equal_opportunity_difference":
                round(equal_opportunity_difference, 4),
            "tpr_difference":
                round(tpr_difference, 4),
            "fpr_difference":
                round(fpr_difference, 4)
        })


results_df = pd.DataFrame(results)


os.makedirs("results/fairness", exist_ok=True)

results_df.to_csv(
    "results/fairness/fairness_comparison.csv",
    index=False
)


print("\nCOMPLETE FAIRNESS COMPARISON")
print("=" * 100)

print(
    results_df.to_string(index=False)
)

print("\nSaved to:")
print(
    "results/fairness/fairness_comparison.csv"
) 