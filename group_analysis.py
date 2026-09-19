import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("data/loan_data.csv")

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

test_results = X_test.copy()

test_results["actual"] = y_test.values
test_results["prediction"] = predictions

test_results["city"] = df.loc[
    X_test.index,
    "city"
].values

print("\nGROUP PERFORMANCE ANALYSIS")
print("=" * 70)

cities = test_results["city"].unique()

results = []

for city in cities:

    group = test_results[
        test_results["city"] == city
    ]

    accuracy = accuracy_score(
        group["actual"],
        group["prediction"]
    )

    precision = precision_score(
        group["actual"],
        group["prediction"],
        zero_division=0
    )

    recall = recall_score(
        group["actual"],
        group["prediction"],
        zero_division=0
    )

    results.append({
        "city": city,
        "samples": len(group),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    "results/group_analysis.csv",
    index=False
)

print("\nSaved to:")
print("results/group_analysis.csv")