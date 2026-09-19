import pandas as pd
import os

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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

        predictions = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions
        ).ravel()

        results.append({
            "model": model_name,
            "threshold": threshold,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        })

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/model_threshold_results.csv",
    index=False
)

print("\nMODEL & THRESHOLD COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False))

print("\nSaved to:")
print("results/model_threshold_results.csv")