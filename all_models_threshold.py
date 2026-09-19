import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("data/loan_data.csv")

print("Dataset loaded!")
print("Rows:", len(df))


# -----------------------------
# 2. Features
# -----------------------------

X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "years_employed"
    ]
]


# -----------------------------
# 3. Target
# -----------------------------

y = df["loan_approved"]


# -----------------------------
# 4. Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 5. Models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# -----------------------------
# 6. Thresholds
# -----------------------------

thresholds = [0.3, 0.5, 0.7]


# -----------------------------
# 7. Train models
# -----------------------------

for name, model in models.items():

    print("\n")
    print("========================================")
    print(name)
    print("========================================")

    model.fit(X_train, y_train)

    # Get probability of True class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Test thresholds
    for threshold in thresholds:

        predictions = probabilities >= threshold

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        print("\nThreshold:", threshold)

        print(
            "Accuracy :",
            round(accuracy, 4)
        )

        print(
            "Precision:",
            round(precision, 4)
        )

        print(
            "Recall   :",
            round(recall, 4)
        )

        print("Confusion Matrix:")

        print(cm)