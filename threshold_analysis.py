import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
print("Columns:", df.columns.tolist())


# -----------------------------
# 2. Features
# -----------------------------

X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "years_employed",
        "points"
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
# 5. Create and train model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# -----------------------------
# 6. Get probabilities
# -----------------------------

probabilities = model.predict_proba(X_test)[:, 1]

print("\nFirst 10 probabilities:")

print(probabilities[:10])


# -----------------------------
# 7. Default prediction
# Threshold = 0.5
# -----------------------------

predictions = (probabilities >= 0.5)


# -----------------------------
# 8. Metrics
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)


print("\n--- Threshold 0.5 ---")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))


# -----------------------------
# 9. Test different thresholds
# -----------------------------

thresholds = [0.3, 0.5, 0.7]

print("\n==============================")
print("THRESHOLD COMPARISON")
print("==============================")


for threshold in thresholds:

    predictions = (probabilities >= threshold)

    accuracy = accuracy_score(y_test, predictions)

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

    print("\nThreshold:", threshold)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, predictions))