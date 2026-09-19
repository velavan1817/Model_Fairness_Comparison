import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# Load dataset
df = pd.read_csv("data/loan_data.csv")


# Features WITHOUT points
X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "years_employed"
    ]
]


# Target
y = df["loan_approved"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


print("Model trained without points!")


# Probability
probabilities = model.predict_proba(X_test)[:, 1]


print("\nFirst 10 probabilities:")

print(probabilities[:10])


# Test thresholds
thresholds = [0.3, 0.5, 0.7]


print("\n==============================")
print("THRESHOLD COMPARISON")
print("==============================")


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

    print("\nThreshold:", threshold)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))

    print("\nConfusion Matrix:")

    print(confusion_matrix(
        y_test,
        predictions
    ))