import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("data/loan_data.csv")

print("Dataset loaded!")
print(df.head())

# Features
X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "years_employed",
        "points"
    ]
]

# Target
y = df["loan_approved"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Prediction
predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test.values)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)