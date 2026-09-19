import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

os.makedirs("results/confusion_matrices", exist_ok=True)

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    print("\n" + name)
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Approved", "Approved"]
    )

    display.plot()

    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        f"results/confusion_matrices/{filename}.png"
    )

    plt.show()

print("\nConfusion matrices saved successfully.")