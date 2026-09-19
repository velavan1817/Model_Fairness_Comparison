import pandas as pd

df = pd.read_csv("results/model_threshold_results.csv")

best_accuracy = df.loc[df["accuracy"].idxmax()]
best_precision = df.loc[df["precision"].idxmax()]
best_recall = df.loc[df["recall"].idxmax()]

print("\nBEST ACCURACY")
print(best_accuracy)

print("\nBEST PRECISION")
print(best_precision)

print("\nBEST RECALL")
print(best_recall)

# Balanced score
df["balanced_score"] = (df["precision"] + df["recall"]) / 2

best_balanced = df.loc[df["balanced_score"].idxmax()]

print("\nBEST BALANCED MODEL")
print(best_balanced)