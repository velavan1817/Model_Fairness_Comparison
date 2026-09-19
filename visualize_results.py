import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/model_threshold_results.csv")

os.makedirs("results/graphs", exist_ok=True)

# 1. Accuracy
plt.figure(figsize=(10, 6))

for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(
        data["threshold"],
        data["accuracy"],
        marker="o",
        label=model
    )

plt.title("Accuracy vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.xticks([0.3, 0.5, 0.7])
plt.legend()
plt.grid(True)

plt.savefig("results/graphs/accuracy_vs_threshold.png")
plt.show()


# 2. Precision
plt.figure(figsize=(10, 6))

for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(
        data["threshold"],
        data["precision"],
        marker="o",
        label=model
    )

plt.title("Precision vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.xticks([0.3, 0.5, 0.7])
plt.legend()
plt.grid(True)

plt.savefig("results/graphs/precision_vs_threshold.png")
plt.show()


# 3. Recall
plt.figure(figsize=(10, 6))

for model in df["model"].unique():
    data = df[df["model"] == model]
    plt.plot(
        data["threshold"],
        data["recall"],
        marker="o",
        label=model
    )

plt.title("Recall vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.xticks([0.3, 0.5, 0.7])
plt.legend()
plt.grid(True)

plt.savefig("results/graphs/recall_vs_threshold.png")
plt.show()


print("\nGraphs saved successfully:")
print("results/graphs/accuracy_vs_threshold.png")
print("results/graphs/precision_vs_threshold.png")
print("results/graphs/recall_vs_threshold.png")