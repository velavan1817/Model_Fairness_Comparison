import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/group_analysis.csv")

os.makedirs("results/group_graphs", exist_ok=True)

# Accuracy
plt.figure(figsize=(10, 6))
plt.bar(df["city"], df["accuracy"])
plt.title("Accuracy by City")
plt.xlabel("City")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/group_graphs/accuracy_by_city.png")
plt.show()


# Precision
plt.figure(figsize=(10, 6))
plt.bar(df["city"], df["precision"])
plt.title("Precision by City")
plt.xlabel("City")
plt.ylabel("Precision")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/group_graphs/precision_by_city.png")
plt.show()


# Recall
plt.figure(figsize=(10, 6))
plt.bar(df["city"], df["recall"])
plt.title("Recall by City")
plt.xlabel("City")
plt.ylabel("Recall")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/group_graphs/recall_by_city.png")
plt.show()

print("Group graphs saved successfully.")