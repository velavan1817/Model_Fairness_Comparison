import pandas as pd
import os

model_results = pd.read_csv(
    "results/model_threshold_results.csv"
)

fairness_results = pd.read_csv(
    "results/fairness/fairness_comparison.csv"
)

# Find best balanced model
model_results["balanced_score"] = (
    model_results["precision"] +
    model_results["recall"]
) / 2

best_model = model_results.loc[
    model_results["balanced_score"].idxmax()
]

# Find fairness result for the selected model
selected_fairness = fairness_results[
    (fairness_results["model"] == best_model["model"]) &
    (fairness_results["threshold"] == best_model["threshold"])
]

print("\nFINAL MODEL SUMMARY")
print("=" * 70)

print("\nBest Model:")
print(best_model["model"])

print("\nBest Threshold:")
print(best_model["threshold"])

print("\nAccuracy:")
print(f"{best_model['accuracy'] * 100:.2f}%")

print("\nPrecision:")
print(f"{best_model['precision'] * 100:.2f}%")

print("\nRecall:")
print(f"{best_model['recall'] * 100:.2f}%")

print("\nFAIRNESS METRICS")
print("=" * 70)

print(
    selected_fairness.to_string(index=False)
)

# Create final summary
summary = {
    "model": best_model["model"],
    "threshold": best_model["threshold"],
    "accuracy": best_model["accuracy"],
    "precision": best_model["precision"],
    "recall": best_model["recall"],
    "demographic_parity_difference":
        selected_fairness[
            "demographic_parity_difference"
        ].iloc[0],
    "disparate_impact":
        selected_fairness[
            "disparate_impact"
        ].iloc[0],
    "equal_opportunity_difference":
        selected_fairness[
            "equal_opportunity_difference"
        ].iloc[0],
    "tpr_difference":
        selected_fairness[
            "tpr_difference"
        ].iloc[0],
    "fpr_difference":
        selected_fairness[
            "fpr_difference"
        ].iloc[0]
}

summary_df = pd.DataFrame([summary])

os.makedirs("results", exist_ok=True)

summary_df.to_csv(
    "results/final_model_summary.csv",
    index=False
)

print("\nFinal summary saved to:")
print("results/final_model_summary.csv")