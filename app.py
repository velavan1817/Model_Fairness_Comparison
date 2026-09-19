import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="Model Fairness Dashboard",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------

model_results = pd.read_csv(
    "results/model_threshold_results.csv"
)

fairness_results = pd.read_csv(
    "results/fairness/fairness_comparison.csv"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("🤖 Model Fairness Comparison Dashboard")

st.write(
    "Compare machine learning models, prediction thresholds, "
    "performance and fairness metrics."
)
# -----------------------------
# LOAN PREDICTION
# -----------------------------

st.header("🔮 Loan Approval Prediction")

import joblib

model = joblib.load(
    "results/final_model.pkl"
)

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=50000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=900,
        value=650
    )

with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=20000.0
    )

    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        value=3.0
    )

# Threshold selector

threshold = st.select_slider(
    "Select Prediction Threshold",
    options=[0.3, 0.5, 0.7],
    value=0.5
)

st.write(
    f"Current Threshold: **{threshold}**"
)

if st.button("Predict Loan"):

    input_data = pd.DataFrame({
        "income": [income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "years_employed": [years_employed]
    })

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = probability >= threshold

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric(
        "Approval Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Selected Threshold",
        f"{threshold:.2f}"
    )

    if prediction:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.info(
        f"Decision rule: Approval Probability "
        f"{probability:.2f} >= Threshold {threshold:.2f}"
    )
# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Model Controls")

model = st.sidebar.selectbox(
    "Select Model",
    model_results["model"].unique()
)

threshold = st.sidebar.selectbox(
    "Select Threshold",
    [0.3, 0.5, 0.7]
)

# -----------------------------
# SELECTED RESULTS
# -----------------------------

selected = model_results[
    (model_results["model"] == model) &
    (model_results["threshold"] == threshold)
]

fairness = fairness_results[
    (fairness_results["model"] == model) &
    (fairness_results["threshold"] == threshold)
]

row = selected.iloc[0]

# -----------------------------
# PERFORMANCE
# -----------------------------

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{row['accuracy'] * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{row['precision'] * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{row['recall'] * 100:.2f}%"
)

# -----------------------------
# CONFIGURATION
# -----------------------------

st.header("⚙️ Selected Configuration")

col1, col2 = st.columns(2)

col1.info(
    f"**Model:** {model}"
)

col2.info(
    f"**Threshold:** {threshold}"
)

# -----------------------------
## -----------------------------
# CONFUSION MATRIX
# -----------------------------

st.header("🔲 Confusion Matrix")

tn = int(float(row["true_negative"]))
fp = int(float(row["false_positive"]))
fn = int(float(row["false_negative"]))
tp = int(float(row["true_positive"]))

cm = [
    [tn, fp],
    [fn, tp]
]

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Not Approved",
    "Approved"
])

ax.set_yticklabels([
    "Not Approved",
    "Approved"
])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_title(
    f"{model} - Threshold {threshold}"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

# -----------------------------
# THRESHOLD COMPARISON
# -----------------------------

st.header("🎚️ Threshold Analysis")

threshold_data = model_results[
    model_results["model"] == model
]

st.dataframe(
    threshold_data[
        [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall"
        ]
    ],
    use_container_width=True
)

# Accuracy chart

st.subheader("Accuracy vs Threshold")

fig, ax = plt.subplots()

ax.plot(
    threshold_data["threshold"],
    threshold_data["accuracy"],
    marker="o"
)

ax.set_xlabel("Threshold")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Threshold")

st.pyplot(fig)

# Precision chart

st.subheader("Precision vs Threshold")

fig, ax = plt.subplots()

ax.plot(
    threshold_data["threshold"],
    threshold_data["precision"],
    marker="o"
)

ax.set_xlabel("Threshold")
ax.set_ylabel("Precision")
ax.set_title("Precision vs Threshold")

st.pyplot(fig)

# Recall chart

st.subheader("Recall vs Threshold")

fig, ax = plt.subplots()

ax.plot(
    threshold_data["threshold"],
    threshold_data["recall"],
    marker="o"
)

ax.set_xlabel("Threshold")
ax.set_ylabel("Recall")
ax.set_title("Recall vs Threshold")

st.pyplot(fig)

# -----------------------------
# FAIRNESS
# -----------------------------

st.header("⚖️ Fairness Metrics")

if not fairness.empty:

    fairness_row = fairness.iloc[0]

    col1, col2 = st.columns(2)

    col1.metric(
        "Demographic Parity Difference",
        fairness_row[
            "demographic_parity_difference"
        ]
    )

    col2.metric(
        "Disparate Impact",
        fairness_row[
            "disparate_impact"
        ]
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Equal Opportunity Difference",
        fairness_row[
            "equal_opportunity_difference"
        ]
    )

    col4.metric(
        "FPR Difference",
        fairness_row[
            "fpr_difference"
        ]
    )

# -----------------------------
# COMPLETE MODEL COMPARISON
# -----------------------------

st.header("🏆 Complete Model Comparison")

st.dataframe(
    model_results,
    use_container_width=True
)

# -----------------------------
# FAIRNESS COMPARISON
# -----------------------------

st.header("⚖️ Complete Fairness Comparison")

st.dataframe(
    fairness_results,
    use_container_width=True
)

# -----------------------------
# FINAL MESSAGE
# -----------------------------

st.success(
    "ML model comparison and fairness analysis completed."
)