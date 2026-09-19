# Model Fairness Comparison & Threshold Analysis Dashboard

An interactive machine learning dashboard for comparing classification models, analyzing decision thresholds, and identifying potential fairness differences across demographic groups.

## 📌 Project Overview

Machine learning models can produce different outcomes for different demographic groups, even when they achieve high overall accuracy.

This project provides a practical dashboard for analyzing both **model performance** and **fairness**. It allows users to compare multiple classification models at different probability thresholds and understand how changing the threshold affects predictions and fairness-related metrics.

The project is designed as an educational and engineering-focused ML application that demonstrates how model evaluation can go beyond accuracy.

---

## 🎯 Objectives

* Compare multiple machine learning classification models.
* Analyze model performance at different decision thresholds.
* Evaluate confusion matrices and classification metrics.
* Measure fairness across demographic groups.
* Identify differences in model outcomes between groups.
* Visualize model performance and fairness metrics.
* Provide an interactive dashboard for easier analysis.

---

## 🚀 Key Features

### 1. Model Comparison

The dashboard supports comparison of different classification algorithms:

* Logistic Regression
* Decision Tree
* Random Forest

### 2. Threshold Analysis

Users can analyze model predictions using different probability thresholds.

Example:

```text
Thresholds:
0.3
0.4
0.5
0.6
0.7
```

Changing the threshold can affect:

* Accuracy
* Precision
* Recall
* True Positive Rate
* False Positive Rate
* Number of positive predictions

### 3. Performance Evaluation

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* True Positive Rate
* False Positive Rate

### 4. Fairness Analysis

The dashboard can evaluate fairness-related metrics across demographic groups.

Metrics include:

* Demographic Parity
* Equal Opportunity
* Equalized Odds
* Disparate Impact

### 5. Visualization

The dashboard provides visual representations of:

* Confusion matrices
* Model comparison
* Threshold performance
* Group-wise prediction rates
* Fairness metrics

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Probability Prediction
   ↓
Threshold Analysis
   ↓
Performance Evaluation
   ↓
Fairness Evaluation
   ↓
Dashboard Visualization
```

---

## 📊 Dataset

The current dataset contains features such as:

| Feature        | Description           |
| -------------- | --------------------- |
| name           | Record identifier     |
| city           | Location information  |
| income         | Income value          |
| credit_score   | Credit-related score  |
| loan_amount    | Requested loan amount |
| years_employed | Employment experience |
| points         | Additional feature    |
| loan_approved  | Target variable       |

> The dataset is used for demonstrating machine-learning model evaluation and fairness analysis.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Dashboard

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
Model-Fairness-Dashboard/
│
├── data/
│   └── loan_data.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── train_models.py
│   ├── threshold_analysis.py
│   ├── fairness_metrics.py
│   └── visualization.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/velavan1817/Model-Fairness-Dashboard.git
```

### 2. Open the Project

```bash
cd Model-Fairness-Dashboard
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will normally open in your browser at:

```text
http://localhost:8501
```

---

## 📈 Example Model Evaluation

The project evaluates models using multiple metrics rather than relying only on accuracy.

Example:

```text
Model              Accuracy    Precision    Recall
---------------------------------------------------
Logistic Regression   0.88       0.88        0.86
Decision Tree         0.96       0.96        0.96
Random Forest         ...        ...         ...
```

The actual values depend on the dataset, preprocessing, model configuration, and selected threshold.

---

## 🎚️ Threshold Analysis

Classification models commonly produce probabilities.

For example:

```text
Prediction Probability = 0.72
```

With a threshold of:

```text
0.50
```

the prediction becomes:

```text
0.72 >= 0.50
Prediction = Positive
```

If the threshold changes to:

```text
0.80
```

the same prediction becomes:

```text
0.72 < 0.80
Prediction = Negative
```

Therefore, threshold selection can significantly change model behavior.

---

## ⚖️ Fairness Metrics

### Demographic Parity

Measures whether positive prediction rates are similar across demographic groups.

```text
Positive Prediction Rate
=
Positive Predictions / Total Predictions
```

### Equal Opportunity

Focuses on whether groups have similar true-positive rates.

```text
TPR = TP / (TP + FN)
```

### Equalized Odds

Considers both:

```text
True Positive Rate
False Positive Rate
```

across groups.

### Disparate Impact

Compares positive prediction rates between groups.

```text
Disparate Impact
=
Positive Rate of Group A /
Positive Rate of Group B
```

These metrics provide different perspectives, and improving one fairness measure does not necessarily improve all others.

---

## 🔍 Why Threshold Analysis Matters

A model with high accuracy may still produce substantially different outcomes across groups.

This project therefore examines:

```text
Model Performance
        +
Threshold Behavior
        +
Group-Level Outcomes
        ↓
Fairness Analysis
```

This gives a more complete picture of model behavior.

---

## 📊 Dashboard Sections

The dashboard can contain the following sections:

### Overview

* Dataset summary
* Number of records
* Feature information
* Target distribution

### Model Comparison

* Logistic Regression
* Decision Tree
* Random Forest

### Threshold Analysis

* Threshold selector
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Fairness Analysis

* Group comparison
* Demographic parity
* Equal opportunity
* Equalized odds
* Disparate impact

### Visualizations

* Confusion matrix
* Metric comparison
* Threshold curves
* Group-wise prediction rates

---

## 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Classification
* Model training
* Model evaluation
* Probability prediction
* Decision thresholds
* Confusion matrices
* Precision and recall
* Fairness-aware ML evaluation
* Data visualization
* Streamlit dashboard development
* Git and GitHub project management

---

## 🔮 Future Improvements

Possible future enhancements include:

* Add more classification algorithms.
* Add automated threshold optimization.
* Add interactive fairness reports.
* Add SHAP-based explainability.
* Add model performance monitoring.
* Add downloadable evaluation reports.
* Add support for custom datasets.
* Add model version tracking.
* Add REST API using FastAPI.
* Containerize the application using Docker.

---

## ⚠️ Important Note

Fairness metrics should be interpreted in the context of the dataset, protected/group attributes, prediction task, and application requirements. A single metric does not provide a complete assessment of whether a model is fair.

This project is intended for educational and experimental purposes and should not be used as the sole basis for real-world high-impact decisions.

---

## 👨‍💻 Authors

**Velavan A**
AI & Data Science

**Vishakan V**

### Guide

**Dr. K. Manivannan**

---

## ⭐ Project Highlights

```text
Machine Learning
        +
Threshold Analysis
        +
Fairness Metrics
        +
Interactive Dashboard
        =
Model Fairness Analysis Platform
```

---

## 📜 License

This project is intended for educational and research purposes.
