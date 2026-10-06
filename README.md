# Predicting Student Dropout Risk with Machine Learning

**Post Graduate Diploma in Artificial Intelligence and Machine Learning — Capstone Project**

A data-driven solution for identifying students at risk of dropping out early, enabling proactive and targeted academic support.

---

## Overview

This project develops a machine learning pipeline to predict student dropout risk using academic, demographic, and financial data. The goal is to support advising teams with earlier, more informed intervention decisions.

- **Dataset:** Predict Students' Dropout and Academic Success (UCI Machine Learning Repository)
- **Records:** 4,424 students
- **Features:** 37 variables
- **Problem Type:** Binary classification (Dropout vs. non-dropout)
- **Best Model:** Logistic Regression

## Results

The primary model meets the capstone performance targets for the dropout class:

| Metric | Target | Achieved |
| --- | --- | --- |
| Recall | ≥ 0.80 | **0.841** |
| Precision | ≥ 0.70 | **0.773** |
| AUC-ROC | ≥ 0.85 | **0.926** |
| Accuracy | — | **0.812** |

Logistic Regression was selected because it achieves the required performance while remaining highly interpretable. This is important in academic settings where explanations for intervention decisions should be understandable to advisors and stakeholders.

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Bunz825/student-dropout-prediction.git
cd student-dropout-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python run_all.py
```

This executes the project in sequence and generates trained models, evaluation metrics, and visual diagnostics in the `reports/` and `models/` folders.

## Repository Structure

```text
student-dropout-prediction/
├── README.md
├── requirements.txt
├── run_all.py
├── data/
│   ├── data_dictionary.md
│   └── raw/
│       └── students_dropout_academic_success.csv
├── notebooks/
│   └── 01_results_overview.ipynb
├── src/
│   ├── 01_data_quality_eda.py
│   ├── 02_eda_feature_engineering.py
│   ├── 03_train_logistic_regression.py
│   ├── 04_train_random_forest.py
│   ├── 05_train_xgboost.py
│   ├── 06_train_svm.py
│   ├── 07_train_neural_network.py
│   ├── 08_unsupervised_analysis.py
│   └── 09_bias_fairness_audit.py
├── reports/
│   ├── Einar_R_Abuyuan_Capstone_Project.pdf
│   ├── Einar_Ramos_Abuyuan_Business_Presentation.pptx
│   ├── Einar_Ramos_Abuyuan_Technical_Presentation.pptx
│   ├── model_metrics.csv
│   ├── step5_results.txt
│   └── figures/
├── models/
│   └── generated during execution
└── .gitignore
```

## Key Components

- **Feature engineering:** 22 domain-specific features capturing academic performance, financial stress, and engagement risk
- **Model comparison:** Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network
- **Bias and fairness analysis:** SHAP, LIME, demographic parity, and equal opportunity checks
- **Decision support:** interpretable features that help advisors justify intervention decisions

## Dataset

| Property | Value |
| --- | --- |
| **Source** | UCI Machine Learning Repository |
| **Dataset Name** | Predict Students' Dropout and Academic Success |
| **Records** | 4,424 |
| **Features** | 37 |
| **Target** | Multi-class label, modeled as dropout vs. non-dropout |
| **Missing Values** | None |

Full field descriptions are available in `data/data_dictionary.md`.

## Why Logistic Regression?

The logistic regression model was selected as the primary solution because it balances strong predictive performance with the interpretability required for real academic interventions.

It offers:
- strong discrimination for the dropout class
- transparent coefficients for feature-level explanations
- strong suitability for advising workflows and stakeholder communication
- lower complexity than more opaque models

## Fairness and Responsible Use

The project includes fairness and ethical assessment to understand whether the model behaves equitably across relevant demographic groups. This is important because predictive models in education must support student success without creating unintended harm or exclusion.

## Deliverables

- Final capstone report: `reports/Einar_R_Abuyuan_Capstone_Project.pdf`
- Business presentation: `reports/Einar_Ramos_Abuyuan_Business_Presentation.pptx`
- Technical presentation: `reports/Einar_Ramos_Abuyuan_Technical_Presentation.pptx`
- Notebook summary: `notebooks/01_results_overview.ipynb`

## Technologies

- **Languages:** Python (93.3%), Jupyter Notebook (6.7%)
- **Core libraries:** scikit-learn, pandas, NumPy, XGBoost, TensorFlow, matplotlib, seaborn
- **Explainability:** SHAP, LIME
- **Evaluation:** precision, recall, ROC-AUC, cross-validation, fairness metrics

## Notes

- The project is designed to run from the repository root without hard-coded local paths.
- Model artifacts and generated files are produced during execution.
- A fixed random seed is used to promote reproducibility.
- The fairness and explainability workflow is included to support responsible deployment decisions.

---

**Author:** Einar Ramos Abuyuan  
**Program:** Post Graduate Diploma in Artificial Intelligence and Machine Learning
