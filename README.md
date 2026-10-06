# Predicting Student Dropout Risk with Machine Learning

**Post Graduate Diploma in Artificial Intelligence and Machine Learning — Capstone Project**

An end-to-end machine learning system to identify at-risk students early, enabling timely academic intervention.

---

## Overview

This project applies classification modeling to predict student dropout risk using academic, financial, and demographic data. Early identification allows academic advisors to provide targeted support before students disengage.

**Dataset:** 4,424 students, 37 features (UCI ML Repository)  
**Problem:** Binary classification (Dropout vs. Enrolled/Graduate)  
**Best Model:** Logistic Regression — achieves all performance targets with high interpretability for real-world deployment.

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall | ≥ 0.80 | **0.841** |
| Precision | ≥ 0.70 | **0.773** |
| AUC-ROC | ≥ 0.85 | **0.926** |
| Accuracy | — | **0.812** |

The logistic regression model was selected for its transparent feature coefficients and strong performance across all metrics, making it ideal for explainable, high-stakes decision support.

## Quick Start

### Installation
```bash
git clone https://github.com/Bunz825/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python run_all.py
```

This executes all 9 modeling steps in sequence, producing trained models, performance metrics, and diagnostic figures in the `reports/` directory.

## Project Structure

```
student-dropout-prediction/
├── README.md
├── requirements.txt               # Python dependencies
├── run_all.py                     # Entry point: orchestrates the pipeline
│
├── data/
│   ├── data_dictionary.md
│   └── raw/
│       └── students_dropout_academic_success.csv
│
├── src/                           # Numbered scripts (run in order)
│   ├── 01_data_quality_eda.py            # Data validation and EDA
│   ├── 02_eda_feature_engineering.py     # Feature creation (22 new features)
│   ├── 03_train_logistic_regression.py   # Primary model
│   ├── 04_train_random_forest.py         # Model comparison
│   ├── 05_train_xgboost.py               # Model comparison
│   ├── 06_train_svm.py                   # Model comparison
│   ├── 07_train_neural_network.py        # Model comparison
│   ├── 08_unsupervised_analysis.py       # Clustering and dimensionality
│   └── 09_bias_fairness_audit.py         # Fairness, SHAP, LIME
│
├── notebooks/
│   └── 01_results_overview.ipynb         # Results summary notebook
│
├── reports/
│   ├── Einar_R_Abuyuan_Capstone_Project.pdf
│   ├── Einar_Ramos_Abuyuan_Business_Presentation.pptx
│   ├── Einar_Ramos_Abuyuan_Technical_Presentation.pptx
│   ├── model_metrics.csv                 # Generated: model comparison table
│   ├── step5_results.txt                 # Generated: fairness audit output
│   └── figures/                          # Generated: diagnostic plots
│
└── models/                        # Generated: trained model artifacts
```

## Key Contributions

- **Feature Engineering:** 22 domain-derived features capturing academic engagement, financial hardship, and risk signals
- **Model Comparison:** 5 algorithms (Logistic Regression, Random Forest, XGBoost, SVM, Neural Network)
- **Fairness & Explainability:** SHAP values, LIME local explanations, demographic parity analysis
- **Production-Ready:** Class weighting for imbalance, threshold optimization, fairness-aware mitigations

## Dataset

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository: "Predict Students' Dropout and Academic Success" |
| **Records** | 4,424 students |
| **Features** | 37 (academic, demographic, financial) |
| **Class Distribution** | Graduate 50%, Dropout 32%, Enrolled 18% |
| **Missing Values** | None |

See `data/data_dictionary.md` for full feature descriptions.

## Model Selection

**Why Logistic Regression?**
- Meets all three success targets (Recall, Precision, AUC-ROC)
- Coefficients directly interpretable for advisor communication
- Fast inference; low computational overhead
- Calibrated decision thresholds for fairness mitigations
- Robust 5-fold cross-validation results

Fairness audit confirms the model maintains parity across gender, age, and economic status without systematic bias.

## Deliverables

- **Final Report:** `reports/Einar_R_Abuyuan_Capstone_Project.pdf`
- **Business Presentation:** `reports/Einar_Ramos_Abuyuan_Business_Presentation.pptx` (stakeholder-focused)
- **Technical Presentation:** `reports/Einar_Ramos_Abuyuan_Technical_Presentation.pptx` (methods & results)
- **Results Notebook:** `notebooks/01_results_overview.ipynb`

## Technologies

- **Languages:** Python 93.3%, Jupyter Notebook 6.7%
- **Core Libraries:** scikit-learn, XGBoost, TensorFlow, pandas, NumPy
- **Explainability:** SHAP, LIME
- **Fairness:** Demographic parity, equal opportunity, disparate impact analysis

## Notes

- All scripts use dynamic path resolution; run from the repository root
- Model artifacts and metrics are generated during execution
- Random seed (42) ensures reproducibility; neural network may vary in third decimal place on different hardware
- Fairness mitigations tested: unawareness, reweighting, and group-aware thresholds

---

**Author:** Einar Ramos Abuyuan  
**Institution:** Post Graduate Diploma in AI & Machine Learning  
**Language Composition:** Python (93.3%), Jupyter (6.7%)
