# Predicting Student Dropout Risk with Machine Learning

**Post Graduate Diploma in Artificial Intelligence and Machine Learning — Capstone Project**
Author: *[Your full name]* · Submitted: *[date]*

> Early identification of students at risk of dropping out, so advisors can intervene while there is still time to help.

---

## 1. Problem statement

Student dropout costs institutions tuition revenue and costs students their investment and future opportunities. This project frames dropout prediction as a **binary classification** task: *Dropout* (1) versus *Graduate or still Enrolled* (0), using data available at enrolment and after the first two semesters.

| Success metric (Dropout class) | Target | Achieved (best model) |
| --- | --- | --- |
| Recall | ≥ 0.80 | 0.841 |
| Precision | ≥ 0.70 | 0.773 |
| AUC-ROC | ≥ 0.85 | 0.926 |

Business KPI: retention lift of 5–10% through early, targeted intervention.

## 2. Results summary

*Replace the placeholders with the numbers your scripts write to `reports/model_metrics.csv`.*

| Model | Accuracy | Precision | Recall | AUC-ROC |
| --- | --- | --- | --- | --- |
| Logistic Regression (selected) | 0.812 | 0.773 | 0.841 | 0.926 |
| Random Forest | [ ] | [ ] | [ ] | [ ] |
| XGBoost | [ ] | [ ] | [ ] | [ ] |
| SVM | [ ] | [ ] | [ ] | [ ] |
| Neural Network | [ ] | [ ] | [ ] | [ ] |

**Why Logistic Regression was selected:** it meets all three targets and its coefficients explain each prediction, which matters when advisors must justify interventions to students. The fairness audit (Step 5) found no systematic disparity across gender, age or international status.

## 3. Dataset

- **Source:** *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/697 (Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L., 2021). Licensed CC BY 4.0. *[Check the link and licence, and adjust if you used the Kaggle copy.]*
- **Size:** 4,424 students × 37 columns, no missing values.
- **Target:** `target` — Graduate (50%), Dropout (32%), Enrolled (18%).
- **File:** `data/raw/students_dropout_academic_success.csv` (unchanged original).
- **Data dictionary:** [`data/data_dictionary.md`](data/data_dictionary.md)

## 4. Repository structure

```
student-dropout-prediction/
├── README.md              ← you are here
├── requirements.txt       ← Python libraries with exact versions
├── run_all.py             ← runs the full pipeline in order
├── data/
│   ├── raw/               ← original dataset (never edited)
│   ├── processed/         ← cleaned / engineered data (created by the code)
│   └── data_dictionary.md
├── notebooks/             ← Jupyter notebooks: EDA and model comparison
├── src/                   ← Python scripts, numbered in run order
│   └── config.py          ← all paths, random seed and settings
├── models/                ← trained models (.joblib)
└── reports/
    ├── final_report.pdf
    ├── Technical_Presentation.pptx
    ├── Business_Presentation.pptx
    ├── model_metrics.csv  ← created by the code
    └── figures/           ← charts created by the code
```

## 5. How to reproduce the results

Requires **Python [3.x.x]** — tested with Anaconda on Windows. *[Fill in from `python --version`.]*

```bash
# 1. Download the repo (or use Code → Download ZIP on GitHub and unzip)
git clone https://github.com/<your-username>/student-dropout-prediction.git
cd student-dropout-prediction

# 2. Create a clean environment and install the libraries
conda create -n dropout python=3.11 -y
conda activate dropout
pip install -r requirements.txt

# 3. Run the full pipeline (about 15–20 minutes)
python run_all.py
```

Outputs appear in `models/`, `reports/figures/` and `reports/model_metrics.csv`.
A fixed random seed (`SEED = 42` in `src/config.py`) makes results repeatable. Neural network results may differ in the third decimal place on different hardware.

**Optional — GenAI advisor chatbot (Step 9):** runs in demo mode with no setup. For live responses, set an Anthropic API key as an environment variable first (never put the key in code):

```bash
set ANTHROPIC_API_KEY=your-key-here        # Windows Command Prompt
export ANTHROPIC_API_KEY=your-key-here     # macOS / Linux
```

## 6. Pipeline steps

| Script | What it does |
| --- | --- |
| `src/01_preprocess_features.py` | Cleaning, 22 engineered features, scaling, feature selection, PCA |
| `src/02_train_logistic_regression.py` | Logistic Regression + threshold tuning |
| `src/03_train_random_forest.py` | Random Forest + GridSearchCV |
| `src/04_train_xgboost.py` | XGBoost + GridSearchCV + early stopping |
| `src/05_train_svm.py` | SVM (RBF) + GridSearchCV |
| `src/06_train_neural_network.py` | Keras MLP (128-64-32), dropout, L2 |
| `src/07_bias_fairness_audit.py` | SHAP/LIME explanations, demographic parity, equalised odds, disparate impact |
| `src/08_genai_advisor_chatbot.py` | LLM-backed advisor chatbot (demo mode without API key) |

*[Edit this table to match your actual file names.]*

## 7. Ethics and limitations

Predictions are a prompt for supportive outreach, never for punitive action. Key limitations: moderate class imbalance (mitigated with class weighting), second-semester features are only available mid-year, and the data comes from a single institution, so results may not transfer. Full analysis: *Bias & Fairness Analysis* section of `reports/final_report.pdf`.

## 8. Reports

- [Final report (PDF)](reports/final_report.pdf)
- [Technical presentation](reports/Technical_Presentation.pptx)
- [Business presentation](reports/Business_Presentation.pptx)

## 9. Licence

Code released under the MIT Licence (see `LICENSE`). The dataset remains under its original licence.
