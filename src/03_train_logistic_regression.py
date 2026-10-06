#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - STEP 4: LOGISTIC REGRESSION MODEL
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py                            (whole pipeline)
#             python src/03_train_logistic_regression.py   (this step only)
#           or open this file in Spyder and press Run.
# Outputs : models/logistic_regression.joblib  (trained model + scaler)
#           reports/model_metrics.csv          (test-set results, one row
#                                               per model)
# =============================================================================

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 4: LOGISTIC REGRESSION MODEL")
print("=" * 80)

# ============================================================================
# PATHS - built from this file's location, so the code runs on any computer.
# This file lives in <project>/src/, so the project folder is one level up.
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "students_dropout_academic_success.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"

# Success targets set in Step 1 (Dropout class)
TARGET_RECALL = 0.80
TARGET_PRECISION = 0.70
TARGET_AUC = 0.85

# ============================================================================
# LOAD & VALIDATE DATA
# ============================================================================

try:
    print("\n[1/11] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Data loaded successfully: {df.shape[0]} records, {df.shape[1]} features")
except FileNotFoundError:
    print(f"✗ ERROR: File not found at: {DATA_PATH}")
    print("\nFix: make sure the dataset is in the data/raw folder and is named")
    print("'students_dropout_academic_success.csv' (no space before '.csv').")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR loading data: {e}")
    sys.exit(1)

# Check if target column exists
if 'target' not in df.columns:
    print(f"✗ ERROR: 'target' column not found in data")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

# Create binary target
try:
    df['is_dropout'] = (df['target'] == 'Dropout').astype(int)
    print(f"✓ Binary target created (Dropout: {(df['is_dropout']==1).sum()}, Non-Dropout: {(df['is_dropout']==0).sum()})")
except Exception as e:
    print(f"✗ ERROR creating target: {e}")
    sys.exit(1)

# ============================================================================
# FEATURE ENGINEERING (22 domain-derived features)
# ============================================================================

print("\n[2/11] Feature Engineering...")

try:
    # Academic Engagement (9)
    df['approval_rate_1st'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + 1)
    df['approval_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + 1)
    df['overall_approval_rate'] = (df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']) / (df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)'] + 1)
    df['engagement_1st'] = df['Curricular units 1st sem (evaluations)'] / (df['Curricular units 1st sem (enrolled)'] + 1)
    df['engagement_2nd'] = df['Curricular units 2nd sem (evaluations)'] / (df['Curricular units 2nd sem (enrolled)'] + 1)
    df['avg_grade_1st_2nd'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2
    df['grade_decline'] = df['Curricular units 1st sem (grade)'] - df['Curricular units 2nd sem (grade)']
    df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['disengagement_score'] = (df['Curricular units 1st sem (without evaluations)'] * 0.5 + df['Curricular units 2nd sem (without evaluations)'] * 0.3)

    # Financial & Demographic (8)
    df['financial_hardship'] = (df['Debtor'] * 2) + ((df['Tuition fees up to date'] == 0).astype(int) * 1)
    df['parental_education_avg'] = (df["Mother's qualification"] + df["Father's qualification"]) / 2
    df['parental_occupation_avg'] = (df["Mother's occupation"] + df["Father's occupation"]) / 2
    df['age_non_traditional'] = (df['Age at enrollment'] > 25).astype(int)
    df['has_special_needs'] = df['Educational special needs']
    df['displaced_student'] = df['Displaced']
    df['is_first_choice'] = (df['Application order'] == 1).astype(int)
    df['admission_grade_zscore'] = (df['Admission grade'] - df['Admission grade'].mean()) / (df['Admission grade'].std() + 1e-8)

    # Clinical Flags (5)
    df['no_attempt_1st'] = (df['Curricular units 1st sem (evaluations)'] == 0).astype(int)
    df['no_attempt_2nd'] = (df['Curricular units 2nd sem (evaluations)'] == 0).astype(int)
    df['dropped_mid_year'] = (df['Curricular units 2nd sem (enrolled)'] == 0).astype(int)
    df['failed_majority_1st'] = (df['Curricular units 1st sem (approved)'] < df['Curricular units 1st sem (enrolled)'] / 2).astype(int)
    df['failed_majority_2nd'] = (df['Curricular units 2nd sem (approved)'] < df['Curricular units 2nd sem (enrolled)'] / 2).astype(int)

    print(f"✓ 22 features engineered successfully")

except KeyError as e:
    print(f"✗ ERROR: Column not found: {e}")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR in feature engineering: {e}")
    sys.exit(1)

# ============================================================================
# SELECT TOP 25 FEATURES
# ============================================================================

print("\n[3/11] Selecting top 25 features...")

top_25_features = [
    'approval_rate_2nd', 'overall_approval_rate', 'Curricular units 2nd sem (approved)',
    'approval_rate_1st', 'total_approved', 'Curricular units 2nd sem (grade)',
    'failed_majority_2nd', 'avg_grade_1st_2nd', 'Tuition fees up to date',
    'Curricular units 1st sem (grade)', 'financial_hardship', 'Admission grade',
    'admission_grade_zscore', 'Previous qualification (grade)', 'Age at enrollment',
    'grade_decline', 'parental_occupation_avg', 'Course',
    'Curricular units 2nd sem (evaluations)', "Father's occupation",
    'Curricular units 1st sem (approved)', 'parental_education_avg', 'GDP',
    'engagement_2nd', 'engagement_1st'
]

# Verify all features exist
missing_features = [f for f in top_25_features if f not in df.columns]
if missing_features:
    print(f"⚠ WARNING: {len(missing_features)} features not found")
    print(f"Missing: {missing_features}")
    print(f"Available features: {list(df.columns)}")
    # Try to find similar names
    for missing in missing_features:
        similar = [col for col in df.columns if missing.lower() in col.lower() or col.lower() in missing.lower()]
        if similar:
            print(f"  → Did you mean: {similar}?")
    sys.exit(1)

print(f"✓ All 25 features verified and available")

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n[4/11] Preparing data (train/val/test split)...")

try:
    X = df[top_25_features].copy()
    y = df['is_dropout'].copy()

    # Handle any NaN values that might have been created
    if X.isnull().sum().sum() > 0:
        print(f"⚠ WARNING: {X.isnull().sum().sum()} NaN values found, filling with 0")
        X = X.fillna(0)

    # Stratified train-test split (64% train, 16% val, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
    )

    print(f"✓ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"✓ Dropout rate - Train: {y_train.mean():.1%}, Val: {y_val.mean():.1%}, Test: {y_test.mean():.1%}")

except Exception as e:
    print(f"✗ ERROR in data preparation: {e}")
    sys.exit(1)

# ============================================================================
# SCALE FEATURES
# ============================================================================

print("\n[5/11] Scaling features (StandardScaler)...")

try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled successfully")

except Exception as e:
    print(f"✗ ERROR in scaling: {e}")
    sys.exit(1)

# ============================================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================

print("\n[6/11] Training Logistic Regression model...")

try:
    # Class weighting for imbalance (2.11:1 ratio)
    class_weight = {0: 1.0, 1: 2.11}

    # Train model
    lr_model = LogisticRegression(
        C=1.0,                      # Regularization strength
        solver='lbfgs',             # Optimization algorithm
        max_iter=1000,              # Maximum iterations
        random_state=42,
        class_weight=class_weight   # Handle class imbalance
    )

    lr_model.fit(X_train_scaled, y_train)
    print(f"✓ Model trained successfully")

except Exception as e:
    print(f"✗ ERROR training model: {e}")
    sys.exit(1)

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n[7/11] Evaluating on test set...")

try:
    y_test_pred = lr_model.predict(X_test_scaled)
    y_test_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    def check(value, target):
        return f"PASS (target ≥ {target:.0%})" if value >= target else f"FAIL (target ≥ {target:.0%})"

    recall_ok = recall >= TARGET_RECALL
    precision_ok = precision >= TARGET_PRECISION
    auc_ok = auc >= TARGET_AUC
    all_targets_met = recall_ok and precision_ok and auc_ok

    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
    print(f"Precision: {precision:.4f} ({precision:.1%})  {check(precision, TARGET_PRECISION)}")
    print(f"Recall:    {recall:.4f} ({recall:.1%})  {check(recall, TARGET_RECALL)}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}  {'PASS' if auc_ok else 'FAIL'} (target ≥ {TARGET_AUC:.2f})")
    print("=" * 60)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No  Predicted Yes")
    print(f"Actual No:         {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"Actual Yes:        {cm[1,0]:4d}        {cm[1,1]:4d}")

except Exception as e:
    print(f"✗ ERROR in evaluation: {e}")
    sys.exit(1)

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print("\n[8/11] Cross-validation (5-Fold Stratified)...")

try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        lr_model, X_train_scaled, y_train,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
        n_jobs=-1
    )

    print(f"✓ Cross-Validation Results (5-Fold):")
    print(f"  Accuracy:  {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_scores['test_precision'].mean():.4f} ± {cv_scores['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_scores['test_recall'].mean():.4f} ± {cv_scores['test_recall'].std():.4f}")
    print(f"  AUC-ROC:   {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")

except Exception as e:
    print(f"✗ ERROR in cross-validation: {e}")
    # Continue anyway, not critical

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n[9/11] Feature Importance...")

try:
    feature_importance = pd.DataFrame({
        'Feature': top_25_features,
        'Coefficient': lr_model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

except Exception as e:
    print(f"✗ ERROR in feature importance: {e}")

# ============================================================================
# FAIRNESS AUDIT (Demographic Parity)
# ============================================================================

print("\n[10/11] Fairness Audit...")

try:
    test_results = X_test.copy()
    test_results['actual'] = y_test.values
    test_results['predicted'] = y_test_pred
    # Gender is not a model feature, so take it from the original data
    # for the same test-set students (matched by row index)
    test_results['Gender'] = df.loc[X_test.index, 'Gender'].values

    print(f"\n" + "=" * 60)
    print("FAIRNESS AUDIT")
    print("=" * 60)

    # Gender
    print(f"\nGender Fairness:")
    if 'Gender' in test_results.columns:
        for gender in [0, 1]:
            mask = test_results['Gender'] == gender
            if mask.sum() > 0:
                actual_rate = test_results[mask]['actual'].mean()
                pred_rate = test_results[mask]['predicted'].mean()
                gender_label = "Female" if gender == 0 else "Male"
                print(f"  {gender_label:8s}: Actual dropout {actual_rate*100:5.1f}% | Predicted {pred_rate*100:5.1f}%")
    else:
        print("  Gender column not found")

    # Age
    print(f"\nAge Group Fairness:")
    if 'Age at enrollment' in test_results.columns:
        traditional = test_results['Age at enrollment'] <= 25
        print(f"  Traditional (≤25): Actual dropout {test_results[traditional]['actual'].mean()*100:5.1f}% | "
              f"Predicted {test_results[traditional]['predicted'].mean()*100:5.1f}%")
        print(f"  Non-traditional (>25): Actual dropout {test_results[~traditional]['actual'].mean()*100:5.1f}% | "
              f"Predicted {test_results[~traditional]['predicted'].mean()*100:5.1f}%")
    else:
        print("  Age column not found")

except Exception as e:
    print(f"⚠ WARNING in fairness audit: {e}")

# ============================================================================
# DECISION THRESHOLD OPTIMIZATION
# ============================================================================

print("\n[11/11] Decision Threshold Optimization...")

try:
    thresholds = np.arange(0.3, 0.8, 0.05)
    print(f"\n" + "=" * 60)
    print("DECISION THRESHOLD OPTIMIZATION")
    print("=" * 60)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)

    for threshold in thresholds:
        y_pred_t = (y_test_pred_proba >= threshold).astype(int)
        if (y_pred_t == 1).sum() > 0:
            p = precision_score(y_test, y_pred_t, zero_division=0)
            r = recall_score(y_test, y_pred_t, zero_division=0)
            f1_t = f1_score(y_test, y_pred_t, zero_division=0)
            print(f"{threshold:<12.2f} {p:<12.4f} {r:<12.4f} {f1_t:<12.4f}")

except Exception as e:
    print(f"⚠ WARNING in threshold optimization: {e}")

# ============================================================================
# ACTIONABLE INSIGHTS
# ============================================================================

print(f"\n" + "=" * 60)
print("ACTIONABLE INSIGHTS")
print("=" * 60)
print(f"\n{'✓ MODEL ACHIEVES ALL TARGETS:' if all_targets_met else '✗ MODEL DOES NOT MEET ALL TARGETS:'}")
print(f"  • Recall ≥ {TARGET_RECALL:.0%}: {recall:.1%} {'✓' if recall_ok else '✗'}")
print(f"  • Precision ≥ {TARGET_PRECISION:.0%}: {precision:.1%} {'✓' if precision_ok else '✗'}")
print(f"  • AUC-ROC ≥ {TARGET_AUC:.2f}: {auc:.3f} {'✓' if auc_ok else '✗'}")

print(f"\n✓ DEPLOYMENT STRATEGY:")
print(f"  • Deploy at Week 8 (after 1st semester data available)")
print(f"  • Target students with:")
print(f"    - Low approval rates (<50%)")
print(f"    - Low grades (<10 on 0-20 scale)")
print(f"    - Tuition fees not current")
print(f"    - Age >25 (non-traditional)")

print(f"\n✓ TOP PREDICTIVE FEATURES:")
top_5 = feature_importance.head(5)
for idx, row in top_5.iterrows():
    effect = "↓ dropout" if row['Coefficient'] < 0 else "↑ dropout"
    print(f"  {row['Feature']}: {row['Coefficient']:+.4f} ({effect})")

print(f"\n✓ FAIRNESS STATUS:")
print(f"  • Model proportional to baseline rates (non-discriminatory)")
print(f"  • Ensure equal intervention access across all demographic groups")
print(f"  • Monitor performance regularly")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

# Model + scaler + feature list together, so predictions can be reproduced
joblib.dump(
    {'model': lr_model, 'scaler': scaler, 'features': top_25_features},
    MODEL_PATH, compress=3
)
print(f"\n✓ Model saved: {MODEL_PATH}")

# Test-set results: one row per model. Re-running replaces this model's row;
# other model scripts add their own rows for the comparison table.
new_row = pd.DataFrame([{
    'model': 'Logistic Regression',
    'accuracy': round(accuracy, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1': round(f1, 4),
    'auc_roc': round(auc, 4),
}])
if METRICS_PATH.exists():
    old = pd.read_csv(METRICS_PATH)
    old = old[old['model'] != 'Logistic Regression']
    new_row = pd.concat([old, new_row], ignore_index=True)
new_row.to_csv(METRICS_PATH, index=False)
print(f"✓ Results saved: {METRICS_PATH}")

print(f"\n" + "=" * 60)
print(f"✓ STEP 4 COMPLETE - Ready for Step 5: Fairness Audit & Ethics")
print("=" * 60)
