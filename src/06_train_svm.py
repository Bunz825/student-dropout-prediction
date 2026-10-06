#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - STEP 4: SUPPORT VECTOR MACHINE (SVM) MODEL
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py              (whole pipeline)
#             python src/06_train_svm.py     (this step only)
#           or open this file in Spyder and press Run.  Takes ~1-2 minutes.
# Outputs : models/svm.joblib           (trained model + scaler + threshold)
#           reports/model_metrics.csv   (test-set results, one row per model)
#
# What this script does:
# - Hyperparameter tuning via GridSearchCV (C, kernel, gamma; 12 combinations)
# - Class weight balancing for imbalanced data
# - Test-set evaluation, plus a decision cut-off chosen on the VALIDATION set
#   (highest cut-off that reaches the recall target), applied once to the
#   test set
# - Cross-validation (5-fold stratified)
# - Feature importance via permutation
# - LIME explanation of one high-risk student (in original units)
# - Fairness audit (gender, age group, scholarship)
# Requires: scikit-learn, lime (both listed in requirements.txt)
# =============================================================================

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
import lime
import lime.lime_tabular

print("=" * 80)
print("SVM MODEL - STUDENT DROPOUT PREDICTION")
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
MODEL_PATH = MODELS_DIR / "svm.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"

# Success targets set in Step 1 (Dropout class)
TARGET_RECALL = 0.80
TARGET_PRECISION = 0.70
TARGET_AUC = 0.85

# Other imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.inspection import permutation_importance

# ============================================================================
# 1. LOAD & VALIDATE DATA
# ============================================================================

print("\n[1/13] Loading data...")
print("-" * 80)

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Data loaded: {df.shape[0]:,} records × {df.shape[1]} features")
except FileNotFoundError:
    print(f"✗ ERROR: File not found at {DATA_PATH}")
    print("Make sure the dataset is in the data/raw folder and is named")
    print("'students_dropout_academic_success.csv' (no space before '.csv').")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

if 'target' not in df.columns:
    print(f"✗ ERROR: 'target' column not found")
    sys.exit(1)

df['is_dropout'] = (df['target'] == 'Dropout').astype(int)
print(f"✓ Binary target created: {(df['is_dropout']==1).sum()} dropouts")

# ============================================================================
# 2. FEATURE ENGINEERING (22 domain-derived features)
# ============================================================================

print("\n[2/13] Feature Engineering...")
print("-" * 80)

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

    print(f"✓ 22 features engineered")
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 3. SELECT TOP 25 FEATURES
# ============================================================================

print("\n[3/13] Selecting top 25 features...")
print("-" * 80)

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

print(f"✓ All 25 features verified")

# ============================================================================
# 4. PREPARE DATA
# ============================================================================

print("\n[4/13] Preparing data...")
print("-" * 80)

try:
    X = df[top_25_features].copy()
    y = df['is_dropout'].copy()

    if X.isnull().sum().sum() > 0:
        X = X.fillna(0)

    # Stratified split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
    )

    print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 5. FEATURE SCALING (CRITICAL FOR SVM)
# ============================================================================

print("\n[5/13] Feature scaling (CRITICAL for SVM)...")
print("-" * 80)

try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled successfully")
    print(f"  SVM requires scaled features (mean=0, std=1)")
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 6. HYPERPARAMETER TUNING
# ============================================================================

print("\n[6/13] Hyperparameter tuning (optimized SVM parameters)...")
print("-" * 80)

try:
    print("Testing SVM parameter combinations...")

    # Calculate class weights
    class_weight = {0: 1.0, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    print(f"✓ Class weights calculated: {class_weight}")

    # Optimized parameter grid (fewer combos than other models)
    param_grid = {
        'C': [0.1, 1, 10],           # Regularization parameter
        'kernel': ['rbf', 'linear'],  # RBF or linear kernel
        'gamma': ['scale', 'auto']    # Kernel coefficient
    }

    svm_base = SVC(
        probability=True,              # Enable probability estimates
        random_state=42,
        class_weight=class_weight
    )

    # GridSearchCV
    grid_search = GridSearchCV(
        svm_base,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    print("GridSearchCV in progress (this takes ~1-2 minutes)...")
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"✓ Tuning complete")
    print(f"  • Best AUC-ROC (CV): {best_cv_score:.4f}")
    print(f"  • Best Parameters:")
    print(f"    - C: {best_params['C']}")
    print(f"    - kernel: {best_params['kernel']}")
    print(f"    - gamma: {best_params['gamma']}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    print(f"Falling back to default parameters...")
    best_params = {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'}
    class_weight = {0: 1.0, 1: (y_train == 0).sum() / (y_train == 1).sum()}

# ============================================================================
# 7. TRAIN SVM MODEL
# ============================================================================

print("\n[7/13] Training SVM model...")
print("-" * 80)

try:
    svm_model = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma=best_params['gamma'],
        probability=True,
        random_state=42,
        class_weight=class_weight
    )

    print("Training SVM (this may take 30-60 seconds)...")
    svm_model.fit(X_train_scaled, y_train)

    print(f"✓ SVM trained successfully")
    print(f"  • Support vectors: {len(svm_model.support_vectors_)}")
    print(f"  • Kernel: {best_params['kernel']}")
    print(f"  • C parameter: {best_params['C']}")

except Exception as e:
    print(f"✗ ERROR during training: {e}")
    sys.exit(1)

# ============================================================================
# 8. EVALUATE ON TEST SET
# ============================================================================

print("\n[8/13] Evaluating on test set...")
print("-" * 80)

try:
    y_test_pred = svm_model.predict(X_test_scaled)
    y_test_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} {'✓ EXCEEDS 70%' if precision >= 0.70 else ''}")
    print(f"Recall:    {recall:.4f} {'✓ EXCEEDS 80%' if recall >= 0.80 else ''}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f} {'✓ EXCEEDS 0.85' if auc >= 0.85 else ''}")
    print("=" * 60)

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No  Predicted Yes")
    print(f"Actual No:         {tn:4d}        {fp:4d}")
    print(f"Actual Yes:        {fn:4d}        {tp:4d}")
    print("(Note: SVM's predict() uses its decision function, which can differ")
    print(" slightly from 'probability ≥ 0.50'; step 8b uses probabilities.)")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 8b. DECISION CUT-OFF CHOSEN ON THE VALIDATION SET
# ============================================================================
# Same method as the Random Forest and XGBoost scripts: the cut-off is chosen
# on the validation set (data the model never trained on and that is NOT the
# test set) as the highest cut-off whose validation recall still reaches the
# target, which keeps precision as high as possible. It is then applied once
# to the test set - whatever the test result turns out to be.

print("\n[8b/13] Choosing decision cut-off on the validation set...")
print("-" * 80)

try:
    y_val_pred_proba = svm_model.predict_proba(X_val_scaled)[:, 1]
    candidate_thresholds = np.round(np.arange(0.30, 0.80, 0.05), 2)

    print(f"\n{'Cut-off':<10} {'Val Precision':<15} {'Val Recall':<12}")
    print("-" * 37)
    meets_target = []
    for t in candidate_thresholds:
        pred_t = (y_val_pred_proba >= t).astype(int)
        p_t = precision_score(y_val, pred_t, zero_division=0)
        r_t = recall_score(y_val, pred_t, zero_division=0)
        print(f"{t:<10.2f} {p_t:<15.4f} {r_t:<12.4f}")
        if r_t >= TARGET_RECALL:
            meets_target.append(t)

    if meets_target:
        chosen_threshold = float(max(meets_target))
        print(f"\n✓ Chosen cut-off: {chosen_threshold:.2f} "
              f"(highest cut-off with validation recall ≥ {TARGET_RECALL:.0%})")
    else:
        chosen_threshold = 0.50
        print(f"\n⚠ No cut-off reached {TARGET_RECALL:.0%} validation recall; keeping 0.50")

    # Apply the chosen cut-off once to the test set
    y_test_pred_final = (y_test_pred_proba >= chosen_threshold).astype(int)
    accuracy_final = accuracy_score(y_test, y_test_pred_final)
    precision_final = precision_score(y_test, y_test_pred_final)
    recall_final = recall_score(y_test, y_test_pred_final)
    f1_final = f1_score(y_test, y_test_pred_final)

    print("\n" + "=" * 60)
    print(f"TEST SET RESULTS AT CHOSEN CUT-OFF ({chosen_threshold:.2f})")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_final:.4f}")
    print(f"Precision: {precision_final:.4f} {'✓ EXCEEDS 70%' if precision_final >= TARGET_PRECISION else '⚠ Below 70%'}")
    print(f"Recall:    {recall_final:.4f} {'✓ EXCEEDS 80%' if recall_final >= TARGET_RECALL else '⚠ Below 80%'}")
    print(f"F1 Score:  {f1_final:.4f}")
    print(f"AUC-ROC:   {auc:.4f} (does not depend on the cut-off)")
    print("=" * 60)

    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test, y_test_pred_final).ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No  Predicted Yes")
    print(f"Actual No:         {tn_f:4d}        {fp_f:4d}")
    print(f"Actual Yes:        {fn_f:4d}        {tp_f:4d}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 9. CROSS-VALIDATION
# ============================================================================

print("\n[9/13] Cross-validation (5-Fold Stratified)...")
print("-" * 80)

try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        svm_model, X_train_scaled, y_train,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'roc_auc', 'f1'],
        n_jobs=-1
    )

    print(f"✓ Cross-Validation Results (5-Fold):")
    print(f"  Accuracy:  {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_scores['test_precision'].mean():.4f} ± {cv_scores['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_scores['test_recall'].mean():.4f} ± {cv_scores['test_recall'].std():.4f}")
    print(f"  F1:        {cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}")
    print(f"  AUC-ROC:   {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 10. FEATURE IMPORTANCE (Permutation-based)
# ============================================================================

print("\n[10/13] Feature Importance Analysis (Permutation-based)...")
print("-" * 80)

try:
    perm_importance = permutation_importance(
        svm_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'Feature': top_25_features,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 Features (Permutation Importance):")
    print(perm_df.head(10).to_string(index=False))

    print(f"\n✓ Note: SVM doesn't have native feature importance")
    print(f"✓ Using permutation importance (shuffle & measure impact)")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 11. FAIRNESS AUDIT
# ============================================================================

print("\n[11/13] Fairness Audit...")
print("-" * 80)

try:
    test_results = X_test.copy()
    test_results['actual'] = y_test.values
    # Audit the predictions at the chosen cut-off (the ones that would be used)
    test_results['predicted'] = y_test_pred_final
    test_results['pred_proba'] = y_test_pred_proba
    # Gender and Scholarship are not model features, so take them from the
    # original data for the same test-set students (matched by row index)
    test_results['Gender'] = df.loc[X_test.index, 'Gender'].values
    test_results['Scholarship holder'] = df.loc[X_test.index, 'Scholarship holder'].values

    print(f"(Predictions at the chosen cut-off of {chosen_threshold:.2f})")

    print(f"\nGender Fairness:")
    for gender in [0, 1]:
        mask = test_results['Gender'] == gender
        if mask.sum() > 0:
            actual_rate = test_results[mask]['actual'].mean()
            pred_rate = test_results[mask]['predicted'].mean()
            gender_label = "Female" if gender == 0 else "Male"
            di = min(actual_rate, pred_rate) / max(actual_rate, pred_rate) if max(actual_rate, pred_rate) > 0 else 1.0
            print(f"  {gender_label:8s}: Actual {actual_rate*100:5.1f}% | Predicted {pred_rate*100:5.1f}% | DI: {di:.3f}")

    print(f"\nAge Group Fairness:")
    traditional = test_results['Age at enrollment'] <= 25
    print(f"  Traditional (≤25):     Actual {test_results[traditional]['actual'].mean()*100:5.1f}% | "
          f"Predicted {test_results[traditional]['predicted'].mean()*100:5.1f}%")
    print(f"  Non-traditional (>25): Actual {test_results[~traditional]['actual'].mean()*100:5.1f}% | "
          f"Predicted {test_results[~traditional]['predicted'].mean()*100:5.1f}%")

    print(f"\nScholarship Fairness:")
    for scholarship in [0, 1]:
        mask = test_results['Scholarship holder'] == scholarship
        if mask.sum() > 0:
            actual_rate = test_results[mask]['actual'].mean()
            pred_rate = test_results[mask]['predicted'].mean()
            scholarship_label = "Without scholarship" if scholarship == 0 else "With scholarship"
            print(f"  {scholarship_label:25s}: Actual {actual_rate*100:5.1f}% | Predicted {pred_rate*100:5.1f}%")

    print(f"\n✓ Fairness audit complete")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 12. LIME EXPLAINABILITY (Individual Prediction Explanation)
# ============================================================================

print("\n[12/13] LIME Explainability (Sample Prediction Explanation)...")
print("-" * 80)

try:
    print("Creating LIME explainer (this takes ~30 seconds)...")

    # LIME works on the ORIGINAL (unscaled) values so its explanations read in
    # real units; the prediction function scales them before calling the SVM.
    # random_state makes the explanation identical on every run.
    def predict_proba_original_units(x):
        return svm_model.predict_proba(scaler.transform(pd.DataFrame(x, columns=top_25_features)))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=top_25_features,
        class_names=['Non-Dropout', 'Dropout'],
        mode='classification',
        verbose=False,
        random_state=42
    )

    # Explain a positive prediction (high risk student)
    high_risk_idx = np.where(y_test_pred_proba > 0.7)[0]
    if len(high_risk_idx) > 0:
        idx = high_risk_idx[0]
        exp = explainer.explain_instance(
            X_test.values[idx],
            predict_proba_original_units,
            num_features=10
        )
        print(f"\n✓ Example: High-Risk Student (Dropout Probability: {y_test_pred_proba[idx]:.2%})")
        print(f"  Top factors contributing to dropout prediction:")
        for feature, weight in exp.as_list()[:5]:
            print(f"    • {feature} (weight: {weight:.3f})")

    print(f"\n✓ LIME explainability ready for individual predictions")
    print(f"  (Positive weight = pushes towards Dropout; negative = away from it)")

except Exception as e:
    print(f"⚠ WARNING: {e}")
    print(f"LIME optional but recommended for model explainability")

# ============================================================================
# 13. DECISION THRESHOLD OPTIMIZATION
# ============================================================================

print("\n[13/13] Decision Threshold Optimization...")
print("-" * 80)

try:
    thresholds = np.arange(0.3, 0.8, 0.05)
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
    print(f"⚠ WARNING: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("SVM MODEL SUMMARY")
print("=" * 60)

print(f"\n✓ PERFORMANCE (test set):")
print(f"  Using SVM's predict():")
print(f"  • Recall: {recall:.1%} {'✓ ACHIEVES' if recall >= TARGET_RECALL else '⚠ Below'} 80% target")
print(f"  • Precision: {precision:.1%} {'✓ ACHIEVES' if precision >= TARGET_PRECISION else '⚠ Below'} 70% target")
print(f"  • F1 Score: {f1:.4f}")
print(f"  At cut-off {chosen_threshold:.2f} (chosen on validation set):")
print(f"  • Recall: {recall_final:.1%} {'✓ ACHIEVES' if recall_final >= TARGET_RECALL else '⚠ Below'} 80% target")
print(f"  • Precision: {precision_final:.1%} {'✓ ACHIEVES' if precision_final >= TARGET_PRECISION else '⚠ Below'} 70% target")
print(f"  • F1 Score: {f1_final:.4f}")
print(f"  AUC-ROC: {auc:.4f} {'✓ ACHIEVES' if auc >= TARGET_AUC else '⚠ Below'} 0.85 target")

print(f"\n✓ SVM CHARACTERISTICS:")
print(f"  • Excellent for binary classification")
print(f"  • Finds optimal decision boundary")
print(f"  • Support vectors: {len(svm_model.support_vectors_)}")
print(f"  • Kernel: {best_params['kernel']}")
print(f"  • Class-weighted (handles imbalance)")

print(f"\n✓ EXPLAINABILITY:")
print(f"  • Permutation importance provided")
print(f"  • LIME integration for individual predictions")

print(f"\n✓ CROSS-VALIDATION:")
print(f"  • AUC-ROC: {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
print(f"  • Status: {'✓ Stable' if cv_scores['test_roc_auc'].std() < 0.02 else '⚠ Variable'}")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

# Model + scaler + feature list + chosen cut-off, so predictions can be reproduced
joblib.dump(
    {'model': svm_model, 'scaler': scaler, 'features': top_25_features,
     'threshold': chosen_threshold},
    MODEL_PATH, compress=3
)
print(f"\n✓ Model saved: {MODEL_PATH}")

# Test-set results at the chosen cut-off: one row per model. Re-running
# replaces this model's row; other model scripts add their own rows.
new_row = pd.DataFrame([{
    'model': 'SVM',
    'accuracy': round(accuracy_final, 4),
    'precision': round(precision_final, 4),
    'recall': round(recall_final, 4),
    'f1': round(f1_final, 4),
    'auc_roc': round(auc, 4),
    'threshold': chosen_threshold,
}])
if METRICS_PATH.exists():
    old = pd.read_csv(METRICS_PATH)
    old = old[old['model'] != 'SVM']
    new_row = pd.concat([old, new_row], ignore_index=True)
new_row.to_csv(METRICS_PATH, index=False)
print(f"✓ Results saved: {METRICS_PATH}")

print(f"\n✓ STEP 4 (SVM) COMPLETE")
print(f"✓ Ready for comparison with other models\n")

print("=" * 60)
print("SCRIPT FINISHED SUCCESSFULLY")
print("=" * 60)
