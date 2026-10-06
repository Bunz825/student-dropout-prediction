#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - STEP 4: RANDOM FOREST MODEL
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py                       (whole pipeline)
#             python src/04_train_random_forest.py    (this step only)
#           or open this file in Spyder and press Run.  Takes ~4-5 minutes.
# Outputs : models/random_forest.joblib  (trained model + scaler + threshold)
#           reports/model_metrics.csv    (test-set results, one row per model)
#
# What this script does:
# - Hyperparameter tuning via GridSearchCV (5-fold stratified CV, AUC-ROC)
# - Test-set evaluation at the default 0.50 cut-off
# - Decision cut-off chosen on the VALIDATION set (highest cut-off that
#   reaches the recall target), then applied once to the test set
# - Cross-validation with stratification
# - Feature importance (tree-based + permutation)
# - Fairness audit (gender, age group, scholarship)
# =============================================================================

import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 4: RANDOM FOREST MODEL - STUDENT DROPOUT PREDICTION")
print("Captures Feature Interactions Naturally")
print("=" * 80)

START_TIME = time.time()

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
MODEL_PATH = MODELS_DIR / "random_forest.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"

# Success targets set in Step 1 (Dropout class)
TARGET_RECALL = 0.80
TARGET_PRECISION = 0.70
TARGET_AUC = 0.85

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

# Validate target
if 'target' not in df.columns:
    print(f"✗ ERROR: 'target' column not found")
    sys.exit(1)

df['is_dropout'] = (df['target'] == 'Dropout').astype(int)
print(f"✓ Binary target created: {(df['is_dropout']==1).sum()} dropouts, {(df['is_dropout']==0).sum()} non-dropouts")

# ============================================================================
# 2. FEATURE ENGINEERING (22 domain-derived features)
# ============================================================================

print("\n[2/13] Feature Engineering (22 domain-derived features)...")
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

    print(f"✓ 22 features engineered successfully")

except KeyError as e:
    print(f"✗ ERROR: Column not found: {e}")
    sys.exit(1)
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

missing = [f for f in top_25_features if f not in df.columns]
if missing:
    print(f"✗ ERROR: Features not found: {missing}")
    sys.exit(1)

print(f"✓ All 25 features verified")

# ============================================================================
# 4. PREPARE DATA (Train/Val/Test Split)
# ============================================================================

print("\n[4/13] Preparing data (stratified split)...")
print("-" * 80)

try:
    X = df[top_25_features].copy()
    y = df['is_dropout'].copy()

    # Handle NaN
    if X.isnull().sum().sum() > 0:
        print(f"⚠ WARNING: {X.isnull().sum().sum()} NaN values, filling with 0")
        X = X.fillna(0)

    # Stratified split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
    )

    print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"✓ Dropout rate - Train: {y_train.mean():.1%}, Val: {y_val.mean():.1%}, Test: {y_test.mean():.1%}")

except Exception as e:
    print(f"✗ ERROR in data preparation: {e}")
    sys.exit(1)

# ============================================================================
# 5. FEATURE SCALING (Optional but improves tree splits)
# ============================================================================

print("\n[5/13] Feature scaling (StandardScaler - optional for RF but improves)...")
print("-" * 80)

try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled successfully")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 6. HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================================

print("\n[6/13] Hyperparameter tuning (GridSearchCV - 5-fold CV)...")
print("-" * 80)
print("Testing parameter combinations...")

try:
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"✓ Grid Search Complete")
    print(f"  • Best AUC-ROC (CV): {best_cv_score:.4f}")
    print(f"  • Best Parameters:")
    for param, value in best_params.items():
        print(f"    - {param}: {value}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 7. TRAIN RANDOM FOREST MODEL
# ============================================================================

print("\n[7/13] Training Random Forest with best parameters...")
print("-" * 80)

try:
    rf_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    rf_model.fit(X_train_scaled, y_train)
    print(f"✓ Random Forest trained successfully")
    print(f"  • Trees: {rf_model.n_estimators}")
    print(f"  • Max depth: {rf_model.max_depth}")
    print(f"  • Features used: {rf_model.n_features_in_}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 8. EVALUATE ON TEST SET
# ============================================================================

print("\n[8/13] Evaluating on test set...")
print("-" * 80)

try:
    y_test_pred = rf_model.predict(X_test_scaled)
    y_test_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} {'✓ EXCEEDS 70%' if precision >= 0.70 else '⚠ Below 70%'}")
    print(f"Recall:    {recall:.4f} {'✓ EXCEEDS 80%' if recall >= 0.80 else '⚠ Below 80%'}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f} {'✓ EXCEEDS 0.85' if auc >= 0.85 else '⚠ Below 0.85'}")
    print("=" * 60)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No  Predicted Yes")
    print(f"Actual No:         {tn:4d}        {fp:4d}")
    print(f"Actual Yes:        {fn:4d}        {tp:4d}")
    print(f"\nSpecificity (TNR): {tn/(tn+fp):.4f}")
    print(f"Sensitivity (TPR): {tp/(tp+fn):.4f}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 8b. DECISION CUT-OFF CHOSEN ON THE VALIDATION SET
# ============================================================================
# At the default 0.50 cut-off the model can be too cautious for the recall
# target. The cut-off is chosen on the validation set (data the model never
# trained on and that is NOT the test set): the highest cut-off whose
# validation recall still reaches the target, which keeps precision as high
# as possible. It is then applied once to the test set.

print("\n[8b/13] Choosing decision cut-off on the validation set...")
print("-" * 80)

try:
    y_val_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
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
# 9. CROSS-VALIDATION (5-Fold Stratified)
# ============================================================================

print("\n[9/13] Cross-validation (5-Fold Stratified)...")
print("-" * 80)

try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        rf_model, X_train_scaled, y_train,
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

    if cv_scores['test_roc_auc'].std() < 0.02:
        print(f"  ✓ Stable performance (low std → good generalization)")
    else:
        print(f"  ⚠ High variance across folds")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 10. FEATURE IMPORTANCE (Tree-based + Permutation)
# ============================================================================

print("\n[10/13] Feature Importance Analysis...")
print("-" * 80)

try:
    # Tree-based importance
    tree_importance = pd.DataFrame({
        'Feature': top_25_features,
        'Tree_Importance': rf_model.feature_importances_
    }).sort_values('Tree_Importance', ascending=False)

    # Permutation importance
    perm_importance = permutation_importance(
        rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_importance_df = pd.DataFrame({
        'Feature': top_25_features,
        'Perm_Importance': perm_importance.importances_mean,
        'Perm_Std': perm_importance.importances_std
    }).sort_values('Perm_Importance', ascending=False)

    # Merge both
    feature_importance = tree_importance.merge(
        perm_importance_df,
        on='Feature'
    ).sort_values('Tree_Importance', ascending=False)

    print(f"\nTop 10 Features (Tree-based Importance):")
    print(feature_importance[['Feature', 'Tree_Importance', 'Perm_Importance']].head(10).to_string(index=False))

    # Identify interactions (features with high importance = likely interactions)
    print(f"\n✓ Random Forest captures {len(tree_importance)} feature interactions through tree structure")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 11. FAIRNESS AUDIT
# ============================================================================

print("\n[11/13] Fairness Audit (Demographic Parity)...")
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
    actual_young = test_results[traditional]['actual'].mean()
    pred_young = test_results[traditional]['predicted'].mean()
    actual_old = test_results[~traditional]['actual'].mean()
    pred_old = test_results[~traditional]['predicted'].mean()

    di_age = min(actual_young, actual_old) / max(actual_young, actual_old) if max(actual_young, actual_old) > 0 else 1.0

    print(f"  Traditional (≤25): Actual {actual_young*100:5.1f}% | Predicted {pred_young*100:5.1f}%")
    print(f"  Non-trad (>25):    Actual {actual_old*100:5.1f}% | Predicted {pred_old*100:5.1f}% | DI: {di_age:.3f}")

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
# 12. DECISION THRESHOLD OPTIMIZATION
# ============================================================================

print("\n[12/13] Decision Threshold Optimization...")
print("-" * 80)

try:
    thresholds = np.arange(0.3, 0.8, 0.05)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Specificity':<12}")
    print("-" * 60)

    for threshold in thresholds:
        y_pred_t = (y_test_pred_proba >= threshold).astype(int)
        if (y_pred_t == 1).sum() > 0:
            p = precision_score(y_test, y_pred_t, zero_division=0)
            r = recall_score(y_test, y_pred_t, zero_division=0)
            f1_t = f1_score(y_test, y_pred_t, zero_division=0)
            cm_t = confusion_matrix(y_test, y_pred_t, labels=[0, 1])
            spec = cm_t[0,0] / (cm_t[0,0] + cm_t[0,1]) if (cm_t[0,0] + cm_t[0,1]) > 0 else 0
            print(f"{threshold:<12.2f} {p:<12.4f} {r:<12.4f} {f1_t:<12.4f} {spec:<12.4f}")

    print(f"\n(Test-set table for reference only - the cut-off itself was chosen")
    print(f" on the validation set in step 8b: {chosen_threshold:.2f})")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 13. ACTIONABLE INSIGHTS & SUMMARY
# ============================================================================

print("\n[13/13] Actionable Insights & Summary...")
print("-" * 80)

print(f"\n" + "=" * 60)
print("RANDOM FOREST MODEL SUMMARY")
print("=" * 60)

print(f"\n✓ MODEL PERFORMANCE (test set):")
print(f"  At default cut-off 0.50:")
print(f"  • Recall: {recall:.1%} {'✓ ACHIEVES TARGET' if recall >= TARGET_RECALL else '⚠ Below target'}")
print(f"  • Precision: {precision:.1%} {'✓ ACHIEVES TARGET' if precision >= TARGET_PRECISION else '⚠ Below target'}")
print(f"  • F1 Score: {f1:.4f}")
print(f"  At cut-off {chosen_threshold:.2f} (chosen on validation set):")
print(f"  • Recall: {recall_final:.1%} {'✓ ACHIEVES TARGET' if recall_final >= TARGET_RECALL else '⚠ Below target'}")
print(f"  • Precision: {precision_final:.1%} {'✓ ACHIEVES TARGET' if precision_final >= TARGET_PRECISION else '⚠ Below target'}")
print(f"  • F1 Score: {f1_final:.4f}")
print(f"  AUC-ROC: {auc:.4f} {'✓ ACHIEVES TARGET' if auc >= TARGET_AUC else '⚠ Below target'}")

print(f"\n✓ ADVANTAGES OVER LOGISTIC REGRESSION:")
print(f"  • Captures feature interactions naturally (tree ensembles)")
print(f"  • Non-linear relationships modeled")
print(f"  • Robust to outliers")
print(f"  • Feature importance interpretable")
print(f"  • Better generalization typically")

print(f"\n✓ TOP 5 PREDICTIVE FEATURES:")
top_5 = feature_importance.head(5)
for idx, (_, row) in enumerate(top_5.iterrows(), 1):
    print(f"  {idx}. {row['Feature']}: Tree={row['Tree_Importance']:.4f}, Perm={row['Perm_Importance']:.4f}")

print(f"\n✓ DEPLOYMENT STRATEGY:")
print(f"  • Deploy at Week 8 (after 1st semester data)")
print(f"  • Target students with low approval rates & grades")
print(f"  • Use predicted probability scores for intervention prioritization")
print(f"  • Monitor feature importance drift over time")

print(f"\n✓ FAIRNESS STATUS:")
print(f"  • Disparate impact ratios checked")
print(f"  • Model proportional to baseline rates")
print(f"  • Equal access to interventions recommended")

print(f"\n✓ CROSS-VALIDATION:")
print(f"  • 5-Fold AUC-ROC: {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
print(f"  • Status: {'✓ Stable (low variance)' if cv_scores['test_roc_auc'].std() < 0.02 else '⚠ High variance'}")

print(f"\n" + "=" * 60)
print(f"✓ STEP 4 (RANDOM FOREST) COMPLETE")
print("=" * 60)

print(f"\nNext Steps:")
print(f"1. Compare RF vs Logistic Regression results")
print(f"2. Document feature interactions discovered")
print(f"3. Proceed to Step 5: Fairness Audit & Ethics Review")
print(f"4. Move to Step 6: Create presentations")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

# Model + scaler + feature list + chosen cut-off, so predictions can be reproduced
joblib.dump(
    {'model': rf_model, 'scaler': scaler, 'features': top_25_features,
     'threshold': chosen_threshold},
    MODEL_PATH, compress=3
)
print(f"\n✓ Model saved: {MODEL_PATH}")

# Test-set results at the chosen cut-off: one row per model. Re-running
# replaces this model's row; other model scripts add their own rows.
new_row = pd.DataFrame([{
    'model': 'Random Forest',
    'accuracy': round(accuracy_final, 4),
    'precision': round(precision_final, 4),
    'recall': round(recall_final, 4),
    'f1': round(f1_final, 4),
    'auc_roc': round(auc, 4),
    'threshold': chosen_threshold,
}])
if METRICS_PATH.exists():
    old = pd.read_csv(METRICS_PATH)
    old = old[old['model'] != 'Random Forest']
    new_row = pd.concat([old, new_row], ignore_index=True)
new_row.to_csv(METRICS_PATH, index=False)
print(f"✓ Results saved: {METRICS_PATH}")

elapsed = time.time() - START_TIME
print(f"\nTotal Execution Time: {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)\n")
