#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - STEP 4: NEURAL NETWORK MODEL
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py                       (whole pipeline)
#             python src/07_train_neural_network.py   (this step only)
#           or open this file in Spyder and press Run.
# Outputs : models/neural_network.keras         (trained network)
#           models/neural_network_extras.joblib (scaler + features + cut-off)
#           reports/model_metrics.csv           (test-set results, one row
#                                                per model)
#
# What this script does:
# - Deep neural network (3 hidden layers: 128 -> 64 -> 32)
# - Batch normalization, dropout (0.3, 0.3, 0.2) and L2 regularization
# - Early stopping on validation AUC; learning-rate reduction on plateau
# - Class weighting for imbalanced data
# - Fixed random seed so results repeat on the same computer
# - Test-set evaluation, plus a decision cut-off chosen on the VALIDATION set
#   (highest cut-off that reaches the recall target), applied once to the
#   test set
# - Cross-validation (5-fold stratified)
# - Permutation feature importance (AUC drop when a feature is shuffled)
# - LIME explanation of one high-risk student (in original units)
# - Fairness audit (gender, age group, scholarship)
# Requires: tensorflow, lime (both listed in requirements.txt)
#
# Note: neural networks can still give slightly different numbers on a
# different computer (other hardware / TensorFlow version), even with a seed.
# =============================================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow info logs

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

print("=" * 80)
print("NEURAL NETWORK MODEL - STUDENT DROPOUT PREDICTION")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# REPRODUCIBILITY - fix every source of randomness
# ============================================================================
SEED = 42
tf.keras.utils.set_random_seed(SEED)   # seeds Python, NumPy and TensorFlow
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

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
MODEL_PATH = MODELS_DIR / "neural_network.keras"
EXTRAS_PATH = MODELS_DIR / "neural_network_extras.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"

# Success targets set in Step 1 (Dropout class)
TARGET_RECALL = 0.80
TARGET_PRECISION = 0.70
TARGET_AUC = 0.85

# ============================================================================
# 1. LOAD & VALIDATE DATA
# ============================================================================

print("\n[1/14] Loading data...")
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

print("\n[2/14] Feature Engineering...")
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

print("\n[3/14] Selecting top 25 features...")
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
# 4. PREPARE DATA
# ============================================================================

print("\n[4/14] Preparing data...")
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
# 5. FEATURE SCALING (CRITICAL for Neural Networks)
# ============================================================================

print("\n[5/14] Feature scaling...")
print("-" * 80)

try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled (mean=0, std=1)")
    print(f"✓ CRITICAL: Neural networks require scaled features")
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 6. BUILD NEURAL NETWORK ARCHITECTURE
# ============================================================================

print("\n[6/14] Building neural network architecture...")
print("-" * 80)


def build_model():
    """Create and compile the network (same architecture for the main model
    and every cross-validation fold)."""
    m = keras.Sequential([
        keras.Input(shape=(len(top_25_features),)),

        # Hidden layer 1
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 2
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 3
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    # The AUC metric gets a FIXED name ('auc'), so early stopping and the
    # learning-rate schedule can always find 'val_auc'. (Without a name,
    # later models are called auc_1, auc_2, ... and early stopping silently
    # stops working.)
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.Precision(name='precision')]
    )
    return m


def make_callbacks():
    """Fresh callbacks for each training run."""
    early_stop = EarlyStopping(
        monitor='val_auc', mode='max',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc', mode='max',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=0
    )
    return [early_stop, reduce_lr]


try:
    # Calculate class weights for imbalanced data
    class_weight = {0: 1.0, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    print(f"✓ Class weights calculated: {class_weight}")

    model = build_model()

    print(f"✓ Neural network created")
    print(f"  • Architecture: 25 → 128 → 64 → 32 → 1")
    print(f"  • Batch Normalization: 3 layers")
    print(f"  • Dropout: 0.3, 0.3, 0.2")
    print(f"  • L2 Regularization: 0.001")
    print(f"  • Optimizer: Adam (lr=0.001)")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 7. TRAIN NEURAL NETWORK
# ============================================================================

print("\n[7/14] Training neural network...")
print("-" * 80)

try:
    print("Starting training...")

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        callbacks=make_callbacks(),
        verbose=0
    )

    best_epoch = int(np.argmax(history.history['val_auc'])) + 1
    print(f"✓ Neural network trained successfully")
    print(f"  • Epochs completed: {len(history.history['loss'])} (early stopping)")
    print(f"  • Best epoch (weights restored): {best_epoch}")
    print(f"  • Best validation AUC: {max(history.history['val_auc']):.4f}")

except Exception as e:
    print(f"✗ ERROR during training: {e}")
    sys.exit(1)

# ============================================================================
# 8. EVALUATE ON TEST SET
# ============================================================================

print("\n[8/14] Evaluating on test set...")
print("-" * 80)

try:
    y_test_pred_proba = model.predict(X_test_scaled, verbose=0)[:, 0]
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS (default cut-off 0.50)")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} {'✓ EXCEEDS 70%' if precision >= TARGET_PRECISION else '⚠ Below 70%'}")
    print(f"Recall:    {recall:.4f} {'✓ EXCEEDS 80%' if recall >= TARGET_RECALL else '⚠ Below 80%'}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f} {'✓ EXCEEDS 0.85' if auc >= TARGET_AUC else '⚠ Below 0.85'}")
    print("=" * 60)

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No  Predicted Yes")
    print(f"Actual No:         {tn:4d}        {fp:4d}")
    print(f"Actual Yes:        {fn:4d}        {tp:4d}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 8b. DECISION CUT-OFF CHOSEN ON THE VALIDATION SET
# ============================================================================
# Same method as the Random Forest, XGBoost and SVM scripts: the cut-off is
# chosen on the validation set (NOT the test set) as the highest cut-off whose
# validation recall still reaches the target, which keeps precision as high
# as possible. It is then applied once to the test set.
# Note: this network also uses the validation set for early stopping, so the
# validation set does double duty; the test set stays untouched either way.

print("\n[8b/14] Choosing decision cut-off on the validation set...")
print("-" * 80)

try:
    y_val_pred_proba = model.predict(X_val_scaled, verbose=0)[:, 0]
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

print("\n[9/14] Cross-validation (5-Fold Stratified)...")
print("-" * 80)

cv_auc_scores = []
cv_recall_scores = []
cv_precision_scores = []

try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
        print(f"  Fold {fold}/5...", end=' ')

        # Re-seed per fold so each fold is repeatable on its own
        tf.keras.utils.set_random_seed(SEED + fold)

        X_cv_train = X_train_scaled[train_idx]
        y_cv_train = y_train.iloc[train_idx]
        X_cv_val = X_train_scaled[val_idx]
        y_cv_val = y_train.iloc[val_idx]

        cv_model = build_model()
        cv_history = cv_model.fit(
            X_cv_train, y_cv_train,
            validation_data=(X_cv_val, y_cv_val),
            epochs=200,
            batch_size=32,
            class_weight=class_weight,
            callbacks=make_callbacks(),
            verbose=0
        )

        cv_pred_proba = cv_model.predict(X_cv_val, verbose=0)[:, 0]
        cv_pred = (cv_pred_proba >= 0.5).astype(int)

        cv_auc_scores.append(roc_auc_score(y_cv_val, cv_pred_proba))
        cv_recall_scores.append(recall_score(y_cv_val, cv_pred))
        cv_precision_scores.append(precision_score(y_cv_val, cv_pred))

        print(f"AUC={cv_auc_scores[-1]:.4f} (stopped after {len(cv_history.history['loss'])} epochs)")

    print(f"\n✓ Cross-Validation Results (5-Fold, cut-off 0.50):")
    print(f"  AUC-ROC:   {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
    print(f"  Recall:    {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}")
    print(f"  Precision: {np.mean(cv_precision_scores):.4f} ± {np.std(cv_precision_scores):.4f}")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 10. FEATURE IMPORTANCE (Permutation-based)
# ============================================================================
# Shuffle one feature at a time on the test set and measure how much the
# AUC-ROC drops. A bigger drop = the model relies more on that feature.

print("\n[10/14] Feature Importance Analysis (Permutation-based)...")
print("-" * 80)

try:
    rng = np.random.default_rng(SEED)
    n_repeats = 10
    baseline_auc = auc
    importances = []
    for j, feature in enumerate(top_25_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test_scaled.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_proba = model.predict(X_perm, verbose=0)[:, 0]
            drops.append(baseline_auc - roc_auc_score(y_test, perm_proba))
        importances.append((feature, np.mean(drops), np.std(drops)))

    perm_df = pd.DataFrame(importances, columns=['Feature', 'AUC_Drop', 'Std']) \
                .sort_values('AUC_Drop', ascending=False)

    print(f"\nTop 10 Features (mean AUC-ROC drop when shuffled, {n_repeats} repeats):")
    print(perm_df.head(10).to_string(index=False))

except Exception as e:
    print(f"⚠ WARNING in permutation importance: {e}")
    print(f"Continuing without this step...")

# ============================================================================
# 11. FAIRNESS AUDIT
# ============================================================================

print("\n[11/14] Fairness Audit...")
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
            print(f"  {gender_label:8s}: Actual {actual_rate*100:5.1f}% | Predicted {pred_rate*100:5.1f}%")

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
# 12. LIME EXPLAINABILITY
# ============================================================================

print("\n[12/14] LIME Explainability...")
print("-" * 80)

try:
    print("Creating LIME explainer...")

    # LIME works on the ORIGINAL (unscaled) values so its explanations read in
    # real units; the prediction function scales them before calling the
    # network. random_state makes the explanation identical on every run.
    def predict_proba_original_units(x):
        p = model.predict(scaler.transform(pd.DataFrame(x, columns=top_25_features)),
                          verbose=0)[:, 0]
        return np.column_stack((1 - p, p))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=top_25_features,
        class_names=['Non-Dropout', 'Dropout'],
        mode='classification',
        verbose=False,
        random_state=SEED
    )

    # Explain a high-risk prediction
    high_risk_idx = np.where(y_test_pred_proba > 0.7)[0]
    if len(high_risk_idx) > 0:
        idx = high_risk_idx[0]
        exp = explainer.explain_instance(
            X_test.values[idx],
            predict_proba_original_units,
            num_features=10
        )
        print(f"\n✓ Example: High-Risk Student (Dropout Probability: {y_test_pred_proba[idx]:.2%})")
        print(f"  Top factors contributing to dropout:")
        for feature, weight in exp.as_list()[:5]:
            print(f"    • {feature} (weight: {weight:.3f})")

    print(f"\n✓ LIME explainability ready")
    print(f"  (Positive weight = pushes towards Dropout; negative = away from it)")

except Exception as e:
    print(f"⚠ WARNING: {e}")
    print(f"LIME optional but recommended for explainability")

# ============================================================================
# 13. DECISION THRESHOLD TABLE (test set, for reference)
# ============================================================================

print("\n[13/14] Decision Threshold Table (test set)...")
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

    print(f"\n(Test-set table for reference only - the cut-off itself was chosen")
    print(f" on the validation set in step 8b: {chosen_threshold:.2f})")

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 14. SUMMARY
# ============================================================================

print("\n[14/14] Summary...")
print("-" * 80)

print("\n" + "=" * 60)
print("NEURAL NETWORK MODEL SUMMARY")
print("=" * 60)

print(f"\n✓ PERFORMANCE (test set):")
print(f"  At default cut-off 0.50:")
print(f"  • Recall: {recall:.1%} {'✓ ACHIEVES' if recall >= TARGET_RECALL else '⚠ Below'} 80% target")
print(f"  • Precision: {precision:.1%} {'✓ ACHIEVES' if precision >= TARGET_PRECISION else '⚠ Below'} 70% target")
print(f"  • F1 Score: {f1:.4f}")
print(f"  At cut-off {chosen_threshold:.2f} (chosen on validation set):")
print(f"  • Recall: {recall_final:.1%} {'✓ ACHIEVES' if recall_final >= TARGET_RECALL else '⚠ Below'} 80% target")
print(f"  • Precision: {precision_final:.1%} {'✓ ACHIEVES' if precision_final >= TARGET_PRECISION else '⚠ Below'} 70% target")
print(f"  • F1 Score: {f1_final:.4f}")
print(f"  AUC-ROC: {auc:.4f} {'✓ ACHIEVES' if auc >= TARGET_AUC else '⚠ Below'} 0.85 target")

print(f"\n✓ ARCHITECTURE:")
print(f"  • Layers: 25 → 128 → 64 → 32 → 1")
print(f"  • Batch Normalization: 3 layers")
print(f"  • Dropout: 0.3, 0.3, 0.2 (regularization)")
print(f"  • L2 Regularization: 0.001")
print(f"  • Training epochs: {len(history.history['loss'])}")

print(f"\n✓ ADVANTAGES:")
print(f"  • Deep learning captures complex non-linear patterns")
print(f"  • Multiple layer learning representations")
print(f"  • Built-in batch normalization for stability")
print(f"  • Dropout prevents overfitting")
print(f"  • Early stopping prevents training too long")

if cv_auc_scores:
    print(f"\n✓ CROSS-VALIDATION:")
    print(f"  • AUC-ROC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
    print(f"  • Status: {'✓ Stable' if np.std(cv_auc_scores) < 0.02 else '⚠ Variable'}")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

# Keras models are saved in their own format; the scaler, feature list and
# chosen cut-off are saved alongside, so predictions can be reproduced
model.save(MODEL_PATH)
joblib.dump(
    {'scaler': scaler, 'features': top_25_features, 'threshold': chosen_threshold},
    EXTRAS_PATH, compress=3
)
print(f"\n✓ Model saved: {MODEL_PATH}")
print(f"✓ Scaler/features/cut-off saved: {EXTRAS_PATH}")

# Test-set results at the chosen cut-off: one row per model. Re-running
# replaces this model's row; other model scripts add their own rows.
new_row = pd.DataFrame([{
    'model': 'Neural Network',
    'accuracy': round(accuracy_final, 4),
    'precision': round(precision_final, 4),
    'recall': round(recall_final, 4),
    'f1': round(f1_final, 4),
    'auc_roc': round(auc, 4),
    'threshold': chosen_threshold,
}])
if METRICS_PATH.exists():
    old = pd.read_csv(METRICS_PATH)
    old = old[old['model'] != 'Neural Network']
    new_row = pd.concat([old, new_row], ignore_index=True)
new_row.to_csv(METRICS_PATH, index=False)
print(f"✓ Results saved: {METRICS_PATH}")

print(f"\n✓ STEP 4 (NEURAL NETWORK) COMPLETE")
print(f"✓ Ready for comparison with other models\n")

print("=" * 60)
print("SCRIPT FINISHED SUCCESSFULLY")
print("=" * 60)
