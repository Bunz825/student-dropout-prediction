#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - STEP 3: EDA & FEATURE ENGINEERING
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py                          (whole pipeline)
#             python src/02_eda_feature_engineering.py   (this step only)
#           or open this file in Spyder and press Run.
# Output  : data/processed/feature_list.txt  (25 selected features,
#           in Random Forest importance order)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
# Paths are built from this file's location, so the code runs on any computer.
# This file lives in <project>/src/, so the project folder is one level up.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "students_dropout_academic_success.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_LIST_PATH = PROCESSED_DIR / "feature_list.txt"

# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================

print("=" * 80)
print("STEP 3: EDA & FEATURE ENGINEERING")
print("=" * 80)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"\n✓ Dataset loaded: {df.shape[0]} records × {df.shape[1]} features")

# Create binary target
df['is_dropout'] = (df['target'] == 'Dropout').astype(int)

# ============================================================================
# 1.1 MISSING VALUES ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
print("1. DATA CLEANING & QUALITY ASSESSMENT")
print("=" * 80)

print("\n1.1 Missing Values")
missing_count = df.isnull().sum().sum()
print(f"Total missing values: {missing_count}")
print(f"Data completeness: {(1 - missing_count / (df.shape[0] * df.shape[1])) * 100:.1f}%")

# ============================================================================
# 1.2 DUPLICATE RECORDS
# ============================================================================

print("\n1.2 Duplicate Records")
duplicates = df.duplicated().sum()
print(f"Exact duplicates: {duplicates}")

# ============================================================================
# 1.3 OUTLIER DETECTION
# ============================================================================

print("\n1.3 Outlier Detection (IQR Method)")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('is_dropout')

outlier_counts = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    if outliers > 0:
        outlier_counts[col] = outliers

print(f"Features with outliers: {len(outlier_counts)}")
print(f"Total outliers: {sum(outlier_counts.values())}")
print("Recommendation: RETAIN outliers (real data, not errors)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("2. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ============================================================================
# 2.1 TARGET DISTRIBUTION
# ============================================================================

print("\n2.1 Target Variable Distribution")
print(df['target'].value_counts())
print(f"\nBinary target (Dropout=1):")
print(f"  Dropout: {df['is_dropout'].sum()} ({df['is_dropout'].mean()*100:.1f}%)")
print(f"  Non-Dropout: {(1-df['is_dropout']).sum()} ({(1-df['is_dropout']).mean()*100:.1f}%)")
print(f"  Imbalance ratio: 2.11:1")

# ============================================================================
# 2.2 FEATURE CORRELATIONS WITH DROPOUT
# ============================================================================

print("\n2.2 Top Predictive Features (by Correlation)")
correlations = df[numeric_cols + ['is_dropout']].corr()['is_dropout'].drop('is_dropout')
top_corr = correlations.abs().sort_values(ascending=False).head(15)
print("Top 15 features by absolute correlation:")
for i, (feat, corr) in enumerate(top_corr.items(), 1):
    print(f"  {i:2d}. {feat:45s} {corr:+.4f}")

# ============================================================================
# 2.3 DEMOGRAPHIC DISPARITIES
# ============================================================================

print("\n2.3 Demographic Disparities (Equity Analysis)")
print("\nGender disparities:")
print(f"  Gender 0: {df[df['Gender']==0]['is_dropout'].mean()*100:.1f}% dropout")
print(f"  Gender 1: {df[df['Gender']==1]['is_dropout'].mean()*100:.1f}% dropout")

print("\nAge disparities:")
print(f"  Age ≤25: {df[df['Age at enrollment']<=25]['is_dropout'].mean()*100:.1f}% dropout")
print(f"  Age >25: {df[df['Age at enrollment']>25]['is_dropout'].mean()*100:.1f}% dropout")

print("\nFinancial status:")
print(f"  Scholarship: {df[df['Scholarship holder']==1]['is_dropout'].mean()*100:.1f}% dropout")
print(f"  No scholarship: {df[df['Scholarship holder']==0]['is_dropout'].mean()*100:.1f}% dropout")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("3. FEATURE ENGINEERING")
print("=" * 80)

df_eng = df.copy()

# ============================================================================
# 3.1 ACADEMIC ENGAGEMENT FEATURES (9)
# ============================================================================

print("\n3.1 Creating Academic Engagement Features (9)...")

df_eng['approval_rate_1st'] = df_eng['Curricular units 1st sem (approved)'] / (df_eng['Curricular units 1st sem (enrolled)'] + 1)
df_eng['approval_rate_2nd'] = df_eng['Curricular units 2nd sem (approved)'] / (df_eng['Curricular units 2nd sem (enrolled)'] + 1)
df_eng['overall_approval_rate'] = (df_eng['Curricular units 1st sem (approved)'] + df_eng['Curricular units 2nd sem (approved)']) / (df_eng['Curricular units 1st sem (enrolled)'] + df_eng['Curricular units 2nd sem (enrolled)'] + 1)
df_eng['engagement_1st'] = df_eng['Curricular units 1st sem (evaluations)'] / (df_eng['Curricular units 1st sem (enrolled)'] + 1)
df_eng['engagement_2nd'] = df_eng['Curricular units 2nd sem (evaluations)'] / (df_eng['Curricular units 2nd sem (enrolled)'] + 1)
df_eng['avg_grade_1st_2nd'] = (df_eng['Curricular units 1st sem (grade)'] + df_eng['Curricular units 2nd sem (grade)']) / 2
df_eng['grade_decline'] = df_eng['Curricular units 1st sem (grade)'] - df_eng['Curricular units 2nd sem (grade)']
df_eng['total_approved'] = df_eng['Curricular units 1st sem (approved)'] + df_eng['Curricular units 2nd sem (approved)']
df_eng['disengagement_score'] = (df_eng['Curricular units 1st sem (without evaluations)'] * 0.5 + df_eng['Curricular units 2nd sem (without evaluations)'] * 0.3)

# ============================================================================
# 3.2 FINANCIAL HARDSHIP (1)
# ============================================================================

print("3.2 Creating Financial Hardship Indicator (1)...")
df_eng['financial_hardship'] = (df_eng['Debtor'] * 2) + ((df_eng['Tuition fees up to date'] == 0).astype(int) * 1)

# ============================================================================
# 3.3 DEMOGRAPHIC INTERACTIONS (5)
# ============================================================================

print("3.3 Creating Demographic Interaction Features (5)...")

df_eng['parental_education_avg'] = (df_eng['Mother\'s qualification'] + df_eng['Father\'s qualification']) / 2
df_eng['parental_occupation_avg'] = (df_eng['Mother\'s occupation'] + df_eng['Father\'s occupation']) / 2
df_eng['age_non_traditional'] = (df_eng['Age at enrollment'] > 25).astype(int)
df_eng['has_special_needs'] = df_eng['Educational special needs']
df_eng['displaced_student'] = df_eng['Displaced']

# ============================================================================
# 3.4 MOTIVATION SIGNALS (2)
# ============================================================================

print("3.4 Creating Motivation Signals (2)...")

df_eng['is_first_choice'] = (df_eng['Application order'] == 1).astype(int)
df_eng['admission_grade_zscore'] = (df_eng['Admission grade'] - df_eng['Admission grade'].mean()) / df_eng['Admission grade'].std()

# ============================================================================
# 3.5 CLINICAL FLAGS (5)
# ============================================================================

print("3.5 Creating Clinical Flags (5)...")

df_eng['no_attempt_1st'] = (df_eng['Curricular units 1st sem (evaluations)'] == 0).astype(int)
df_eng['no_attempt_2nd'] = (df_eng['Curricular units 2nd sem (evaluations)'] == 0).astype(int)
df_eng['dropped_mid_year'] = (df_eng['Curricular units 2nd sem (enrolled)'] == 0).astype(int)
df_eng['failed_majority_1st'] = (df_eng['Curricular units 1st sem (approved)'] < df_eng['Curricular units 1st sem (enrolled)'] / 2).astype(int)
df_eng['failed_majority_2nd'] = (df_eng['Curricular units 2nd sem (approved)'] < df_eng['Curricular units 2nd sem (enrolled)'] / 2).astype(int)

# Engineered features = columns in df_eng that were not in df
# (df already includes the is_dropout flag, so it is not counted)
engineered_features = [c for c in df_eng.columns if c not in df.columns]
print(f"\n✓ Total engineered features: {len(engineered_features)}")
print(f"✓ Final dataset shape: {df_eng.shape[0]} × {df_eng.shape[1]} features")

# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("4. FEATURE SELECTION (3 METHODS)")
print("=" * 80)

# Prepare data
numeric_all = df_eng.select_dtypes(include=[np.number]).columns.tolist()
numeric_all.remove('is_dropout')

X = df_eng[numeric_all]
y = df_eng['is_dropout']

# ============================================================================
# 4.1 FILTER METHOD
# ============================================================================

print("\n4.1 Filter Method (Univariate F-Statistic)")
selector_filter = SelectKBest(f_classif, k='all')
selector_filter.fit(X, y)
filter_scores = pd.DataFrame({
    'feature': numeric_all,
    'score': selector_filter.scores_
}).sort_values('score', ascending=False)

filter_top20 = set(filter_scores.head(20)['feature'].tolist())
print(f"Top 20 features by F-statistic:")
for i, (idx, row) in enumerate(filter_scores.head(20).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:45s} Score: {row['score']:.2f}")

# ============================================================================
# 4.2 EMBEDDED METHOD
# ============================================================================

print("\n4.2 Embedded Method (Random Forest Importance)")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X, y)

rf_importance = pd.DataFrame({
    'feature': numeric_all,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

rf_top25 = set(rf_importance.head(25)['feature'].tolist())
variance_explained = rf_importance.head(25)['importance'].sum() / rf_importance['importance'].sum() * 100

print(f"Top 25 features by Random Forest importance")
print(f"Cumulative variance: {variance_explained:.1f}%")
for i, (idx, row) in enumerate(rf_importance.head(25).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:45s} Importance: {row['importance']:.6f}")

# ============================================================================
# 4.3 WRAPPER METHOD
# ============================================================================

print("\n4.3 Wrapper Method (RFE - Recursive Feature Elimination)")
rfe = RFE(
    RandomForestClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    ),
    n_features_to_select=20
)
rfe.fit(X, y)
rfe_features = set([col for col, keep in zip(numeric_all, rfe.support_) if keep])

print(f"Top 20 features selected by RFE")
for i, feat in enumerate(sorted(rfe_features), 1):
    print(f"  {i:2d}. {feat}")

# ============================================================================
# 4.4 CONSENSUS ANALYSIS
# ============================================================================

print("\n4.4 Feature Consensus Analysis")
consensus = filter_top20 & rf_top25 & rfe_features
print(f"Consensus features (all 3 methods): {len(consensus)}")
for feat in sorted(consensus):
    print(f"  • {feat}")

# ============================================================================
# 4.5 FINAL FEATURE SELECTION
# ============================================================================

print("\n4.5 Final Feature Selection")
# Use RF top 25, kept in importance order (most important first) so the
# list is identical on every run (a set has no fixed order)
final_features = rf_importance.head(25)['feature'].tolist()
print(f"✓ Selected {len(final_features)} features (Random Forest importance)")
print(f"✓ Explains {variance_explained:.1f}% of variance")

# Save feature list
with open(FEATURE_LIST_PATH, 'w', encoding='utf-8') as f:
    for feat in final_features:
        f.write(f"{feat}\n")
print(f"✓ Feature list saved: {FEATURE_LIST_PATH}")

# ============================================================================
# 5. DIMENSIONALITY REDUCTION (PCA)
# ============================================================================

print("\n" + "=" * 80)
print("5. DIMENSIONALITY REDUCTION (PCA)")
print("=" * 80)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(X_scaled)

cumsum_var = np.cumsum(pca.explained_variance_ratio_)
n_80 = np.argmax(cumsum_var >= 0.80) + 1
n_85 = np.argmax(cumsum_var >= 0.85) + 1

print(f"\nPCA Variance Analysis")
print(f"Total features: {len(numeric_all)}")
print(f"Components for 80% variance: {n_80} (reduction: {(1 - n_80/len(numeric_all))*100:.1f}%)")
print(f"Components for 85% variance: {n_85} (reduction: {(1 - n_85/len(numeric_all))*100:.1f}%)")

print(f"\nRecommendation:")
print(f"  Tree models: Use all {len(numeric_all)} features (no PCA)")
print(f"  Linear models: Use selected {len(final_features)} features (preferred)")

# ============================================================================
# 6. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("6. DATA PREPROCESSING SUMMARY")
print("=" * 80)

print("\n6.1 Scaling Strategy")
print("  • Tree models: NO scaling (scale-invariant)")
print("  • Linear models: StandardScaler (mean=0, std=1)")
print("  • Fit on TRAINING data only (prevent leakage)")

print("\n6.2 Class Imbalance Handling")
imbalance_ratio = (df['is_dropout'] == 0).sum() / (df['is_dropout'] == 1).sum()
print(f"  • Imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"  • Recommended: Class weighting")
print(f"    - Dropout class weight: {imbalance_ratio:.2f}")
print(f"    - Non-dropout class weight: 1.0")

print("\n6.3 Train-Test Split Strategy")
train_size = int(0.64 * len(df))
val_size = int(0.16 * len(df))
test_size = len(df) - train_size - val_size
print(f"  • Train: {train_size} ({train_size/len(df)*100:.1f}%)")
print(f"  • Validation: {val_size} ({val_size/len(df)*100:.1f}%)")
print(f"  • Test: {test_size} ({test_size/len(df)*100:.1f}%)")
print(f"  • Method: Stratified split (maintain class ratio)")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3 COMPLETE")
print("=" * 80)
print("\n✓ Data cleaning and quality assessment")
print("✓ Exploratory data analysis (EDA) performed")
print("✓ 22 domain-derived features engineered")
print("✓ Feature selection (3 methods)")
print(f"✓ Final feature set: {len(final_features)} features identified")
print("✓ Dimensionality reduction (PCA) analyzed")
print("✓ Preprocessing strategy documented")
print("\n→ Ready for Step 4: Model Implementation")
