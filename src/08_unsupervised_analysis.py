#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - UNSUPERVISED LEARNING (complements the 5 supervised models)
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py                        (whole pipeline)
#             python src/08_unsupervised_analysis.py   (this step only)
#           or open this file in Spyder and press Run.  Takes a few seconds.
# Outputs : reports/figures/unsupervised/unsupervised_pca_analysis.png
#           reports/figures/unsupervised/unsupervised_dendrogram.png
#           reports/unsupervised_cluster_profiles.csv
#
# What this script does:
# 1. Checks how many clusters the data supports (silhouette scores, K=2-6)
# 2. Segments students into 3 risk tiers (K-Means)
# 3. Visualizes the feature space (PCA)
# 4. Shows natural grouping structure (hierarchical clustering dendrogram)
# 5. Finds unusual students (Isolation Forest)
# 6. Density-based alternative clustering (DBSCAN)
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

# Show charts on screen only inside Jupyter/Spyder; as a plain script
# (e.g. via run_all.py) charts are saved to files without opening windows.
try:
    get_ipython  # exists only inside Jupyter / Spyder
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

SEED = 42

print("=" * 80)
print("UNSUPERVISED LEARNING - COMPLEMENTARY ANALYSIS")
print("=" * 80)

# ============================================================================
# PATHS - built from this file's location, so the code runs on any computer.
# This file lives in <project>/src/, so the project folder is one level up.
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "students_dropout_academic_success.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "unsupervised"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PCA_FIG_PATH = FIGURES_DIR / "unsupervised_pca_analysis.png"
DENDRO_FIG_PATH = FIGURES_DIR / "unsupervised_dendrogram.png"
PROFILES_CSV_PATH = REPORTS_DIR / "unsupervised_cluster_profiles.csv"

N_CLUSTERS = 3   # three practical risk tiers (see silhouette check in step 1)

# ============================================================================
# 0. LOAD DATA & PREPARE
# ============================================================================

print("\n[0/8] Loading and preparing data...")
print("-" * 80)

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Data loaded: {df.shape[0]:,} records × {df.shape[1]} features")
except FileNotFoundError:
    print(f"✗ ERROR: File not found at {DATA_PATH}")
    print("Make sure the dataset is in the data/raw folder and is named")
    print("'students_dropout_academic_success.csv' (no space before '.csv').")
    sys.exit(1)

# Binary target
df['is_dropout'] = (df['target'] == 'Dropout').astype(int)

# Feature engineering (same as supervised models)
df['approval_rate_1st'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + 1)
df['approval_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + 1)
df['overall_approval_rate'] = (df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']) / (df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)'] + 1)
df['engagement_1st'] = df['Curricular units 1st sem (evaluations)'] / (df['Curricular units 1st sem (enrolled)'] + 1)
df['engagement_2nd'] = df['Curricular units 2nd sem (evaluations)'] / (df['Curricular units 2nd sem (enrolled)'] + 1)
df['avg_grade_1st_2nd'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2
df['grade_decline'] = df['Curricular units 1st sem (grade)'] - df['Curricular units 2nd sem (grade)']
df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
df['disengagement_score'] = (df['Curricular units 1st sem (without evaluations)'] * 0.5 + df['Curricular units 2nd sem (without evaluations)'] * 0.3)
df['financial_hardship'] = (df['Debtor'] * 2) + ((df['Tuition fees up to date'] == 0).astype(int) * 1)
df['parental_education_avg'] = (df["Mother's qualification"] + df["Father's qualification"]) / 2
df['parental_occupation_avg'] = (df["Mother's occupation"] + df["Father's occupation"]) / 2
df['age_non_traditional'] = (df['Age at enrollment'] > 25).astype(int)
df['has_special_needs'] = df['Educational special needs']
df['displaced_student'] = df['Displaced']
df['is_first_choice'] = (df['Application order'] == 1).astype(int)
df['admission_grade_zscore'] = (df['Admission grade'] - df['Admission grade'].mean()) / (df['Admission grade'].std() + 1e-8)
df['no_attempt_1st'] = (df['Curricular units 1st sem (evaluations)'] == 0).astype(int)
df['no_attempt_2nd'] = (df['Curricular units 2nd sem (evaluations)'] == 0).astype(int)
df['dropped_mid_year'] = (df['Curricular units 2nd sem (enrolled)'] == 0).astype(int)
df['failed_majority_1st'] = (df['Curricular units 1st sem (approved)'] < df['Curricular units 1st sem (enrolled)'] / 2).astype(int)
df['failed_majority_2nd'] = (df['Curricular units 2nd sem (approved)'] < df['Curricular units 2nd sem (enrolled)'] / 2).astype(int)

# Top 25 features
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

X = df[top_25_features].copy().fillna(0)
y = df['is_dropout'].copy()

# Scale features (critical for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Data prepared: {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")

# ============================================================================
# 1. HOW MANY CLUSTERS? (silhouette check) + K-MEANS CLUSTERING
# ============================================================================

print(f"\n[1/8] Choosing K and running K-Means (K={N_CLUSTERS})...")
print("-" * 80)

try:
    # Silhouette score: how well separated the clusters are (higher = better)
    print("Silhouette scores (higher = better-separated clusters):")
    silhouette_by_k = {}
    for k in range(2, 7):
        labels_k = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X_scaled)
        silhouette_by_k[k] = silhouette_score(X_scaled, labels_k, sample_size=2000, random_state=SEED)
        print(f"  K={k}: {silhouette_by_k[k]:.3f}")
    best_k = max(silhouette_by_k, key=silhouette_by_k.get)
    print(f"\n  → Best-separated: K={best_k} (silhouette {silhouette_by_k[best_k]:.3f})")
    if best_k != N_CLUSTERS:
        print(f"  → K={N_CLUSTERS} is used anyway (silhouette {silhouette_by_k[N_CLUSTERS]:.3f}) because it gives")
        print(f"    three practical risk tiers (low / medium / high) for intervention planning.")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10, verbose=0)
    clusters = kmeans.fit_predict(X_scaled)

    print("\n✓ K-Means clustering complete")
    print("\nCluster Analysis:")

    cluster_info = []
    for cluster_id in range(N_CLUSTERS):
        mask = clusters == cluster_id
        n_students = mask.sum()
        dropout_rate = y[mask].mean()
        engagement_avg = X[mask]['engagement_2nd'].mean()
        approval_avg = X[mask]['approval_rate_2nd'].mean()
        hardship_avg = X[mask]['financial_hardship'].mean()

        cluster_info.append({
            'Cluster': cluster_id,
            'Size': n_students,
            'Dropout %': dropout_rate * 100,
            'Engagement': engagement_avg,
            'Approval %': approval_avg * 100,
            'Hardship': hardship_avg
        })

        print(f"\n  Cluster {cluster_id}: {n_students:4d} students ({n_students/len(y)*100:5.1f}%)")
        print(f"    • Dropout rate: {dropout_rate*100:5.1f}%")
        print(f"    • Avg engagement: {engagement_avg:6.2f}")
        print(f"    • Avg approval rate: {approval_avg*100:5.1f}%")
        print(f"    • Financial hardship: {hardship_avg:5.2f}")

    cluster_df = pd.DataFrame(cluster_info)

    # Cluster numbers are arbitrary labels, so rank clusters by dropout rate
    # and name the risk tiers from that ranking
    tier_names = ['Low', 'Medium', 'High'] if N_CLUSTERS == 3 else \
                 [f"Tier {i+1}" for i in range(N_CLUSTERS)]
    ranked = cluster_df.sort_values('Dropout %').reset_index(drop=True)
    risk_tier = {int(row['Cluster']): tier_names[i] for i, row in ranked.iterrows()}
    cluster_df['Risk Tier'] = cluster_df['Cluster'].map(risk_tier)

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 2. PCA VISUALIZATION
# ============================================================================

print("\n[2/8] Principal Component Analysis (PCA)...")
print("-" * 80)

try:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"✓ PCA complete")
    print(f"  • PC1 explains: {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"  • PC2 explains: {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"  • Total: {pca.explained_variance_ratio_.sum():.1%}")

    # Top features for PC1
    top_pc1_features = np.argsort(np.abs(pca.components_[0]))[-5:][::-1]
    print(f"\n  Top features in PC1:")
    for idx in top_pc1_features:
        print(f"    • {top_25_features[idx]}: {pca.components_[0][idx]:.3f}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 3. CREATE VISUALIZATIONS
# ============================================================================

print("\n[3/8] Creating PCA visualizations...")
print("-" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: PCA colored by K-Means clusters
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis',
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA: Colored by K-Means Clusters', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(alpha=0.3)

    # Plot 2: PCA colored by actual dropout
    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r',
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA: Colored by Actual Dropout', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Dropout (1=Yes, 0=No)')
    ax.grid(alpha=0.3)

    # Plot 3: Cluster profiles
    ax = axes[1, 0]
    cluster_df.set_index('Cluster')[['Dropout %', 'Approval %']].plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'])
    ax.set_title('Cluster Profiles: Dropout vs Approval Rates', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Cluster')
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # Plot 4: Variance explained
    ax = axes[1, 1]
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, 11), cumulative_variance[:10], 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('PCA: Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PCA_FIG_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ PCA visualizations saved: {PCA_FIG_PATH}")
    plt.show()   # displays in Jupyter/Spyder; does nothing as a plain script
    plt.close()

except Exception as e:
    print(f"✗ ERROR creating visualizations: {e}")

# ============================================================================
# 4. HIERARCHICAL CLUSTERING
# ============================================================================

print("\n[4/8] Hierarchical Clustering (Dendrogram)...")
print("-" * 80)

try:
    # Sample data for faster computation (seeded, so the same 500 students
    # are drawn - and the same dendrogram is produced - on every run)
    rng = np.random.default_rng(SEED)
    sample_size = min(500, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[sample_idx]

    linkage_matrix = linkage(X_sample, method='ward')

    plt.figure(figsize=(14, 7))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Clustering: Dendrogram (Truncated to 30 Clusters)',
              fontsize=12, fontweight='bold')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(DENDRO_FIG_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Dendrogram saved: {DENDRO_FIG_PATH}")
    print(f"  (Read alongside the silhouette scores in step 1 when judging K)")
    plt.show()
    plt.close()

except Exception as e:
    print(f"⚠ WARNING: {e}")

# ============================================================================
# 5. ANOMALY DETECTION (ISOLATION FOREST)
# ============================================================================

print("\n[5/8] Anomaly Detection (Isolation Forest)...")
print("-" * 80)

try:
    iso_forest = IsolationForest(contamination=0.03, random_state=SEED, n_estimators=100)
    anomalies = iso_forest.fit_predict(X_scaled)

    anomaly_mask = anomalies == -1
    n_anomalies = anomaly_mask.sum()
    dropout_rate_anomalies = y[anomaly_mask].mean()
    dropout_rate_normal = y[~anomaly_mask].mean()

    print(f"✓ Anomaly detection complete")
    print(f"  • Anomalies detected: {n_anomalies} ({n_anomalies/len(y)*100:.1f}%)")
    print(f"  • Anomaly dropout rate: {dropout_rate_anomalies*100:.1f}%")
    print(f"  • Normal dropout rate: {dropout_rate_normal*100:.1f}%")
    print(f"  • Difference: {(dropout_rate_anomalies - dropout_rate_normal)*100:+.1f} pp")

    if dropout_rate_anomalies > dropout_rate_normal:
        print(f"  → Anomalies are {dropout_rate_anomalies/dropout_rate_normal:.1f}x more likely to dropout")

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 6. CLUSTER-BASED ANALYSIS
# ============================================================================

print("\n[6/8] Cluster-Based Detailed Analysis...")
print("-" * 80)

try:
    cluster_summary = []

    for cluster_id in range(N_CLUSTERS):
        mask = clusters == cluster_id
        cluster_dropout = y[mask]
        cluster_X = X[mask]

        summary = {
            'Cluster': cluster_id,
            'Risk Tier': risk_tier[cluster_id],
            'Size': mask.sum(),
            'Dropout Rate': cluster_dropout.mean() * 100,
            'Low Approval (<50%)': (cluster_X['approval_rate_2nd'] < 0.5).mean() * 100,
            'High Hardship (>2)': (cluster_X['financial_hardship'] > 2).mean() * 100,
            'No Engagement': (cluster_X['engagement_2nd'] == 0).mean() * 100,
            'Traditional Age': ((df.loc[mask, 'Age at enrollment'] <= 25).sum() / mask.sum()) * 100
        }
        cluster_summary.append(summary)

    summary_df = pd.DataFrame(cluster_summary)
    print("\nDetailed Cluster Profiles:")
    print(summary_df.to_string(index=False))

    # Save to CSV
    summary_df.to_csv(PROFILES_CSV_PATH, index=False)
    print(f"\n✓ Cluster profiles saved: {PROFILES_CSV_PATH}")

except Exception as e:
    print(f"✗ ERROR: {e}")

# ============================================================================
# 7. DBSCAN CLUSTERING (Alternative to K-Means)
# ============================================================================

print("\n[7/8] DBSCAN Clustering (Density-Based Alternative)...")
print("-" * 80)

try:
    # eps (neighbourhood radius) set to 3.0 on the scaled features
    dbscan = DBSCAN(eps=3.0, min_samples=10)
    dbscan_clusters = dbscan.fit_predict(X_scaled)

    n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    n_noise = list(dbscan_clusters).count(-1)

    print(f"✓ DBSCAN complete")
    print(f"  • Clusters found: {n_clusters_dbscan}")
    print(f"  • Noise points: {n_noise} ({n_noise/len(dbscan_clusters)*100:.1f}%)")

    # Analyze noise points (anomalies)
    noise_mask = dbscan_clusters == -1
    if noise_mask.sum() > 0:
        noise_dropout = y[noise_mask].mean()
        print(f"  • Noise point dropout rate: {noise_dropout*100:.1f}%")

except Exception as e:
    print(f"⚠ WARNING DBSCAN: {e}")

# ============================================================================
# 8. SUMMARY & INSIGHTS
# ============================================================================

print("\n[8/8] Summary & Insights...")
print("-" * 80)

print("\n" + "=" * 80)
print("UNSUPERVISED LEARNING FINDINGS")
print("=" * 80)

print("\n✓ K-MEANS SEGMENTATION:")
print(f"  • {N_CLUSTERS} student clusters (silhouette {silhouette_by_k[N_CLUSTERS]:.3f}; "
      f"best-separated K={best_k} at {silhouette_by_k[best_k]:.3f})")
print(f"  • Cluster sizes: " + ", ".join(str(c['Size']) for c in cluster_info))
print(f"  • Dropout rates: " + ", ".join(f"{c['Dropout %']:.1f}%" for c in cluster_info))
lowest, highest = cluster_df['Dropout %'].min(), cluster_df['Dropout %'].max()
print(f"  • Dropout rate ranges from {lowest:.1f}% to {highest:.1f}% across clusters "
      f"(overall {y.mean()*100:.1f}%)")

print("\n✓ PCA VISUALIZATION:")
print(f"  • 2 components explain {pca.explained_variance_ratio_.sum():.1%} of variance")
n_80 = int(np.argmax(cumulative_variance >= 0.80)) + 1
print(f"  • {n_80} components needed to explain 80% of variance")

print("\n✓ ANOMALY DETECTION:")
print(f"  • {n_anomalies} anomalous students identified ({n_anomalies/len(y)*100:.1f}%)")
print(f"  • Anomalies show {'HIGHER' if dropout_rate_anomalies > dropout_rate_normal else 'LOWER'} dropout rate "
      f"({dropout_rate_anomalies*100:.1f}% vs {dropout_rate_normal*100:.1f}%)")
print(f"  • May need specialized interventions")

# Advice is attached to the RISK TIER (ranked by dropout rate), not to the
# arbitrary cluster number
tier_focus = {
    'Low': "Monitor engagement and financial support",
    'Medium': "Targeted academic support",
    'High': "Early intervention and intensive support",
}
print("\n✓ ACTIONABLE INSIGHTS (clusters ranked by dropout rate):")
for _, row in ranked.iterrows():
    cid = int(row['Cluster'])
    tier = risk_tier[cid]
    print(f"  • Cluster {cid}: {tier}-risk students ({row['Dropout %']:.1f}% dropout, "
          f"{row['Approval %']:.1f}% avg approval)")
    print(f"    → Focus: {tier_focus.get(tier, 'Review cluster profile')}")

print("\n✓ INTEGRATION WITH SUPERVISED MODELS:")
print(f"  • Use cluster membership as additional feature")
print(f"  • Apply cluster-specific thresholds for predictions")
print(f"  • Combine supervised predictions with cluster insights")

print("\n" + "=" * 80)
print("UNSUPERVISED LEARNING COMPLETE")
print("=" * 80)

print("\n✓ Output Files:")
print(f"  • {PCA_FIG_PATH.relative_to(ROOT)}")
print(f"  • {DENDRO_FIG_PATH.relative_to(ROOT)}")
print(f"  • {PROFILES_CSV_PATH.relative_to(ROOT)}")
