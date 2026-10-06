#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CAPSTONE PROJECT - SECTION 2: DATA QUALITY AND INITIAL UNDERSTANDING
# Dataset : Predict Students' Dropout and Academic Success
# Run     : from the project folder, either
#             python run_all.py                    (whole pipeline)
#             python src/01_data_quality_eda.py    (this step only)
#           or open this file in Spyder and press Run.
# Outputs : reports/figures/section2_eda/  (8 charts + Excel tables)
# =============================================================================

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Show charts on screen only when running inside Jupyter or Spyder.
# When run as a plain script (e.g. via run_all.py), charts are saved to
# files without opening windows, so the script never pauses.
try:
    get_ipython  # exists only inside Jupyter / Spyder
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except ImportError:
    display = print

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:,.2f}".format)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# -----------------------------------------------------------------------------
# 0. PATHS AND HELPERS
# -----------------------------------------------------------------------------
# Paths are built from this file's location, so the code runs on any computer.
# This file lives in <project>/src/, so the project folder is one level up.
ROOT_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = str(ROOT_DIR / "data" / "raw" / "students_dropout_academic_success.csv")
OUTPUT_DIR = str(ROOT_DIR / "reports" / "figures" / "section2_eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLORS = {"Graduate": "#2E8B57", "Dropout": "#C0392B", "Enrolled": "#F39C12"}


def header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()   # displays in Jupyter/Spyder; does nothing when run as a script
    plt.close()  # frees memory once the chart is saved


def find_col(*keywords, exclude=None):
    """Find the first column whose name contains all keywords (case-insensitive)."""
    for c in df.columns:
        lc = c.lower()
        if all(k.lower() in lc for k in keywords):
            if exclude and exclude.lower() in lc:
                continue
            return c
    return None


# -----------------------------------------------------------------------------
# 1. LOAD DATA (auto-detects comma or semicolon delimiter)
# -----------------------------------------------------------------------------
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"File not found:\n{FILE_PATH}\n"
        "Make sure the dataset is in the data/raw folder and is named "
        "'students_dropout_academic_success.csv' (no space before '.csv')."
    )

df = None
for enc in ("utf-8-sig", "latin-1"):
    try:
        df = pd.read_csv(FILE_PATH, sep=None, engine="python", encoding=enc)
        break
    except UnicodeDecodeError:
        continue

# Clean column names (removes stray tabs / double spaces found in some versions)
df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

target_col = next((c for c in df.columns if c.lower() == "target"), df.columns[-1])
df[target_col] = df[target_col].astype(str).str.strip()

print(f"Dataset loaded successfully from:\n{FILE_PATH}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
display(df.head())

# -----------------------------------------------------------------------------
# 2. DATA QUALITY SUMMARY
# -----------------------------------------------------------------------------
header("2. DATA QUALITY AND INITIAL UNDERSTANDING")

n_rows, n_cols = df.shape
total_cells = n_rows * n_cols
missing_total = int(df.isna().sum().sum())
missing_pct = missing_total / total_cells * 100
dup_total = int(df.duplicated().sum())
dup_pct = dup_total / n_rows * 100
completeness = (1 - missing_total / total_cells) * 100

quality_table = pd.DataFrame({
    "Metric": ["Total Records", "Total Columns", "Missing Values",
               "Duplicate Records", "Data Completeness"],
    "Value": [f"{n_rows:,}",
              f"{n_cols}",
              f"{missing_total:,} ({missing_pct:.1f}%)",
              f"{dup_total:,} ({dup_pct:.1f}%)",
              f"{completeness:.1f}%".replace(".0%", "%")]
})
print(quality_table.to_string(index=False))

# Column-level profile
header("2.1 Column Profile (data type, completeness, cardinality)")
column_profile = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Non-Null": df.notna().sum(),
    "Missing": df.isna().sum(),
    "Missing (%)": df.isna().mean() * 100,
    "Unique Values": df.nunique(),
})
column_profile.index.name = "Column"
display(column_profile)

print("\nData type counts:")
print(df.dtypes.astype(str).value_counts().to_string())

# --- Visual 1: Data quality overview ---
fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})

sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap=["#DDEEFF", "#C0392B"], ax=axes[0])
axes[0].set_title("Missing-Value Map\n(red = missing, light blue = present)", fontsize=11)
axes[0].tick_params(axis="x", labelsize=7, rotation=90)

comp = (df.notna().mean() * 100).sort_values()
axes[1].barh(comp.index, comp.values, color="#2E86C1")
axes[1].set_xlim(0, 105)
axes[1].set_title("Completeness per Column (%)", fontsize=11)
axes[1].tick_params(axis="y", labelsize=7)
axes[1].axvline(100, color="green", ls="--", lw=1)

summary_vals = [n_rows, n_cols, missing_total, dup_total]
summary_lbls = ["Records", "Columns", "Missing\ncells", "Duplicate\nrows"]
bars = axes[2].bar(summary_lbls, summary_vals, color=["#2E86C1", "#5DADE2", "#C0392B", "#E67E22"])
for b, v in zip(bars, summary_vals):
    axes[2].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}",
                 ha="center", va="bottom", fontweight="bold")
axes[2].set_title(f"Quality Summary\nCompleteness = {completeness:.1f}%", fontsize=11)
axes[2].set_yscale("symlog")
fig.suptitle("Figure 2.1 - Data Quality Overview", fontsize=14, fontweight="bold")
save_fig("Fig2_1_data_quality_overview.png")

# -----------------------------------------------------------------------------
# 3. TARGET VARIABLE
# -----------------------------------------------------------------------------
header(f"2.2 Target Variable: {target_col}")

target_desc = {
    "Graduate": "Successfully completed degree program",
    "Dropout": "Discontinued enrollment",
    "Enrolled": "Still enrolled, not yet graduated",
}
counts = df[target_col].value_counts()
order = [c for c in ["Graduate", "Dropout", "Enrolled"] if c in counts.index] + \
        [c for c in counts.index if c not in ["Graduate", "Dropout", "Enrolled"]]
counts = counts.reindex(order)
pcts = counts / counts.sum() * 100
palette = {k: TARGET_COLORS.get(k, "#7F8C8D") for k in order}

target_table = pd.DataFrame({
    "Class": order,
    "Records": counts.values,
    "Percent (%)": pcts.round(1).values,
    "Description": [target_desc.get(k, "") for k in order],
})
print(f"Data Type: {'String (Categorical)' if df[target_col].dtype == object else df[target_col].dtype}")
print("Allowed Values:")
for _, r in target_table.iterrows():
    print(f"  \u2022 {r['Class']} - {r['Records']:,} records ({r['Percent (%)']:.1f}%) - {r['Description']}")
display(target_table)

imbalance_ratio = counts.max() / counts.min()
print(f"\nClass imbalance ratio (largest / smallest class): {imbalance_ratio:.2f} : 1")
if "Dropout" in counts.index:
    d = counts["Dropout"]
    print(f"Binary dropout view: Dropout = {d:,} ({d / n_rows * 100:.1f}%), "
          f"Non-dropout = {n_rows - d:,} ({(n_rows - d) / n_rows * 100:.1f}%)")

# --- Visual 2: Target distribution ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
bars = axes[0].bar(order, counts.values, color=[palette[k] for k in order])
for b, c, p in zip(bars, counts.values, pcts.values):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c:,}\n({p:.1f}%)",
                 ha="center", va="bottom", fontweight="bold")
axes[0].set_title("Class Counts")
axes[0].set_ylabel("Number of students")
axes[0].set_ylim(0, counts.max() * 1.18)

axes[1].pie(counts.values, labels=order, autopct="%1.1f%%", startangle=90,
            colors=[palette[k] for k in order], wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Proportions")

if "Dropout" in counts.index:
    bin_counts = pd.Series({"Dropout": counts["Dropout"], "Non-dropout": n_rows - counts["Dropout"]})
    axes[2].pie(bin_counts.values, labels=bin_counts.index, autopct="%1.1f%%", startangle=90,
                colors=["#C0392B", "#95A5A6"], wedgeprops={"width": 0.45, "edgecolor": "white"})
    axes[2].set_title("Focused Dropout-Risk View (binary)")
else:
    axes[2].set_visible(False)
fig.suptitle(f"Figure 2.2 - Distribution of Target Variable ({target_col})", fontsize=14, fontweight="bold")
save_fig("Fig2_2_target_distribution.png")

# -----------------------------------------------------------------------------
# 4. KEY NUMERIC PREDICTORS
# -----------------------------------------------------------------------------
header("2.3 Distribution of Key Numeric Predictors")

numeric_keys = [
    ("admission grade",), ("previous qualification", "grade"), ("age at enrollment",),
    ("1st sem", "enrolled"), ("1st sem", "approved"), ("1st sem", "grade"),
    ("2nd sem", "enrolled"), ("2nd sem", "approved"), ("2nd sem", "grade"),
    ("unemployment",), ("inflation",), ("gdp",),
]
num_cols = [c for c in (find_col(*k) for k in numeric_keys) if c is not None]
num_cols = list(dict.fromkeys(num_cols))  # remove any duplicates, keep order
print(f"Key numeric predictors analysed ({len(num_cols)}):")
for c in num_cols:
    print(f"  - {c}")


def expected_range(col):
    lc = col.lower()
    if "admission grade" in lc or "qualification (grade)" in lc:
        return (0, 200)
    if "age at" in lc:
        return (15, 80)
    if "sem" in lc and "grade" in lc:
        return (0, 20)
    if "sem" in lc:
        return (0, None)
    if "unemployment" in lc:
        return (0, 100)
    if "inflation" in lc:
        return (-10, 50)
    if "gdp" in lc:
        return (-20, 20)
    return (None, None)


def skew_label(s):
    a = abs(s)
    shape = "approx. symmetric" if a < 0.5 else ("moderately skewed" if a < 1 else "highly skewed")
    if a >= 0.5:
        shape += " (right)" if s > 0 else " (left)"
    return shape


rows, plaus_rows = [], []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce").dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < lo_f) | (s > hi_f)).sum())
    rows.append({
        "Predictor": c, "Min": s.min(), "Q1": q1, "Median": s.median(), "Mean": s.mean(),
        "Q3": q3, "Max": s.max(), "Std Dev": s.std(), "IQR": iqr,
        "CV (%)": s.std() / s.mean() * 100 if s.mean() != 0 else np.nan,
        "Skewness": s.skew(), "Kurtosis": s.kurt(), "Shape": skew_label(s.skew()),
        "Zeros (%)": (s == 0).mean() * 100,
        "Outliers (IQR)": n_out, "Outliers (%)": n_out / len(s) * 100,
    })
    lo, hi = expected_range(c)
    below = int((s < lo).sum()) if lo is not None else 0
    above = int((s > hi).sum()) if hi is not None else 0
    plaus_rows.append({
        "Predictor": c,
        "Expected Range": f"{lo if lo is not None else '-inf'} to {hi if hi is not None else '+inf'}",
        "Observed Range": f"{s.min():,.2f} to {s.max():,.2f}",
        "Values Outside Range": below + above,
        "Plausible?": "Yes" if below + above == 0 else "Check",
    })

numeric_summary = pd.DataFrame(rows).set_index("Predictor")
plausibility = pd.DataFrame(plaus_rows).set_index("Predictor")

print("\nDescriptive statistics (location, spread, shape, outliers):")
display(numeric_summary)
print("\nPlausibility check against expected value ranges:")
display(plausibility)

# Zero-grade note (students with no approved units receive a grade of 0)
for c in num_cols:
    if "sem" in c.lower() and "grade" in c.lower():
        z = int((df[c] == 0).sum())
        print(f"Note: {c} has {z:,} zero values ({z / n_rows * 100:.1f}%) "
              f"- typically students with no evaluated/approved units, not data errors.")

# --- Visual 3: Histograms with KDE, mean and median ---
ncols_grid = 3
nrows_grid = int(np.ceil(len(num_cols) / ncols_grid))
fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(18, 4.2 * nrows_grid))
axes = np.array(axes).flatten()
for ax, c in zip(axes, num_cols):
    sns.histplot(df[c], kde=True, bins=30, color="#2E86C1", ax=ax)
    ax.axvline(df[c].mean(), color="red", ls="--", lw=1.5, label=f"Mean = {df[c].mean():.2f}")
    ax.axvline(df[c].median(), color="green", ls="-", lw=1.5, label=f"Median = {df[c].median():.2f}")
    ax.set_title(f"{c}\nSkewness = {df[c].skew():.2f}", fontsize=10)
    ax.set_xlabel("")
    ax.legend(fontsize=8)
for ax in axes[len(num_cols):]:
    ax.set_visible(False)
fig.suptitle("Figure 2.3 - Distribution of Key Numeric Predictors", fontsize=14, fontweight="bold")
save_fig("Fig2_3_numeric_histograms.png")

# --- Visual 4: Boxplots (outlier detection) ---
fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(18, 3.2 * nrows_grid))
axes = np.array(axes).flatten()
for ax, c in zip(axes, num_cols):
    sns.boxplot(x=df[c], color="#AED6F1", ax=ax,
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#C0392B"})
    ax.set_title(f"{c}\nIQR outliers: {numeric_summary.loc[c, 'Outliers (IQR)']:,} "
                 f"({numeric_summary.loc[c, 'Outliers (%)']:.1f}%)", fontsize=10)
    ax.set_xlabel("")
for ax in axes[len(num_cols):]:
    ax.set_visible(False)
fig.suptitle("Figure 2.4 - Boxplots of Key Numeric Predictors (IQR Outliers)", fontsize=14, fontweight="bold")
save_fig("Fig2_4_numeric_boxplots.png")

# --- Visual 5: Numeric predictors by target class ---
fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(18, 4.2 * nrows_grid))
axes = np.array(axes).flatten()
for ax, c in zip(axes, num_cols):
    sns.boxplot(data=df, x=target_col, y=c, order=order, palette=palette, ax=ax, fliersize=2)
    ax.set_title(c, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
for ax in axes[len(num_cols):]:
    ax.set_visible(False)
fig.suptitle("Figure 2.5 - Key Numeric Predictors by Academic Outcome", fontsize=14, fontweight="bold")
save_fig("Fig2_5_numeric_by_target.png")

# --- Visual 6: Correlation heatmap ---
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8})
plt.title("Figure 2.6 - Correlation Matrix of Key Numeric Predictors", fontsize=14, fontweight="bold")
save_fig("Fig2_6_numeric_correlation.png")

# -----------------------------------------------------------------------------
# 5. KEY CATEGORICAL PREDICTORS
# -----------------------------------------------------------------------------
header("2.4 Frequency Distribution of Key Categorical Predictors")

YES_NO = {1: "Yes", 0: "No"}
LABEL_MAPS = {
    "gender": {1: "Male", 0: "Female"},
    "daytime/evening attendance": {1: "Daytime", 0: "Evening"},
    "marital status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
                       5: "Facto union", 6: "Legally separated"},
    "scholarship holder": YES_NO, "debtor": YES_NO, "tuition fees up to date": YES_NO,
    "displaced": YES_NO, "educational special needs": YES_NO, "international": YES_NO,
    "course": {33: "Biofuel Production Tech", 171: "Animation & Multimedia Design",
               8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
               9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
               9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
               9556: "Oral Hygiene", 9670: "Advertising & Marketing Mgmt",
               9773: "Journalism & Communication", 9853: "Basic Education",
               9991: "Management (evening)"},
}

categorical_keys = [
    ("gender",), ("marital status",), ("daytime",), ("scholarship holder",), ("debtor",),
    ("tuition fees",), ("displaced",), ("educational special needs",), ("international",),
    ("course",), ("application mode",), ("previous qualification",),
]
cat_cols = []
for k in categorical_keys:
    c = find_col(*k, exclude="grade")
    if c is not None and c != target_col and c not in cat_cols:
        cat_cols.append(c)


def labelled(col):
    """Return the column with readable labels where the coding is recognised."""
    s = df[col]
    mapping = next((m for key, m in LABEL_MAPS.items() if key in col.lower()), None)
    if mapping is not None and set(pd.unique(s.dropna())).issubset(mapping.keys()):
        return s.map(mapping).astype(str)
    return s.astype(str)


def collapse(s, top=8):
    vc = s.value_counts()
    if len(vc) > top:
        return s.where(s.isin(vc.index[:top]), "Other")
    return s


freq_tables = []
cat_labelled = {}
for c in cat_cols:
    s = labelled(c)
    cat_labelled[c] = s
    vc = s.value_counts()
    ct = pd.crosstab(s, df[target_col], normalize="index").reindex(columns=order) * 100
    tbl = pd.DataFrame({"Count": vc, "Percent (%)": vc / n_rows * 100})
    tbl = tbl.join(ct.add_suffix(" rate (%)"))
    tbl.index.name = "Category"
    print(f"\n--- {c}  ({s.nunique()} categories) ---")
    display(tbl)
    t = tbl.reset_index()
    t.insert(0, "Variable", c)
    freq_tables.append(t)

categorical_summary = pd.concat(freq_tables, ignore_index=True)

# --- Visual 7: Frequency bar charts ---
ncols_grid = 3
nrows_grid = int(np.ceil(len(cat_cols) / ncols_grid))
fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(20, 4.5 * nrows_grid))
axes = np.array(axes).flatten()
for ax, c in zip(axes, cat_cols):
    s = collapse(cat_labelled[c])
    vc = s.value_counts()
    labels = [str(x)[:28] for x in vc.index]
    ax.barh(labels, vc.values, color="#5DADE2")
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v:,} ({v / n_rows * 100:.1f}%)", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, vc.max() * 1.35)
    ax.set_title(c, fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
for ax in axes[len(cat_cols):]:
    ax.set_visible(False)
fig.suptitle("Figure 2.7 - Frequency Distribution of Key Categorical Predictors "
             "(top 8 categories shown; rest grouped as 'Other')", fontsize=14, fontweight="bold")
save_fig("Fig2_7_categorical_frequencies.png")

# --- Visual 8: Outcome composition within each category ---
fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(20, 4.5 * nrows_grid))
axes = np.array(axes).flatten()
for ax, c in zip(axes, cat_cols):
    s = collapse(cat_labelled[c])
    vc = s.value_counts()
    ct = pd.crosstab(s, df[target_col], normalize="index").reindex(index=vc.index, columns=order) * 100
    ct.index = [str(x)[:28] for x in ct.index]
    ct.plot(kind="barh", stacked=True, ax=ax, color=[palette[k] for k in order], legend=False, width=0.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of students")
    ax.set_title(c, fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
for ax in axes[len(cat_cols):]:
    ax.set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in order]
fig.legend(handles, order, loc="upper right", ncol=len(order), fontsize=11)
fig.suptitle("Figure 2.8 - Academic Outcome Composition by Categorical Predictor",
             fontsize=14, fontweight="bold")
save_fig("Fig2_8_categorical_by_target.png")

# -----------------------------------------------------------------------------
# 6. AUTO-GENERATED KEY OBSERVATIONS
# -----------------------------------------------------------------------------
header("2.5 Key Observations (supporting data preparation decisions)")

obs = []
obs.append(f"Dataset has {n_rows:,} records and {n_cols} columns; {missing_total:,} missing cells "
           f"({missing_pct:.1f}%) and {dup_total:,} duplicate rows ({dup_pct:.1f}%) - completeness "
           f"{completeness:.1f}%.")
obs.append(f"Target is imbalanced (ratio {imbalance_ratio:.2f}:1); '{counts.idxmin()}' is the minority "
           f"class - consider stratified splitting and class weighting / resampling.")

top_skew = numeric_summary["Skewness"].abs().sort_values(ascending=False).head(3)
obs.append("Most skewed numeric predictors: " +
           ", ".join(f"{c} ({numeric_summary.loc[c, 'Skewness']:.2f})" for c in top_skew.index) +
           " - candidates for transformation or robust scaling.")

top_out = numeric_summary["Outliers (%)"].sort_values(ascending=False).head(3)
obs.append("Highest IQR-outlier shares: " +
           ", ".join(f"{c} ({v:.1f}%)" for c, v in top_out.items()) +
           " - review before capping; many reflect genuine zero-grade/zero-unit students.")

n_implausible = int(plausibility["Values Outside Range"].sum())
obs.append("All key numeric values fall within expected ranges." if n_implausible == 0
           else f"{n_implausible:,} values fall outside expected ranges - see plausibility table.")

for c in cat_cols:
    share = cat_labelled[c].value_counts(normalize=True).iloc[0] * 100
    if share >= 90:
        obs.append(f"'{c}' is near-constant ({share:.1f}% in one category) - low information value.")

if "Dropout" in order:
    best = []
    for c in cat_cols:
        s = cat_labelled[c]
        vc = s.value_counts()
        rate = pd.crosstab(s, df[target_col], normalize="index")["Dropout"] * 100
        rate = rate[vc.reindex(rate.index) >= 30]
        if len(rate):
            best.append((c, rate.idxmax(), rate.max()))
    best = sorted(best, key=lambda x: x[2], reverse=True)[:3]
    overall = counts["Dropout"] / n_rows * 100
    obs.append(f"Highest dropout-rate categories (n >= 30; overall rate {overall:.1f}%): " +
               "; ".join(f"{c} = {cat} ({r:.1f}%)" for c, cat, r in best) + ".")

for i, o in enumerate(obs, 1):
    print(f"{i}. {o}")

# -----------------------------------------------------------------------------
# 7. EXPORT TABLES TO EXCEL
# -----------------------------------------------------------------------------
excel_path = os.path.join(OUTPUT_DIR, "Section2_Data_Quality_Tables.xlsx")
try:
    with pd.ExcelWriter(excel_path) as writer:
        quality_table.to_excel(writer, sheet_name="Data_Quality", index=False)
        column_profile.to_excel(writer, sheet_name="Column_Profile")
        target_table.to_excel(writer, sheet_name="Target_Distribution", index=False)
        numeric_summary.to_excel(writer, sheet_name="Numeric_Summary")
        plausibility.to_excel(writer, sheet_name="Plausibility_Check")
        categorical_summary.to_excel(writer, sheet_name="Categorical_Frequencies", index=False)
        df.describe().T.to_excel(writer, sheet_name="All_Numeric_Describe")
        pd.DataFrame({"Observation": obs}).to_excel(writer, sheet_name="Key_Observations", index=False)
    print(f"\nTables exported to: {excel_path}")
except Exception as e:
    print(f"\nExcel export skipped ({e}). Tables are still displayed above.")

print(f"All figures saved to: {OUTPUT_DIR}")
print("\nSection 2 analysis complete.")
