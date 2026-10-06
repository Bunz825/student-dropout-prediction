#!/usr/bin/env python
# coding: utf-8

"""
STEP 5 - Critical Thinking: Ethical AI & Bias Auditing  (Student Dropout Capstone)
==================================================================================
Reproduces every number in the Step 5 report section (Step5_Ethical_AI_Bias_Audit.docx).

Primary model : Logistic Regression (class_weight='balanced'), all 40 features
                (36 original + 4 engineered) - its own configuration, separate
                from the Step 4 model scripts (03-07)
Comparator    : XGBoost (300 trees, depth 4)
Target        : Dropout (1) vs Graduate/Enrolled (0)
Split         : stratified 80/20, random_state = 42  (test n = 885)

HOW TO RUN (from the project folder)
  - Whole pipeline           :  python run_all.py
  - This step only           :  python src/09_bias_fairness_audit.py
  - Spyder                   :  open this file and press Run
Requires: numpy, pandas, scikit-learn, matplotlib, xgboost, shap, lime
          (all listed in requirements.txt)
Outputs: reports/figures/step5_explainability.png  and  reports/step5_results.txt
Runtime: under a minute.
"""

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
from pathlib import Path

# Paths are built from this file's location, so the code runs on any computer.
# This file lives in <project>/src/, so the project folder is one level up.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "students_dropout_academic_success.csv"
OUT_DIR = ROOT / "reports"
RS = 42                   # random seed used throughout

# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
import sys, os, warnings
warnings.filterwarnings("ignore")

# Windows consoles (cp1252) crash on some characters; force UTF-8 where possible
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def step(msg):
    print("\n" + "=" * 78 + "\n" + msg + "\n" + "=" * 78, flush=True)

step("[1/9] Loading libraries")
import numpy as np
import pandas as pd
import matplotlib
# Use a non-GUI backend only when NOT inside Jupyter (so plots still show in notebooks)
_IN_NOTEBOOK = "ipykernel" in sys.modules
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, accuracy_score)
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from xgboost import XGBClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import sklearn, xgboost, lime
for _name, _mod in [("numpy", np), ("pandas", pd), ("sklearn", sklearn), ("matplotlib", matplotlib),
                    ("xgboost", xgboost), ("shap", shap), ("lime", lime)]:
    print(f"  OK  {_name:<11} {getattr(_mod, '__version__', '')}")

_log_lines = []            # saved to reports/step5_results.txt at the end

def out(*args):
    """Print to screen AND save to reports/step5_results.txt."""
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    _log_lines.append(s)

def rnd(d, k=3):
    return {a: round(float(b), k) for a, b in d.items()}

# ============================================================================
# 2. LOAD DATA
# ============================================================================
step("[2/9] Loading data")

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\nMake sure it is in the data/raw folder and is "
        f"named 'students_dropout_academic_success.csv' (no space before '.csv').")
csv_path = str(DATA_PATH)

# Outputs go to the project's reports folder
os.makedirs(OUT_DIR / "figures", exist_ok=True)
FIG_PATH = str(OUT_DIR / "figures" / "step5_explainability.png")
TXT_PATH = str(OUT_DIR / "step5_results.txt")
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]           # remove stray spaces
if "target" not in df.columns:                          # some versions use 'Target'
    _t = [c for c in df.columns if c.lower() == "target"]
    if not _t:
        raise KeyError("No 'target' column found in the CSV.")
    df = df.rename(columns={_t[0]: "target"})
df["target"] = df["target"].astype(str).str.strip()
out(f"  Loaded: {DATA_PATH.relative_to(ROOT)}")
out(f"  Shape : {df.shape}   duplicates: {df.duplicated().sum()}   missing: {df.isna().sum().sum()}")
out("  Class mix:", rnd(df["target"].value_counts(normalize=True).to_dict()))

# ============================================================================
# 3. FEATURES, SPLIT, MODELS
# ============================================================================
step("[3/9] Feature engineering, split and model training")
y = (df["target"] == "Dropout").astype(int)
X = df.drop(columns="target").copy()

e1, e2 = "Curricular units 1st sem (enrolled)", "Curricular units 2nd sem (enrolled)"
a1, a2 = "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"
X["approval_rate_1st"] = np.where(X[e1] > 0, X[a1] / X[e1].replace(0, 1), 0)
X["approval_rate_2nd"] = np.where(X[e2] > 0, X[a2] / X[e2].replace(0, 1), 0)
X["total_approved"] = X[a1] + X[a2]
X["overall_approval_rate"] = np.where((X[e1] + X[e2]) > 0,
                                      X["total_approved"] / (X[e1] + X[e2]).replace(0, 1), 0)
X = X.astype(float)                                     # avoids int/float errors in PDP

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
out(f"  Train n = {len(Xtr)}, Test n = {len(Xte)}")

def make_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced",
                                            C=1.0, random_state=RS))

lr = make_lr().fit(Xtr, ytr)
spw = (ytr == 0).sum() / (ytr == 1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=spw, random_state=RS,
                    eval_metric="logloss", n_jobs=1)
xgb.fit(Xtr, ytr)

def metrics(p, ya, t=0.5):
    yh = (p >= t).astype(int)
    return dict(auc=roc_auc_score(ya, p), rec=recall_score(ya, yh),
                prec=precision_score(ya, yh, zero_division=0),
                f1=f1_score(ya, yh), acc=accuracy_score(ya, yh))

def m(model, Xa, ya, t=0.5):
    return metrics(model.predict_proba(Xa)[:, 1], ya, t)

# ============================================================================
# 4. OVERFITTING, IMBALANCE, LEAKAGE  (Limitations table)
# ============================================================================
step("[4/9] Limitations: overfitting, class imbalance, temporal leakage")
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
for name, mod in [("LR", lr), ("XGB", xgb)]:
    cvs = cross_val_score(mod, Xtr, ytr, cv=cv, scoring="roc_auc")
    out(f"  {name} train {rnd(m(mod, Xtr, ytr))}")
    out(f"  {name} test  {rnd(m(mod, Xte, yte))}")
    out(f"  {name} 5-fold CV AUC {cvs.mean():.3f} +/- {cvs.std():.3f}")
out(f"  XGB scale_pos_weight = {spw:.2f}")

uw = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
out("  Imbalance - UNWEIGHTED LR test:", rnd(m(uw, Xte, yte)))

sem2 = [c for c in X if "2nd sem" in c] + ["approval_rate_2nd", "total_approved",
                                           "overall_approval_rate"]
sem1 = [c for c in X if "1st sem" in c] + ["approval_rate_1st"]
ablation = {"Enrolment-only": [c for c in X if c not in sem1 + sem2],
            "Enrolment+Sem1": [c for c in X if c not in sem2],
            "All (Sem1+Sem2)": list(X.columns)}
for k, cols in ablation.items():
    mm = make_lr().fit(Xtr[cols], ytr)
    out(f"  Leakage ablation - {k:<16} ({len(cols)} feats): {rnd(m(mm, Xte[cols], yte))}")

# ============================================================================
# 5. EXPLAINABILITY: SHAP (global), PDP/ICE, LIME + SHAP (local)
# ============================================================================
step("[5/9] Explainability: SHAP")

def xgb_shap_values(model, Xdata):
    """SHAP for XGBoost. Falls back to XGBoost's native SHAP (pred_contribs) if the
    installed shap and xgboost versions are incompatible (common in Anaconda)."""
    try:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(Xdata)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = float(np.ravel(ex.expected_value)[-1])
        return sv, base, "shap.TreeExplainer"
    except Exception as err:
        import xgboost as xgb_lib
        print(f"  (TreeExplainer failed: {type(err).__name__} - using XGBoost native SHAP)")
        contrib = model.get_booster().predict(xgb_lib.DMatrix(Xdata), pred_contribs=True)
        return contrib[:, :-1], float(contrib[0, -1]), "xgboost pred_contribs"

sv, base_val, shap_src = xgb_shap_values(xgb, Xte)
out(f"  SHAP source: {shap_src}")
imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values(ascending=False)
out("  XGBoost mean |SHAP| top 10:")
for f, v in imp.head(10).items():
    out(f"    {f:<40} {v:.3f}")
out("  Direction (corr of feature value with its SHAP):")
cols_list = list(Xte.columns)
for f in imp.head(8).index:
    c = np.corrcoef(Xte[f], sv[:, cols_list.index(f)])[0, 1]
    out(f"    {f:<40} {c:+.2f}")

# Linear SHAP for LR (on standardised data). For a linear model SHAP = coef*(x-mean);
# computed directly so it works with every shap version.
sc, clf = lr[0], lr[1]
Zte = sc.transform(Xte)
Ztr_mean = sc.transform(Xtr).mean(0)
lsv = (Zte - Ztr_mean) * clf.coef_[0]
limp = pd.Series(np.abs(lsv).mean(0), index=Xte.columns).sort_values(ascending=False)
out("  LR linear SHAP top 10:")
for f, v in limp.head(10).items():
    out(f"    {f:<40} {v:.3f}")
out(f"  Overlap of top-10 features (XGB vs LR): "
    f"{len(set(imp.head(10).index) & set(limp.head(10).index))}")

step("[6/9] Explainability: PDP / ICE and figure")

def pd_grid(res):
    # sklearn >= 1.3 uses 'grid_values'; older versions use 'values'
    return res["grid_values"] if "grid_values" in res else res["values"]

for f in ["approval_rate_2nd", "Age at enrollment", "Tuition fees up to date"]:
    res = partial_dependence(xgb, Xte, [f], kind="average", grid_resolution=10)
    out(f"  PDP {f}: grid {np.round(pd_grid(res)[0], 2).tolist()}")
    out(f"      avg P(dropout) {np.round(res['average'][0], 3).tolist()}")

sub = Xte.sample(150, random_state=1)
ice = partial_dependence(xgb, sub, ["approval_rate_2nd"], kind="individual",
                         grid_resolution=10)["individual"][0]
drop = ice[:, 0] - ice[:, -1]
out(f"  ICE (approval 0 -> 1): median drop {np.median(drop):.3f}, "
    f"IQR {np.percentile(drop, 25):.3f}-{np.percentile(drop, 75):.3f}, "
    f"share > 0.30 = {(drop > 0.3).mean():.2f}")

# ---- Figure 1: SHAP beeswarm + PDP/ICE ----
try:
    # SHAP's summary plot spreads its dots with random jitter; seeding NumPy
    # makes the saved image identical on every run
    np.random.seed(RS)
    fig = plt.figure(figsize=(11, 4.2))
    ax1 = fig.add_subplot(1, 3, 1)
    plt.sca(ax1)
    shap.summary_plot(sv, Xte, max_display=8, show=False, plot_size=None, color_bar=False)
    ax1.set_title("(a) SHAP summary - XGBoost", fontsize=9)
    ax1.set_xlabel("SHAP value (log-odds of Dropout)", fontsize=7)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    for ax, feat, title in [(ax2, "approval_rate_2nd", "(b) PDP + ICE: 2nd-sem approval rate"),
                            (ax3, "Age at enrollment", "(c) PDP + ICE: age at enrolment")]:
        PartialDependenceDisplay.from_estimator(
            xgb, sub, [feat], kind="both", ax=ax,
            ice_lines_kw={"alpha": 0.15, "color": "steelblue"},
            pd_line_kw={"color": "crimson", "lw": 2.5})
        ax.set_title(title, fontsize=9)
    for a in fig.axes:
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
        if a.get_legend():
            a.get_legend().remove()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    if _IN_NOTEBOOK:
        plt.show()
    plt.close("all")
    out(f"  Figure saved: {FIG_PATH}")
except Exception as err:          # never let a plotting issue stop the audit
    out(f"  WARNING - figure not created ({type(err).__name__}: {err}); numbers above are unaffected")

step("[7/9] Explainability: LIME vs SHAP for one student")
p_xgb = xgb.predict_proba(Xte)[:, 1]
cand = np.where((yte.values == 1) & (p_xgb > 0.85))[0]
idx = int(cand[0]) if len(cand) else int(np.argmax(p_xgb))
lime_exp = LimeTabularExplainer(Xtr.values, feature_names=list(Xtr.columns),
                                class_names=["Not dropout", "Dropout"],
                                discretize_continuous=True, random_state=RS)
e = lime_exp.explain_instance(Xte.values[idx], xgb.predict_proba, num_features=6)
out(f"  Student index {Xte.index[idx]}, predicted P(dropout) = {p_xgb[idx]:.3f}")
out("  LIME top reasons:")
for rule, w in e.as_list():
    out(f"    {rule:<45} {w:+.3f}")
out("  Student values:", rnd(Xte.iloc[idx][["approval_rate_2nd", "Tuition fees up to date",
                                            "Debtor", "Age at enrollment",
                                            "Scholarship holder"]].to_dict(), 2))
s_local = pd.Series(sv[idx], index=Xte.columns)
s_local = s_local.reindex(s_local.abs().sort_values(ascending=False).index).head(6)
out(f"  SHAP top reasons (base value {base_val:.3f}):")
for f, v in s_local.items():
    out(f"    {f:<45} {v:+.3f}")

# ============================================================================
# 8. FAIRNESS AUDIT
# ============================================================================
step("[8/9] Fairness audit (demographic parity, disparate impact, equalised odds)")
# ASCII labels ("<=20") avoid encoding errors on Windows consoles
def groups(Xd):
    return {"Gender": Xd["Gender"].map({0: "Female", 1: "Male"}),
            "Age": pd.cut(Xd["Age at enrollment"], [0, 20, 25, 100],
                          labels=["<=20", "21-25", ">25"]).astype(str),
            "Scholarship": Xd["Scholarship holder"].map({0: "No", 1: "Yes"}),
            "Debtor": Xd["Debtor"].map({0: "No", 1: "Yes"}),
            "International": Xd["International"].map({0: "Domestic", 1: "International"})}
REF = {"Gender": "Female", "Age": "<=20", "Scholarship": "No",
       "Debtor": "No", "International": "Domestic"}

def safe_mean(a):
    return float(np.mean(a)) if len(a) else np.nan

def audit(yh, Xd=Xte, yd=yte, verbose=True):
    """Group table + summary gaps for a vector of 0/1 predictions."""
    G = groups(Xd)
    summary = {}
    for a, g in G.items():
        rows = []
        for lv in sorted(g.unique()):
            mk = (g == lv).values
            yy, h = yd.values[mk], yh[mk]
            rows.append(dict(attr=a, group=lv, n=int(mk.sum()), base=yy.mean(),
                             flag=h.mean(), TPR=safe_mean(h[yy == 1]),
                             FPR=safe_mean(h[yy == 0]),
                             prec=safe_mean(yy[h == 1])))
        R = pd.DataFrame(rows)
        ref = R[R.group == REF[a]].iloc[0]
        R["DI_flag"] = R.flag / ref.flag
        R["DI_base"] = R.base / ref.base
        summary[a] = dict(DPD=R.flag.max() - R.flag.min(),
                          DI_min=R.flag.min() / R.flag.max(),
                          dTPR=R.TPR.max() - R.TPR.min(),
                          dFPR=R.FPR.max() - R.FPR.min(),
                          EOD=max(R.TPR.max() - R.TPR.min(), R.FPR.max() - R.FPR.min()),
                          minTPR=R.TPR.min())
        if verbose:
            out(R.round(3).to_string(index=False))
            out("")
    S = pd.DataFrame(summary).T
    out("  Summary gaps:")
    out(S.round(3).to_string())
    return S

def perf(yh, p):
    return rnd(dict(auc=roc_auc_score(yte, p), rec=recall_score(yte, yh),
                    prec=precision_score(yte, yh, zero_division=0), f1=f1_score(yte, yh)))

p = lr.predict_proba(Xte)[:, 1]
yb = (p >= 0.5).astype(int)
out("  --- BASELINE LR (threshold 0.50) ---")
S0 = audit(yb)
out("  Performance:", perf(yb, p))

# Bootstrap 95% CIs for the key gaps
rng = np.random.default_rng(0)
G_te = groups(Xte)
def boot_gap(attr, g1, g2, label, B=1000):
    res = []
    yv, gv = yte.values, G_te[attr].values
    for _ in range(B):
        i = rng.integers(0, len(yv), len(yv))
        h, yy, gg = yb[i], yv[i], gv[i]
        r1, r2 = h[(gg == g1) & (yy == label)], h[(gg == g2) & (yy == label)]
        if len(r1) and len(r2):
            res.append(r1.mean() - r2.mean())
    return np.round(np.percentile(res, [2.5, 97.5]), 3).tolist()

out("  95% bootstrap CI  TPR Male-Female:", boot_gap("Gender", "Male", "Female", 1),
    "  FPR Male-Female:", boot_gap("Gender", "Male", "Female", 0))
out("  95% bootstrap CI  TPR >25 - <=20 :", boot_gap("Age", ">25", "<=20", 1),
    "  FPR >25 - <=20 :", boot_gap("Age", ">25", "<=20", 0))

t = pd.DataFrame(dict(Gender=G_te["Gender"].values, Age=G_te["Age"].values,
                      y=yte.values, h=yb))
out("  Intersectional TPR (actual dropouts only):")
out(t[t.y == 1].groupby(["Gender", "Age"]).agg(n=("h", "size"), TPR=("h", "mean"))
    .round(3).to_string())

# ============================================================================
# 9. MITIGATIONS
# ============================================================================
step("[9/9] Mitigations: M1 unawareness, M2 reweighing, M3 group thresholds")

# M1 - fairness through unawareness
sens = ["Gender", "Age at enrollment", "International", "Nacionality", "Marital Status",
        "Scholarship holder", "Debtor", "Displaced"]
sens = [c for c in sens if c in Xtr.columns]
keep = [c for c in Xtr.columns if c not in sens]
un = make_lr().fit(Xtr[keep], ytr)
pu = un.predict_proba(Xte[keep])[:, 1]
out("  --- M1 UNAWARENESS (drop sensitive attributes) ---")
out("  Performance:", perf((pu >= 0.5).astype(int), pu))
audit((pu >= 0.5).astype(int), verbose=False)
for nm, tgt in [("Gender", Xtr["Gender"]),
                ("Age>25", (Xtr["Age at enrollment"] > 25).astype(int))]:
    auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                          Xtr[keep], tgt, cv=5, scoring="roc_auc").mean()
    out(f"  Proxy test - remaining features predict {nm}: AUC {auc:.3f}")

# M2 - reweighing (Kamiran & Calders) on gender x age band x label
g_tr = groups(Xtr)
key = (g_tr["Gender"] + "|" + g_tr["Age"]).values
w = np.ones(len(ytr))
for k in np.unique(key):
    for lab in (0, 1):
        mk = (key == k) & (ytr.values == lab)
        if mk.sum():
            w[mk] = ((key == k).mean() * (ytr.values == lab).mean()) / mk.mean()
rw = make_lr()
rw.fit(Xtr, ytr, logisticregression__sample_weight=w)
pr = rw.predict_proba(Xte)[:, 1]
out("\n  --- M2 REWEIGHING ---")
out("  Performance:", perf((pr >= 0.5).astype(int), pr))
audit((pr >= 0.5).astype(int), verbose=False)

# M3 - equal-opportunity floor: lower threshold only for under-served groups,
# calibrated on OUT-OF-FOLD training predictions (no test-set peeking)
oof = cross_val_predict(make_lr(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                        method="predict_proba")[:, 1]
def under(g):
    return ((g["Age"] == "<=20") | (g["Scholarship"] == "Yes")).values
m_tr = under(g_tr)
pos_scores = oof[m_tr & (ytr.values == 1)]
for tgt in (0.78, 0.80, 0.82):
    t_u = float(np.quantile(pos_scores, 1 - tgt))
    yh = np.where(under(G_te), p >= t_u, p >= 0.5).astype(int)
    out(f"\n  --- M3 GROUP THRESHOLDS  target TPR {tgt:.2f}: threshold {t_u:.3f} "
        f"for age<=20 or scholarship holders, 0.50 otherwise ---")
    out("  Performance:", perf(yh, p))
    audit(yh, verbose=(tgt == 0.82))

with open(TXT_PATH, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_log_lines))
step("DONE - all Step 5 numbers reproduced")
print(f"  Results log : {os.path.abspath(TXT_PATH)}")
print(f"  Figure      : {os.path.abspath(FIG_PATH)}")
