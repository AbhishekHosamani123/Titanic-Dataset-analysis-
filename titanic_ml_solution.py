"""
============================================================
  TITANIC SURVIVAL PREDICTION – ML Challenge Solution
============================================================
Models: Logistic Regression, Random Forest, Gradient Boosting
Extras: Feature Engineering, Cross-Validation, Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION")
print("=" * 55)

df = pd.read_csv("/mnt/user-data/uploads/Titanic-Dataset.csv")
print(f"\n[DATA] Shape: {df.shape}")
print(f"[DATA] Target distribution:\n{df['Survived'].value_counts().to_string()}")
print(f"\n[MISSING] Before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 2: DATA CLEANING")
print("─" * 55)

# Age: fill with median per Pclass & Sex (more accurate than global median)
age_median = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(age_median)
print("[FIX] Age: filled with median grouped by Pclass + Sex")

# Embarked: fill with mode (only 2 missing)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("[FIX] Embarked: filled with mode 'S'")

# Cabin: too many nulls (77%) → binary HasCabin flag
df["HasCabin"] = df["Cabin"].notna().astype(int)
print("[FIX] Cabin: converted to binary HasCabin feature (77% missing)")

# Drop columns not useful for modelling
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
print("[DROP] Removed: PassengerId, Name, Ticket, Cabin")
print(f"\n[MISSING] After cleaning: {df.isnull().sum().sum()} nulls remaining")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 3: FEATURE ENGINEERING")
print("─" * 55)

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
print("[FEAT] FamilySize = SibSp + Parch + 1")

# IsAlone flag
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
print("[FEAT] IsAlone flag")

# Age buckets
df["AgeBin"] = pd.cut(df["Age"],
                      bins=[0, 12, 18, 35, 60, 100],
                      labels=[0, 1, 2, 3, 4]).astype(int)
print("[FEAT] AgeBin (Child/Teen/Young Adult/Adult/Senior)")

# Fare per person (some tickets shared)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
print("[FEAT] FarePerPerson = Fare / FamilySize")

# Pclass × Sex interaction
df["Pclass_Sex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
print("[FEAT] Pclass_Sex interaction feature")

# ─────────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 4: ENCODING CATEGORICALS")
print("─" * 55)

le = LabelEncoder()
for col in ["Sex", "Embarked", "Pclass_Sex"]:
    df[col] = le.fit_transform(df[col])
    print(f"[ENCODE] {col} → Label Encoded")

# ─────────────────────────────────────────────
# 5. PREPARE FEATURES
# ─────────────────────────────────────────────
X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 5: MODEL TRAINING")
print("─" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                   min_samples_split=4, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "acc":       accuracy_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_proba),
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "cv_scores": cv_scores,
    }
    print(f"\n  ▶ {name}")
    print(f"    Test Accuracy  : {results[name]['acc']:.4f}")
    print(f"    ROC-AUC        : {results[name]['auc']:.4f}")
    print(f"    CV Accuracy    : {results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f}")

# ─────────────────────────────────────────────
# 7. CLASSIFICATION REPORTS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  STEP 6: CLASSIFICATION REPORTS")
print("─" * 55)
for name, r in results.items():
    print(f"\n  ── {name} ──")
    print(classification_report(y_test, r["y_pred"],
                                 target_names=["Not Survived", "Survived"]))

# ─────────────────────────────────────────────
# 8. VISUALIZATIONS  (5-panel figure)
# ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"Logistic Regression": "#4C72B0",
          "Random Forest":       "#DD8452",
          "Gradient Boosting":   "#55A868"}

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel A: Accuracy & AUC bar chart ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
names   = list(results.keys())
accs    = [results[n]["acc"]      for n in names]
aucs    = [results[n]["auc"]      for n in names]
cv_mean = [results[n]["cv_mean"]  for n in names]
cv_std  = [results[n]["cv_std"]   for n in names]

x  = np.arange(len(names))
bw = 0.25
bars1 = ax1.bar(x - bw, accs,    bw, label="Test Accuracy", color=[COLORS[n] for n in names], alpha=0.9)
bars2 = ax1.bar(x,       aucs,   bw, label="ROC-AUC",       color=[COLORS[n] for n in names], alpha=0.55)
bars3 = ax1.bar(x + bw, cv_mean, bw, yerr=cv_std,
                label="CV Accuracy (5-fold)", color=[COLORS[n] for n in names],
                alpha=0.4, capsize=5)
for bar in [*bars1, *bars2, *bars3]:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11)
ax1.set_ylim(0.70, 1.02)
ax1.set_ylabel("Score", fontsize=11)
ax1.set_title("Model Comparison – Test Accuracy, AUC & CV Accuracy", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right", fontsize=9)

# ── Panel B: CV score violin ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
cv_data   = [results[n]["cv_scores"] for n in names]
vp = ax2.violinplot(cv_data, positions=range(len(names)), showmeans=True)
for i, (pc, n) in enumerate(zip(vp["bodies"], names)):
    pc.set_facecolor(COLORS[n]); pc.set_alpha(0.7)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax2.set_ylabel("Accuracy", fontsize=10)
ax2.set_title("5-Fold CV Distribution", fontsize=12, fontweight="bold")

# ── Panel C: Confusion matrices (3 side by side) ────────────────
for i, name in enumerate(names):
    ax = fig.add_subplot(gs[1, i])
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not\nSurvived", "Survived"],
                yticklabels=["Not\nSurvived", "Survived"])
    ax.set_title(f"{name}\nAcc={results[name]['acc']:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)

# ── Panel D: ROC Curves ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax5.plot(fpr, tpr, lw=2, color=COLORS[name],
             label=f"{name}  (AUC = {r['auc']:.3f})")
ax5.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
ax5.fill_between([0,1],[0,1], alpha=0.05, color="gray")
ax5.set_xlabel("False Positive Rate", fontsize=11)
ax5.set_ylabel("True Positive Rate", fontsize=11)
ax5.set_title("ROC Curves", fontsize=13, fontweight="bold")
ax5.legend(loc="lower right", fontsize=10)

# ── Panel E: Random Forest Feature Importance ───────────────────
ax6 = fig.add_subplot(gs[2, 2])
rf_model  = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_imp)))
ax6.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor="white")
ax6.set_xlabel("Importance", fontsize=10)
ax6.set_title("Feature Importance\n(Random Forest)", fontsize=11, fontweight="bold")
for i, v in enumerate(feat_imp.values):
    ax6.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

# ── Title ────────────────────────────────────────────────────────
fig.suptitle("🚢  Titanic Survival Prediction – ML Challenge Results",
             fontsize=16, fontweight="bold", y=0.98, color="#1a1a2e")

plt.savefig("/mnt/user-data/outputs/titanic_ml_results.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n[SAVED] Visualization → titanic_ml_results.png")

# ─────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────
best = max(results, key=lambda n: results[n]["cv_mean"])
print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)
for name, r in results.items():
    marker = " ◀ BEST" if name == best else ""
    print(f"  {name:<25} Acc={r['acc']:.4f}  AUC={r['auc']:.4f}  CV={r['cv_mean']:.4f}±{r['cv_std']:.4f}{marker}")
print("\n[DONE]")
