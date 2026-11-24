"""
Customer Churn Prediction - training and evaluation pipeline.

Dataset: IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
Source:  https://github.com/IBM/telco-customer-churn-on-icp4d
         (7,043 customers, 21 columns, binary target 'Churn')

Evaluation design (leakage-safe):
- Stratified 60/20/20 train/validation/test split.
- The three models are compared on the VALIDATION set; the winner is
  selected there.
- The decision threshold is tuned on the VALIDATION set.
- The winner is refit on train+validation and evaluated ONCE on the
  untouched TEST set. Test metrics are the reported final results.

Runs: data loading, cleaning, EDA plots, three sklearn models, threshold
analysis, feature importance. Saves the final pipeline (joblib), all plots
and metrics to outputs/.
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = BASE_DIR / "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def scores(y_true, proba, threshold=0.50):
    pred = (proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": auc(fpr, tpr),
            "roc": (fpr, tpr)}


# ---------------------------------------------------------------- load data
print("=" * 64)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("=" * 64)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Raw dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nChurn distribution (raw):")
print(df["Churn"].value_counts().to_string())

# -------------------------------------------------------------- cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
blank_tc = df["TotalCharges"].isna().sum()
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df = df.drop(columns=["customerID"])  # identifier, not a predictor
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print(f"\n[2] Cleaning:")
print(f"  - TotalCharges converted to numeric ({blank_tc} blank values -> rows dropped)")
print(f"  - customerID dropped (identifier, not a predictor)")
print(f"  - Churn encoded Yes=1 / No=0")
print(f"  - Clean dataset: {df.shape[0]} rows x {df.shape[1]} columns")
churn_rate = df["Churn"].mean()
print(f"  - Churn rate: {churn_rate:.1%}  ({df['Churn'].sum()} churned / {len(df)} customers)")

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in df.columns
                        if c not in numeric_features + ["Churn"]]
print(f"\n  Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

# ------------------------------------------------------------------ EDA
print("\n[3] Exploratory data analysis - saving plots")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
counts = df["Churn"].value_counts().sort_index()
labels = ["No (stayed)", "Yes (churned)"]
colors = ["#2ca02c", "#d62728"]
axes[0].bar(labels, counts.values, color=colors, edgecolor="black")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 40, f"{v}\n({v / len(df):.1%})", ha="center", fontsize=10)
axes[0].set_title("Churn class distribution")
axes[0].set_ylabel("Customers")
axes[0].set_ylim(0, counts.max() * 1.2)
axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%", colors=colors,
            startangle=90, wedgeprops={"edgecolor": "black"})
axes[1].set_title("Churn share")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eda_churn_distribution.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df.loc[df.Churn == 0, "tenure"], bins=30, alpha=0.65,
        label="No (stayed)", color="#2ca02c", edgecolor="black")
ax.hist(df.loc[df.Churn == 1, "tenure"], bins=30, alpha=0.65,
        label="Yes (churned)", color="#d62728", edgecolor="black")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Customers")
ax.set_title("Tenure distribution by churn")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eda_tenure_by_churn.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
data = [df.loc[df.Churn == 0, "MonthlyCharges"], df.loc[df.Churn == 1, "MonthlyCharges"]]
bp = ax.boxplot(data, tick_labels=["No (stayed)", "Yes (churned)"], patch_artist=True)
for patch, c in zip(bp["boxes"], ["#2ca02c", "#d62728"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_ylabel("Monthly charges ($)")
ax.set_title("Monthly charges by churn")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eda_monthly_charges.png"), dpi=150)
plt.close(fig)

contract_rate = df.groupby("Contract")["Churn"].mean().reindex(
    ["Month-to-month", "One year", "Two year"])
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(contract_rate.index, contract_rate.values,
              color=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black")
for b, v in zip(bars, contract_rate.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
ax.set_ylabel("Churn rate")
ax.set_title("Churn rate by contract type")
ax.set_ylim(0, contract_rate.max() * 1.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eda_contract_churn.png"), dpi=150)
plt.close(fig)

print("  saved: eda_churn_distribution.png, eda_tenure_by_churn.png,")
print("         eda_monthly_charges.png, eda_contract_churn.png")
print("\n  Quick facts:")
print(f"  - Avg tenure   churned {df.loc[df.Churn==1,'tenure'].mean():.1f} mo "
      f"vs stayed {df.loc[df.Churn==0,'tenure'].mean():.1f} mo")
print(f"  - Avg monthly  churned ${df.loc[df.Churn==1,'MonthlyCharges'].mean():.2f} "
      f"vs stayed ${df.loc[df.Churn==0,'MonthlyCharges'].mean():.2f}")
print("  - Churn rate by contract: " +
      ", ".join(f"{k} {v:.1%}" for k, v in contract_rate.items()))

# ------------------------------------------------- split and preprocessing
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
print(f"\n[4] Train/validation/test split (stratified 60/20/20, random_state={RANDOM_STATE}):")
print(f"  train {X_train.shape[0]} | validation {X_val.shape[0]} | test {X_test.shape[0]} rows")
print(f"  churn rate: train {y_train.mean():.1%} | val {y_val.mean():.1%} "
      f"| test {y_test.mean():.1%} (stratified, all match overall)")
print("  Validation set is used for model selection and threshold tuning;")
print("  the test set stays untouched until the single final evaluation.")

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# --------------------------------- train on train, select on VALIDATION
print("\n[5] Model selection on the validation set (models fit on train only)")
print("-" * 64)
val_results = {}
roc_val = {}
fitted = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    proba = pipe.predict_proba(X_val)[:, 1]
    r = scores(y_val, proba)
    val_results[name] = r
    roc_val[name] = (r["roc"][0], r["roc"][1], r["roc_auc"])
    print(f"{name:<21} acc {r['accuracy']:.4f} | prec {r['precision']:.4f} "
          f"| rec {r['recall']:.4f} | f1 {r['f1']:.4f} | ROC-AUC {r['roc_auc']:.4f}")

best_name = max(val_results, key=lambda n: val_results[n]["roc_auc"])
best_val_auc = val_results[best_name]["roc_auc"]
print("-" * 64)
print(f"Selected model (best validation ROC-AUC): {best_name} ({best_val_auc:.4f})")

# ROC curves on validation (the selection evidence)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, (fpr, tpr, roc_auc) in roc_val.items():
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curves - validation set")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_roc_curves.png"), dpi=150)
plt.close(fig)

# --------------------------------- threshold tuning on VALIDATION
print("\n[6] Decision-threshold tuning on the validation set")
val_proba = fitted[best_name].predict_proba(X_val)[:, 1]
thresholds = np.arange(0.20, 0.71, 0.05)
rows = []
for t in thresholds:
    pred_t = (val_proba >= t).astype(int)
    rows.append((t, precision_score(y_val, pred_t, zero_division=0),
                 recall_score(y_val, pred_t),
                 f1_score(y_val, pred_t, zero_division=0)))
print("  threshold | precision | recall | f1")
for t, p, r, f1t in rows:
    print(f"     {t:.2f}   |   {p:.3f}   | {r:.3f}  | {f1t:.3f}")
chosen_t, chosen_p, chosen_r, chosen_f1 = max(rows, key=lambda r: r[3])
print(f"  -> best validation F1 at threshold {chosen_t:.2f} "
      f"(precision {chosen_p:.3f}, recall {chosen_r:.3f})")
ts = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(ts, [r[1] for r in rows], "o-", label="Precision", color="#1f77b4")
ax.plot(ts, [r[2] for r in rows], "s-", label="Recall", color="#d62728")
ax.plot(ts, [r[3] for r in rows], "^-", label="F1", color="#2ca02c")
ax.axvline(0.50, color="gray", ls="--", lw=1)
ax.text(0.505, 0.05, "default 0.50", rotation=90, color="gray", fontsize=8)
ax.axvline(chosen_t, color="purple", ls=":", lw=1.5)
ax.text(chosen_t + 0.005, 0.05, f"chosen {chosen_t:.2f}", rotation=90,
        color="purple", fontsize=8)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title(f"Precision / recall trade-off - {best_name} (validation set)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_threshold_curve.png"), dpi=150)
plt.close(fig)

# ----------------------- refit winner on train+val, ONE final test pass
print("\n[7] Final evaluation on the untouched test set")
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_pipe = Pipeline([("prep", preprocess), ("model", models[best_name])])
final_pipe.fit(X_trainval, y_trainval)
print(f"  {best_name} refit on train+validation ({X_trainval.shape[0]} rows)")
test_proba = final_pipe.predict_proba(X_test)[:, 1]
test_default = scores(y_test, test_proba, 0.50)
test_chosen = scores(y_test, test_proba, chosen_t)
print(f"  TEST @ 0.50 default : acc {test_default['accuracy']:.4f} | "
      f"prec {test_default['precision']:.4f} | rec {test_default['recall']:.4f} "
      f"| f1 {test_default['f1']:.4f} | ROC-AUC {test_default['roc_auc']:.4f}")
print(f"  TEST @ {chosen_t:.2f} (chosen on val): acc {test_chosen['accuracy']:.4f} | "
      f"prec {test_chosen['precision']:.4f} | rec {test_chosen['recall']:.4f} "
      f"| f1 {test_chosen['f1']:.4f}")

cm = confusion_matrix(y_test, (test_proba >= 0.50).astype(int))
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(cm, cmap="RdPu")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred No", "Pred Yes"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No", "Actual Yes"])
ax.set_title(f"Confusion matrix on the test set\n{best_name} @ 0.50",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_confusion_matrix.png"), dpi=150)
plt.close(fig)

# -------------------------------------------------- feature importance
print("\n[8] Feature importance (permutation importance, final model, test set)")
pi = permutation_importance(final_pipe, X_test, y_test, n_repeats=10,
                            random_state=RANDOM_STATE, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
for feat, val in imp.head(10).items():
    print(f"  {feat:<22} {val:.4f}")
fig, ax = plt.subplots(figsize=(8, 5))
top = imp.head(10).iloc[::-1]
ax.barh(top.index, top.values, color="#7b2d8b", edgecolor="black")
ax.set_xlabel("Mean ROC-AUC decrease when shuffled")
ax.set_title(f"Top 10 features - permutation importance ({best_name}, test set)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_feature_importance.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------- save artefacts
model_path = os.path.join(OUT_DIR, "churn_model.joblib")
joblib.dump(final_pipe, model_path)

metrics = {
    "dataset": {"source": "IBM Telco Customer Churn (github.com/IBM/telco-customer-churn-on-icp4d)",
                "raw_rows": 7043, "clean_rows": int(len(df)),
                "churn_rate": round(float(churn_rate), 4)},
    "split": {"scheme": "stratified 60/20/20 train/validation/test",
              "random_state": RANDOM_STATE,
              "train_rows": int(len(X_train)), "val_rows": int(len(X_val)),
              "test_rows": int(len(X_test)),
              "note": "validation used for model selection and threshold tuning; "
                      "test evaluated once at the end"},
    "validation_results": {k: {m: round(v, 4) for m, v in r.items() if m != "roc"}
                           for k, r in val_results.items()},
    "selected_model": best_name,
    "chosen_threshold": round(float(chosen_t), 2),
    "threshold_table_validation": [
        {"threshold": round(float(t), 2), "precision": round(p, 4),
         "recall": round(r, 4), "f1": round(f1t, 4)}
        for t, p, r, f1t in rows],
    "test_results": {
        "at_0.50": {m: round(v, 4) for m, v in test_default.items() if m != "roc"},
        f"at_{chosen_t:.2f}": {m: round(v, 4) for m, v in test_chosen.items()
                               if m != "roc"},
    },
    "top_features": {k: round(float(v), 4) for k, v in imp.head(10).items()},
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2)

print(f"\n[9] Saved model -> {model_path}")
print(f"    Saved metrics -> {os.path.join(OUT_DIR, 'metrics.json')}")
print("    All plots saved in outputs/")
print("\nTraining complete.")
