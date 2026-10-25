"""Credit default prediction: does demographic information add predictive value?



Put 'default of credit card clients.xls' in the same folder as this script.
An alternative CSV can be passed as the first argument for testing.

This is a retrospective, educational audit. It is NOT a lending policy.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


folder = Path(__file__).resolve().parent
input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "default of credit card clients.xls"
plots = folder / "results"
plots.mkdir(exist_ok=True)

if input_file.suffix.lower() == ".csv":
    df = pd.read_csv(input_file)
else:
    # Original UCI spreadsheet: the first row is a title, not column names.
    df = pd.read_excel(input_file, header=1, engine="xlrd")

df.columns = [str(c).strip() for c in df.columns]
target = "default payment next month"
if target not in df.columns:
    raise ValueError(f"Can't find '{target}'. Found columns: {df.columns.tolist()}")

# The dataset has some undocumented education codes (0, 5, 6).
# Keep them as an 'unknown' category, rather than guessing their meaning.
df["EDUCATION"] = df["EDUCATION"].replace({0: -1, 5: -1, 6: -1})
y = df[target].astype(int)

# Drop the other demographic variable (marriage) in every experiment.
# This way we're explicitly adding only age, sex and/or education.
base = [c for c in df.columns if c not in
        [target, "ID", "AGE", "SEX", "EDUCATION", "MARRIAGE"]]

experiments = {
    "Financial only": [],
    "+ Age": ["AGE"],
    "+ Sex": ["SEX"],
    "+ Education": ["EDUCATION"],
    "+ All three": ["AGE", "SEX", "EDUCATION"],
}

# Use identical train/test customers for every model.
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.25, random_state=42, stratify=y
)
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx].to_numpy()

interest = 0.15
recovery = 0.30
breakeven = interest / (1 + interest - recovery)
print(f"Rows: {len(df):,}; historical default rate: {y.mean():.2%}")
print(f"Illustrative break-even predicted default probability: {breakeven:.2%}")


def paper_profit(probabilities, default_outcomes):
    # £1 loan, +15p if repaid, -70p if defaulted, and 0 if not lent.
    # Assumes all loans are identical and recovery occurs at year-end.
    would_lend = probabilities < breakeven
    earned = np.where(default_outcomes == 0, interest, recovery - 1)
    return 1000 * np.mean(earned * would_lend), np.mean(would_lend)


rows = []
preds = {}
for name, extra in experiments.items():
    cols = base + extra
    categories = [c for c in extra if c in ["SEX", "EDUCATION"]]
    numbers = [c for c in cols if c not in categories]

    prep = ColumnTransformer([
        ("numeric", StandardScaler(), numbers),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categories),
    ])
    model = make_pipeline(prep, LogisticRegression(max_iter=1000))
    model.fit(df.iloc[train_idx][cols], y_train)

    p = model.predict_proba(df.iloc[test_idx][cols])[:, 1]
    preds[name] = p
    profit, approval = paper_profit(p, y_test)
    rows.append({
        "Model": name,
        "ROC AUC": roc_auc_score(y_test, p),
        "Brier score": brier_score_loss(y_test, p),
        "Simulated return / 1000 customers": profit,
        "Approval rate": approval,
    })

results = pd.DataFrame(rows).set_index("Model")
print("\nResults on the same held-out test set:\n")
print(results.round(4).to_string())
results.to_csv(plots / "model_comparison.csv")

# Paired bootstrap for the ROC-AUC difference: same resampled test customers
# for both models. This treats the fitted models as fixed, and therefore does
# not represent uncertainty from retraining on new training samples.
rng_auc = np.random.default_rng(124)
auc_diffs = []
for _ in range(600):
    idx = rng_auc.integers(0, len(y_test), size=len(y_test))
    y_sample = y_test[idx]
    if len(np.unique(y_sample)) < 2:
        continue
    auc_diffs.append(
        roc_auc_score(y_sample, preds["+ All three"][idx])
        - roc_auc_score(y_sample, preds["Financial only"][idx])
    )
ci_auc = np.percentile(auc_diffs, [2.5, 97.5])
auc_diff = results.loc["+ All three", "ROC AUC"] - results.loc["Financial only", "ROC AUC"]
print(f"\nAUC difference (full - baseline): {auc_diff:.5f}; "
      f"paired bootstrap 95% CI [{ci_auc[0]:.5f}, {ci_auc[1]:.5f}]")

# Rough paired bootstrap: does the extra information change the toy return?
loan_outcome = np.where(y_test == 0, interest, recovery - 1)
diff = loan_outcome * (
    (preds["+ All three"] < breakeven).astype(int)
    - (preds["Financial only"] < breakeven).astype(int)
)
rng = np.random.default_rng(21)
boot = []
for _ in range(400):
    sample = rng.integers(0, len(diff), size=len(diff))
    boot.append(1000 * diff[sample].mean())
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nSimulated return difference (+ All vs financial), per 1000: "
      f"{1000 * diff.mean():.2f}; bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")

# Plot 1: better discrimination = higher AUC, better probabilities = lower Brier.
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(results.index, results["ROC AUC"])
ax.set_title("Default prediction: does adding demographics help?")
ax.set_ylabel("ROC AUC (higher is better)")
ax.set_ylim(0.5, 1)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(plots / "01_auc.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(results.index, results["Brier score"])
ax.set_title("Probability accuracy: Brier score")
ax.set_ylabel("Mean squared probability error (lower is better)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(plots / "02_brier.png", dpi=160)
plt.close(fig)

# Plot 3: ONLY a toy, retrospective profit calculation, not actual bank revenue.
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(results.index, results["Simulated return / 1000 customers"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Hypothetical net return under the same lending rule")
ax.set_ylabel("Return per 1,000 observed customers (units of principal)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(plots / "03_simulated_returns.png", dpi=160)
plt.close(fig)

# Plot 4: demographic audit, not a recommendation for demographic-based lending.
# Even the 'Financial only' model can show differences due to correlated features.
sex = df.iloc[test_idx]["SEX"].to_numpy()
labels = {1: "Male (coded 1)", 2: "Female (coded 2)"}
xs = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, name in enumerate(["Financial only", "+ All three"]):
    approvals = preds[name] < breakeven
    rates = [approvals[sex == code].mean() for code in labels]
    ax.bar(xs + (i - .5)*.33, rates, width=.33, label=name)
ax.set_xticks(xs, list(labels.values()))
ax.set_ylim(0, 1)
ax.set_ylabel("Simulated approval rate")
ax.set_title("Audit: simulated approval rates by recorded sex")
ax.legend()
fig.tight_layout()
fig.savefig(plots / "04_group_audit.png", dpi=160)
plt.close(fig)

print("\nSaved model_comparison.csv and four charts in:", plots)
print("NOTE: These are historical Taiwanese credit-card customers, not all applicants.")
print("The simulated return is based on assumed loan terms, not observed profits.")
print("Do not use demographic-based models as real lending policies.")
