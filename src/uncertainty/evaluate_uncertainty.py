import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SLOWWATCH - UNCERTAINTY AND CONFIDENCE EVALUATION
# ============================================================

INPUT = r"results/metrics/slowwatch_core_results.csv"

OUT_CSV = r"results/metrics/slowwatch_uncertainty_evaluation.csv"

OUT_FIG = r"results/figures/slowwatch_uncertainty_analysis.png"

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

print("=" * 70)
print("SLOWWATCH - UNCERTAINTY EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print(f"\nRecords: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Identify columns
# ------------------------------------------------------------

risk_col = "temporal_risk_score"
uncertainty_col = "uncertainty"
confidence_col = "confidence"

required = [
    risk_col,
    uncertainty_col,
    confidence_col
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ------------------------------------------------------------
# Numeric conversion
# ------------------------------------------------------------

risk = pd.to_numeric(
    df[risk_col],
    errors="coerce"
)

uncertainty = pd.to_numeric(
    df[uncertainty_col],
    errors="coerce"
)

confidence = pd.to_numeric(
    df[confidence_col],
    errors="coerce"
)

valid = (
    risk.notna()
    & uncertainty.notna()
    & confidence.notna()
)

d = df.loc[valid].copy()

d["risk"] = risk[valid].values
d["uncertainty_value"] = uncertainty[valid].values
d["confidence_value"] = confidence[valid].values

# ------------------------------------------------------------
# Uncertainty statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNCERTAINTY STATISTICS")
print("=" * 70)

print(
    f"Mean uncertainty     : "
    f"{d['uncertainty_value'].mean():.4f}"
)

print(
    f"Median uncertainty   : "
    f"{d['uncertainty_value'].median():.4f}"
)

print(
    f"Minimum uncertainty  : "
    f"{d['uncertainty_value'].min():.4f}"
)

print(
    f"Maximum uncertainty  : "
    f"{d['uncertainty_value'].max():.4f}"
)

print(
    f"\nMean confidence      : "
    f"{d['confidence_value'].mean():.4f}"
)

print(
    f"Median confidence    : "
    f"{d['confidence_value'].median():.4f}"
)

# ------------------------------------------------------------
# Risk / uncertainty relationship
# ------------------------------------------------------------

risk_uncertainty_corr = d[
    "risk"
].corr(
    d["uncertainty_value"]
)

risk_confidence_corr = d[
    "risk"
].corr(
    d["confidence_value"]
)

print(
    f"\nRisk–uncertainty correlation : "
    f"{risk_uncertainty_corr:.4f}"
)

print(
    f"Risk–confidence correlation  : "
    f"{risk_confidence_corr:.4f}"
)

# ------------------------------------------------------------
# Confidence categories
# ------------------------------------------------------------

def confidence_category(x):

    if x >= 0.70:
        return "HIGH"

    elif x >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


d["confidence_category"] = (
    d["confidence_value"]
    .apply(confidence_category)
)

print("\nConfidence categories:")

print(
    d["confidence_category"]
    .value_counts()
)

# ------------------------------------------------------------
# Uncertainty categories
# ------------------------------------------------------------

def uncertainty_category(x):

    if x <= 0.33:
        return "LOW"

    elif x <= 0.66:
        return "MEDIUM"

    else:
        return "HIGH"


d["uncertainty_category"] = (
    d["uncertainty_value"]
    .apply(uncertainty_category)
)

print("\nUncertainty categories:")

print(
    d["uncertainty_category"]
    .value_counts()
)

# ------------------------------------------------------------
# Save detailed evaluation
# ------------------------------------------------------------

output = d[
    [
        "date",
        risk_col,
        uncertainty_col,
        confidence_col,
        "confidence_category",
        "uncertainty_category"
    ]
].copy()

output.to_csv(
    OUT_CSV,
    index=False
)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    d["risk"],
    d["uncertainty_value"],
    alpha=0.75
)

plt.xlabel(
    "Temporal Risk Score"
)

plt.ylabel(
    "Uncertainty"
)

plt.title(
    "SLOWWATCH Risk–Uncertainty Relationship"
)

plt.tight_layout()

plt.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SLOWWATCH UNCERTAINTY EVALUATION COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    f"Metrics : {OUT_CSV}"
)

print(
    f"Figure  : {OUT_FIG}"
)

print("\nDone.")