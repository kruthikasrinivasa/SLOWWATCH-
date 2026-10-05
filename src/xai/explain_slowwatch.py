import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SLOWWATCH - EXPLAINABILITY ANALYSIS
# ============================================================

INPUT = r"results/metrics/slowwatch_core_results.csv"

OUT_CSV = r"results/metrics/slowwatch_xai_results.csv"

OUT_FIG = r"results/figures/slowwatch_feature_contribution.png"


# ------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


print("=" * 70)
print("SLOWWATCH - EXPLAINABILITY ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Load SLOWWATCH core results
# ------------------------------------------------------------

print("\nLoading SLOWWATCH core results...")

df = pd.read_csv(INPUT)

print(f"Records: {len(df)}")

print("\nAvailable columns:")
for i, col in enumerate(df.columns):
    print(f"{i:02d} : {col}")


# ------------------------------------------------------------
# SLOWWATCH temporal risk score
# ------------------------------------------------------------

TARGET_COLUMN = "temporal_risk_score"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"\nERROR: '{TARGET_COLUMN}' was not found.\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )


# ------------------------------------------------------------
# Features used for explainability
# ------------------------------------------------------------

candidate_features = [
    "ndvi_mean",
    "ndvi_change",
    "slope_6m",
    "anomaly_zscore",
    "temperature_c",
    "precipitation_raw",
    "temperature_change",
    "precipitation_change"
]


available_features = [
    feature
    for feature in candidate_features
    if feature in df.columns
]


print("\nExplainable features found:")

for feature in available_features:
    print(" -", feature)


if len(available_features) == 0:
    raise ValueError(
        "\nERROR: No explainable feature columns were found."
    )


# ------------------------------------------------------------
# Calculate feature-risk relationships
# ------------------------------------------------------------

print("\nCalculating feature contributions...")

rows = []

for feature in available_features:

    x = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    valid = (
        x.notna()
        & y.notna()
    )

    valid_count = int(valid.sum())

    if valid_count >= 3:

        correlation = x[valid].corr(
            y[valid]
        )

    else:

        correlation = np.nan


    absolute_importance = (
        abs(correlation)
        if pd.notna(correlation)
        else np.nan
    )


    rows.append({

        "feature": feature,

        "correlation_with_temporal_risk_score":
            correlation,

        "absolute_importance":
            absolute_importance,

        "valid_observations":
            valid_count
    })


# ------------------------------------------------------------
# Create explainability table
# ------------------------------------------------------------

importance = pd.DataFrame(rows)

importance = importance.sort_values(
    "absolute_importance",
    ascending=False
)


# ------------------------------------------------------------
# Save explainability results
# ------------------------------------------------------------

importance.to_csv(
    OUT_CSV,
    index=False
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE-RISK CONTRIBUTION RESULTS")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Plot feature relationships
# ------------------------------------------------------------

plot_df = importance.dropna(
    subset=[
        "correlation_with_temporal_risk_score"
    ]
).sort_values(
    "absolute_importance"
)


plt.figure(
    figsize=(10, 6)
)


plt.barh(
    plot_df["feature"],
    plot_df[
        "correlation_with_temporal_risk_score"
    ]
)


plt.axvline(
    0,
    linewidth=1
)


plt.xlabel(
    "Correlation with Temporal Risk Score"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "SLOWWATCH Feature–Risk Relationship"
)


plt.tight_layout()


plt.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# Final information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SLOWWATCH XAI COMPLETED")
print("=" * 70)

print("\nOutput files:")

print(
    f"Feature analysis : {OUT_CSV}"
)

print(
    f"Feature figure   : {OUT_FIG}"
)

print(
    "\nExplainability method:"
)

print(
    "Pearson correlation between temporal/environmental "
    "features and the SLOWWATCH temporal risk score."
)

print(
    "\nNote: This is a feature-risk relationship analysis, "
    "not SHAP-based explainability."
)

print("\nDone.")