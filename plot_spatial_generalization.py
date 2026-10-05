from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SLOWWATCH - SPATIAL CROSS-REGION GENERALIZATION FIGURE
# ============================================================

PROJECT_DIR = Path(r"F:\SLOWWATCH_DATA\SLOWWATCH_PROJECT")

INPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "final_tables"
    / "table_06_spatial_cross_region_generalization.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "spatial_cross_region_generalization.png"


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nSPATIAL CROSS-REGION GENERALIZATION")
print("=" * 60)
print(df.to_string(index=False))


# ------------------------------------------------------------
# Prepare labels
# ------------------------------------------------------------

df["Experiment"] = (
    df["Train"].str.title()
    + " → "
    + df["Test"].str.title()
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]

# Support either lowercase or uppercase column names
column_map = {}

for col in df.columns:
    normalized = col.strip().lower().replace("-", "_")

    if normalized == "accuracy":
        column_map["Accuracy"] = col
    elif normalized == "precision":
        column_map["Precision"] = col
    elif normalized == "recall":
        column_map["Recall"] = col
    elif normalized == "f1":
        column_map["F1"] = col
    elif normalized == "roc_auc":
        column_map["ROC-AUC"] = col
    elif normalized == "pr_auc":
        column_map["PR-AUC"] = col


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

x = np.arange(len(metrics))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6.5))

values_1 = [
    df.loc[0, column_map[m]]
    for m in metrics
]

values_2 = [
    df.loc[1, column_map[m]]
    for m in metrics
]

bars1 = ax.bar(
    x - width / 2,
    values_1,
    width,
    label=f"{df.loc[0, 'Train'].title()} → {df.loc[0, 'Test'].title()}",
)

bars2 = ax.bar(
    x + width / 2,
    values_2,
    width,
    label=f"{df.loc[1, 'Train'].title()} → {df.loc[1, 'Test'].title()}",
)


# ------------------------------------------------------------
# Value labels
# ------------------------------------------------------------

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


add_labels(bars1)
add_labels(bars2)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_title(
    "Spatial Cross-Region Generalization of SLOWWATCH",
    fontsize=15,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel(
    "Evaluation Metric",
    fontsize=11,
)

ax.set_ylabel(
    "Score",
    fontsize=11,
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.set_ylim(0, 1.08)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    frameon=False,
)

plt.tight_layout()


# ------------------------------------------------------------
# Save high-resolution figure
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("\n" + "=" * 60)
print("FIGURE GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Saved to:")
print(OUTPUT_FILE)
print("=" * 60)