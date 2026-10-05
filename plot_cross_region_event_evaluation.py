from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SLOWWATCH
# CROSS-REGION EVENT-LEVEL EVALUATION FIGURE
# ============================================================

PROJECT_DIR = Path(r"F:\SLOWWATCH_DATA\SLOWWATCH_PROJECT")

INPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "final_tables"
    / "table_07_cross_region_event_evaluation.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cross_region_event_level_evaluation.png"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print()
print("CROSS-REGION EVENT-LEVEL EVALUATION")
print("=" * 70)
print(df.to_string(index=False))
print()


# ============================================================
# SHOW ACTUAL CSV COLUMNS
# ============================================================

print("Detected columns:")
for col in df.columns:
    print(" -", repr(col))

print()


# ============================================================
# ROBUST COLUMN FINDER
# ============================================================

def find_column(keyword_list):
    """
    Finds a column using flexible substring matching.
    """

    normalized_columns = {}

    for col in df.columns:

        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(".", "_")
        )

        normalized_columns[normalized] = col

    # Exact match first
    for keyword in keyword_list:

        key = (
            keyword
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(".", "_")
        )

        if key in normalized_columns:
            return normalized_columns[key]

    # Substring match
    for normalized, original in normalized_columns.items():

        for keyword in keyword_list:

            key = (
                keyword
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
                .replace(".", "_")
            )

            if key in normalized:
                return original

    return None


# ============================================================
# IDENTIFY REQUIRED COLUMNS
# ============================================================

reference_col = find_column([
    "reference_events",
    "ref_events",
    "reference"
])

detected_col = find_column([
    "detected_events",
    "detected"
])

recall_col = find_column([
    "event_recall",
    "recall"
])

mean_lead_col = find_column([
    "mean_lead_time_months",
    "mean_lead_time_mo",
    "mean_lead_months",
    "mean_lead_mo",
    "mean_lead"
])


# ============================================================
# CHECK COLUMNS
# ============================================================

print("Column mapping:")
print("Reference Events :", reference_col)
print("Detected Events  :", detected_col)
print("Event Recall     :", recall_col)
print("Mean Lead Time   :", mean_lead_col)
print()


if reference_col is None:
    raise ValueError(
        "Could not identify Reference Events column."
    )

if detected_col is None:
    raise ValueError(
        "Could not identify Detected Events column."
    )

if recall_col is None:
    raise ValueError(
        "Could not identify Event Recall column."
    )

if mean_lead_col is None:
    raise ValueError(
        "Could not identify Mean Lead Time column."
    )


# ============================================================
# EXPERIMENT LABELS
# ============================================================

if "Experiment" in df.columns:

    labels = (
        df["Experiment"]
        .astype(str)
        .tolist()
    )

else:

    labels = []

    for _, row in df.iterrows():

        train = str(row["Train"]).title()
        test = str(row["Test"]).title()

        labels.append(
            f"{train} → {test}"
        )


# ============================================================
# NUMERIC VALUES
# ============================================================

reference_values = (
    pd.to_numeric(
        df[reference_col],
        errors="coerce"
    )
    .fillna(0)
    .values
)

detected_values = (
    pd.to_numeric(
        df[detected_col],
        errors="coerce"
    )
    .fillna(0)
    .values
)

recall_values = (
    pd.to_numeric(
        df[recall_col],
        errors="coerce"
    )
    .fillna(0)
    .values
)

lead_values = (
    pd.to_numeric(
        df[mean_lead_col],
        errors="coerce"
    )
    .fillna(0)
    .values
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)


# ============================================================
# PANEL A — EVENT DETECTION
# ============================================================

x = np.arange(len(df))

width = 0.36


bars_reference = axes[0].bar(
    x - width / 2,
    reference_values,
    width,
    label="Reference Events"
)

bars_detected = axes[0].bar(
    x + width / 2,
    detected_values,
    width,
    label="Detected Events"
)


# Value labels
for bar in bars_reference:

    height = bar.get_height()

    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.8,
        f"{int(round(height))}",
        ha="center",
        va="bottom",
        fontsize=9
    )


for bar in bars_detected:

    height = bar.get_height()

    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.8,
        f"{int(round(height))}",
        ha="center",
        va="bottom",
        fontsize=9
    )


axes[0].set_title(
    "Event Detection",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_ylabel(
    "Number of Events",
    fontsize=10
)

axes[0].set_xticks(x)

axes[0].set_xticklabels(
    labels,
    rotation=15,
    ha="right"
)

axes[0].grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

axes[0].legend(
    frameon=False
)


# ============================================================
# PANEL B — RECALL AND LEAD TIME
# ============================================================

bars_recall = axes[1].bar(
    x - width / 2,
    recall_values,
    width,
    label="Event Recall"
)

bars_lead = axes[1].bar(
    x + width / 2,
    lead_values,
    width,
    label="Mean Lead Time (months)"
)


# Recall labels
for bar, value in zip(
    bars_recall,
    recall_values
):

    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


# Lead-time labels
for bar, value in zip(
    bars_lead,
    lead_values
):

    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


axes[1].set_title(
    "Event Recall and Warning Lead Time",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_ylabel(
    "Value",
    fontsize=10
)

axes[1].set_xticks(x)

axes[1].set_xticklabels(
    labels,
    rotation=15,
    ha="right"
)


maximum_value = max(
    max(recall_values),
    max(lead_values)
)

axes[1].set_ylim(
    0,
    maximum_value + 0.25
)

axes[1].grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

axes[1].legend(
    frameon=False
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Cross-Region Event-Level Evaluation of SLOWWATCH",
    fontsize=15,
    fontweight="bold"
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

print()
print("=" * 70)
print("FIGURE GENERATED SUCCESSFULLY")
print("=" * 70)
print()
print("Saved to:")
print(OUTPUT_FILE)
print()
print("=" * 70)