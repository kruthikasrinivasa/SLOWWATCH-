from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"F:\SLOWWATCH_DATA\SLOWWATCH_PROJECT")
METRICS = PROJECT / "results" / "metrics"
OUT = PROJECT / "results" / "final_tables"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER
# ============================================================

def show(df, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(df.to_string(index=False))


def save_table(df, filename):
    df.to_csv(OUT / filename, index=False)


# ============================================================
# TABLE 1 — FINAL MODEL COMPARISON
# ============================================================

p = METRICS / "final_model_comparison.csv"

if p.exists():
    d = pd.read_csv(p)

    columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    # Normalize possible column names
    rename_map = {
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    d = d.rename(columns=rename_map)

    available = [c for c in columns if c in d.columns]
    d = d[available]

    for c in available:
        if c != "Model":
            d[c] = pd.to_numeric(d[c], errors="coerce").round(4)

    save_table(d, "table_01_model_comparison.csv")
    show(d, "TABLE 1 - FINAL MODEL COMPARISON")


# ============================================================
# TABLE 2 — ABLATION STUDY
# ============================================================

p = METRICS / "final_ablation_table.csv"

if p.exists():
    d = pd.read_csv(p)

    rename_map = {
        "experiment": "Experiment",
        "feature_count": "Features",
        "features": "Features",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    d = d.rename(columns=rename_map)

    columns = [
        "Experiment",
        "Features",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    available = [c for c in columns if c in d.columns]
    d = d[available]

    for c in available:
        if c not in ["Experiment"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(4)

    save_table(d, "table_02_ablation_study.csv")
    show(d, "TABLE 2 - ABLATION STUDY")


# ============================================================
# TABLE 3 — SLOWWATCH CORE / UNCERTAINTY SUMMARY
# ============================================================

p = METRICS / "slowwatch_core_results.csv"

if p.exists():
    d = pd.read_csv(p)

    rows = []

    # Number of observations
    rows.append({
        "Metric": "Observations",
        "Value": len(d)
    })

    # TRS
    if "temporal_risk_score" in d.columns:
        rows.append({
            "Metric": "Mean TRS",
            "Value": d["temporal_risk_score"].mean()
        })

    # Uncertainty
    if "uncertainty" in d.columns:
        rows.append({
            "Metric": "Mean Uncertainty",
            "Value": d["uncertainty"].mean()
        })

    # Confidence
    if "confidence" in d.columns:
        rows.append({
            "Metric": "Mean Confidence",
            "Value": d["confidence"].mean()
        })

    # Risk categories
    if "risk_category" in d.columns:
        counts = d["risk_category"].value_counts()

        for category in ["LOW", "MINIMAL", "MODERATE", "HIGH"]:
            if category in counts.index:
                rows.append({
                    "Metric": f"{category} Risk",
                    "Value": int(counts[category])
                })

    d3 = pd.DataFrame(rows)

    d3["Value"] = pd.to_numeric(
        d3["Value"], errors="coerce"
    ).round(4)

    save_table(d3, "table_03_slowwatch_core_summary.csv")
    show(d3, "TABLE 3 - SLOWWATCH CORE / UNCERTAINTY SUMMARY")


# ============================================================
# TABLE 4 — RISK CATEGORY DISTRIBUTION
# ============================================================

p = METRICS / "slowwatch_core_results.csv"

if p.exists():
    d = pd.read_csv(p)

    if "risk_category" in d.columns:

        d4 = (
            d["risk_category"]
            .value_counts()
            .rename_axis("Risk Category")
            .reset_index(name="Count")
        )

        # Keep logical order
        order = {
            "LOW": 0,
            "MINIMAL": 1,
            "MODERATE": 2,
            "HIGH": 3
        }

        d4["_order"] = d4["Risk Category"].map(order)
        d4 = (
            d4.sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        save_table(d4, "table_04_risk_distribution.csv")
        show(d4, "TABLE 4 - SLOWWATCH RISK CATEGORY DISTRIBUTION")


# ============================================================
# TABLE 5 — XAI / FEATURE CONTRIBUTION
# ============================================================

p = METRICS / "slowwatch_xai_results.csv"

if p.exists():
    d = pd.read_csv(p)

    # Exact columns from the actual XAI output
    feature_col = "feature"
    correlation_col = "correlation_with_temporal_risk_score"
    importance_col = "absolute_importance"
    valid_col = "valid_observations"

    if (
        feature_col in d.columns
        and correlation_col in d.columns
    ):

        columns = [feature_col, correlation_col]

        if importance_col in d.columns:
            columns.append(importance_col)

        if valid_col in d.columns:
            columns.append(valid_col)

        d5 = d[columns].copy()

        rename_map = {
            feature_col: "Feature",
            correlation_col: "Correlation",
            importance_col: "Absolute Importance",
            valid_col: "Valid Observations",
        }

        d5 = d5.rename(columns=rename_map)

        numeric_columns = [
            "Correlation",
            "Absolute Importance",
            "Valid Observations"
        ]

        for c in numeric_columns:
            if c in d5.columns:
                d5[c] = pd.to_numeric(
                    d5[c], errors="coerce"
                )

        if "Correlation" in d5.columns:
            d5["Correlation"] = d5["Correlation"].round(4)

        if "Absolute Importance" in d5.columns:
            d5["Absolute Importance"] = (
                d5["Absolute Importance"].round(4)
            )

        if "Valid Observations" in d5.columns:
            d5["Valid Observations"] = (
                d5["Valid Observations"]
                .fillna(0)
                .astype(int)
            )

        # Sort by absolute contribution
        if "Absolute Importance" in d5.columns:
            d5 = d5.sort_values(
                "Absolute Importance",
                ascending=False
            )

        d5 = d5.reset_index(drop=True)

        save_table(
            d5,
            "table_05_xai_feature_contribution.csv"
        )

        show(
            d5,
            "TABLE 5 - XAI / FEATURE CONTRIBUTION"
        )

    else:
        print("\nTABLE 5 - XAI / FEATURE CONTRIBUTION")
        print("Required XAI columns were not found.")


# ============================================================
# TABLE 6 — SPATIAL CROSS-REGION GENERALIZATION
# ============================================================

p = METRICS / "regional_cross_region_results.csv"

if p.exists():
    d = pd.read_csv(p)

    rename_map = {
        "train_region": "Train",
        "test_region": "Test",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    d = d.rename(columns=rename_map)

    columns = [
        "Train",
        "Test",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    available = [c for c in columns if c in d.columns]
    d6 = d[available].copy()

    for c in available:
        if c not in ["Train", "Test"]:
            d6[c] = pd.to_numeric(
                d6[c], errors="coerce"
            ).round(4)

    save_table(
        d6,
        "table_06_spatial_cross_region_generalization.csv"
    )

    show(
        d6,
        "TABLE 6 - SPATIAL CROSS-REGION GENERALIZATION"
    )


# ============================================================
# TABLE 7 — CROSS-REGION EVENT-LEVEL EVALUATION
# ============================================================

p = METRICS / "regional_event_evaluation_summary.csv"

if p.exists():
    d = pd.read_csv(p)

    rename_map = {
        "experiment": "Experiment",
        "reference_events": "Ref Events",
        "detected_events": "Detected",
        "event_recall": "Recall",
        "mean_lead_time_months": "Mean Lead (mo)",
        "median_lead_time_months": "Median Lead (mo)",
        "max_lead_time_months": "Max Lead (mo)",
        "false_warning_rate": "False Warn. Rate",
    }

    d = d.rename(columns=rename_map)

    columns = [
        "Experiment",
        "Ref Events",
        "Detected",
        "Recall",
        "Mean Lead (mo)",
        "Median Lead (mo)",
        "Max Lead (mo)",
        "False Warn. Rate",
    ]

    available = [c for c in columns if c in d.columns]
    d7 = d[available].copy()

    for c in available:
        if c != "Experiment":
            d7[c] = pd.to_numeric(
                d7[c], errors="coerce"
            ).round(4)

    save_table(
        d7,
        "table_07_cross_region_event_evaluation.csv"
    )

    show(
        d7,
        "TABLE 7 - CROSS-REGION EVENT-LEVEL EVALUATION"
    )


# ============================================================
# TABLE 8 — REGIONAL DATASET SUMMARY
# ============================================================

rows = []

for region in ["wayanad", "nilgiris"]:

    p = (
        PROJECT
        / "data"
        / "features"
        / "regional"
        / f"{region}_ml_dataset.csv"
    )

    if p.exists():

        d = pd.read_csv(p)

        rows.append({
            "Region": region.title(),
            "Rows": len(d),
            "Events": (
                int(d["firms_event_target"].sum())
                if "firms_event_target" in d.columns
                else "-"
            ),
            "Pre-event": (
                int(d["firms_pre_event_3m"].sum())
                if "firms_pre_event_3m" in d.columns
                else "-"
            ),
            "Early Warning": (
                int(d["early_warning_target"].sum())
                if "early_warning_target" in d.columns
                else "-"
            ),
            "Missing Cells": int(d.isna().sum().sum())
        })

if rows:

    d8 = pd.DataFrame(rows)

    save_table(
        d8,
        "table_08_regional_dataset_summary.csv"
    )

    show(
        d8,
        "TABLE 8 - REGIONAL DATASET SUMMARY"
    )


# ============================================================
# TABLE 9 — FIRMS REGIONAL SUMMARY
# ============================================================

rows = []

FIRMS_REGIONAL = Path(r"F:\SLOWWATCH_DATA\FIRMS_Regional")

for region in ["wayanad", "nilgiris"]:

    p = FIRMS_REGIONAL / f"{region}_firms_2018_2025.csv"

    if p.exists():

        d = pd.read_csv(p)

        date_col = "acq_date"

        if date_col in d.columns:

            dates = pd.to_datetime(
                d[date_col],
                errors="coerce"
            )

            rows.append({
                "Region": region.title(),
                "Detections": len(d),
                "Start Date": str(dates.min().date()),
                "End Date": str(dates.max().date())
            })

if rows:

    d9 = pd.DataFrame(rows)

    save_table(
        d9,
        "table_09_firms_regional_summary.csv"
    )

    show(
        d9,
        "TABLE 9 - FIRMS REGIONAL SUMMARY"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 72)
print("CLEAN RESULTS TABLE GENERATION COMPLETE")
print("=" * 72)

print(f"Output folder:")
print(OUT)

print("\nOriginal experiment outputs were NOT modified.")

print("\nGenerated tables:")

for f in sorted(OUT.glob("*.csv")):
    print(f" - {f.name}")

print("=" * 72)