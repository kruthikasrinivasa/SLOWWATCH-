"""
SLOWWATCH - FINAL RESEARCH RESULTS
==================================

Purpose:
    1. Read already-generated model/evaluation CSV files.
    2. Display clean boxed research tables in the terminal.
    3. Save final summary CSV files.
    4. Generate final comparison figures.
    5. Generate a concise research summary.

No model training is performed in this script.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALUE FORMATTER
# ============================================================

def format_value(value, decimals=4):

    if pd.isna(value):
        return ""

    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"

    if isinstance(value, (int, np.integer)):
        return str(value)

    return str(value)


# ============================================================
# CLEAN TABLE DISPLAY
# ============================================================

def print_box_table(title, df, decimals=4):

    """
    Prints a compact, fully bordered table.

    Designed specifically for:
        - VS Code terminal
        - PowerShell
        - Review screenshots
        - Research demonstrations
    """

    print()
    print("=" * 96)
    print(title.center(96))
    print("=" * 96)

    if df is None or df.empty:

        print("+--------------------------------------+")
        print("| No data available                    |")
        print("+--------------------------------------+")
        return

    table = df.copy()

    # --------------------------------------------------------
    # STANDARD COLUMN NAMES
    # --------------------------------------------------------

    table = table.rename(columns={

        # Model comparison
        "Model": "Model",
        "Evaluation": "Eval.",
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1",
        "ROC-AUC": "ROC-AUC",
        "PR-AUC": "PR-AUC",

        # B0
        "Dataset": "Dataset",
        "Ref.Events": "Ref.",
        "Detected": "Detect.",
        "Mean Lead": "Mean Lead",
        "Median Lead": "Median",
        "False Warn.": "False Warn.",

        # Ablation
        "Experiment": "Experiment",
        "Features": "Feat.",

        # XAI
        "Correlation": "Corr.",
        "Importance": "Importance",
        "N": "N",

        # SLOWWATCH
        "Date": "Date",
        "TRS": "TRS",
        "Uncertainty": "Uncert.",
        "Confidence": "Conf.",
        "Risk": "Risk",
        "Warning": "Warn.",

        # Uncertainty
        "Statistic": "Statistic",
        "Value": "Value"
    })

    # --------------------------------------------------------
    # FORMAT NUMBERS
    # --------------------------------------------------------

    for col in table.columns:

        if pd.api.types.is_numeric_dtype(table[col]):

            table[col] = table[col].apply(
                lambda x: format_value(x, decimals)
            )

        else:

            table[col] = table[col].astype(str)

    # --------------------------------------------------------
    # COMPACT EVALUATION LABELS
    # --------------------------------------------------------

    if "Eval." in table.columns:

        table["Eval."] = table["Eval."].replace({

            "Chronological Test": "Chron. Test",
            "Chronological": "Chron.",
            "Validation": "Valid.",
            "Training": "Train"
        })

    # --------------------------------------------------------
    # COMPACT ABLATION LABELS
    # --------------------------------------------------------

    if "Experiment" in table.columns:

        table["Experiment"] = table["Experiment"].replace({

            "A_NDVI_ONLY":
                "A: NDVI only",

            "B_NDVI_PLUS_TEMPORAL":
                "B: NDVI + temporal",

            "C_PLUS_ENVIRONMENT":
                "C: + environment",

            "D_FULL_SLOWWATCH_FEATURES":
                "D: Full SLOWWATCH"
        })

    # --------------------------------------------------------
    # FIXED WIDTHS FOR MODEL COMPARISON
    # --------------------------------------------------------

    if set([
        "Model",
        "Eval.",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ]).issubset(table.columns):

        widths = {

            "Model": 14,
            "Eval.": 12,
            "Accuracy": 8,
            "Precision": 9,
            "Recall": 8,
            "F1": 7,
            "ROC-AUC": 8,
            "PR-AUC": 8
        }

    # --------------------------------------------------------
    # FIXED WIDTHS FOR B0
    # --------------------------------------------------------

    elif set([
        "Dataset",
        "Ref.",
        "Detect.",
        "Recall",
        "Mean Lead",
        "Median",
        "False Warn."
    ]).issubset(table.columns):

        widths = {

            "Dataset": 24,
            "Ref.": 7,
            "Detect.": 8,
            "Recall": 8,
            "Mean Lead": 10,
            "Median": 9,
            "False Warn.": 10
        }

    # --------------------------------------------------------
    # AUTOMATIC WIDTHS FOR OTHER TABLES
    # --------------------------------------------------------

    else:

        widths = {}

        for col in table.columns:

            values = [
                str(x)
                for x in table[col]
            ]

            max_length = max(
                [len(str(col))]
                + [len(x) for x in values]
            )

            widths[col] = min(
                max(max_length, 6),
                22
            )

    # --------------------------------------------------------
    # ENSURE EVERY COLUMN HAS A WIDTH
    # --------------------------------------------------------

    for col in table.columns:

        if col not in widths:

            widths[col] = min(
                max(len(str(col)), 6),
                22
            )

    # --------------------------------------------------------
    # BORDER FUNCTION
    # --------------------------------------------------------

    def border():

        return (
            "+"
            + "+".join(
                "-" * (widths[col] + 2)
                for col in table.columns
            )
            + "+"
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print(border())

    header = []

    for col in table.columns:

        header.append(
            " "
            + str(col).center(widths[col])
            + " "
        )

    print(
        "|"
        + "|".join(header)
        + "|"
    )

    print(border())

    # --------------------------------------------------------
    # DATA ROWS
    # --------------------------------------------------------

    for _, row in table.iterrows():

        cells = []

        for col in table.columns:

            value = str(row[col])

            # Prevent wrapping
            if len(value) > widths[col]:

                value = (
                    value[:widths[col] - 3]
                    + "..."
                )

            # Right-align numeric values
            try:

                float(value)

                value = value.rjust(
                    widths[col]
                )

            except ValueError:

                value = value.ljust(
                    widths[col]
                )

            cells.append(
                " "
                + value
                + " "
            )

        print(
            "|"
            + "|".join(cells)
            + "|"
        )

        print(border())

    print()


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(filename):

    path = METRICS_DIR / filename

    if not path.exists():

        print(
            f"[WARNING] Missing file: {filename}"
        )

        return None

    try:

        return pd.read_csv(path)

    except Exception as e:

        print(
            f"[WARNING] Could not read {filename}: {e}"
        )

        return None


# ============================================================
# TABLE 1
# CHRONOLOGICAL MODEL COMPARISON
# ============================================================

def build_model_comparison():

    rf = load_csv(
        "random_forest_metrics.csv"
    )

    lstm = load_csv(
        "lstm_metrics.csv"
    )

    rows = []

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    if rf is not None:

        test_rf = rf[
            rf["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]

        if not test_rf.empty:

            r = test_rf.iloc[0]

            rows.append({

                "Model":
                    "Random Forest",

                "Evaluation":
                    "Chronological Test",

                "Accuracy":
                    r.get(
                        "accuracy",
                        np.nan
                    ),

                "Precision":
                    r.get(
                        "precision",
                        np.nan
                    ),

                "Recall":
                    r.get(
                        "recall",
                        np.nan
                    ),

                "F1":
                    r.get(
                        "f1",
                        np.nan
                    ),

                "ROC-AUC":
                    r.get(
                        "roc_auc",
                        np.nan
                    ),

                "PR-AUC":
                    r.get(
                        "pr_auc",
                        np.nan
                    )
            })

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    if lstm is not None:

        test_lstm = lstm[
            lstm["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]

        if not test_lstm.empty:

            r = test_lstm.iloc[0]

            rows.append({

                "Model":
                    "LSTM",

                "Evaluation":
                    "Chronological Test",

                "Accuracy":
                    r.get(
                        "accuracy",
                        np.nan
                    ),

                "Precision":
                    r.get(
                        "precision",
                        np.nan
                    ),

                "Recall":
                    r.get(
                        "recall",
                        np.nan
                    ),

                "F1":
                    r.get(
                        "f1",
                        np.nan
                    ),

                "ROC-AUC":
                    r.get(
                        "roc_auc",
                        np.nan
                    ),

                "PR-AUC":
                    r.get(
                        "pr_auc",
                        np.nan
                    )
            })

    df = pd.DataFrame(rows)

    if not df.empty:

        df.to_csv(
            METRICS_DIR
            / "final_model_comparison.csv",
            index=False
        )

    print_box_table(
        "TABLE 1 - CHRONOLOGICAL MODEL COMPARISON",
        df,
        decimals=4
    )

    return df


# ============================================================
# TABLE 2
# B0 EARLY WARNING
# ============================================================

def build_b0_table():

    summary = load_csv(
        "b0_chronological_evaluation_summary.csv"
    )

    if summary is None:

        return pd.DataFrame()

    df = summary.copy()

    rename_map = {

        "dataset":
            "Dataset",

        "total_reference_events":
            "Ref.Events",

        "detected_events":
            "Detected",

        "event_recall":
            "Recall",

        "mean_first_warning_lead_months":
            "Mean Lead",

        "median_first_warning_lead_months":
            "Median Lead",

        "false_warning_rate":
            "False Warn."
    }

    df = df.rename(
        columns=rename_map
    )

    preferred = [

        "Dataset",
        "Ref.Events",
        "Detected",
        "Recall",
        "Mean Lead",
        "Median Lead",
        "False Warn."
    ]

    available = [

        c
        for c in preferred
        if c in df.columns
    ]

    if available:

        df = df[available]

    df.to_csv(
        METRICS_DIR
        / "final_b0_summary.csv",
        index=False
    )

    print_box_table(
        "TABLE 2 - B0 CHRONOLOGICAL EARLY-WARNING BASELINE",
        df,
        decimals=4
    )

    return df


# ============================================================
# TABLE 3
# ABLATION
# ============================================================

def build_ablation_table():

    df = load_csv(
        "slowwatch_ablation_results.csv"
    )

    if df is None:

        return pd.DataFrame()

    df = df.copy()

    rename_map = {

        "experiment":
            "Experiment",

        "feature_count":
            "Features",

        "accuracy":
            "Accuracy",

        "precision":
            "Precision",

        "recall":
            "Recall",

        "f1":
            "F1",

        "roc_auc":
            "ROC-AUC",

        "pr_auc":
            "PR-AUC"
    }

    df = df.rename(
        columns=rename_map
    )

    preferred = [

        "Experiment",
        "Features",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ]

    available = [

        c
        for c in preferred
        if c in df.columns
    ]

    if available:

        df = df[available]

    df.to_csv(
        METRICS_DIR
        / "final_ablation_table.csv",
        index=False
    )

    print_box_table(
        "TABLE 3 - FEATURE ABLATION",
        df,
        decimals=4
    )

    return df


# ============================================================
# TABLE 4
# SLOWWATCH CORE
# ============================================================

def build_slowwatch_table():

    df = load_csv(
        "slowwatch_core_results.csv"
    )

    if df is None:

        return pd.DataFrame()

    preferred = [

        "date",
        "temporal_risk_score",
        "uncertainty",
        "confidence",
        "risk_category",
        "slowwatch_warning"
    ]

    available = [

        c
        for c in preferred
        if c in df.columns
    ]

    if available:

        display_df = df[
            available
        ].copy()

    else:

        display_df = df.copy()

    rename_map = {

        "date":
            "Date",

        "temporal_risk_score":
            "TRS",

        "uncertainty":
            "Uncertainty",

        "confidence":
            "Confidence",

        "risk_category":
            "Risk",

        "slowwatch_warning":
            "Warning"
    }

    display_df = display_df.rename(
        columns=rename_map
    )

    # Only show warnings in terminal.
    # Full 77-row output remains in CSV.

    if "Warning" in display_df.columns:

        warning_mask = (

            display_df["Warning"]
            .astype(str)
            .str.lower()
            .isin([
                "1",
                "true",
                "yes"
            ])
        )

        warnings = display_df[
            warning_mask
        ].copy()

        if not warnings.empty:

            display_df = warnings

    print_box_table(
        "TABLE 4 - SLOWWATCH TEMPORAL RISK OUTPUT",
        display_df,
        decimals=4
    )

    return df


# ============================================================
# TABLE 5
# UNCERTAINTY
# ============================================================

def build_uncertainty_table():

    df = load_csv(
        "slowwatch_uncertainty_evaluation.csv"
    )

    if df is None:

        return pd.DataFrame()

    rows = []

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    if "uncertainty" in df.columns:

        uncertainty = pd.to_numeric(
            df["uncertainty"],
            errors="coerce"
        ).dropna()

        if not uncertainty.empty:

            rows.append({

                "Statistic":
                    "Mean uncertainty",

                "Value":
                    uncertainty.mean()
            })

            rows.append({

                "Statistic":
                    "Median uncertainty",

                "Value":
                    uncertainty.median()
            })

            rows.append({

                "Statistic":
                    "Minimum uncertainty",

                "Value":
                    uncertainty.min()
            })

            rows.append({

                "Statistic":
                    "Maximum uncertainty",

                "Value":
                    uncertainty.max()
            })

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if "confidence" in df.columns:

        confidence = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        ).dropna()

        if not confidence.empty:

            rows.append({

                "Statistic":
                    "Mean confidence",

                "Value":
                    confidence.mean()
            })

            rows.append({

                "Statistic":
                    "Median confidence",

                "Value":
                    confidence.median()
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        METRICS_DIR
        / "final_uncertainty_summary.csv",
        index=False
    )

    print_box_table(
        "TABLE 5 - UNCERTAINTY SUMMARY",
        result,
        decimals=4
    )

    return result


# ============================================================
# TABLE 6
# XAI
# ============================================================

def build_xai_table():

    df = load_csv(
        "slowwatch_xai_results.csv"
    )

    if df is None:

        return pd.DataFrame()

    preferred = [

        "feature",
        "correlation_with_temporal_risk_score",
        "absolute_importance",
        "valid_observations"
    ]

    available = [

        c
        for c in preferred
        if c in df.columns
    ]

    if available:

        result = df[
            available
        ].copy()

    else:

        result = df.copy()

    rename_map = {

        "feature":
            "Feature",

        "correlation_with_temporal_risk_score":
            "Correlation",

        "absolute_importance":
            "Importance",

        "valid_observations":
            "N"
    }

    result = result.rename(
        columns=rename_map
    )

    print_box_table(
        "TABLE 6 - FEATURE-RISK CONTRIBUTION ANALYSIS",
        result,
        decimals=4
    )

    return result


# ============================================================
# MODEL COMPARISON FIGURE
# ============================================================

def create_model_figure(model_df):

    if model_df is None:
        return

    if model_df.empty:
        return

    metrics = [

        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ]

    available = [

        m
        for m in metrics
        if m in model_df.columns
    ]

    if not available:
        return

    plot_df = model_df.set_index(
        "Model"
    )[available]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title(
        "SLOWWATCH - Chronological Model Comparison"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylim(
        0,
        1.05
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ABLATION FIGURE
# ============================================================

def create_ablation_figure(ablation_df):

    if ablation_df is None:
        return

    if ablation_df.empty:
        return

    if "Experiment" not in ablation_df.columns:
        return

    if "F1" not in ablation_df.columns:
        return

    values = pd.to_numeric(
        ablation_df["F1"],
        errors="coerce"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        ablation_df["Experiment"].astype(str),
        values
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.xlabel(
        "Experiment"
    )

    plt.title(
        "SLOWWATCH - Feature Ablation Comparison"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_ablation_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# RESEARCH SUMMARY FILE
# ============================================================

def create_research_summary(
    model_df,
    b0_df,
    ablation_df,
    uncertainty_df
):

    lines = []

    lines.append(
        "SLOWWATCH - FINAL RESEARCH RESULTS SUMMARY"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "Study region: Kodagu, Karnataka, India"
    )

    lines.append(
        "Temporal period: 2018-01 to 2025-12"
    )

    lines.append(
        "Monthly observations: 77"
    )

    lines.append("")

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    lines.append(
        "MODEL EVALUATION"
    )

    lines.append(
        "-" * 70
    )

    if (
        model_df is not None
        and not model_df.empty
    ):

        for _, row in model_df.iterrows():

            lines.append(

                f"{row['Model']}: "
                f"Accuracy={row['Accuracy']:.4f}, "
                f"Precision={row['Precision']:.4f}, "
                f"Recall={row['Recall']:.4f}, "
                f"F1={row['F1']:.4f}, "
                f"ROC-AUC={row['ROC-AUC']:.4f}, "
                f"PR-AUC={row['PR-AUC']:.4f}"
            )

    lines.append("")

    # --------------------------------------------------------
    # B0
    # --------------------------------------------------------

    lines.append(
        "EARLY-WARNING BASELINE"
    )

    lines.append(
        "-" * 70
    )

    if (
        b0_df is not None
        and not b0_df.empty
    ):

        for _, row in b0_df.iterrows():

            dataset = row.get(
                "Dataset",
                "Unknown"
            )

            recall = row.get(
                "Recall",
                np.nan
            )

            mean_lead = row.get(
                "Mean Lead",
                np.nan
            )

            lines.append(

                f"{dataset}: "
                f"Recall={recall}, "
                f"Mean Lead={mean_lead}"
            )

    lines.append("")

    # --------------------------------------------------------
    # ABLATION
    # --------------------------------------------------------

    lines.append(
        "ABLATION"
    )

    lines.append(
        "-" * 70
    )

    if (
        ablation_df is not None
        and not ablation_df.empty
    ):

        for _, row in ablation_df.iterrows():

            try:
                f1 = float(
                    row.get(
                        "F1",
                        np.nan
                    )
                )
            except:
                f1 = np.nan

            try:
                roc = float(
                    row.get(
                        "ROC-AUC",
                        np.nan
                    )
                )
            except:
                roc = np.nan

            try:
                pr = float(
                    row.get(
                        "PR-AUC",
                        np.nan
                    )
                )
            except:
                pr = np.nan

            lines.append(

                f"{row.get('Experiment', '')}: "
                f"F1={f1:.4f}, "
                f"ROC-AUC={roc:.4f}, "
                f"PR-AUC={pr:.4f}"
            )

    lines.append("")

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    lines.append(
        "UNCERTAINTY"
    )

    lines.append(
        "-" * 70
    )

    if (
        uncertainty_df is not None
        and not uncertainty_df.empty
    ):

        for _, row in uncertainty_df.iterrows():

            try:

                value = float(
                    row["Value"]
                )

                lines.append(
                    f"{row['Statistic']}: "
                    f"{value:.4f}"
                )

            except:

                pass

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "Results generated from chronological evaluation outputs."
    )

    lines.append(
        "Results are experiment-specific and should be interpreted "
        "with the corresponding sample sizes and evaluation protocol."
    )

    with open(
        METRICS_DIR
        / "final_research_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 96
    )

    print(
        "SLOWWATCH - FINAL RESEARCH RESULTS".center(96)
    )

    print(
        "=" * 96
    )

    print()

    print(
        "Generating publication-style research tables..."
    )

    print()

    # --------------------------------------------------------
    # TABLE 1
    # --------------------------------------------------------

    model_df = build_model_comparison()

    # --------------------------------------------------------
    # TABLE 2
    # --------------------------------------------------------

    b0_df = build_b0_table()

    # --------------------------------------------------------
    # TABLE 3
    # --------------------------------------------------------

    ablation_df = build_ablation_table()

    # --------------------------------------------------------
    # TABLE 4
    # --------------------------------------------------------

    slowwatch_df = build_slowwatch_table()

    # --------------------------------------------------------
    # TABLE 5
    # --------------------------------------------------------

    uncertainty_df = build_uncertainty_table()

    # --------------------------------------------------------
    # TABLE 6
    # --------------------------------------------------------

    xai_df = build_xai_table()

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    create_model_figure(
        model_df
    )

    create_ablation_figure(
        ablation_df
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    create_research_summary(

        model_df,

        b0_df,

        ablation_df,

        uncertainty_df
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()

    print(
        "=" * 96
    )

    print(
        "SLOWWATCH FINAL RESULTS PACKAGE COMPLETED".center(96)
    )

    print(
        "=" * 96
    )

    print()

    print(
        "Created / updated:"
    )

    print(
        "  results/metrics/final_model_comparison.csv"
    )

    print(
        "  results/metrics/final_b0_summary.csv"
    )

    print(
        "  results/metrics/final_ablation_table.csv"
    )

    print(
        "  results/metrics/final_uncertainty_summary.csv"
    )

    print(
        "  results/metrics/final_research_summary.txt"
    )

    print(
        "  results/figures/final_model_comparison.png"
    )

    print(
        "  results/figures/final_ablation_comparison.png"
    )

    print()

    print(
        "READY FOR CONFERENCE PAPER."
    )

    print()


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    main()