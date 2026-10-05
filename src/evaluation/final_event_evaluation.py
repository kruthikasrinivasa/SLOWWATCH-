"""
SLOWWATCH - FINAL EVENT-LEVEL EVALUATION
========================================

Purpose:
    Finalize the Kodagu event-level early-warning evaluation
    using the already-generated SLOWWATCH / RF / LSTM outputs.

Inputs:
    data/features/slowwatch_multisource_dataset.csv
    data/events/event_labels.csv
    results/metrics/random_forest_test_predictions.csv
    results/metrics/lstm_test_predictions.csv
    results/metrics/slowwatch_core_results.csv

Outputs:
    results/metrics/final_event_level_evaluation.csv
    results/metrics/final_warning_lead_times.csv
    results/metrics/final_region_summary.csv
    results/figures/final_warning_event_timeline.png
    results/figures/final_lead_time_distribution.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data" / "features" / "slowwatch_multisource_dataset.csv"
EVENT_FILE = ROOT / "data" / "events" / "event_labels.csv"

RF_FILE = ROOT / "results" / "metrics" / "random_forest_test_predictions.csv"
LSTM_FILE = ROOT / "results" / "metrics" / "lstm_test_predictions.csv"
SW_FILE = ROOT / "results" / "metrics" / "slowwatch_core_results.csv"

METRIC_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"

METRIC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Event is considered detected if a warning occurs
# within this many months before/at the event start.
WARNING_WINDOW_MONTHS = 3

# Maximum warning lead time reported.
MAX_LEAD_MONTHS = 12


# ============================================================
# HELPERS
# ============================================================

def load_csv(path, name):
    """Load CSV safely."""
    if not path.exists():
        print(f"[WARNING] {name} not found:")
        print(f"         {path}")
        return None

    df = pd.read_csv(path)

    print(f"[OK] {name}")
    print(f"     Shape: {df.shape}")

    return df


def find_column(df, candidates):
    """
    Return the first matching column from candidates.
    Matching is case-insensitive.
    """
    if df is None:
        return None

    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def convert_date_column(df, candidates):
    """Find and convert date column."""
    if df is None:
        return df, None

    col = find_column(df, candidates)

    if col is None:
        return df, None

    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")

    return df, col


def month_difference(start, warning):
    """
    Event start minus warning date in months.
    Positive = warning before event.
    """
    return (
        (start.year - warning.year) * 12
        + (start.month - warning.month)
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("SLOWWATCH - FINAL KODAGU EVENT-LEVEL EVALUATION")
print("=" * 78)
print()

features = load_csv(
    FEATURE_FILE,
    "Feature dataset"
)

events = load_csv(
    EVENT_FILE,
    "Reference event labels"
)

rf = load_csv(
    RF_FILE,
    "Random Forest predictions"
)

lstm = load_csv(
    LSTM_FILE,
    "LSTM predictions"
)

slowwatch = load_csv(
    SW_FILE,
    "SLOWWATCH core results"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if features is None:
    raise FileNotFoundError(
        "Feature dataset is missing."
    )

if events is None:
    raise FileNotFoundError(
        "Reference event file is missing."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

features, feature_date_col = convert_date_column(
    features,
    ["date", "year_month"]
)

if feature_date_col is None:
    raise ValueError(
        "Could not find a date column in the feature dataset."
    )

features = features.dropna(
    subset=[feature_date_col]
).sort_values(
    feature_date_col
).reset_index(drop=True)


# ============================================================
# PREPARE EVENT LABELS
# ============================================================

events = events.copy()

event_start_col = find_column(
    events,
    [
        "event_start",
        "start_date",
        "event_date",
        "date"
    ]
)

if event_start_col is None:
    raise ValueError(
        "Could not find event_start/date column in event_labels.csv."
    )

events[event_start_col] = pd.to_datetime(
    events[event_start_col],
    errors="coerce"
)

events = events.dropna(
    subset=[event_start_col]
).sort_values(
    event_start_col
).reset_index(drop=True)


# Event ID if available
event_id_col = find_column(
    events,
    [
        "event_id",
        "id",
        "event"
    ]
)

if event_id_col is None:
    events["event_id_generated"] = [
        f"KODAGU_EVENT_{i+1:03d}"
        for i in range(len(events))
    ]
    event_id_col = "event_id_generated"


# ============================================================
# DETERMINE EVENT DRIVER
# ============================================================

driver_col = find_column(
    events,
    [
        "driver",
        "event_type",
        "type",
        "cause"
    ]
)

if driver_col is None:
    events["driver_generated"] = "reference_event"
    driver_col = "driver_generated"


# ============================================================
# PREPARE MODEL PREDICTIONS
# ============================================================

def prepare_predictions(df, model_name):
    """
    Convert model prediction CSV into:
        date
        predicted_class
        probability
    """

    if df is None:
        return None

    df = df.copy()

    date_col = find_column(
        df,
        [
            "date",
            "year_month"
        ]
    )

    pred_class_col = find_column(
        df,
        [
            "predicted_class",
            "prediction",
            "predicted",
            "class",
            "pred"
        ]
    )

    prob_col = find_column(
        df,
        [
            "predicted_probability",
            "probability",
            "predicted_prob",
            "positive_probability",
            "risk_probability"
        ]
    )

    if date_col is None:
        print(
            f"[WARNING] {model_name}: date column not found."
        )
        return None

    if pred_class_col is None:
        print(
            f"[WARNING] {model_name}: prediction column not found."
        )
        return None

    df["date_eval"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["prediction_eval"] = pd.to_numeric(
        df[pred_class_col],
        errors="coerce"
    )

    if prob_col is not None:
        df["probability_eval"] = pd.to_numeric(
            df[prob_col],
            errors="coerce"
        )
    else:
        df["probability_eval"] = np.nan

    df = df[
        ["date_eval", "prediction_eval", "probability_eval"]
    ].dropna(
        subset=["date_eval", "prediction_eval"]
    ).sort_values(
        "date_eval"
    )

    df["prediction_eval"] = (
        df["prediction_eval"]
        .round()
        .astype(int)
    )

    df["model"] = model_name

    return df


rf_pred = prepare_predictions(
    rf,
    "Random Forest"
)

lstm_pred = prepare_predictions(
    lstm,
    "LSTM"
)


# ============================================================
# PREPARE SLOWWATCH CORE
# ============================================================

def prepare_slowwatch(df):
    """
    Convert SLOWWATCH core output to warning dates.
    """

    if df is None:
        return None

    df = df.copy()

    date_col = find_column(
        df,
        [
            "date",
            "year_month"
        ]
    )

    warning_col = find_column(
        df,
        [
            "slowwatch_warning",
            "warning",
            "risk_warning",
            "alert"
        ]
    )

    if date_col is None:
        print(
            "[WARNING] SLOWWATCH core: date column not found."
        )
        return None

    if warning_col is None:
        print(
            "[WARNING] SLOWWATCH core: warning column not found."
        )
        print(
            "Available columns:",
            list(df.columns)
        )
        return None

    df["date_eval"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["warning_eval"] = pd.to_numeric(
        df[warning_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date_eval", "warning_eval"]
    ).sort_values(
        "date_eval"
    )

    df["warning_eval"] = (
        df["warning_eval"]
        .round()
        .astype(int)
    )

    # Only actual warnings
    warnings_df = df[
        df["warning_eval"] == 1
    ].copy()

    # Optional risk information
    risk_col = find_column(
        df,
        [
            "temporal_risk_score",
            "risk_score",
            "trs",
            "risk"
        ]
    )

    uncertainty_col = find_column(
        df,
        [
            "uncertainty",
            "risk_uncertainty"
        ]
    )

    confidence_col = find_column(
        df,
        [
            "confidence",
            "risk_confidence"
        ]
    )

    if risk_col:
        warnings_df["risk_score"] = pd.to_numeric(
            warnings_df[risk_col],
            errors="coerce"
        )
    else:
        warnings_df["risk_score"] = np.nan

    if uncertainty_col:
        warnings_df["uncertainty"] = pd.to_numeric(
            warnings_df[uncertainty_col],
            errors="coerce"
        )
    else:
        warnings_df["uncertainty"] = np.nan

    if confidence_col:
        warnings_df["confidence"] = pd.to_numeric(
            warnings_df[confidence_col],
            errors="coerce"
        )
    else:
        warnings_df["confidence"] = np.nan

    return warnings_df[
        [
            "date_eval",
            "risk_score",
            "uncertainty",
            "confidence"
        ]
    ].reset_index(drop=True)


sw_warnings = prepare_slowwatch(
    slowwatch
)


# ============================================================
# EVENT MATCHING FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    warning_dates,
    events_df,
    event_start_column
):
    """
    Match model warnings against reference events.

    A reference event is detected when at least one warning
    occurs in the preceding WARNING_WINDOW_MONTHS.
    """

    if warning_dates is None:
        return pd.DataFrame()

    warning_dates = sorted(
        pd.to_datetime(warning_dates).tolist()
    )

    records = []

    for _, event in events_df.iterrows():

        event_id = event[event_id_col]

        event_start = event[event_start_column]

        matched_warning = None
        lead_months = None

        # Search warnings closest to event first
        candidate_warnings = []

        for warning_date in warning_dates:

            lead = month_difference(
                event_start,
                warning_date
            )

            # warning before or at event
            if 0 <= lead <= WARNING_WINDOW_MONTHS:

                candidate_warnings.append(
                    (warning_date, lead)
                )

        if candidate_warnings:

            # earliest warning = maximum lead
            matched_warning, lead_months = max(
                candidate_warnings,
                key=lambda x: x[1]
            )

            detected = 1

        else:

            detected = 0

        records.append(
            {
                "model": model_name,
                "event_id": event_id,
                "driver": event[driver_col],
                "event_start": event_start,
                "detected": detected,
                "warning_date": matched_warning,
                "lead_time_months": lead_months
            }
        )

    return pd.DataFrame(records)


# ============================================================
# SLOWWATCH EVENT EVALUATION
# ============================================================

if sw_warnings is not None:

    sw_event_results = evaluate_model(
        "SLOWWATCH",
        sw_warnings["date_eval"].tolist(),
        events,
        event_start_col
    )

else:

    sw_event_results = pd.DataFrame()


# ============================================================
# RANDOM FOREST EVENT EVALUATION
# ============================================================

if rf_pred is not None:

    rf_warning_dates = rf_pred.loc[
        rf_pred["prediction_eval"] == 1,
        "date_eval"
    ].tolist()

    rf_event_results = evaluate_model(
        "Random Forest",
        rf_warning_dates,
        events,
        event_start_col
    )

else:

    rf_event_results = pd.DataFrame()


# ============================================================
# LSTM EVENT EVALUATION
# ============================================================

if lstm_pred is not None:

    lstm_warning_dates = lstm_pred.loc[
        lstm_pred["prediction_eval"] == 1,
        "date_eval"
    ].tolist()

    lstm_event_results = evaluate_model(
        "LSTM",
        lstm_warning_dates,
        events,
        event_start_col
    )

else:

    lstm_event_results = pd.DataFrame()


# ============================================================
# COMBINE EVENT RESULTS
# ============================================================

event_result_frames = [
    sw_event_results,
    rf_event_results,
    lstm_event_results
]

event_result_frames = [
    x for x in event_result_frames
    if x is not None and not x.empty
]

if not event_result_frames:
    raise RuntimeError(
        "No model event results could be generated."
    )

all_event_results = pd.concat(
    event_result_frames,
    ignore_index=True
)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary_records = []

for model_name, group in all_event_results.groupby("model"):

    total_events = len(group)

    detected_events = int(
        group["detected"].sum()
    )

    missed_events = (
        total_events -
        detected_events
    )

    recall = (
        detected_events / total_events
        if total_events > 0
        else np.nan
    )

    detected_leads = group.loc[
        group["detected"] == 1,
        "lead_time_months"
    ].dropna()

    if len(detected_leads) > 0:

        mean_lead = detected_leads.mean()
        median_lead = detected_leads.median()
        max_lead = detected_leads.max()

    else:

        mean_lead = np.nan
        median_lead = np.nan
        max_lead = np.nan

    summary_records.append(
        {
            "model": model_name,
            "reference_events": total_events,
            "detected_events": detected_events,
            "missed_events": missed_events,
            "event_recall": recall,
            "mean_lead_months": mean_lead,
            "median_lead_months": median_lead,
            "maximum_lead_months": max_lead
        }
    )


summary = pd.DataFrame(
    summary_records
)


# ============================================================
# WARNING COUNTS + FALSE WARNING RATE
# ============================================================

def add_warning_statistics(
    summary_df,
    model_name,
    warning_dates
):

    if warning_dates is None:

        return summary_df

    warning_dates = pd.to_datetime(
        warning_dates
    )

    warning_dates = sorted(
        set(warning_dates)
    )

    model_rows = summary_df[
        summary_df["model"] == model_name
    ]

    if model_rows.empty:
        return summary_df

    detected_events = int(
        model_rows.iloc[0]["detected_events"]
    )

    total_warnings = len(
        warning_dates
    )

    false_warnings = max(
        total_warnings -
        detected_events,
        0
    )

    false_warning_rate = (
        false_warnings / total_warnings
        if total_warnings > 0
        else np.nan
    )

    summary_df.loc[
        summary_df["model"] == model_name,
        "warning_periods"
    ] = total_warnings

    summary_df.loc[
        summary_df["model"] == model_name,
        "false_warning_periods"
    ] = false_warnings

    summary_df.loc[
        summary_df["model"] == model_name,
        "false_warning_rate"
    ] = false_warning_rate

    return summary_df


summary = add_warning_statistics(
    summary,
    "SLOWWATCH",
    (
        sw_warnings["date_eval"].tolist()
        if sw_warnings is not None
        else None
    )
)

summary = add_warning_statistics(
    summary,
    "Random Forest",
    (
        rf_warning_dates
        if rf_pred is not None
        else None
    )
)

summary = add_warning_statistics(
    summary,
    "LSTM",
    (
        lstm_warning_dates
        if lstm_pred is not None
        else None
    )
)


# ============================================================
# CLEAN SUMMARY TYPES
# ============================================================

for col in [
    "reference_events",
    "detected_events",
    "missed_events",
    "warning_periods",
    "false_warning_periods"
]:

    if col in summary.columns:

        summary[col] = (
            pd.to_numeric(
                summary[col],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )


# ============================================================
# SAVE RESULTS
# ============================================================

event_output = (
    METRIC_DIR /
    "final_event_level_evaluation.csv"
)

summary_output = (
    METRIC_DIR /
    "final_region_summary.csv"
)

lead_output = (
    METRIC_DIR /
    "final_warning_lead_times.csv"
)

all_event_results.to_csv(
    event_output,
    index=False
)

summary.to_csv(
    summary_output,
    index=False
)

lead_results = all_event_results[
    all_event_results["detected"] == 1
].copy()

lead_results.to_csv(
    lead_output,
    index=False
)


# ============================================================
# TERMINAL TABLE
# ============================================================

print()
print("=" * 78)
print("FINAL EVENT-LEVEL RESULTS")
print("=" * 78)
print()

display_summary = summary.copy()

numeric_cols = [
    "event_recall",
    "mean_lead_months",
    "median_lead_months",
    "maximum_lead_months",
    "false_warning_rate"
]

for col in numeric_cols:

    if col in display_summary.columns:

        display_summary[col] = (
            pd.to_numeric(
                display_summary[col],
                errors="coerce"
            )
            .round(4)
        )


print(
    display_summary.to_string(
        index=False
    )
)


# ============================================================
# LEAD-TIME SUMMARY
# ============================================================

print()
print("=" * 78)
print("DETECTED EVENT LEAD-TIME SUMMARY")
print("=" * 78)
print()

for model_name in all_event_results["model"].unique():

    leads = all_event_results.loc[
        (all_event_results["model"] == model_name)
        &
        (all_event_results["detected"] == 1),
        "lead_time_months"
    ].dropna()

    print(f"{model_name}")
    print("-" * 40)

    if len(leads) == 0:

        print("No matched events.")

    else:

        print(
            f"Detected events : {len(leads)}"
        )

        print(
            f"Mean lead       : {leads.mean():.2f} months"
        )

        print(
            f"Median lead     : {leads.median():.2f} months"
        )

        print(
            f"Maximum lead    : {leads.max():.2f} months"
        )

    print()


# ============================================================
# FIGURE 1 - LEAD TIME DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plotted = False

for model_name in all_event_results["model"].unique():

    leads = all_event_results.loc[
        (all_event_results["model"] == model_name)
        &
        (all_event_results["detected"] == 1),
        "lead_time_months"
    ].dropna()

    if len(leads) > 0:

        plt.hist(
            leads,
            bins=np.arange(
                -0.5,
                max(
                    leads.max() + 1,
                    1.5
                ),
                1
            ),
            alpha=0.5,
            label=model_name
        )

        plotted = True

if plotted:

    plt.xlabel(
        "Warning Lead Time (months)"
    )

    plt.ylabel(
        "Number of Detected Events"
    )

    plt.title(
        "SLOWWATCH Event-Level Warning Lead Time"
    )

    plt.legend()
    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        "final_lead_time_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 2 - EVENT TIMELINE
# ============================================================

plt.figure(figsize=(14, 7))

event_plot = events.copy()

event_plot["event_index"] = np.arange(
    1,
    len(event_plot) + 1
)

plt.scatter(
    event_plot[event_start_col],
    event_plot["event_index"],
    marker="x",
    s=60,
    label="Reference events"
)

if sw_warnings is not None and not sw_warnings.empty:

    plt.scatter(
        sw_warnings["date_eval"],
        np.full(
            len(sw_warnings),
            len(event_plot) + 2
        ),
        marker="o",
        s=55,
        label="SLOWWATCH warnings"
    )

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Reference Event Index"
)

plt.title(
    "Kodagu Reference Events and SLOWWATCH Warning Periods"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "final_warning_event_timeline.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 78)
print("KODAGU EVENT-LEVEL EVALUATION COMPLETED")
print("=" * 78)
print()

print("Created:")
print(
    f"  {event_output.relative_to(ROOT)}"
)
print(
    f"  {summary_output.relative_to(ROOT)}"
)
print(
    f"  {lead_output.relative_to(ROOT)}"
)
print(
    f"  {FIG_DIR / 'final_warning_event_timeline.png'}"
)
print(
    f"  {FIG_DIR / 'final_lead_time_distribution.png'}"
)

print()
print("Next stage:")
print("  1. Freeze Kodagu results")
print("  2. Build Wayanad dataset")
print("  3. Build Nilgiris dataset")
print("  4. Run cross-region validation")
print()