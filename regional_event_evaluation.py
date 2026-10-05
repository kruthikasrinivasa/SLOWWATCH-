"""
SLOWWATCH Regional Event-Level Early-Warning Evaluation

Evaluates cross-region Random Forest predictions against
FIRMS-derived regional event references.

Experiments:
    Wayanad -> Nilgiris
    Nilgiris -> Wayanad

Metrics:
    - reference events
    - detected events
    - event recall
    - mean warning lead time
    - median warning lead time
    - maximum warning lead time
    - false warning rate

IMPORTANT:
FIRMS events are treated as disturbance-reference events,
NOT confirmed vegetation-degradation ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

REGIONAL_DIR = (
    PROJECT_DIR
    / "data"
    / "features"
    / "regional"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

EXPERIMENTS = {
    "Wayanad_to_Nilgiris": {
        "predictions": (
            RESULTS_DIR
            / "regional_cv_wayanad_to_nilgiris_predictions.csv"
        ),
        "labels": (
            REGIONAL_DIR
            / "nilgiris_event_labels.csv"
        ),
    },

    "Nilgiris_to_Wayanad": {
        "predictions": (
            RESULTS_DIR
            / "regional_cv_nilgiris_to_wayanad_predictions.csv"
        ),
        "labels": (
            REGIONAL_DIR
            / "wayanad_event_labels.csv"
        ),
    },
}


# ============================================================
# HELPERS
# ============================================================

def month_difference(later_date, earlier_date):
    """
    Number of calendar months between two dates.
    """

    later = pd.Timestamp(later_date)
    earlier = pd.Timestamp(earlier_date)

    return (
        (later.year - earlier.year) * 12
        + (later.month - earlier.month)
    )


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Label file not found:\n{path}"
        )

    labels = pd.read_csv(path)

    labels["year_month"] = pd.to_datetime(
        labels["date"],
        errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()

    return labels


# ============================================================
# EVALUATE ONE EXPERIMENT
# ============================================================

def evaluate_experiment(
    experiment_name,
    prediction_path,
    label_path,
):

    print("\n" + "=" * 75)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{prediction_path}"
        )

    predictions = pd.read_csv(
        prediction_path
    )

    labels = load_labels(
        label_path
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    predictions["year_month"] = pd.to_datetime(
        predictions["year_month"],
        errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()

    # --------------------------------------------------------
    # PREDICTED WARNING MONTHS
    # --------------------------------------------------------

    predictions["predicted_event"] = (
        pd.to_numeric(
            predictions["predicted_event"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    warning_months = (
        predictions[
            predictions["predicted_event"] == 1
        ]
        .copy()
        .sort_values("year_month")
    )

    # --------------------------------------------------------
    # REFERENCE EVENT MONTHS
    # --------------------------------------------------------

    event_labels = labels[
        labels["firms_event_target"] == 1
    ].copy()

    event_labels = event_labels[
        event_labels["firms_event_id"].notna()
    ].copy()

    event_ids = (
        event_labels[
            "firms_event_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    total_reference_events = len(
        event_ids
    )

    # --------------------------------------------------------
    # EVENT START DATE LOOKUP
    # --------------------------------------------------------

    event_start_lookup = (
        event_labels[
            [
                "firms_event_id",
                "year_month",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["firms_event_id"]
        )
        .copy()
    )

    event_start_lookup[
        "firms_event_id"
    ] = (
        event_start_lookup[
            "firms_event_id"
        ]
        .astype(str)
    )

    event_start_lookup = dict(
        zip(
            event_start_lookup[
                "firms_event_id"
            ],
            event_start_lookup[
                "year_month"
            ],
        )
    )

    # --------------------------------------------------------
    # EVENT-LEVEL DETECTION
    # --------------------------------------------------------

    detected_events = []

    for event_id in event_ids:

        event_id = str(event_id)

        if event_id not in event_start_lookup:
            continue

        event_start = event_start_lookup[
            event_id
        ]

        # Warnings associated with this event.
        event_warnings = warning_months[
            warning_months[
                "firms_event_id"
            ].astype(str) == event_id
        ].copy()

        if len(event_warnings) == 0:
            continue

        # Keep only warnings at or before event month.
        event_warnings = event_warnings[
            event_warnings["year_month"]
            <= event_start
        ].copy()

        if len(event_warnings) == 0:
            continue

        # Earliest warning gives maximum lead time.
        earliest_warning = (
            event_warnings[
                "year_month"
            ].min()
        )

        lead_time = month_difference(
            event_start,
            earliest_warning
        )

        # We only count warnings occurring within the
        # defined 3-month pre-event window or event month.
        if 0 <= lead_time <= 3:

            detected_events.append({
                "event_id": event_id,
                "event_start": event_start,
                "warning_month": earliest_warning,
                "lead_time_months": lead_time,
            })

    detected_df = pd.DataFrame(
        detected_events
    )

    detected_event_count = len(
        detected_df
    )

    # --------------------------------------------------------
    # EVENT RECALL
    # --------------------------------------------------------

    if total_reference_events > 0:

        event_recall = (
            detected_event_count
            / total_reference_events
        )

    else:

        event_recall = np.nan

    # --------------------------------------------------------
    # LEAD TIME STATISTICS
    # --------------------------------------------------------

    if detected_event_count > 0:

        mean_lead = (
            detected_df[
                "lead_time_months"
            ].mean()
        )

        median_lead = (
            detected_df[
                "lead_time_months"
            ].median()
        )

        max_lead = (
            detected_df[
                "lead_time_months"
            ].max()
        )

    else:

        mean_lead = np.nan
        median_lead = np.nan
        max_lead = np.nan

    # --------------------------------------------------------
    # FALSE WARNINGS
    # --------------------------------------------------------

    valid_warning_months = 0

    false_warning_months = 0

    for _, warning in warning_months.iterrows():

        warning_date = warning[
            "year_month"
        ]

        event_id = warning.get(
            "firms_event_id",
            np.nan
        )

        if pd.isna(event_id):

            false_warning_months += 1
            continue

        event_id = str(event_id)

        if event_id not in event_start_lookup:

            false_warning_months += 1
            continue

        event_start = event_start_lookup[
            event_id
        ]

        lead = month_difference(
            event_start,
            warning_date
        )

        # Valid warning = event month or
        # up to 3 months before event.
        if 0 <= lead <= 3:

            valid_warning_months += 1

        else:

            false_warning_months += 1

    total_warning_months = len(
        warning_months
    )

    if total_warning_months > 0:

        false_warning_rate = (
            false_warning_months
            / total_warning_months
        )

    else:

        false_warning_rate = np.nan

    # --------------------------------------------------------
    # SAVE EVENT-LEVEL RESULTS
    # --------------------------------------------------------

    event_output = (
        RESULTS_DIR
        / (
            "regional_event_lead_times_"
            f"{experiment_name.lower()}.csv"
        )
    )

    if len(detected_df) > 0:

        detected_df.to_csv(
            event_output,
            index=False
        )

    else:

        pd.DataFrame(
            columns=[
                "event_id",
                "event_start",
                "warning_month",
                "lead_time_months",
            ]
        ).to_csv(
            event_output,
            index=False
        )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nReference events      :", total_reference_events)
    print("Detected events       :", detected_event_count)
    print(
        "Event recall          : "
        f"{event_recall:.4f}"
    )

    print(
        "Mean lead time (mo)   : "
        f"{mean_lead:.4f}"
        if not np.isnan(mean_lead)
        else "Mean lead time (mo)   : N/A"
    )

    print(
        "Median lead time (mo) : "
        f"{median_lead:.4f}"
        if not np.isnan(median_lead)
        else "Median lead time (mo) : N/A"
    )

    print(
        "Maximum lead time (mo) : "
        f"{max_lead:.4f}"
        if not np.isnan(max_lead)
        else "Maximum lead time (mo) : N/A"
    )

    print(
        "Warning months         : "
        f"{total_warning_months}"
    )

    print(
        "Valid warning months   : "
        f"{valid_warning_months}"
    )

    print(
        "False warning months   : "
        f"{false_warning_months}"
    )

    print(
        "False warning rate     : "
        f"{false_warning_rate:.4f}"
        if not np.isnan(false_warning_rate)
        else "False warning rate     : N/A"
    )

    print("\nSaved:")
    print(event_output)

    return {
        "experiment": experiment_name,
        "reference_events": total_reference_events,
        "detected_events": detected_event_count,
        "event_recall": event_recall,
        "mean_lead_time_months": mean_lead,
        "median_lead_time_months": median_lead,
        "max_lead_time_months": max_lead,
        "warning_months": total_warning_months,
        "valid_warning_months": valid_warning_months,
        "false_warning_months": false_warning_months,
        "false_warning_rate": false_warning_rate,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SLOWWATCH REGIONAL EVENT-LEVEL EVALUATION")
    print("=" * 75)

    results = []

    for experiment_name, config in EXPERIMENTS.items():

        try:

            result = evaluate_experiment(
                experiment_name,
                config["predictions"],
                config["labels"],
            )

            results.append(
                result
            )

        except Exception as exc:

            print("\n" + "!" * 75)
            print(
                f"FAILED: {experiment_name}"
            )
            print("!" * 75)
            print(exc)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    if not results:

        print(
            "\nNo experiments completed."
        )
        return

    summary = pd.DataFrame(
        results
    )

    summary_path = (
        RESULTS_DIR
        / "regional_event_evaluation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 75)
    print("FINAL REGIONAL EVENT-LEVEL RESULTS")
    print("=" * 75)

    display_columns = [
        "experiment",
        "reference_events",
        "detected_events",
        "event_recall",
        "mean_lead_time_months",
        "median_lead_time_months",
        "max_lead_time_months",
        "warning_months",
        "false_warning_rate",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSaved:")
    print(summary_path)

    print("\n" + "=" * 75)
    print(
        "REGIONAL EVENT EVALUATION COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()