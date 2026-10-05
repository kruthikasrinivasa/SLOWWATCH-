"""
SLOWWATCH — B0 Chronological Baseline Evaluation

B0:
    Negative NDVI anomaly + negative 6-month NDVI slope

Reference events:
    Independent FIRMS-derived fire disturbance events.

Evaluation:
    6-month pre-event warning window.

Important limitation:
    The current SLOWWATCH feature dataset is a temporal
    regional summary and does not yet contain spatial
    coordinates matching individual FIRMS events.
    Therefore this first evaluation is TEMPORAL ONLY.

No future event information is used as a model input.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

FEATURE_FILE = Path(
    "data/features/slowwatch_multisource_dataset.csv"
)

EVENT_FILE = Path(
    "data/events/event_labels.csv"
)

SENSITIVITY_FILE = Path(
    "data/events/event_labels_sensitivity_10detections.csv"
)

OUTPUT_DIR = Path(
    "results/metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_FILE = (
    OUTPUT_DIR /
    "b0_chronological_evaluation_summary.csv"
)

EVENT_RESULTS_FILE = (
    OUTPUT_DIR /
    "b0_event_level_results.csv"
)

SENSITIVITY_RESULTS_FILE = (
    OUTPUT_DIR /
    "b0_sensitivity_event_results.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

LOOKBACK_MONTHS = 6


# ============================================================
# LOAD FEATURE DATA
# ============================================================

print("=" * 70)
print("SLOWWATCH — B0 CHRONOLOGICAL BASELINE EVALUATION")
print("=" * 70)

print("\nLoading feature dataset...")

features = pd.read_csv(
    FEATURE_FILE
)

events = pd.read_csv(
    EVENT_FILE
)

sensitivity_events = pd.read_csv(
    SENSITIVITY_FILE
)


# ============================================================
# PREPARE FEATURE DATA
# ============================================================

features["date"] = pd.to_datetime(
    features["date"],
    errors="coerce"
)

features["year_month"] = pd.to_datetime(
    features["year_month"].astype(str),
    errors="coerce"
)

features = features.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# NORMALIZE B0 WARNING COLUMN
# ============================================================

if "baseline_warning" not in features.columns:

    raise KeyError(
        "baseline_warning column not found "
        "in the feature dataset."
    )


def normalize_warning(value):

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    value = str(value).strip().lower()

    return value in [
        "true",
        "1",
        "yes",
        "warning"
    ]


features["b0_warning"] = (
    features["baseline_warning"]
    .apply(normalize_warning)
)


# ============================================================
# PREPARE EVENT DATES
# ============================================================

events["event_start"] = pd.to_datetime(
    events["event_start"].astype(str),
    format="%Y-%m",
    errors="coerce"
)

events["event_end"] = pd.to_datetime(
    events["event_end"].astype(str),
    format="%Y-%m",
    errors="coerce"
)

sensitivity_events["event_start"] = pd.to_datetime(
    sensitivity_events["event_start"].astype(str),
    format="%Y-%m",
    errors="coerce"
)

sensitivity_events["event_end"] = pd.to_datetime(
    sensitivity_events["event_end"].astype(str),
    format="%Y-%m",
    errors="coerce"
)


# ============================================================
# BASIC DATA CHECKS
# ============================================================

print(
    f"\nFeature records: {len(features)}"
)

print(
    f"Reference events: {len(events)}"
)

print(
    f"Sensitivity events: "
    f"{len(sensitivity_events)}"
)

print(
    f"B0 warning months: "
    f"{features['b0_warning'].sum()}"
)

print(
    f"Feature period: "
    f"{features['date'].min().date()} "
    f"to "
    f"{features['date'].max().date()}"
)


# ============================================================
# HELPER
# ============================================================

def evaluate_events(
    event_df,
    feature_df
):

    results = []

    warning_dates = (
        feature_df.loc[
            feature_df["b0_warning"],
            "date"
        ]
        .sort_values()
        .tolist()
    )


    for _, event in event_df.iterrows():

        event_id = event["event_id"]

        event_start = event["event_start"]

        event_end = event["event_end"]


        # ----------------------------------------------------
        # PRE-EVENT WINDOW
        # ----------------------------------------------------

        window_start = (
            event_start
            - pd.DateOffset(
                months=LOOKBACK_MONTHS
            )
        )

        window_end = (
            event_start
            - pd.DateOffset(
                months=1
            )
        )


        # ----------------------------------------------------
        # WARNINGS IN PRE-EVENT WINDOW
        # ----------------------------------------------------

        event_warnings = [
            d
            for d in warning_dates
            if (
                d >= window_start
                and d <= window_end
            )
        ]


        detected = (
            len(event_warnings) > 0
        )


        # ----------------------------------------------------
        # LEAD TIME
        # ----------------------------------------------------

        if detected:

            # Earliest warning in the
            # allowed pre-event window.
            first_warning = min(
                event_warnings
            )

            # Closest warning to event onset.
            last_warning = max(
                event_warnings
            )

            first_lead_days = (
                event_start
                - first_warning
            ).days

            last_lead_days = (
                event_start
                - last_warning
            ).days

        else:

            first_warning = pd.NaT

            last_warning = pd.NaT

            first_lead_days = np.nan

            last_lead_days = np.nan


        results.append({

            "event_id": event_id,

            "event_start": event_start,

            "event_end": event_end,

            "pre_event_window_start": (
                window_start
            ),

            "pre_event_window_end": (
                window_end
            ),

            "b0_detected": detected,

            "warning_count": len(
                event_warnings
            ),

            "first_warning_date": (
                first_warning
            ),

            "last_warning_date": (
                last_warning
            ),

            "first_warning_lead_days": (
                first_lead_days
            ),

            "last_warning_lead_days": (
                last_lead_days
            ),

            "first_warning_lead_months": (
                first_lead_days / 30.44
                if detected
                else np.nan
            ),

            "last_warning_lead_months": (
                last_lead_days / 30.44
                if detected
                else np.nan
            ),

            "total_detections": (
                event.get(
                    "total_detections",
                    np.nan
                )
            ),

            "label_confidence": (
                event.get(
                    "label_confidence",
                    ""
                )
            ),
        })


    return pd.DataFrame(results)


# ============================================================
# PRIMARY EVENT EVALUATION
# ============================================================

print(
    "\nEvaluating primary reference events..."
)

primary_results = evaluate_events(
    events,
    features
)


# ============================================================
# SENSITIVITY EVENT EVALUATION
# ============================================================

print(
    "Evaluating strict sensitivity events..."
)

sensitivity_results = evaluate_events(
    sensitivity_events,
    features
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    event_results,
    feature_df
):

    total_events = len(
        event_results
    )

    detected_events = int(
        event_results[
            "b0_detected"
        ].sum()
    )

    undetected_events = (
        total_events
        - detected_events
    )


    if total_events > 0:

        event_recall = (
            detected_events
            / total_events
        )

    else:

        event_recall = np.nan


    detected = event_results[
        event_results["b0_detected"]
    ].copy()


    if len(detected) > 0:

        mean_first_lead = (
            detected[
                "first_warning_lead_months"
            ].mean()
        )

        median_first_lead = (
            detected[
                "first_warning_lead_months"
            ].median()
        )

        mean_last_lead = (
            detected[
                "last_warning_lead_months"
            ].mean()
        )

        median_last_lead = (
            detected[
                "last_warning_lead_months"
            ].median()
        )

    else:

        mean_first_lead = np.nan

        median_first_lead = np.nan

        mean_last_lead = np.nan

        median_last_lead = np.nan


    # --------------------------------------------------------
    # WARNING MONTHS
    # --------------------------------------------------------

    warning_dates = (
        feature_df.loc[
            feature_df["b0_warning"],
            "date"
        ]
        .tolist()
    )


    # A warning is considered associated with an event
    # if it falls inside at least one pre-event window.

    associated_warning_dates = set()

    for _, row in event_results.iterrows():

        window_start = (
            row[
                "pre_event_window_start"
            ]
        )

        window_end = (
            row[
                "pre_event_window_end"
            ]
        )

        for date in warning_dates:

            if (
                date >= window_start
                and date <= window_end
            ):

                associated_warning_dates.add(
                    date
                )


    total_warning_months = len(
        warning_dates
    )

    associated_warning_months = len(
        associated_warning_dates
    )

    false_warning_months = (
        total_warning_months
        - associated_warning_months
    )


    if total_warning_months > 0:

        false_warning_rate = (
            false_warning_months
            / total_warning_months
        )

    else:

        false_warning_rate = np.nan


    return {

        "total_reference_events": (
            total_events
        ),

        "detected_events": (
            detected_events
        ),

        "undetected_events": (
            undetected_events
        ),

        "event_recall": (
            event_recall
        ),

        "mean_first_warning_lead_months": (
            mean_first_lead
        ),

        "median_first_warning_lead_months": (
            median_first_lead
        ),

        "mean_last_warning_lead_months": (
            mean_last_lead
        ),

        "median_last_warning_lead_months": (
            median_last_lead
        ),

        "total_b0_warning_months": (
            total_warning_months
        ),

        "associated_warning_months": (
            associated_warning_months
        ),

        "false_warning_months": (
            false_warning_months
        ),

        "false_warning_rate": (
            false_warning_rate
        ),

        "evaluation_window_months": (
            LOOKBACK_MONTHS
        ),

        "evaluation_type": (
            "temporal_only"
        ),

    }


# ============================================================
# CALCULATE
# ============================================================

primary_metrics = calculate_metrics(
    primary_results,
    features
)

sensitivity_metrics = calculate_metrics(
    sensitivity_results,
    features
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(
    [
        {
            "dataset": "primary_ge5",
            **primary_metrics
        },
        {
            "dataset": "sensitivity_ge10",
            **sensitivity_metrics
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

primary_results.to_csv(
    EVENT_RESULTS_FILE,
    index=False
)

sensitivity_results.to_csv(
    SENSITIVITY_RESULTS_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("B0 EVALUATION RESULTS")
print("=" * 70)

print(
    "\nPRIMARY REFERENCE EVENTS (>=5 detections)"
)

for key, value in primary_metrics.items():

    print(
        f"{key}: {value}"
    )


print(
    "\nSTRICT SENSITIVITY EVENTS (>=10 detections)"
)

for key, value in sensitivity_metrics.items():

    print(
        f"{key}: {value}"
    )


print("\nOutputs:")

print(
    f"  {SUMMARY_FILE}"
)

print(
    f"  {EVENT_RESULTS_FILE}"
)

print(
    f"  {SENSITIVITY_RESULTS_FILE}"
)


print(
    "\nB0 chronological baseline evaluation completed."
)

print(
    "\nNOTE:"
)

print(
    "This is a temporal-only evaluation because the "
    "current feature dataset is a regional temporal "
    "summary and does not yet contain spatial coordinates "
    "for matching individual FIRMS events."
)