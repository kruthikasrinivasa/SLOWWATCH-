"""
SLOWWATCH — Leakage-Safe ML Target Construction

Purpose:
    Construct chronological prediction targets for classical ML
    and temporal models.

Target:
    target_event_3m = 1 if an independent FIRMS-derived
    fire-disturbance reference event begins within the next
    3 months after the current observation.

Important:
    Event information is used ONLY to construct the target.
    Event dates, event IDs and any future-derived information
    are NOT included as model features.

Chronological split:
    Train      : 2018–2022
    Validation : 2023
    Test       : 2024–2025
"""

from pathlib import Path

import pandas as pd
import numpy as np


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
    "data/splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "slowwatch_ml_targets.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

PREDICTION_HORIZON_MONTHS = 3


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SLOWWATCH — LEAKAGE-SAFE ML TARGET CONSTRUCTION")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

events = pd.read_csv(
    EVENT_FILE
)

sensitivity_events = pd.read_csv(
    SENSITIVITY_FILE
)


print(
    f"\nFeature records: {len(features)}"
)

print(
    f"Primary reference events: {len(events)}"
)

print(
    f"Strict sensitivity events: "
    f"{len(sensitivity_events)}"
)


# ============================================================
# PREPARE MONTHLY FEATURE DATA
# ============================================================

features["date"] = pd.to_datetime(
    features["date"],
    errors="coerce"
)

features["year_month"] = pd.to_datetime(
    features["year_month"].astype(str),
    errors="coerce"
)

features = (
    features
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# PREPARE EVENT DATES
# ============================================================

def prepare_events(df):

    df = df.copy()

    df["event_start"] = pd.to_datetime(
        df["event_start"].astype(str),
        format="%Y-%m",
        errors="coerce"
    )

    df["event_end"] = pd.to_datetime(
        df["event_end"].astype(str),
        format="%Y-%m",
        errors="coerce"
    )

    return df


events = prepare_events(events)

sensitivity_events = prepare_events(
    sensitivity_events
)


# ============================================================
# TARGET CONSTRUCTION
# ============================================================

def construct_future_event_target(
    feature_dates,
    event_dates,
    horizon_months=3
):
    """
    For each observation month t:

        target = 1

    if an independent event starts during:

        t + 1 month ... t + horizon_months

    Otherwise:

        target = 0

    Current-month events are NOT counted.

    This prevents the target from representing an event
    that has already started at prediction time.
    """

    targets = []
    event_ids = []
    days_to_event = []

    for current_date in feature_dates:

        current_period = (
            current_date.to_period("M")
        )

        future_start = (
            current_period + 1
        )

        future_end = (
            current_period
            + horizon_months
        )

        future_events = event_dates[
            (
                event_dates["event_start"]
                .dt.to_period("M")
                >= future_start
            )
            &
            (
                event_dates["event_start"]
                .dt.to_period("M")
                <= future_end
            )
        ]

        if len(future_events) > 0:

            # Earliest upcoming event.
            first_event = (
                future_events
                .sort_values("event_start")
                .iloc[0]
            )

            targets.append(1)

            event_ids.append(
                first_event["event_id"]
            )

            days_to_event.append(
                (
                    first_event["event_start"]
                    - current_date
                ).days
            )

        else:

            targets.append(0)

            event_ids.append("")

            days_to_event.append(
                np.nan
            )

    return (
        targets,
        event_ids,
        days_to_event
    )


# ============================================================
# PRIMARY TARGET
# ============================================================

(
    target,
    target_event_id,
    days_to_event
) = construct_future_event_target(
    features["date"],
    events,
    PREDICTION_HORIZON_MONTHS
)


features[
    "target_event_3m"
] = target

features[
    "target_event_id_3m"
] = target_event_id

features[
    "days_to_event_3m"
] = days_to_event


# ============================================================
# STRICT SENSITIVITY TARGET
# ============================================================

(
    sensitivity_target,
    sensitivity_event_id,
    sensitivity_days
) = construct_future_event_target(
    features["date"],
    sensitivity_events,
    PREDICTION_HORIZON_MONTHS
)


features[
    "target_event_3m_strict"
] = sensitivity_target

features[
    "target_event_id_3m_strict"
] = sensitivity_event_id

features[
    "days_to_event_3m_strict"
] = sensitivity_days


# ============================================================
# CHRONOLOGICAL SPLITS
# ============================================================

def assign_split(date):

    year = date.year

    if year <= 2022:
        return "train"

    elif year == 2023:
        return "validation"

    elif year >= 2024:
        return "test"

    return "unknown"


features[
    "split"
] = features["date"].apply(
    assign_split
)


# ============================================================
# REMOVE INVALID SPLITS
# ============================================================

features = features[
    features["split"] != "unknown"
].copy()


# ============================================================
# LEAKAGE AUDIT
# ============================================================

# These columns are targets / future-derived metadata.
# They must NEVER be used as ML input features.

future_columns = [
    "target_event_3m",
    "target_event_id_3m",
    "days_to_event_3m",
    "target_event_3m_strict",
    "target_event_id_3m_strict",
    "days_to_event_3m_strict",
    "split",
]


# Save complete target dataset.
features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TARGET CONSTRUCTION RESULTS")
print("=" * 70)

print(
    f"\nTotal observations: {len(features)}"
)

print(
    "\nChronological split:"
)

print(
    features["split"]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nPrimary 3-month target:"
)

print(
    features[
        "target_event_3m"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nStrict >=10-detection target:"
)

print(
    features[
        "target_event_3m_strict"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SPLIT-WISE TARGET DISTRIBUTION
# ============================================================

print(
    "\nPrimary target by split:"
)

split_summary = (
    features
    .groupby("split")[
        "target_event_3m"
    ]
    .agg(
        observations="count",
        positive_months="sum"
    )
)

split_summary[
    "positive_rate"
] = (
    split_summary["positive_months"]
    / split_summary["observations"]
)

print(
    split_summary.to_string()
)


# ============================================================
# SAMPLE TARGET TABLE
# ============================================================

print(
    "\nSample target records:"
)

print(
    features[
        [
            "year_month",
            "ndvi_mean",
            "ndvi_anomaly",
            "slope_6m",
            "temperature_c",
            "precipitation_raw",
            "target_event_3m",
            "target_event_id_3m",
            "days_to_event_3m",
            "split",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# LEAKAGE WARNING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LEAKAGE CONTROL"
)

print(
    "=" * 70
)

print(
    "\nFuture event information is stored only in target/"
    "metadata columns."
)

print(
    "Those columns must not be supplied to ML models."
)

print(
    "\nChronological split:"
)

print(
    "  TRAIN      : 2018–2022"
)

print(
    "  VALIDATION : 2023"
)

print(
    "  TEST       : 2024–2025"
)


print(
    f"\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nML target construction completed successfully."
)