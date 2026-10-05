"""
SLOWWATCH — FIRMS Candidate Fire Event Construction

Creates candidate disturbance episodes from the cleaned
FIRMS detection dataset.

Important:
    These are CANDIDATE FIRE DISTURBANCE EVENTS.
    They are not yet confirmed vegetation-degradation labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/events/firms_cleaned_diagnostic.csv"
)

OUTPUT_FILE = Path(
    "data/events/firms_candidate_events.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Spatial neighbourhood:
# detections in the same 0.05-degree grid cell are grouped.

GRID_SIZE = 0.05

# Consecutive active months within this gap are considered
# part of the same candidate episode.

MAX_MONTH_GAP = 1

# Minimum number of quality-filtered detections for an episode.

MIN_DETECTIONS = 3


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SLOWWATCH — FIRMS CANDIDATE EVENT CONSTRUCTION")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput records: {len(df):,}")


# ============================================================
# PREPARE
# ============================================================

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "acq_date",
        "latitude",
        "longitude"
    ]
).copy()

df["year_month"] = (
    df["acq_date"]
    .dt.to_period("M")
)

# Spatial grid
df["lat_grid"] = (
    np.floor(df["latitude"] / GRID_SIZE)
    * GRID_SIZE
)

df["lon_grid"] = (
    np.floor(df["longitude"] / GRID_SIZE)
    * GRID_SIZE
)

# Combined spatial-cell identifier
df["grid_id"] = (
    df["lat_grid"].round(3).astype(str)
    + "_"
    + df["lon_grid"].round(3).astype(str)
)


# ============================================================
# MONTHLY SPATIAL AGGREGATION
# ============================================================

monthly = (
    df
    .groupby(
        [
            "grid_id",
            "lat_grid",
            "lon_grid",
            "year_month"
        ]
    )
    .agg(
        detection_count=("acq_date", "size"),

        first_detection=("acq_date", "min"),

        last_detection=("acq_date", "max"),

        mean_frp=("frp", "mean"),

        max_frp=("frp", "max"),

        high_confidence_count=(
            "confidence_label",
            lambda x: (x == "high").sum()
        ),

        nominal_confidence_count=(
            "confidence_label",
            lambda x: (x == "nominal").sum()
        ),

        type_unavailable_count=(
            "fire_type_label",
            lambda x: (x == "type_unavailable").sum()
        ),
    )
    .reset_index()
)

print(
    f"Monthly spatial cells: {len(monthly):,}"
)


# ============================================================
# SORT
# ============================================================

monthly = monthly.sort_values(
    [
        "grid_id",
        "year_month"
    ]
).reset_index(drop=True)


# ============================================================
# TEMPORAL EPISODE GROUPING
# ============================================================

monthly["month_number"] = (
    monthly["year_month"]
    .dt.year * 12
    + monthly["year_month"].dt.month
)

monthly["month_gap"] = (
    monthly
    .groupby("grid_id")["month_number"]
    .diff()
)

# New episode when:
#   1. first observation in a grid cell, OR
#   2. gap > MAX_MONTH_GAP

monthly["new_episode"] = (
    monthly["month_gap"].isna()
    | (monthly["month_gap"] > MAX_MONTH_GAP)
)

monthly["episode_number"] = (
    monthly
    .groupby("grid_id")["new_episode"]
    .cumsum()
)

monthly["candidate_id"] = (
    "FIRMS_"
    + monthly["grid_id"].astype(str)
    + "_"
    + monthly["episode_number"].astype(str)
)


# ============================================================
# BUILD CANDIDATE EVENTS
# ============================================================

events = (
    monthly
    .groupby(
        [
            "candidate_id",
            "grid_id",
            "lat_grid",
            "lon_grid"
        ]
    )
    .agg(
        event_start=("year_month", "min"),

        event_end=("year_month", "max"),

        active_months=(
            "year_month",
            "nunique"
        ),

        total_detections=(
            "detection_count",
            "sum"
        ),

        peak_monthly_detections=(
            "detection_count",
            "max"
        ),

        mean_frp=(
            "mean_frp",
            "mean"
        ),

        max_frp=(
            "max_frp",
            "max"
        ),

        high_confidence_detections=(
            "high_confidence_count",
            "sum"
        ),

        nominal_detections=(
            "nominal_confidence_count",
            "sum"
        ),

        type_unavailable_detections=(
            "type_unavailable_count",
            "sum"
        ),
    )
    .reset_index()
)


# ============================================================
# FILTER SMALL EPISODES
# ============================================================

events = events[
    events["total_detections"] >= MIN_DETECTIONS
].copy()


# ============================================================
# EVENT METADATA
# ============================================================

events["region"] = "Kodagu"

events["driver"] = "fire"

events["label_source"] = "NASA_FIRMS_VIIRS"

events["label_confidence"] = "candidate"

events["notes"] = (
    "Candidate fire disturbance episode derived "
    "from spatial-temporal aggregation of FIRMS "
    "detections; not direct vegetation-degradation ground truth."
)


# ============================================================
# EVENT DURATION
# ============================================================

events["event_start"] = (
    events["event_start"]
    .astype(str)
)

events["event_end"] = (
    events["event_end"]
    .astype(str)
)


# ============================================================
# ORDER COLUMNS
# ============================================================

events = events[
    [
        "candidate_id",
        "region",
        "driver",
        "event_start",
        "event_end",
        "active_months",
        "total_detections",
        "peak_monthly_detections",
        "high_confidence_detections",
        "nominal_detections",
        "type_unavailable_detections",
        "mean_frp",
        "max_frp",
        "lat_grid",
        "lon_grid",
        "label_source",
        "label_confidence",
        "notes",
    ]
]


# ============================================================
# SORT EVENTS
# ============================================================

events = events.sort_values(
    [
        "event_start",
        "total_detections"
    ],
    ascending=[
        True,
        False
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

events.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE EVENT RESULTS")
print("=" * 70)

print(
    f"\nCandidate events generated: {len(events):,}"
)

if len(events) > 0:

    print(
        f"Event period: "
        f"{events['event_start'].min()} "
        f"to "
        f"{events['event_end'].max()}"
    )

    print(
        "\nDetection-count statistics:"
    )

    print(
        events["total_detections"]
        .describe()
        .to_string()
    )

    print(
        "\nTop candidate events:"
    )

    print(
        events[
            [
                "candidate_id",
                "event_start",
                "event_end",
                "active_months",
                "total_detections",
                "peak_monthly_detections",
                "high_confidence_detections",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

print("\nOutput:")
print(OUTPUT_FILE)

print("\nCandidate event construction completed.")