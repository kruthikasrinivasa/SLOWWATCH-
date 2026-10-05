"""
SLOWWATCH — FIRMS Refined Spatial-Temporal Event Construction

Creates localized candidate fire-disturbance episodes from
quality-filtered NASA FIRMS observations.

Protocol:
    1. Aggregate detections to monthly spatial cells.
    2. Identify spatially connected activity.
    3. Connect activity across nearby months only when
       spatially close.
    4. Limit candidate episode duration to 3 months.
    5. Require at least 3 detections.

These are candidate fire-disturbance reference events,
NOT direct vegetation-degradation ground truth.
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
    "data/events/firms_refined_candidate_events.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

GRID_SIZE = 0.05

MAX_EVENT_MONTHS = 3

MIN_DETECTIONS = 3

# Maximum spatial separation between monthly
# component centroids for temporal connection.
MAX_CENTROID_DISTANCE = 0.10


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SLOWWATCH — REFINED FIRMS EVENT CONSTRUCTION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput detections: {len(df):,}")


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


# Only nominal + high confidence
df = df[
    df["confidence_label"].isin(
        ["nominal", "high"]
    )
].copy()


df["year_month"] = (
    df["acq_date"].dt.to_period("M")
)

df["month_number"] = (
    df["year_month"].dt.year * 12
    + df["year_month"].dt.month
)


# ============================================================
# SPATIAL GRID
# ============================================================

df["grid_x"] = np.floor(
    df["longitude"] / GRID_SIZE
).astype(int)

df["grid_y"] = np.floor(
    df["latitude"] / GRID_SIZE
).astype(int)


# ============================================================
# MONTHLY GRID CELLS
# ============================================================

monthly = (
    df
    .groupby(
        [
            "year_month",
            "month_number",
            "grid_x",
            "grid_y"
        ]
    )
    .agg(
        detection_count=("acq_date", "size"),

        mean_lat=("latitude", "mean"),

        mean_lon=("longitude", "mean"),

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
# FIND MONTHLY SPATIAL COMPONENTS
# ============================================================

print("\nFinding monthly spatial components...")


monthly = monthly.sort_values(
    [
        "month_number",
        "grid_y",
        "grid_x"
    ]
).reset_index(drop=True)


# Assign component IDs separately inside each month.

component_records = []

component_counter = 0


for month, group in monthly.groupby(
    "month_number",
    sort=True
):

    group = group.copy()

    coordinates = set(
        zip(
            group["grid_x"].astype(int),
            group["grid_y"].astype(int)
        )
    )

    visited = set()

    coord_to_index = {}

    for idx, row in group.iterrows():

        coord_to_index[
            (
                int(row["grid_x"]),
                int(row["grid_y"])
            )
        ] = idx


    for coordinate in coordinates:

        if coordinate in visited:
            continue

        component_counter += 1

        queue = [coordinate]

        visited.add(coordinate)

        component = []

        while queue:

            current = queue.pop()

            component.append(current)

            x, y = current

            neighbours = [
                (x - 1, y - 1),
                (x - 1, y),
                (x - 1, y + 1),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1),
            ]

            for neighbour in neighbours:

                if (
                    neighbour in coordinates
                    and neighbour not in visited
                ):

                    visited.add(neighbour)

                    queue.append(neighbour)


        for cell in component:

            idx = coord_to_index[cell]

            component_records.append(
                (
                    idx,
                    component_counter
                )
            )


component_map = dict(component_records)

monthly["monthly_component"] = [
    component_map[i]
    for i in range(len(monthly))
]


# ============================================================
# MONTHLY COMPONENT SUMMARY
# ============================================================

components = (
    monthly
    .groupby("monthly_component")
    .agg(
        month_number=("month_number", "first"),

        year_month=("year_month", "first"),

        detection_count=("detection_count", "sum"),

        mean_lat=("mean_lat", "mean"),

        mean_lon=("mean_lon", "mean"),

        mean_frp=("mean_frp", "mean"),

        max_frp=("max_frp", "max"),

        high_confidence_count=(
            "high_confidence_count",
            "sum"
        ),

        nominal_confidence_count=(
            "nominal_confidence_count",
            "sum"
        ),

        type_unavailable_count=(
            "type_unavailable_count",
            "sum"
        ),

        min_grid_x=("grid_x", "min"),

        max_grid_x=("grid_x", "max"),

        min_grid_y=("grid_y", "min"),

        max_grid_y=("grid_y", "max"),
    )
    .reset_index()
)


components = components.sort_values(
    "month_number"
).reset_index(drop=True)


# ============================================================
# TEMPORAL COMPONENT LINKING
# ============================================================

print("Connecting nearby monthly components...")


events = []

active_events = []

event_counter = 0


for _, component in components.iterrows():

    current_month = int(
        component["month_number"]
    )

    current_lat = float(
        component["mean_lat"]
    )

    current_lon = float(
        component["mean_lon"]
    )

    best_event = None

    best_distance = None


    # Look at currently active candidate events.
    for event in active_events:

        month_gap = (
            current_month
            - event["last_month"]
        )

        if month_gap != 1:
            continue

        distance = np.sqrt(
            (
                current_lat
                - event["last_lat"]
            ) ** 2
            +
            (
                current_lon
                - event["last_lon"]
            ) ** 2
        )

        if distance <= MAX_CENTROID_DISTANCE:

            if (
                best_distance is None
                or distance < best_distance
            ):

                best_event = event

                best_distance = distance


    # --------------------------------------------------------
    # CONNECT TO EXISTING EVENT
    # --------------------------------------------------------

    if best_event is not None:

        if (
            current_month
            - best_event["start_month"]
            + 1
            <= MAX_EVENT_MONTHS
        ):

            best_event["last_month"] = current_month

            best_event["last_lat"] = current_lat

            best_event["last_lon"] = current_lon

            best_event["components"].append(
                component
            )

            continue


    # --------------------------------------------------------
    # CREATE NEW EVENT
    # --------------------------------------------------------

    event_counter += 1

    new_event = {
        "event_number": event_counter,

        "start_month": current_month,

        "last_month": current_month,

        "last_lat": current_lat,

        "last_lon": current_lon,

        "components": [component],
    }

    active_events.append(new_event)


# ============================================================
# CONVERT EVENTS TO TABLE
# ============================================================

event_rows = []


for event in active_events:

    parts = event["components"]

    total_detections = sum(
        int(x["detection_count"])
        for x in parts
    )

    if total_detections < MIN_DETECTIONS:
        continue


    start_month = min(
        x["month_number"]
        for x in parts
    )

    end_month = max(
        x["month_number"]
        for x in parts
    )

    active_months = (
        end_month
        - start_month
        + 1
    )


    weighted_lat = np.average(
        [
            x["mean_lat"]
            for x in parts
        ],
        weights=[
            x["detection_count"]
            for x in parts
        ]
    )

    weighted_lon = np.average(
        [
            x["mean_lon"]
            for x in parts
        ],
        weights=[
            x["detection_count"]
            for x in parts
        ]
    )


    high_conf = sum(
        int(x["high_confidence_count"])
        for x in parts
    )

    nominal_conf = sum(
        int(x["nominal_confidence_count"])
        for x in parts
    )

    type_unavailable = sum(
        int(x["type_unavailable_count"])
        for x in parts
    )


    event_rows.append({

        "event_start": (
            pd.Period(
                year=(start_month - 1) // 12,
                month=((start_month - 1) % 12) + 1,
                freq="M"
            ).strftime("%Y-%m")
        ),

        "event_end": (
            pd.Period(
                year=(end_month - 1) // 12,
                month=((end_month - 1) % 12) + 1,
                freq="M"
            ).strftime("%Y-%m")
        ),

        "active_months": active_months,

        "total_detections": total_detections,

        "high_confidence_detections": high_conf,

        "nominal_detections": nominal_conf,

        "type_unavailable_detections": type_unavailable,

        "latitude_center": weighted_lat,

        "longitude_center": weighted_lon,

        "spatial_components": len(parts),

    })


# ============================================================
# DATAFRAME
# ============================================================

events_df = pd.DataFrame(
    event_rows
)


if events_df.empty:

    print("\nNo candidate events generated.")

    events_df = pd.DataFrame(
        columns=[
            "event_id",
            "region",
            "driver",
            "event_start",
            "event_end",
            "active_months",
            "total_detections",
            "high_confidence_detections",
            "nominal_detections",
            "type_unavailable_detections",
            "latitude_center",
            "longitude_center",
            "spatial_components",
            "label_source",
            "label_confidence",
            "notes",
        ]
    )

else:

    events_df.insert(
        0,
        "event_id",
        [
            f"FIRMS_REF_{i:04d}"
            for i in range(
                1,
                len(events_df) + 1
            )
        ]
    )

    events_df["region"] = "Kodagu"

    events_df["driver"] = "fire"

    events_df["label_source"] = (
        "NASA_FIRMS_VIIRS"
    )

    events_df["label_confidence"] = (
        "candidate"
    )

    events_df["notes"] = (
        "Candidate fire disturbance episode "
        "derived from spatial-temporal FIRMS "
        "activity; not direct vegetation-"
        "degradation ground truth."
    )


# ============================================================
# SORT
# ============================================================

events_df = events_df.sort_values(
    "event_start"
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

events_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFINED EVENT RESULTS")
print("=" * 70)

print(
    f"\nCandidate events: "
    f"{len(events_df):,}"
)

if len(events_df) > 0:

    print(
        "\nActive-month statistics:"
    )

    print(
        events_df[
            "active_months"
        ].describe().to_string()
    )

    print(
        "\nDetection statistics:"
    )

    print(
        events_df[
            "total_detections"
        ].describe().to_string()
    )

    print(
        "\nCandidate events:"
    )

    print(
        events_df[
            [
                "event_id",
                "event_start",
                "event_end",
                "active_months",
                "total_detections",
                "high_confidence_detections",
                "spatial_components",
            ]
        ].to_string(index=False)
    )


print(
    f"\nOutput:\n{OUTPUT_FILE}"
)

print(
    "\nRefined FIRMS event construction completed."
)