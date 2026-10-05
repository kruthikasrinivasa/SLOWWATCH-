from pathlib import Path
import pandas as pd


INPUT_FILE = Path(
    "data/events/firms_refined_candidate_events.csv"
)

OUTPUT_FILE = Path(
    "data/events/event_labels.csv"
)

SENSITIVITY_FILE = Path(
    "data/events/event_labels_sensitivity_10detections.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("SLOWWATCH — FINAL FIRMS REFERENCE EVENT LABELS")
print("=" * 70)

print(f"\nRefined candidates: {len(df)}")


# ============================================================
# PRIMARY REFERENCE-EVENT RULE
# ============================================================

primary = df[
    (df["total_detections"] >= 5)
    & (df["active_months"] <= 3)
].copy()


# ============================================================
# ADD REFERENCE-EVENT METADATA
# ============================================================

primary["label_source"] = "NASA_FIRMS_VIIRS"

primary["label_type"] = (
    "fire_disturbance_reference_event"
)

primary["label_confidence"] = primary.apply(
    lambda row:
        "high"
        if row["high_confidence_detections"] > 0
        else "nominal",
    axis=1
)

primary["independent_reference"] = True

primary["notes"] = (
    "Independent FIRMS-derived fire disturbance "
    "reference event. This label is not a direct "
    "vegetation-degradation ground truth."
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

primary = primary[
    [
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
        "label_type",
        "label_confidence",
        "independent_reference",
        "notes",
    ]
].copy()


# ============================================================
# SORT
# ============================================================

primary = primary.sort_values(
    "event_start"
).reset_index(drop=True)


# ============================================================
# SAVE PRIMARY LABELS
# ============================================================

primary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SENSITIVITY SUBSET
# ============================================================

sensitivity = primary[
    primary["total_detections"] >= 10
].copy()

sensitivity.to_csv(
    SENSITIVITY_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL REFERENCE EVENT SUMMARY")
print("=" * 70)

print(
    f"\nPrimary reference events: "
    f"{len(primary)}"
)

print(
    f"Strict sensitivity events (>=10 detections): "
    f"{len(sensitivity)}"
)

print(
    f"\nHigh-confidence primary events: "
    f"{(primary['label_confidence'] == 'high').sum()}"
)

print(
    f"Nominal-only primary events: "
    f"{(primary['label_confidence'] == 'nominal').sum()}"
)

print("\nEvents by year:")

print(
    pd.to_datetime(
        primary["event_start"]
    )
    .dt.year
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nReference-event table:")

print(
    primary[
        [
            "event_id",
            "event_start",
            "event_end",
            "active_months",
            "total_detections",
            "label_confidence",
        ]
    ].to_string(index=False)
)

print("\nPrimary output:")
print(OUTPUT_FILE)

print("\nSensitivity output:")
print(SENSITIVITY_FILE)

print(
    "\nReference-event label construction completed."
)