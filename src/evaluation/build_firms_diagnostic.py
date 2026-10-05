"""
SLOWWATCH — FIRMS Fire Disturbance Diagnostic

Purpose:
    Inspect NASA FIRMS VIIRS detections and identify
    spatial-temporal concentrations that may represent
    candidate fire disturbance episodes.

Important:
    FIRMS detections are treated as independent disturbance
    evidence, NOT as direct vegetation-degradation labels.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

FIRMS_DIR = Path(r"F:\SLOWWATCH_DATA\FIRMS")

OUTPUT_DIR = Path("data/events")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FIRMS FILES
# ============================================================

files = sorted(FIRMS_DIR.rglob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No FIRMS CSV files found in {FIRMS_DIR}"
    )

print("=" * 70)
print("SLOWWATCH — FIRMS FIRE DISTURBANCE DIAGNOSTIC")
print("=" * 70)

print(f"\nFIRMS files found: {len(files)}")

frames = []

for file in files:
    print(f"Loading: {file.name}")

    df = pd.read_csv(file)

    # Record source file for traceability
    df["source_file"] = file.name

    # NOAA-21 NRT file does not contain 'type'
    if "type" not in df.columns:
        df["type"] = pd.NA

    frames.append(df)


# ============================================================
# COMBINE
# ============================================================

df = pd.concat(frames, ignore_index=True)

# Parse date
df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)

# Numeric conversion
for col in ["latitude", "longitude", "frp", "brightness"]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# BASIC VALIDITY FILTER
# ============================================================

study_min_lat = 12.2
study_max_lat = 12.9
study_min_lon = 75.4
study_max_lon = 76.1

valid = (
    df["acq_date"].notna()
    & df["latitude"].between(study_min_lat, study_max_lat)
    & df["longitude"].between(study_min_lon, study_max_lon)
)

df = df.loc[valid].copy()


# ============================================================
# STANDARDIZE CONFIDENCE
# ============================================================

confidence_map = {
    "l": "low",
    "n": "nominal",
    "h": "high"
}

df["confidence_label"] = (
    df["confidence"]
    .astype(str)
    .str.lower()
    .map(confidence_map)
    .fillna("unknown")
)


# ============================================================
# FIRE-TYPE INFORMATION
# ============================================================

# type == 0 is retained as vegetation-fire evidence
# type == 3 is not used as a vegetation-fire reference.
#
# NOAA-21 has no type field, so those records remain
# identifiable as "type_unavailable" and are not silently
# assigned type 0.

df["fire_type_label"] = "type_unavailable"

df.loc[df["type"] == 0, "fire_type_label"] = "vegetation_fire"

df.loc[df["type"] == 3, "fire_type_label"] = "offshore"


# ============================================================
# QUALITY-FOCUSED CANDIDATE SET
# ============================================================

# We keep nominal + high confidence for the primary
# diagnostic. Low-confidence observations remain in the
# raw/clean dataset for sensitivity analysis later.

candidate = df[
    df["confidence_label"].isin(["nominal", "high"])
    & (
        (df["fire_type_label"] == "vegetation_fire")
        | (df["fire_type_label"] == "type_unavailable")
    )
].copy()


# ============================================================
# TEMPORAL FIELDS
# ============================================================

candidate["year_month"] = (
    candidate["acq_date"]
    .dt.to_period("M")
    .astype(str)
)

candidate["year"] = candidate["acq_date"].dt.year

candidate["month"] = candidate["acq_date"].dt.month


# ============================================================
# SPATIAL GRID
# ============================================================

# 0.05 degree grid is used ONLY for diagnostic inspection.
# It is NOT yet the final event-definition rule.

GRID_SIZE = 0.05

candidate["lat_grid"] = (
    (candidate["latitude"] / GRID_SIZE)
    .round()
    * GRID_SIZE
)

candidate["lon_grid"] = (
    (candidate["longitude"] / GRID_SIZE)
    .round()
    * GRID_SIZE
)


# ============================================================
# MONTHLY SPATIAL ACTIVITY
# ============================================================

monthly_grid = (
    candidate
    .groupby(
        [
            "year_month",
            "lat_grid",
            "lon_grid"
        ]
    )
    .agg(
        detection_count=("acq_date", "size"),
        mean_frp=("frp", "mean"),
        max_frp=("frp", "max"),
        high_confidence_count=(
            "confidence_label",
            lambda x: (x == "high").sum()
        ),
        first_detection=("acq_date", "min"),
        last_detection=("acq_date", "max"),
    )
    .reset_index()
)


# ============================================================
# MONTHLY SUMMARY
# ============================================================

monthly_summary = (
    candidate
    .groupby("year_month")
    .agg(
        detection_count=("acq_date", "size"),
        active_grid_cells=(
            ["lat_grid", "lon_grid"],
            "nunique"
        ) if False else ("lat_grid", "nunique"),
        mean_frp=("frp", "mean"),
        max_frp=("frp", "max"),
        high_confidence_count=(
            "confidence_label",
            lambda x: (x == "high").sum()
        ),
    )
    .reset_index()
)

# Correct active-cell count using unique coordinate pairs
active_cells = (
    candidate
    .groupby("year_month")
    .apply(
        lambda g: g[["lat_grid", "lon_grid"]]
        .drop_duplicates()
        .shape[0],
        include_groups=False
    )
    .rename("active_grid_cells")
    .reset_index()
)

monthly_summary = monthly_summary.drop(
    columns=["active_grid_cells"]
)

monthly_summary = monthly_summary.merge(
    active_cells,
    on="year_month",
    how="left"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

clean_path = OUTPUT_DIR / "firms_cleaned_diagnostic.csv"

grid_path = OUTPUT_DIR / "firms_monthly_spatial_activity.csv"

summary_path = OUTPUT_DIR / "firms_monthly_summary.csv"

candidate.to_csv(
    clean_path,
    index=False
)

monthly_grid.to_csv(
    grid_path,
    index=False
)

monthly_summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC RESULTS")
print("=" * 70)

print(f"\nRaw detections: {len(df):,}")
print(f"Candidate detections: {len(candidate):,}")

print(
    f"Date range: "
    f"{candidate['acq_date'].min().date()} "
    f"to "
    f"{candidate['acq_date'].max().date()}"
)

print("\nConfidence distribution:")
print(
    candidate["confidence_label"]
    .value_counts()
    .to_string()
)

print("\nFire-type distribution:")
print(
    candidate["fire_type_label"]
    .value_counts()
    .to_string()
)

print("\nMonthly activity:")
print(
    monthly_summary.to_string(index=False)
)

print("\nOutputs:")
print(f"  {clean_path}")
print(f"  {grid_path}")
print(f"  {summary_path}")

print("\nDiagnostic completed successfully.")