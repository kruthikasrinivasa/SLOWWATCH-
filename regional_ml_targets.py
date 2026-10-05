"""
Regional ML dataset construction for SLOWWATCH.

Combines:
1. HLS temporal vegetation features
2. ERA5-Land environmental context
3. FIRMS-derived event labels

Regions:
- Wayanad
- Nilgiris

Outputs:
data/features/regional/wayanad_ml_dataset.csv
data/features/regional/nilgiris_ml_dataset.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

REGIONAL_FEATURE_DIR = PROJECT_DIR / "data" / "features" / "regional"

REGIONS = {
    "Wayanad": {
        "multisource": REGIONAL_FEATURE_DIR / "wayanad_multisource_dataset.csv",
        "labels": REGIONAL_FEATURE_DIR / "wayanad_event_labels.csv",
        "output": REGIONAL_FEATURE_DIR / "wayanad_ml_dataset.csv",
    },
    "Nilgiris": {
        "multisource": REGIONAL_FEATURE_DIR / "nilgiris_multisource_dataset.csv",
        "labels": REGIONAL_FEATURE_DIR / "nilgiris_event_labels.csv",
        "output": REGIONAL_FEATURE_DIR / "nilgiris_ml_dataset.csv",
    },
}


# ============================================================
# HELPER
# ============================================================

def find_date_column(df):
    """
    Find the temporal column in a dataset.
    """

    candidates = [
        "date",
        "year_month",
        "month",
        "datetime",
        "timestamp",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find a date column. Available columns: {df.columns.tolist()}"
    )


# ============================================================
# BUILD ONE REGIONAL DATASET
# ============================================================

def build_region(region_name, config):

    print("\n" + "=" * 70)
    print(f"BUILDING REGIONAL ML DATASET: {region_name.upper()}")
    print("=" * 70)

    multisource_path = config["multisource"]
    labels_path = config["labels"]
    output_path = config["output"]

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not multisource_path.exists():
        raise FileNotFoundError(
            f"Missing multisource dataset:\n{multisource_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing event labels:\n{labels_path}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    features = pd.read_csv(multisource_path)
    labels = pd.read_csv(labels_path)

    print(f"Features loaded : {len(features)} rows")
    print(f"Labels loaded   : {len(labels)} rows")

    # --------------------------------------------------------
    # CONVERT DATES
    # --------------------------------------------------------

    feature_date_col = find_date_column(features)
    label_date_col = find_date_column(labels)

    features["date"] = pd.to_datetime(
        features[feature_date_col],
        errors="coerce"
    )

    labels["date"] = pd.to_datetime(
        labels[label_date_col],
        errors="coerce"
    )

    # Normalize to month start
    features["year_month"] = (
        features["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    labels["year_month"] = (
        labels["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # --------------------------------------------------------
    # REMOVE INVALID DATES
    # --------------------------------------------------------

    features = features.dropna(subset=["year_month"]).copy()
    labels = labels.dropna(subset=["year_month"]).copy()

    # --------------------------------------------------------
    # KEEP ONE LABEL ROW PER MONTH
    # --------------------------------------------------------

    label_columns = [
        col
        for col in [
            "firms_event_target",
            "firms_pre_event_3m",
            "firms_reference_type",
            "firms_event_id",
            "firms_event_start",
            "firms_event_end",
            "months_to_event",
        ]
        if col in labels.columns
    ]

    # If multiple rows somehow exist for the same month,
    # retain the strongest event information.
    aggregation = {}

    if "firms_event_target" in labels.columns:
        aggregation["firms_event_target"] = "max"

    if "firms_pre_event_3m" in labels.columns:
        aggregation["firms_pre_event_3m"] = "max"

    if "months_to_event" in labels.columns:
        aggregation["months_to_event"] = "min"

    if "firms_reference_type" in labels.columns:
        aggregation["firms_reference_type"] = "first"

    if "firms_event_id" in labels.columns:
        aggregation["firms_event_id"] = "first"

    if "firms_event_start" in labels.columns:
        aggregation["firms_event_start"] = "first"

    if "firms_event_end" in labels.columns:
        aggregation["firms_event_end"] = "first"

    labels_monthly = (
        labels.groupby("year_month", as_index=False)
        .agg(aggregation)
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = features.merge(
        labels_monthly,
        on="year_month",
        how="left",
        suffixes=("", "_label")
    )

    # --------------------------------------------------------
    # FILL TARGET COLUMNS
    # --------------------------------------------------------

    if "firms_event_target" not in df.columns:
        df["firms_event_target"] = 0

    if "firms_pre_event_3m" not in df.columns:
        df["firms_pre_event_3m"] = 0

    df["firms_event_target"] = (
        pd.to_numeric(
            df["firms_event_target"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df["firms_pre_event_3m"] = (
        pd.to_numeric(
            df["firms_pre_event_3m"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # EARLY WARNING TARGET
    #
    # 1 = event month OR within 3-month pre-event window
    # 0 = otherwise
    # --------------------------------------------------------

    df["early_warning_target"] = (
        (
            (df["firms_event_target"] == 1)
            |
            (df["firms_pre_event_3m"] == 1)
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    df["region"] = region_name

    # --------------------------------------------------------
    # TEMPORAL SORTING
    # --------------------------------------------------------

    df = (
        df.sort_values("year_month")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # MONTH INDEX
    #
    # Useful for chronological splitting.
    # --------------------------------------------------------

    df["month_index"] = np.arange(len(df))

    # --------------------------------------------------------
    # CHECK FEATURE AVAILABILITY
    # --------------------------------------------------------

    candidate_features = [
        "ndvi_mean",
        "ndvi_median",
        "quality_fraction",
        "observation_count",
        "ndvi_rolling_mean_3",
        "ndvi_rolling_std_3",
        "ndvi_change",
        "slope_6m",
        "seasonal_expected_ndvi",
        "ndvi_anomaly",
        "anomaly_zscore",
        "temperature_c",
        "precipitation_mm",
        "temperature_change",
        "precipitation_change",
    ]

    available_features = [
        col
        for col in candidate_features
        if col in df.columns
    ]

    missing_features = [
        col
        for col in candidate_features
        if col not in df.columns
    ]

    print("\nAvailable ML features:")
    for col in available_features:
        print(f"  + {col}")

    if missing_features:
        print("\nMissing optional features:")
        for col in missing_features:
            print(f"  - {col}")

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for col in available_features:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMOVE FUTURE/REFERENCE-ONLY COLUMNS FROM FEATURES
    #
    # These columns are labels/reference information and MUST
    # NOT be used as predictive model inputs.
    # --------------------------------------------------------

    forbidden_as_features = [
        "firms_event_target",
        "firms_pre_event_3m",
        "early_warning_target",
        "firms_reference_type",
        "firms_event_id",
        "firms_event_start",
        "firms_event_end",
        "months_to_event",
    ]

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    metadata_columns = [
        "region",
        "year_month",
        "date",
    ]

    target_columns = [
        "firms_event_target",
        "firms_pre_event_3m",
        "early_warning_target",
        "firms_reference_type",
        "firms_event_id",
        "firms_event_start",
        "firms_event_end",
        "months_to_event",
    ]

    ordered_columns = []

    for col in metadata_columns:
        if col in df.columns:
            ordered_columns.append(col)

    for col in available_features:
        if col not in ordered_columns:
            ordered_columns.append(col)

    for col in target_columns:
        if col in df.columns and col not in ordered_columns:
            ordered_columns.append(col)

    if "month_index" in df.columns:
        ordered_columns.append("month_index")

    # Include anything else that was present but not explicitly
    # listed, without allowing duplicate columns.
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)

    df = df[ordered_columns]

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"REGION: {region_name}")
    print("-" * 70)

    print(f"Rows                 : {len(df)}")
    print(
        f"Date range           : "
        f"{df['year_month'].min().strftime('%Y-%m')} "
        f"to "
        f"{df['year_month'].max().strftime('%Y-%m')}"
    )

    print(
        f"Event months         : "
        f"{df['firms_event_target'].sum()}"
    )

    print(
        f"Pre-event months     : "
        f"{df['firms_pre_event_3m'].sum()}"
    )

    print(
        f"Early-warning months : "
        f"{df['early_warning_target'].sum()}"
    )

    print(
        f"Missing feature cells: "
        f"{df[available_features].isna().sum().sum()}"
    )

    print("\nSaved:")
    print(output_path)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SLOWWATCH REGIONAL ML DATASET CONSTRUCTION")
    print("=" * 70)

    results = {}

    for region_name, config in REGIONS.items():

        try:
            results[region_name] = build_region(
                region_name,
                config
            )

        except Exception as exc:

            print("\n" + "!" * 70)
            print(f"FAILED: {region_name}")
            print("!" * 70)
            print(exc)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REGIONAL ML DATASET CONSTRUCTION COMPLETE")
    print("=" * 70)

    for region_name, df in results.items():

        print(
            f"{region_name:10s} | "
            f"rows={len(df):3d} | "
            f"events={int(df['firms_event_target'].sum()):3d} | "
            f"pre_event={int(df['firms_pre_event_3m'].sum()):3d} | "
            f"early_warning={int(df['early_warning_target'].sum()):3d}"
        )


if __name__ == "__main__":
    main()