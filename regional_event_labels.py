"""
SLOWWATCH — Regional FIRMS Event Labels

Connects:
    Regional HLS temporal features
    +
    Regional ERA5-Land environmental context
    +
    Regional FIRMS candidate disturbance events

Regions:
    Wayanad
    Nilgiris

IMPORTANT:
FIRMS events are candidate fire-disturbance reference events.
They are NOT treated as direct vegetation-degradation ground truth.

Outputs:
    data/features/regional/wayanad_event_labels.csv
    data/features/regional/nilgiris_event_labels.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")

PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

FEATURE_DIR = (
    PROJECT_DIR
    / "data"
    / "features"
    / "regional"
)

EVENT_DIR = (
    PROJECT_DIR
    / "data"
    / "events"
    / "regional"
)


# ============================================================
# REGIONS
# ============================================================

REGIONS = {

    "Wayanad": {
        "features":
            FEATURE_DIR
            / "wayanad_multisource_dataset.csv",

        "events":
            EVENT_DIR
            / "wayanad_firms_candidate_events.csv",
    },

    "Nilgiris": {
        "features":
            FEATURE_DIR
            / "nilgiris_multisource_dataset.csv",

        "events":
            EVENT_DIR
            / "nilgiris_firms_candidate_events.csv",
    },
}


# ============================================================
# LABELING PARAMETERS
# ============================================================

# Months immediately BEFORE an event start that are
# considered the pre-event warning window.
PRE_EVENT_MONTHS = 3


# ============================================================
# DATE HELPERS
# ============================================================

def month_period(value):

    return pd.Period(
        pd.to_datetime(value),
        freq="M"
    )


# ============================================================
# PROCESS ONE REGION
# ============================================================

def process_region(region_name, config):

    print("\n" + "=" * 70)
    print(f"PROCESSING REGION: {region_name}")
    print("=" * 70)

    feature_file = config["features"]
    event_file = config["events"]

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not feature_file.exists():

        raise FileNotFoundError(
            f"Missing feature file:\n{feature_file}"
        )

    if not event_file.exists():

        raise FileNotFoundError(
            f"Missing event file:\n{event_file}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    features = pd.read_csv(
        feature_file
    )

    events = pd.read_csv(
        event_file
    )

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Candidate events: {len(events):,}"
    )

    # --------------------------------------------------------
    # Feature date
    # --------------------------------------------------------

    if "date" in features.columns:

        features["date"] = pd.to_datetime(
            features["date"],
            errors="coerce"
        )

    elif "year_month" in features.columns:

        features["date"] = pd.to_datetime(
            features["year_month"].astype(str)
            + "-01",
            errors="coerce"
        )

    else:

        raise ValueError(
            "No date/year_month column found "
            "in regional feature dataset."
        )

    features["year_month"] = (
        features["date"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # Event dates
    # --------------------------------------------------------

    if len(events) > 0:

        events["event_start"] = pd.to_datetime(
            events["event_start"]
            .astype(str)
            + "-01",
            errors="coerce"
        )

        events["event_end"] = pd.to_datetime(
            events["event_end"]
            .astype(str)
            + "-01",
            errors="coerce"
        )

    # ========================================================
    # INITIAL LABEL COLUMNS
    # ========================================================

    features["firms_event_target"] = 0

    features["firms_pre_event_3m"] = 0

    features["firms_event_id"] = ""

    features["firms_event_start"] = pd.NaT

    features["firms_event_end"] = pd.NaT

    features["months_to_event"] = np.nan

    features["firms_reference_type"] = (
        "no_candidate_event"
    )

    # ========================================================
    # APPLY EVENT LABELS
    # ========================================================

    for _, event in events.iterrows():

        event_id = str(
            event["event_id"]
        )

        start = event["event_start"]

        end = event["event_end"]

        if pd.isna(start) or pd.isna(end):

            continue

        # ----------------------------------------------------
        # Event months
        # ----------------------------------------------------

        event_mask = (
            (features["date"] >= start)
            &
            (features["date"] <= end)
        )

        features.loc[
            event_mask,
            "firms_event_target"
        ] = 1

        features.loc[
            event_mask,
            "firms_event_id"
        ] = event_id

        features.loc[
            event_mask,
            "firms_event_start"
        ] = start

        features.loc[
            event_mask,
            "firms_event_end"
        ] = end

        features.loc[
            event_mask,
            "months_to_event"
        ] = 0

        features.loc[
            event_mask,
            "firms_reference_type"
        ] = "candidate_fire_event"

        # ----------------------------------------------------
        # Pre-event warning window
        #
        # Example:
        # Event starts in June
        # Pre-event months = March, April, May
        # ----------------------------------------------------

        pre_start = (
            start
            - pd.DateOffset(
                months=PRE_EVENT_MONTHS
            )
        )

        pre_end = (
            start
            - pd.DateOffset(
                months=1
            )
        )

        pre_mask = (
            (features["date"] >= pre_start)
            &
            (features["date"] <= pre_end)
            &
            (features["firms_event_target"] == 0)
        )

        features.loc[
            pre_mask,
            "firms_pre_event_3m"
        ] = 1

        # ----------------------------------------------------
        # Months-to-event
        #
        # 3 = three months before
        # 2 = two months before
        # 1 = one month before
        # 0 = event month
        # ----------------------------------------------------

        pre_dates = features.loc[
            pre_mask,
            "date"
        ]

        if len(pre_dates) > 0:

            start_period = pd.Period(
                start,
                freq="M"
            )

            months_difference = (
                start_period
                - pre_dates.dt.to_period("M")
            )

            # Period difference → integer months
            features.loc[
                pre_mask,
                "months_to_event"
            ] = [
                int(x.n)
                for x in months_difference
            ]

            features.loc[
                pre_mask,
                "firms_event_id"
            ] = event_id

            features.loc[
                pre_mask,
                "firms_event_start"
            ] = start

            features.loc[
                pre_mask,
                "firms_event_end"
            ] = end

            features.loc[
                pre_mask,
                "firms_reference_type"
            ] = "pre_event_window"

    # ========================================================
    # SORT
    # ========================================================

    features = (
        features
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    event_rows = (
        features["firms_event_target"]
        == 1
    ).sum()

    pre_event_rows = (
        features["firms_pre_event_3m"]
        == 1
    ).sum()

    total_rows = len(features)

    print("\nLABEL SUMMARY")
    print("-" * 70)

    print(
        f"Total temporal rows:       {total_rows:,}"
    )

    print(
        f"Event-month rows:           {event_rows:,}"
    )

    print(
        f"Pre-event rows (3 months):  {pre_event_rows:,}"
    )

    print(
        f"No-event rows:              "
        f"{total_rows - event_rows - pre_event_rows:,}"
    )

    print(
        f"Date range:                 "
        f"{features['date'].min():%Y-%m} "
        f"to "
        f"{features['date'].max():%Y-%m}"
    )

    # ========================================================
    # EVENT COVERAGE
    # ========================================================

    if len(events) > 0:

        matched_events = set(
            features.loc[
                features["firms_event_target"] == 1,
                "firms_event_id"
            ]
        )

        matched_events.discard("")

        print(
            f"Candidate events represented: "
            f"{len(matched_events):,} / "
            f"{len(events):,}"
        )

    # ========================================================
    # PRINT LABEL TABLE
    # ========================================================

    print("\nMONTHLY LABELS")

    display_columns = [
        "date",
        "firms_event_target",
        "firms_pre_event_3m",
        "firms_event_id",
        "months_to_event",
        "firms_reference_type",
    ]

    print(
        features[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        FEATURE_DIR
        / f"{region_name.lower()}_event_labels.csv"
    )

    features.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SLOWWATCH — REGIONAL EVENT LABEL CONSTRUCTION")
    print("=" * 70)

    results = {}

    for region_name, config in REGIONS.items():

        results[region_name] = process_region(
            region_name,
            config
        )

    print("\n" + "=" * 70)
    print("REGIONAL EVENT LABEL CONSTRUCTION COMPLETE")
    print("=" * 70)

    for region_name, df in results.items():

        print(
            f"{region_name}: "
            f"{len(df):,} rows"
        )


if __name__ == "__main__":
    main()