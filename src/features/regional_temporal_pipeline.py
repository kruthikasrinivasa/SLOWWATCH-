from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# SLOWWATCH - REGIONAL TEMPORAL PIPELINE
#
# Wayanad + Nilgiris
#
# HLS NDVI + Fmask
#       ↓
# Monthly NDVI
#       ↓
# Temporal Features
#       ↓
# Seasonal Baseline
#       ↓
# Preliminary Baseline Warning
#
# IMPORTANT:
# Kodagu is NOT processed by this script.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"F:\SLOWWATCH_DATA"
)

PROJECT_DIR = (
    BASE_DIR
    / "SLOWWATCH_PROJECT"
)

FEATURE_DIR = (
    PROJECT_DIR
    / "data"
    / "features"
    / "regional"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "regional"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REGIONAL INPUT DIRECTORIES
# ============================================================

REGIONS = {

    "Wayanad": {

        "ndvi_dir":
            BASE_DIR
            / "Wayanad"
            / "HLS_VI"
            / "NDVI",

        "fmask_dir":
            BASE_DIR
            / "Wayanad"
            / "HLS_VI"
            / "Fmask"
    },

    "Nilgiris": {

        "ndvi_dir":
            BASE_DIR
            / "Nilgiris"
            / "HLS_VI"
            / "NDVI",

        "fmask_dir":
            BASE_DIR
            / "Nilgiris"
            / "HLS_VI"
            / "Fmask"
    }
}


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 512

NDVI_SCALE = 0.0001

NDVI_NODATA = -19999


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_timestamp(filename):

    match = re.search(
        r"\.(\d{7}T\d{6})\.",
        filename
    )

    if not match:
        return None

    timestamp = match.group(1)

    try:

        return pd.to_datetime(
            timestamp,
            format="%Y%jT%H%M%S"
        )

    except Exception:

        return None


# ============================================================
# FIND MATCHING FMASK
# ============================================================

def find_fmask(
    ndvi_file,
    fmask_dir
):

    name = ndvi_file.name

    tile_match = re.search(
        r"\.(T\d{2}[A-Z]{3})\.",
        name
    )

    time_match = re.search(
        r"\.(\d{7}T\d{6})\.",
        name
    )

    if (
        not tile_match
        or not time_match
    ):

        return None

    tile = tile_match.group(1)

    timestamp = time_match.group(1)

    matches = list(
        fmask_dir.glob(
            f"*{tile}*"
            f"{timestamp}*Fmask.tif"
        )
    )

    if not matches:

        return None

    return matches[0]


# ============================================================
# QUALITY MASK
# ============================================================

def quality_mask(fmask):

    # Fmask v2.0

    cirrus = (
        fmask
        & (1 << 0)
    ) != 0

    cloud = (
        fmask
        & (1 << 1)
    ) != 0

    adjacent = (
        fmask
        & (1 << 2)
    ) != 0

    shadow = (
        fmask
        & (1 << 3)
    ) != 0

    snow = (
        fmask
        & (1 << 4)
    ) != 0

    water = (
        fmask
        & (1 << 5)
    ) != 0

    aerosol = (
        fmask >> 6
    ) & 0b11

    high_aerosol = (
        aerosol == 3
    )

    good = ~(
        cirrus
        | cloud
        | adjacent
        | shadow
        | snow
        | water
        | high_aerosol
    )

    return good


# ============================================================
# REPRESENTATIVE WINDOW
# ============================================================

def get_window(src):

    width = src.width

    height = src.height

    col_off = max(
        (width - WINDOW_SIZE) // 2,
        0
    )

    row_off = max(
        (height - WINDOW_SIZE) // 2,
        0
    )

    return rasterio.windows.Window(
        col_off,
        row_off,
        min(
            WINDOW_SIZE,
            width
        ),
        min(
            WINDOW_SIZE,
            height
        )
    )


# ============================================================
# PROCESS ONE OBSERVATION
# ============================================================

def process_observation(
    ndvi_file,
    fmask_dir
):

    fmask_file = find_fmask(
        ndvi_file,
        fmask_dir
    )

    if fmask_file is None:

        return None

    date = extract_timestamp(
        ndvi_file.name
    )

    if date is None:

        return None

    try:

        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        with rasterio.open(
            ndvi_file
        ) as ndvi_src:

            window = get_window(
                ndvi_src
            )

            ndvi = ndvi_src.read(
                1,
                window=window
            ).astype(
                np.float32
            )

        # ----------------------------------------------------
        # FMASK
        # ----------------------------------------------------

        with rasterio.open(
            fmask_file
        ) as fmask_src:

            fmask = fmask_src.read(
                1,
                window=window
            )

        # ----------------------------------------------------
        # SCALE NDVI
        # ----------------------------------------------------

        ndvi = (
            ndvi
            * NDVI_SCALE
        )

        # ----------------------------------------------------
        # NDVI VALIDITY
        # ----------------------------------------------------

        valid_ndvi = (
            np.isfinite(ndvi)
            &
            (ndvi > -1.0)
            &
            (ndvi <= 1.0)
            &
            (
                ndvi
                !=
                NDVI_NODATA
                * NDVI_SCALE
            )
        )

        # ----------------------------------------------------
        # QUALITY FILTER
        # ----------------------------------------------------

        good_quality = quality_mask(
            fmask
        )

        valid = (
            valid_ndvi
            &
            good_quality
        )

        values = ndvi[
            valid
        ]

        # Minimum valid-pixel requirement
        if values.size < 100:

            return None

        # ----------------------------------------------------
        # RETURN OBSERVATION
        # ----------------------------------------------------

        return {

            "date":
                date,

            "year":
                date.year,

            "month":
                date.month,

            "ndvi_mean":
                float(
                    np.mean(values)
                ),

            "ndvi_median":
                float(
                    np.median(values)
                ),

            "valid_pixels":
                int(
                    values.size
                ),

            "quality_fraction":
                float(
                    values.size
                    /
                    ndvi.size
                ),

            "source_file":
                ndvi_file.name
        }

    except Exception as e:

        print(
            f"ERROR: "
            f"{ndvi_file.name}"
        )

        print(e)

        return None


# ============================================================
# BUILD ONE REGIONAL DATASET
# ============================================================

def build_region_dataset(
    region_name,
    ndvi_dir,
    fmask_dir
):

    print()
    print("=" * 70)
    print(
        f"SLOWWATCH REGIONAL TEMPORAL PIPELINE - "
        f"{region_name}"
    )
    print("=" * 70)

    print(
        f"NDVI directory : "
        f"{ndvi_dir}"
    )

    print(
        f"Fmask directory: "
        f"{fmask_dir}"
    )

    print()

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    ndvi_files = sorted(
        ndvi_dir.glob(
            "*.tif"
        )
    )

    print(
        f"NDVI files found: "
        f"{len(ndvi_files)}"
    )

    print(
        f"Representative window: "
        f"{WINDOW_SIZE} x "
        f"{WINDOW_SIZE}"
    )

    print()

    if not ndvi_files:

        raise RuntimeError(
            f"No NDVI files found "
            f"for {region_name}."
        )

    # --------------------------------------------------------
    # PROCESS OBSERVATIONS
    # --------------------------------------------------------

    records = []

    for i, ndvi_file in enumerate(
        ndvi_files,
        start=1
    ):

        result = process_observation(
            ndvi_file,
            fmask_dir
        )

        if result is not None:

            records.append(
                result
            )

        if (
            i % 100 == 0
            or i == len(ndvi_files)
        ):

            print(
                f"Processed "
                f"{i}/{len(ndvi_files)} | "
                f"valid observations: "
                f"{len(records)}"
            )

    if not records:

        raise RuntimeError(
            f"No valid observations "
            f"generated for "
            f"{region_name}."
        )

    # --------------------------------------------------------
    # OBSERVATION DATAFRAME
    # --------------------------------------------------------

    observations = pd.DataFrame(
        records
    )

    observations = (
        observations
        .sort_values("date")
        .reset_index(drop=True)
    )

    observations["date"] = (
        pd.to_datetime(
            observations["date"]
        )
    )

    # --------------------------------------------------------
    # MONTHLY INDEX
    # --------------------------------------------------------

    observations[
        "year_month"
    ] = (
        observations["date"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # MONTHLY AGGREGATION
    # --------------------------------------------------------

    monthly = (

        observations

        .groupby(
            "year_month"
        )

        .agg(

            ndvi_mean=(
                "ndvi_mean",
                "mean"
            ),

            ndvi_median=(
                "ndvi_median",
                "median"
            ),

            quality_fraction=(
                "quality_fraction",
                "mean"
            ),

            observation_count=(
                "ndvi_mean",
                "count"
            ),

            valid_pixels=(
                "valid_pixels",
                "mean"
            )
        )

        .reset_index()
    )

    monthly["date"] = (
        monthly[
            "year_month"
        ]
        .dt
        .to_timestamp()
    )

    monthly = (
        monthly
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    # --------------------------------------------------------
    # 3-month rolling mean
    # --------------------------------------------------------

    monthly[
        "ndvi_rolling_mean_3"
    ] = (

        monthly["ndvi_mean"]

        .rolling(
            window=3,
            min_periods=2
        )

        .mean()
    )

    # --------------------------------------------------------
    # 3-month rolling standard deviation
    # --------------------------------------------------------

    monthly[
        "ndvi_rolling_std_3"
    ] = (

        monthly["ndvi_mean"]

        .rolling(
            window=3,
            min_periods=2
        )

        .std()
    )

    # --------------------------------------------------------
    # Monthly NDVI change
    # --------------------------------------------------------

    monthly[
        "ndvi_change"
    ] = (
        monthly["ndvi_mean"]
        .diff()
    )

    # --------------------------------------------------------
    # 6-month trajectory slope
    # --------------------------------------------------------

    monthly[
        "slope_6m"
    ] = np.nan

    for i in range(
        len(monthly)
    ):

        if i < 5:

            continue

        y = monthly.loc[
            monthly.index[
                i - 5:i + 1
            ],
            "ndvi_mean"
        ].values

        if np.all(
            np.isfinite(y)
        ):

            x = np.arange(
                len(y)
            )

            slope = np.polyfit(
                x,
                y,
                1
            )[0]

            monthly.loc[
                monthly.index[i],
                "slope_6m"
            ] = slope

    # ========================================================
    # SEASONAL BASELINE
    # ========================================================

    monthly[
        "month"
    ] = (
        monthly["date"]
        .dt
        .month
    )

    seasonal_mean = (

        monthly

        .groupby("month")[
            "ndvi_mean"
        ]

        .transform("mean")
    )

    seasonal_std = (

        monthly

        .groupby("month")[
            "ndvi_mean"
        ]

        .transform("std")
    )

    # Expected seasonal NDVI
    monthly[
        "seasonal_expected_ndvi"
    ] = seasonal_mean

    # Seasonal anomaly
    monthly[
        "ndvi_anomaly"
    ] = (

        monthly["ndvi_mean"]

        -
        monthly[
            "seasonal_expected_ndvi"
        ]
    )

    # Anomaly z-score
    monthly[
        "anomaly_zscore"
    ] = (

        monthly[
            "ndvi_anomaly"
        ]

        /

        seasonal_std.replace(
            0,
            np.nan
        )
    )

    # ========================================================
    # PRELIMINARY BASELINE WARNING
    # ========================================================

    # Conservative preliminary rule:
    #
    # Negative seasonal anomaly
    # AND
    # Negative 6-month trajectory slope.
    #
    # This is NOT the final SLOWWATCH model.

    monthly[
        "baseline_warning"
    ] = (

        (
            monthly[
                "ndvi_anomaly"
            ] < 0
        )

        &

        (
            monthly[
                "slope_6m"
            ] < 0
        )
    )

    monthly[
        "baseline_warning"
    ] = (

        monthly[
            "baseline_warning"
        ]

        .fillna(False)

        .astype(int)
    )

    # ========================================================
    # SAVE REGIONAL DATA
    # ========================================================

    observation_file = (

        FEATURE_DIR

        /
        f"{region_name.lower()}"
        "_hls_observation_summary.csv"
    )

    monthly_file = (

        FEATURE_DIR

        /
        f"{region_name.lower()}"
        "_monthly_temporal_dataset.csv"
    )

    observations.to_csv(
        observation_file,
        index=False
    )

    monthly.to_csv(
        monthly_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"TEMPORAL PIPELINE COMPLETE - "
        f"{region_name}"
    )
    print("=" * 70)

    print(
        f"Valid observations : "
        f"{len(observations)}"
    )

    print(
        f"Monthly records    : "
        f"{len(monthly)}"
    )

    print(
        f"Date range         : "
        f"{monthly['date'].min().date()} "
        f"to "
        f"{monthly['date'].max().date()}"
    )

    print(
        f"Baseline warnings  : "
        f"{monthly['baseline_warning'].sum()}"
    )

    print()

    print(
        "Observation file:"
    )

    print(
        observation_file
    )

    print()

    print(
        "Monthly dataset:"
    )

    print(
        monthly_file
    )

    # ========================================================
    # FIGURE 1 - NDVI TRAJECTORY
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        monthly["date"],
        monthly["ndvi_mean"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Monthly NDVI"
    )

    plt.plot(
        monthly["date"],
        monthly[
            "ndvi_rolling_mean_3"
        ],
        linewidth=2,
        label="3-Month Rolling Mean"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "NDVI"
    )

    plt.title(
        f"SLOWWATCH — "
        f"{region_name} "
        f"Monthly Vegetation Trajectory"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    trajectory_file = (

        FIGURE_DIR

        /
        f"{region_name.lower()}"
        "_ndvi_trajectory.png"
    )

    plt.savefig(
        trajectory_file,
        dpi=200
    )

    plt.close()

    # ========================================================
    # FIGURE 2 - NDVI ANOMALY
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.plot(
        monthly["date"],
        monthly["ndvi_anomaly"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="NDVI Anomaly"
    )

    warning_data = monthly[
        monthly[
            "baseline_warning"
        ] == 1
    ]

    if len(warning_data) > 0:

        plt.scatter(
            warning_data["date"],
            warning_data[
                "ndvi_anomaly"
            ],
            s=45,
            label="Baseline Warning"
        )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "NDVI Anomaly"
    )

    plt.title(
        f"SLOWWATCH — "
        f"{region_name} "
        f"Seasonal NDVI Anomaly "
        f"& Preliminary Warnings"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    anomaly_file = (

        FIGURE_DIR

        /
        f"{region_name.lower()}"
        "_ndvi_anomaly.png"
    )

    plt.savefig(
        anomaly_file,
        dpi=200
    )

    plt.close()

    # ========================================================
    # FIGURE 3 - TRAJECTORY SLOPE
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.plot(
        monthly["date"],
        monthly["slope_6m"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="6-Month NDVI Slope"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Slope"
    )

    plt.title(
        f"SLOWWATCH — "
        f"{region_name} "
        f"Vegetation Trajectory Trend"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    slope_file = (

        FIGURE_DIR

        /
        f"{region_name.lower()}"
        "_trajectory_slope.png"
    )

    plt.savefig(
        slope_file,
        dpi=200
    )

    plt.close()

    print()

    print(
        "Generated figures:"
    )

    print(
        trajectory_file
    )

    print(
        anomaly_file
    )

    print(
        slope_file
    )

    return monthly


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SLOWWATCH - "
        "REGIONAL TEMPORAL FEATURE PIPELINE"
    )
    print("=" * 70)

    print()

    print(
        "Processing regions:"
    )

    print(
        "  - Wayanad"
    )

    print(
        "  - Nilgiris"
    )

    print()

    print(
        "Kodagu is intentionally NOT processed."
    )

    print()

    results = {}

    # ========================================================
    # PROCESS WAYANAD
    # ========================================================

    region_name = "Wayanad"

    paths = REGIONS[
        region_name
    ]

    results[
        region_name
    ] = build_region_dataset(
        region_name,
        paths["ndvi_dir"],
        paths["fmask_dir"]
    )

    # ========================================================
    # PROCESS NILGIRIS
    # ========================================================

    region_name = "Nilgiris"

    paths = REGIONS[
        region_name
    ]

    results[
        region_name
    ] = build_region_dataset(
        region_name,
        paths["ndvi_dir"],
        paths["fmask_dir"]
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "REGIONAL TEMPORAL PIPELINE SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        "+-----------+------------------+-----------------+"
    )

    print(
        "| Region    | Observations     | Monthly records |"
    )

    print(
        "+-----------+------------------+-----------------+"
    )

    for (
        region_name,
        monthly
    ) in results.items():

        print(
            f"| "
            f"{region_name:<9}"
            f" | "
            f"{len(monthly):<16}"
            f" | "
            f"{len(monthly):<15}"
            f"|"
        )

    print(
        "+-----------+------------------+-----------------+"
    )

    print()

    print(
        "Regional temporal preprocessing completed."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "ERA5-Land integration for "
        "Wayanad and Nilgiris."
    )


if __name__ == "__main__":

    main()