from pathlib import Path
import warnings
import pandas as pd
import xarray as xr
import cfgrib

warnings.filterwarnings("ignore")

# ============================================================
# SLOWWATCH REGIONAL ERA5-LAND ENVIRONMENTAL INTEGRATION
# Regions: Wayanad + Nilgiris
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

ERA5_DIR = BASE_DIR / "ERA5_Land_Regional"

ERA5_TEMP = ERA5_DIR / "482188f5c82fe32cdf471020a091b176.grib"
ERA5_PRECIP = ERA5_DIR / "2b7c61a9f3351bced429e00c2fb6fccb.grib"

REGIONAL_FEATURE_DIR = (
    PROJECT_DIR / "data" / "features" / "regional"
)

REGIONAL_FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REGIONS = {
    "Wayanad": "wayanad_monthly_temporal_dataset.csv",
    "Nilgiris": "nilgiris_monthly_temporal_dataset.csv",
}


# ============================================================
# FIND DATE COLUMN
# ============================================================

def find_date_column(df):

    candidates = [
        "date",
        "Date",
        "month",
        "Month",
        "timestamp",
        "datetime",
        "time",
        "observation_date",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: detect a datetime-like column
    for col in df.columns:

        parsed = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        if parsed.notna().sum() >= max(
            3,
            int(0.8 * len(df))
        ):
            return col

    raise ValueError(
        "Could not identify a date column."
    )


# ============================================================
# LOAD ERA5-LAND TEMPERATURE
# ============================================================

def load_era5_temperature():

    print("\n" + "=" * 70)
    print("LOADING ERA5-LAND TEMPERATURE")
    print("=" * 70)

    ds = xr.open_dataset(
        ERA5_TEMP,
        engine="cfgrib"
    )

    if "t2m" not in ds:

        raise KeyError(
            "Variable 't2m' was not found "
            "in the ERA5 temperature dataset."
        )

    # Spatial mean over the regional ERA5 grid
    temperature = ds["t2m"].mean(
        dim=["latitude", "longitude"],
        skipna=True
    )

    df = temperature.to_dataframe(
        name="temperature_k"
    ).reset_index()

    # Convert timestamp to month
    df["date"] = (
        pd.to_datetime(df["time"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Kelvin -> Celsius
    df["temperature_c"] = (
        df["temperature_k"] - 273.15
    )

    # One value per month
    df = (
        df[
            [
                "date",
                "temperature_c"
            ]
        ]
        .groupby(
            "date",
            as_index=False
        )["temperature_c"]
        .mean()
        .sort_values("date")
    )

    ds.close()

    print(
        f"Temperature coverage: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    print(
        f"Temperature months: {len(df)}"
    )

    return df


# ============================================================
# LOAD ERA5-LAND PRECIPITATION
# ============================================================

def load_era5_precipitation():

    print("\n" + "=" * 70)
    print("LOADING ERA5-LAND PRECIPITATION")
    print("=" * 70)

    datasets = cfgrib.open_datasets(
        str(ERA5_PRECIP)
    )

    precipitation_parts = []

    for i, ds in enumerate(datasets):

        if "tp" not in ds:
            continue

        print(
            f"Precipitation dataset {i}: "
            f"{len(ds.time)} time steps"
        )

        # Spatial mean
        precipitation = ds["tp"].mean(
            dim=["latitude", "longitude"],
            skipna=True
        )

        part = precipitation.to_dataframe(
            name="precipitation_m"
        ).reset_index()

        # Convert to monthly timestamp
        part["date"] = (
            pd.to_datetime(part["time"])
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        # ERA5 precipitation is expressed in metres
        # Convert metres -> millimetres
        part["precipitation_mm"] = (
            part["precipitation_m"] * 1000.0
        )

        precipitation_parts.append(
            part[
                [
                    "date",
                    "precipitation_mm"
                ]
            ]
        )

        ds.close()

    if not precipitation_parts:

        raise ValueError(
            "No 'tp' precipitation variable "
            "was found in the ERA5 file."
        )

    df = pd.concat(
        precipitation_parts,
        ignore_index=True
    )

    # Combine any overlapping/duplicate months
    df = (
        df
        .groupby(
            "date",
            as_index=False
        )["precipitation_mm"]
        .mean()
        .sort_values("date")
    )

    print(
        f"Precipitation coverage: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    print(
        f"Precipitation months: {len(df)}"
    )

    return df


# ============================================================
# MERGE ENVIRONMENTAL CONTEXT WITH REGIONAL HLS FEATURES
# ============================================================

def load_and_merge_region(
    region_name,
    filename,
    temperature,
    precipitation
):

    print("\n" + "=" * 70)
    print(f"PROCESSING REGION: {region_name}")
    print("=" * 70)

    input_path = (
        REGIONAL_FEATURE_DIR / filename
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"\nRegional temporal dataset not found:\n"
            f"{input_path}"
        )

    print(
        f"Input:\n{input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    original_rows = len(df)

    date_col = find_date_column(df)

    print(
        f"Detected date column: {date_col}"
    )

    df["date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # Remove invalid dates
    df = df[
        df["date"].notna()
    ].copy()

    # Standardise to month start
    df["date"] = (
        df["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # --------------------------------------------------------
    # Merge temperature
    # --------------------------------------------------------

    merged = df.merge(
        temperature,
        on="date",
        how="left"
    )

    # --------------------------------------------------------
    # Merge precipitation
    # --------------------------------------------------------

    merged = merged.merge(
        precipitation,
        on="date",
        how="left"
    )

    # --------------------------------------------------------
    # Environmental temporal changes
    # --------------------------------------------------------

    merged["temperature_change"] = (
        merged["temperature_c"].diff()
    )

    merged["precipitation_change"] = (
        merged["precipitation_mm"].diff()
    )

    # Add region identifier
    merged.insert(
        0,
        "region",
        region_name
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        REGIONAL_FEATURE_DIR
        / f"{region_name.lower()}_multisource_dataset.csv"
    )

    merged.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    temperature_missing = int(
        merged["temperature_c"]
        .isna()
        .sum()
    )

    precipitation_missing = int(
        merged["precipitation_mm"]
        .isna()
        .sum()
    )

    print(
        f"Original rows:          {original_rows}"
    )

    print(
        f"Valid date rows:        {len(df)}"
    )

    print(
        f"Output rows:            {len(merged)}"
    )

    print(
        f"Date range:             "
        f"{merged['date'].min().date()} "
        f"to "
        f"{merged['date'].max().date()}"
    )

    print(
        f"Temperature missing:    "
        f"{temperature_missing}"
    )

    print(
        f"Precipitation missing:  "
        f"{precipitation_missing}"
    )

    print(
        f"Saved:\n{output_path}"
    )

    return {
        "region": region_name,
        "rows": len(merged),
        "date_start": merged["date"].min(),
        "date_end": merged["date"].max(),
        "temperature_missing": temperature_missing,
        "precipitation_missing": precipitation_missing,
        "output": str(output_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "SLOWWATCH REGIONAL ENVIRONMENTAL "
        "CONTEXT INTEGRATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check ERA5 files
    # --------------------------------------------------------

    if not ERA5_TEMP.exists():

        raise FileNotFoundError(
            f"\nERA5 temperature file not found:\n"
            f"{ERA5_TEMP}"
        )

    if not ERA5_PRECIP.exists():

        raise FileNotFoundError(
            f"\nERA5 precipitation file not found:\n"
            f"{ERA5_PRECIP}"
        )

    # --------------------------------------------------------
    # Load environmental data
    # --------------------------------------------------------

    temperature = (
        load_era5_temperature()
    )

    precipitation = (
        load_era5_precipitation()
    )

    # --------------------------------------------------------
    # ERA5 summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ERA5 SUMMARY")
    print("=" * 70)

    print(
        f"Temperature rows:   "
        f"{len(temperature)}"
    )

    print(
        f"Precipitation rows: "
        f"{len(precipitation)}"
    )

    # --------------------------------------------------------
    # Process both regions
    # --------------------------------------------------------

    summaries = []

    for region_name, filename in REGIONS.items():

        result = load_and_merge_region(
            region_name,
            filename,
            temperature,
            precipitation
        )

        summaries.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    print("\n" + "=" * 70)
    print("FINAL REGIONAL INTEGRATION SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print(
        "REGIONAL ERA5 INTEGRATION "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()