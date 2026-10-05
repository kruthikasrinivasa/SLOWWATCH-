"""
Regional ERA5-Land environmental context integration for SLOWWATCH.

Integrates:
- Wayanad HLS temporal features
- Nilgiris HLS temporal features
- Regional ERA5-Land 2m temperature (t2m)
- Regional ERA5-Land total precipitation (tp)

Outputs:
data/features/regional/wayanad_multisource_dataset.csv
data/features/regional/nilgiris_multisource_dataset.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import cfgrib

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

ERA5_DIR = BASE_DIR / "ERA5_Land_Regional"
ERA5_TEMP = ERA5_DIR / "482188f5c82fe32cdf471020a091b176.grib"
ERA5_PRECIP = ERA5_DIR / "2b7c61a9f3351bced429e00c2fb6fccb.grib"

REGIONAL_FEATURE_DIR = PROJECT_DIR / "data" / "features" / "regional"
REGIONAL_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = {
    "wayanad": PROJECT_DIR / "data" / "features" / "regional" / "wayanad_monthly_temporal_dataset.csv",
    "nilgiris": PROJECT_DIR / "data" / "features" / "regional" / "nilgiris_monthly_temporal_dataset.csv",
}


def find_date_column(df):
    candidates = [
        "date", "datetime", "timestamp", "time",
        "month", "month_start", "observation_date"
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: find a column that can be parsed as datetime
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return col

    raise ValueError(
        f"Could not identify a date column. Available columns: {list(df.columns)}"
    )


def load_era5_temperature():
    print("\n[1/4] Loading ERA5-Land temperature...")
    ds = xr.open_dataset(ERA5_TEMP, engine="cfgrib")

    if "t2m" not in ds.data_vars:
        raise ValueError(f"Temperature file does not contain t2m: {list(ds.data_vars)}")

    # Spatial mean over the downloaded regional grid.
    temp = ds["t2m"].mean(dim=["latitude", "longitude"], skipna=True)

    df = temp.to_dataframe(name="temperature_k").reset_index()
    df["date"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df["temperature_c"] = df["temperature_k"] - 273.15

    df = (
        df[["date", "temperature_c"]]
        .groupby("date", as_index=False)["temperature_c"]
        .mean()
        .sort_values("date")
    )

    print(
        f"  Temperature: {len(df)} months | "
        f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"
    )
    return df


def load_era5_precipitation():
    print("\n[2/4] Loading ERA5-Land precipitation...")

    # The CDS GRIB contains two tp message groups. cfgrib.open_datasets()
    # separates them; together they cover the complete 2018-2025 period.
    datasets = cfgrib.open_datasets(ERA5_PRECIP)

    frames = []

    for i, ds in enumerate(datasets):
        if "tp" not in ds.data_vars:
            continue

        precip = ds["tp"].mean(dim=["latitude", "longitude"], skipna=True)

        part = precip.to_dataframe(name="precipitation_m").reset_index()
        part["date"] = pd.to_datetime(part["time"]).dt.to_period("M").dt.to_timestamp()
        part = part[["date", "precipitation_m"]]

        frames.append(part)

        print(
            f"  TP dataset {i}: {len(part)} records | "
            f"{part['date'].min():%Y-%m} to {part['date'].max():%Y-%m}"
        )

    if not frames:
        raise ValueError("No tp variable found in precipitation GRIB.")

    df = pd.concat(frames, ignore_index=True)

    # Remove any accidental duplicate month created by GRIB grouping.
    df = (
        df.groupby("date", as_index=False)["precipitation_m"]
        .mean()
        .sort_values("date")
    )

    # ERA5-Land total precipitation is supplied in metres.
    df["precipitation_mm"] = df["precipitation_m"] * 1000.0

    print(
        f"  Precipitation combined: {len(df)} months | "
        f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"
    )
    return df[["date", "precipitation_mm"]]


def load_and_merge_region(region_name, hls_path, era5_df):
    print(f"\n[3/4] Integrating {region_name.title()}...")

    if not hls_path.exists():
        raise FileNotFoundError(f"HLS temporal dataset not found: {hls_path}")

    hls = pd.read_csv(hls_path)
    date_col = find_date_column(hls)

    hls["date"] = pd.to_datetime(hls[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    hls = hls.dropna(subset=["date"]).copy()

    # If the source has duplicate monthly rows, aggregate numeric columns
    # while retaining the first value for non-numeric metadata.
    hls = hls.sort_values("date")

    merged = hls.merge(era5_df, on="date", how="left")

    merged["temperature_change"] = merged["temperature_c"].diff()
    merged["precipitation_change"] = merged["precipitation_mm"].diff()

    # Keep region explicitly in the research dataset.
    merged.insert(0, "region", region_name)

    out = REGIONAL_FEATURE_DIR / f"{region_name}_multisource_dataset.csv"
    merged.to_csv(out, index=False)

    matched = merged["temperature_c"].notna() & merged["precipitation_mm"].notna()

    print(f"  HLS rows: {len(hls)}")
    print(f"  Output rows: {len(merged)}")
    print(f"  ERA5 matched rows: {int(matched.sum())}")
    print(f"  Missing ERA5 rows: {int((~matched).sum())}")
    print(f"  Saved: {out}")

    return merged


def main():
    print("=" * 72)
    print("SLOWWATCH — REGIONAL ERA5-LAND ENVIRONMENTAL INTEGRATION")
    print("=" * 72)

    temp = load_era5_temperature()
    precip = load_era5_precipitation()

    era5 = temp.merge(precip, on="date", how="outer").sort_values("date")

    print("\n[3/4] Combined ERA5 context:")
    print(
        f"  {len(era5)} months | "
        f"{era5['date'].min():%Y-%m} to {era5['date'].max():%Y-%m}"
    )
    print(f"  Missing temperature: {int(era5['temperature_c'].isna().sum())}")
    print(f"  Missing precipitation: {int(era5['precipitation_mm'].isna().sum())}")

    results = {}
    for region_name, hls_path in REGIONS.items():
        results[region_name] = load_and_merge_region(
            region_name, hls_path, era5
        )

    print("\n[4/4] FINAL SUMMARY")
    print("-" * 72)

    for region_name, df in results.items():
        matched = df["temperature_c"].notna() & df["precipitation_mm"].notna()
        print(
            f"{region_name.title():10s} | "
            f"rows={len(df):3d} | "
            f"ERA5 matched={int(matched.sum()):3d} | "
            f"period={df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"
        )

    print("-" * 72)
    print("REGIONAL ERA5 INTEGRATION COMPLETE.")


if __name__ == "__main__":
    main()
