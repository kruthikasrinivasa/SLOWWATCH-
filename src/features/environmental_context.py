from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib


# ============================================================
# SLOWWATCH - ENVIRONMENTAL CONTEXT INTEGRATION
# ERA5-Land Temperature + Precipitation
# Combined with monthly HLS NDVI temporal dataset
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

ERA5_DIR = BASE_DIR / "ERA5_Land"

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "features"
    / "monthly_temporal_dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "features"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "slowwatch_multisource_dataset.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Kodagu study region
MIN_LAT = 12.2
MAX_LAT = 12.9
MIN_LON = 75.4
MAX_LON = 76.1


# ============================================================
# FIND ERA5 GRIB FILE
# ============================================================

def find_grib_file():

    files = list(ERA5_DIR.glob("*.grib"))

    if not files:
        raise FileNotFoundError(
            f"No GRIB file found in {ERA5_DIR}"
        )

    return files[0]


# ============================================================
# SPATIAL MEAN
# ============================================================

def spatial_mean(ds, variable):

    da = ds[variable]

    lat_name = "latitude"
    lon_name = "longitude"

    # Select Kodagu region.
    # ERA5 latitude is commonly stored in descending order,
    # so we handle both orientations.

    if da[lat_name][0] > da[lat_name][-1]:
        da = da.sel(
            latitude=slice(MAX_LAT, MIN_LAT),
            longitude=slice(MIN_LON, MAX_LON)
        )
    else:
        da = da.sel(
            latitude=slice(MIN_LAT, MAX_LAT),
            longitude=slice(MIN_LON, MAX_LON)
        )

    result = da.mean(
        dim=[lat_name, lon_name],
        skipna=True
    )

    return result


# ============================================================
# LOAD ERA5-LAND
# ============================================================

def load_era5():

    grib_file = find_grib_file()

    print("=" * 70)
    print("SLOWWATCH - ERA5-LAND ENVIRONMENTAL CONTEXT")
    print("=" * 70)

    print(f"ERA5 file: {grib_file}")
    print()

    print("Opening ERA5-Land GRIB...")

    import cfgrib

    datasets = cfgrib.open_datasets(
        str(grib_file)
    )

    print(f"GRIB datasets discovered: {len(datasets)}")
    
    
    temperature = None
    precipitation_datasets = []

    for ds in datasets:

        variables = list(ds.data_vars)

        print("Variables:", variables)

        if "t2m" in ds.data_vars:
            temperature = ds

        if "tp" in ds.data_vars:
            precipitation_datasets.append(ds)


    if temperature is None:
        raise RuntimeError(
            "Temperature variable t2m was not found."
        )


    if len(precipitation_datasets) == 0:
        raise RuntimeError(
            "Precipitation variable tp was not found."
        )


    # Combine all precipitation GRIB groups
    if len(precipitation_datasets) == 1:

        precipitation = precipitation_datasets[0]

    else:

        print(
            f"Combining {len(precipitation_datasets)} "
            "precipitation datasets..."
        )

        precipitation = xr.concat(
            precipitation_datasets,
            dim="time"
        )
    
    

    

    if temperature is None:
        raise RuntimeError(
            "Temperature variable t2m was not found."
        )

    if precipitation is None:
        raise RuntimeError(
            "Precipitation variable tp was not found."
        )

    print()
    print("Calculating Kodagu spatial mean...")

    temp = spatial_mean(
        temperature,
        "t2m"
    )

    precip = spatial_mean(
        precipitation,
        "tp"
    )

    # --------------------------------------------------------
    # Convert temperature Kelvin -> Celsius
    # --------------------------------------------------------

    temp_c = temp - 273.15

    temp_df = temp_c.to_dataframe(
        name="temperature_c"
    ).reset_index()

    precip_df = precip.to_dataframe(
        name="precipitation_raw"
    ).reset_index()

    # --------------------------------------------------------
    # Keep only time + value
    # --------------------------------------------------------

    temp_df = temp_df[
        ["time", "temperature_c"]
    ]

    precip_df = precip_df[
        ["time", "precipitation_raw"]
    ]

    temp_df["date"] = pd.to_datetime(
        temp_df["time"]
    ).dt.to_period("M").dt.to_timestamp()

    precip_df["date"] = pd.to_datetime(
        precip_df["time"]
    ).dt.to_period("M").dt.to_timestamp()

    temp_df = (
        temp_df
        .groupby("date", as_index=False)
        ["temperature_c"]
        .mean()
    )

    precip_df = (
        precip_df
        .groupby("date", as_index=False)
        ["precipitation_raw"]
        .mean()
    )

    # --------------------------------------------------------
    # Combine environmental variables
    # --------------------------------------------------------

    era5 = pd.merge(
        temp_df,
        precip_df,
        on="date",
        how="outer"
    )

    era5 = era5.sort_values("date")

    return era5


# ============================================================
# COMBINE WITH HLS TEMPORAL DATA
# ============================================================

def combine_with_ndvi(era5):

    print()
    print("Loading HLS monthly temporal dataset...")

    ndvi = pd.read_csv(INPUT_FILE)

    ndvi["date"] = pd.to_datetime(
        ndvi["date"]
    )

    ndvi["date"] = (
        ndvi["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    print(
        f"HLS monthly records: {len(ndvi)}"
    )

    print(
        f"ERA5 monthly records: {len(era5)}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    combined = pd.merge(
        ndvi,
        era5,
        on="date",
        how="left"
    )

    combined = combined.sort_values(
        "date"
    )

    # --------------------------------------------------------
    # Add environmental change features
    # --------------------------------------------------------

    combined["temperature_change"] = (
        combined["temperature_c"].diff()
    )

    combined["precipitation_change"] = (
        combined["precipitation_raw"].diff()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return combined


# ============================================================
# MAIN
# ============================================================

def main():

    era5 = load_era5()

    print()
    print(
        f"ERA5 date range: "
        f"{era5['date'].min().date()} "
        f"to "
        f"{era5['date'].max().date()}"
    )

    combined = combine_with_ndvi(
        era5
    )

    print()
    print("=" * 70)
    print("MULTI-SOURCE DATASET CREATED")
    print("=" * 70)

    print(
        f"Records: {len(combined)}"
    )

    print(
        f"Date range: "
        f"{combined['date'].min().date()} "
        f"to "
        f"{combined['date'].max().date()}"
    )

    print()
    print("Columns:")

    for column in combined.columns:
        print(" -", column)

    print()
    print("Missing values:")

    print(
        combined[
            [
                "ndvi_mean",
                "temperature_c",
                "precipitation_raw"
            ]
        ].isna().sum()
    )

    print()
    print("Preview:")

    print(
        combined[
            [
                "date",
                "ndvi_mean",
                "temperature_c",
                "precipitation_raw",
                "ndvi_anomaly",
                "slope_6m"
            ]
        ].head(10).to_string(index=False)
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("ERA5 INTEGRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()