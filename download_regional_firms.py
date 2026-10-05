import os
import time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests


# ============================================================
# SLOWWATCH — Regional FIRMS Historical Downloader
# Wayanad + Nilgiris | 2018–2025
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
OUTPUT_DIR = BASE_DIR / "FIRMS_Regional"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_KEY = os.environ.get("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise RuntimeError(
        "FIRMS_MAP_KEY is not set in this PowerShell session."
    )


REGIONS = {
    "Wayanad": {
        "bbox": "75.55,11.45,76.40,12.05"
    },
    "Nilgiris": {
        "bbox": "76.20,10.85,77.05,11.65"
    },
}


START_DATE = date(2018, 1, 1)
END_DATE = date(2025, 12, 31)

# VIIRS Suomi-NPP provides the long historical record needed here.
SENSOR = "VIIRS_SNPP_SP"


def download_chunk(region_name, bbox, start_day):
    """Download one 5-day FIRMS area-query chunk."""

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{SENSOR}/{bbox}/5/"
        f"{start_day.isoformat()}"
    )

    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        print(
            f"[ERROR] {region_name} "
            f"{start_day} -> HTTP {response.status_code}"
        )
        print(response.text[:300])
        return None

    from io import StringIO

    text = response.text.strip()

    if not text:
        return pd.DataFrame()

    return pd.read_csv(StringIO(text))


def download_region(region_name, bbox):
    """Download the complete 2018–2025 period."""

    print("\n" + "=" * 70)
    print(f"REGION: {region_name}")
    print(f"BBOX:   {bbox}")
    print(f"PERIOD: {START_DATE} -> {END_DATE}")
    print("=" * 70)

    all_chunks = []

    current = START_DATE
    total_requests = 0

    while current <= END_DATE:

        print(
            f"[{region_name}] Requesting "
            f"{current} ..."
        )

        df = download_chunk(
            region_name,
            bbox,
            current
        )

        total_requests += 1

        if df is not None and not df.empty:
            all_chunks.append(df)
            print(
                f"    detections: {len(df)}"
            )
        else:
            print("    detections: 0")

        current += timedelta(days=5)

        # Stay comfortably below the API request-rate limit.
        time.sleep(0.25)

    if all_chunks:
        result = pd.concat(
            all_chunks,
            ignore_index=True
        )

        # Remove duplicate detections caused by
        # overlapping 5-day query boundaries.
        duplicate_columns = [
            c for c in [
                "latitude",
                "longitude",
                "acq_date",
                "acq_time",
                "satellite",
                "instrument",
            ]
            if c in result.columns
        ]

        if duplicate_columns:
            result = result.drop_duplicates(
                subset=duplicate_columns
            )

    else:
        result = pd.DataFrame()

    output_file = (
        OUTPUT_DIR /
        f"{region_name.lower()}_firms_2018_2025.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print("\nRESULT")
    print("-" * 70)
    print(f"Region:      {region_name}")
    print(f"Requests:    {total_requests}")
    print(f"Detections:  {len(result)}")
    print(f"Saved:       {output_file}")

    if not result.empty:
        print(
            f"Date range:  "
            f"{result['acq_date'].min()} -> "
            f"{result['acq_date'].max()}"
        )

        print(
            f"Latitude:    "
            f"{result['latitude'].min():.4f} -> "
            f"{result['latitude'].max():.4f}"
        )

        print(
            f"Longitude:   "
            f"{result['longitude'].min():.4f} -> "
            f"{result['longitude'].max():.4f}"
        )

    return result


def main():

    print("\nSLOWWATCH REGIONAL FIRMS ACQUISITION")
    print("Sensor:", SENSOR)
    print("Period:", START_DATE, "->", END_DATE)

    for region_name, info in REGIONS.items():
        download_region(
            region_name,
            info["bbox"]
        )

    print("\n" + "=" * 70)
    print("REGIONAL FIRMS ACQUISITION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()