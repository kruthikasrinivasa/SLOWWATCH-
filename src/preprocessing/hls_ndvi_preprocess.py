from pathlib import Path
from datetime import datetime
import glob
import os

import numpy as np
import rasterio


# ============================================================
# SLOWWATCH - HLS NDVI PREPROCESSING
# First-stage test: process ONE HLS Landsat granule
# ============================================================

# Input data
NDVI_DIR = Path("F:/SLOWWATCH_DATA/HLS_VI/NDVI")
FMASK_DIR = Path("F:/SLOWWATCH_DATA/HLS_VI/Fmask")

# Output directory inside the project
OUTPUT_DIR = Path(
    "F:/SLOWWATCH_DATA/SLOWWATCH_PROJECT/data/processed/hls_test"
)

# HLS NDVI scale factor
NDVI_SCALE = 0.0001

# HLS NDVI NoData value
NDVI_NODATA = -19999

# Output NoData value
OUTPUT_NODATA = -9999.0


def find_fmask(ndvi_file):
    """
    Find the Fmask corresponding to an NDVI file.

    Matching is based on the tile and acquisition timestamp,
    so differences in product filename prefixes do not matter.
    """

    parts = ndvi_file.name.split(".")

    # Example:
    # HLS-VI.L30.T43PEQ.2018012T051707.v2.0.NDVI.tif
    #
    # Tile:
    # T43PEQ
    #
    # Acquisition:
    # 2018012T051707

    tile = None
    timestamp = None

    for part in parts:
        if part.startswith("T") and len(part) == 6:
            tile = part

        if "T" in part and len(part) == 14:
            timestamp = part

    if tile is None or timestamp is None:
        raise ValueError(
            f"Could not identify tile/timestamp from: {ndvi_file.name}"
        )

    # Search all Fmask files for the same tile and timestamp
    pattern = f"*{tile}*{timestamp}*.Fmask.tif"

    matches = list(FMASK_DIR.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No matching Fmask found.\n"
            f"Tile: {tile}\n"
            f"Timestamp: {timestamp}\n"
            f"Pattern: {pattern}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple matching Fmask files found:\n"
            + "\n".join(str(m) for m in matches)
        )

    return matches[0]


def decode_quality_mask(fmask):
    """
    Decode HLS v2.0 Fmask QA bits.

    Bit 0 = Cirrus
    Bit 1 = Cloud
    Bit 2 = Adjacent to cloud/shadow
    Bit 3 = Cloud shadow
    Bit 4 = Snow/ice
    Bit 5 = Water
    Bits 6-7 = Aerosol level

    Aerosol:
        00 = climatology
        01 = low
        10 = average
        11 = high
    """

    cirrus = (fmask & 1) != 0
    cloud = (fmask & 2) != 0
    adjacent = (fmask & 4) != 0
    shadow = (fmask & 8) != 0
    snow = (fmask & 16) != 0
    water = (fmask & 32) != 0

    aerosol = (fmask >> 6) & 3

    high_aerosol = aerosol == 3

    # Pixels accepted for SLOWWATCH
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


def extract_acquisition_date(filename):
    """
    Extract acquisition date from HLS filename.

    Example:
    HLS-VI.L30.T43PEQ.2018012T051707.v2.0.NDVI.tif

    2018 = year
    012  = day of year
    """

    parts = filename.split(".")

    date_part = None

    for part in parts:
        if "T" in part and len(part) == 14:
            date_part = part
            break

    if date_part is None:
        raise ValueError(
            f"Could not identify acquisition timestamp: {filename}"
        )

    return datetime.strptime(
        date_part,
        "%Y%jT%H%M%S"
    )


def process_one_granule(ndvi_file):
    """
    Process one HLS NDVI + Fmask granule.
    """

    fmask_file = find_fmask(ndvi_file)

    print("=" * 60)
    print("SLOWWATCH HLS PREPROCESSING TEST")
    print("=" * 60)

    print("NDVI file :", ndvi_file.name)
    print("Fmask file:", fmask_file.name)

    acquisition = extract_acquisition_date(
        ndvi_file.name
    )

    print("Acquisition:", acquisition)

    with rasterio.open(ndvi_file) as ndvi_src:

        ndvi_raw = ndvi_src.read(1)

        profile = ndvi_src.profile.copy()

        print("Raster size:", ndvi_src.width, "x", ndvi_src.height)
        print("CRS:", ndvi_src.crs)
        print("Resolution:", ndvi_src.res)

    with rasterio.open(fmask_file) as fmask_src:

        fmask = fmask_src.read(1)

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if ndvi_raw.shape != fmask.shape:
        raise ValueError(
            "NDVI and Fmask dimensions do not match."
        )

    # --------------------------------------------------------
    # Identify valid NDVI pixels
    # --------------------------------------------------------

    valid_ndvi = ndvi_raw != NDVI_NODATA

    # --------------------------------------------------------
    # Decode Fmask quality
    # --------------------------------------------------------

    good_quality = decode_quality_mask(fmask)

    # Final usable pixels
    good_pixels = valid_ndvi & good_quality

    # --------------------------------------------------------
    # Convert NDVI to physical values
    # --------------------------------------------------------

    ndvi = np.full(
        ndvi_raw.shape,
        OUTPUT_NODATA,
        dtype=np.float32
    )

    ndvi[good_pixels] = (
        ndvi_raw[good_pixels].astype(np.float32)
        * NDVI_SCALE
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_pixels = ndvi_raw.size
    valid_count = int(valid_ndvi.sum())
    good_count = int(good_pixels.sum())

    print()
    print("--- QUALITY SUMMARY ---")
    print("Total pixels       :", total_pixels)
    print("Valid NDVI pixels  :", valid_count)
    print("Good pixels        :", good_count)

    if valid_count > 0:
        print(
            "Good of valid NDVI :",
            round(100 * good_count / valid_count, 2),
            "%"
        )

    if good_count > 0:

        good_values = ndvi[good_pixels]

        print()
        print("--- CLEAN NDVI ---")
        print("Minimum :", float(good_values.min()))
        print("Maximum :", float(good_values.max()))
        print("Mean    :", float(good_values.mean()))
        print("Median  :", float(np.median(good_values)))

    # --------------------------------------------------------
    # Save processed GeoTIFF
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_name = (
        ndvi_file.stem
        + "_clean.tif"
    )

    output_file = OUTPUT_DIR / output_name

    profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate"
    )

    with rasterio.open(
        output_file,
        "w",
        **profile
    ) as dst:

        dst.write(ndvi, 1)

    print()
    print("OUTPUT:")
    print(output_file)
    print()
    print("SUCCESS - one HLS granule processed.")
    print("=" * 60)


def main():

    ndvi_files = sorted(
        Path(f)
        for f in glob.glob(
            str(NDVI_DIR / "*.tif")
        )
    )

    if not ndvi_files:
        raise FileNotFoundError(
            f"No NDVI files found in {NDVI_DIR}"
        )

    print("NDVI files found:", len(ndvi_files))

    # IMPORTANT:
    # For now we process ONLY ONE file.
    test_file = ndvi_files[0]

    process_one_granule(test_file)


if __name__ == "__main__":
    main()