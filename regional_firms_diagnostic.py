from pathlib import Path
import pandas as pd


FIRMS_DIR = Path(r"F:\SLOWWATCH_DATA\FIRMS_Regional")


def inspect_file(path):
    print("\n" + "=" * 70)
    print(path.name)
    print("=" * 70)

    df = pd.read_csv(path)

    df["acq_date"] = pd.to_datetime(df["acq_date"])

    print("Total detections:", len(df))
    print("Date range:",
          df["acq_date"].min().date(),
          "to",
          df["acq_date"].max().date())

    print(
        "Latitude:",
        round(df["latitude"].min(), 4),
        "to",
        round(df["latitude"].max(), 4)
    )

    print(
        "Longitude:",
        round(df["longitude"].min(), 4),
        "to",
        round(df["longitude"].max(), 4)
    )

    print("\nMonthly detections:")
    monthly = (
        df.assign(month=df["acq_date"].dt.to_period("M"))
          .groupby("month")
          .size()
    )

    print(monthly.to_string())

    print("\nColumns:")
    print(df.columns.tolist())


def main():

    files = sorted(
        FIRMS_DIR.glob("*_firms_2018_2025.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No regional FIRMS files found in {FIRMS_DIR}"
        )

    print("SLOWWATCH REGIONAL FIRMS DIAGNOSTIC")

    for file in files:
        inspect_file(file)

    print("\n" + "=" * 70)
    print("REGIONAL FIRMS DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()