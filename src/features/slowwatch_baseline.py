from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SLOWWATCH - PRELIMINARY BASELINE ANALYSIS
# Multi-source temporal degradation analysis
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "features"
    / "slowwatch_multisource_dataset.csv"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
)

METRIC_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# BASELINE WARNING REFINEMENT
# ============================================================

def calculate_warning_signal(df):

    # Existing baseline signal:
    # negative seasonal anomaly + negative 6-month slope

    df["trajectory_warning"] = (
        (df["ndvi_anomaly"] < 0)
        &
        (df["slope_6m"] < 0)
    )

    # Environmental context:
    # precipitation below its historical median
    # is used only as contextual information,
    # not as a ground-truth degradation label.

    precipitation_median = (
        df["precipitation_raw"].median()
    )

    df["low_precipitation_context"] = (
        df["precipitation_raw"]
        < precipitation_median
    )

    # Combined preliminary signal
    df["context_supported_warning"] = (
        df["trajectory_warning"]
        &
        df["low_precipitation_context"]
    )

    return df


# ============================================================
# SAVE WARNING SUMMARY
# ============================================================

def save_warning_summary(df):

    warnings = df[
        df["trajectory_warning"]
    ].copy()

    warning_columns = [
        "date",
        "ndvi_mean",
        "ndvi_anomaly",
        "slope_6m",
        "temperature_c",
        "precipitation_raw",
        "trajectory_warning",
        "low_precipitation_context",
        "context_supported_warning"
    ]

    warnings = warnings[warning_columns]

    output = (
        METRIC_DIR
        / "baseline_warning_summary.csv"
    )

    warnings.to_csv(
        output,
        index=False
    )

    return output, warnings


# ============================================================
# FIGURE 1 — NDVI TRAJECTORY
# ============================================================

def plot_trajectory(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["ndvi_mean"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Monthly NDVI"
    )

    plt.plot(
        df["date"],
        df["ndvi_rolling_mean_3"],
        linewidth=2,
        label="3-Month Rolling Mean"
    )

    plt.xlabel("Time")
    plt.ylabel("NDVI")

    plt.title(
        "SLOWWATCH — Vegetation Health Trajectory"
    )

    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "slowwatch_final_ndvi_trajectory.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    return output


# ============================================================
# FIGURE 2 — ANOMALY + WARNING
# ============================================================

def plot_anomaly(df):

    plt.figure(figsize=(12, 6))

    plt.axhline(
        0,
        linewidth=1
    )

    plt.plot(
        df["date"],
        df["ndvi_anomaly"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="NDVI Anomaly"
    )

    warnings = df[
        df["trajectory_warning"]
    ]

    if len(warnings) > 0:

        plt.scatter(
            warnings["date"],
            warnings["ndvi_anomaly"],
            s=55,
            label="Preliminary Baseline Warning"
        )

    plt.xlabel("Time")
    plt.ylabel("NDVI Anomaly")

    plt.title(
        "SLOWWATCH — NDVI Anomaly and Preliminary Warning Signal"
    )

    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "slowwatch_final_warning_analysis.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    return output


# ============================================================
# FIGURE 3 — ENVIRONMENTAL CONTEXT
# ============================================================

def plot_environment(df):

    # Normalize variables so their temporal patterns
    # can be compared on the same scale.

    ndvi = df["ndvi_mean"]

    temp = df["temperature_c"]

    precip = df["precipitation_raw"]

    ndvi_norm = (
        (ndvi - ndvi.mean())
        / ndvi.std()
    )

    temp_norm = (
        (temp - temp.mean())
        / temp.std()
    )

    precip_norm = (
        (precip - precip.mean())
        / precip.std()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        ndvi_norm,
        linewidth=1.8,
        label="NDVI"
    )

    plt.plot(
        df["date"],
        temp_norm,
        linewidth=1.5,
        label="Temperature"
    )

    plt.plot(
        df["date"],
        precip_norm,
        linewidth=1.5,
        label="Precipitation"
    )

    plt.xlabel("Time")
    plt.ylabel("Standardized Value")

    plt.title(
        "SLOWWATCH — Vegetation and Environmental Context"
    )

    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "slowwatch_environmental_context.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SLOWWATCH - PRELIMINARY BASELINE ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(
        f"Records loaded: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    df = calculate_warning_signal(df)

    warning_count = int(
        df["trajectory_warning"].sum()
    )

    context_warning_count = int(
        df["context_supported_warning"].sum()
    )

    print()
    print(
        f"Trajectory warnings: {warning_count}"
    )

    print(
        f"Context-supported warnings: "
        f"{context_warning_count}"
    )

    # Save warning summary

    summary_file, warnings = (
        save_warning_summary(df)
    )

    print()
    print("Warning summary:")
    print(summary_file)

    # Generate figures

    trajectory = plot_trajectory(df)

    anomaly = plot_anomaly(df)

    environment = plot_environment(df)

    print()
    print("Generated figures:")

    print(trajectory)
    print(anomaly)
    print(environment)

    print()
    print("=" * 70)
    print("BASELINE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()