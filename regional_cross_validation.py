"""
SLOWWATCH Regional Cross-Region Validation

Experiments:
    1. Train on Wayanad -> Test on Nilgiris
    2. Train on Nilgiris -> Test on Wayanad

Purpose:
    Evaluate spatial generalization of the SLOWWATCH feature space
    across geographically distinct regions.

Important:
    FIRMS-derived labels are disturbance-reference labels, not
    confirmed vegetation-degradation ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"F:\SLOWWATCH_DATA")
PROJECT_DIR = BASE_DIR / "SLOWWATCH_PROJECT"

REGIONAL_DIR = (
    PROJECT_DIR
    / "data"
    / "features"
    / "regional"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "Wayanad": REGIONAL_DIR / "wayanad_ml_dataset.csv",
    "Nilgiris": REGIONAL_DIR / "nilgiris_ml_dataset.csv",
}


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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


TARGET = "firms_event_target"


# ============================================================
# RANDOM FOREST SETTINGS
# ============================================================

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_region(region_name, path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"{TARGET} missing from {path}"
        )

    missing_features = [
        f for f in FEATURES
        if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in {region_name}: "
            f"{missing_features}"
        )

    print(
        f"{region_name:10s} | "
        f"rows={len(df):3d} | "
        f"events={int(df[TARGET].sum()):3d}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(train_df, test_df):

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = (
        pd.to_numeric(
            train_df[TARGET],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    y_test = (
        pd.to_numeric(
            test_df[TARGET],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # TRAIN-ONLY MEDIAN IMPUTATION
    #
    # Never use test-region statistics to fill training/test
    # features.
    # --------------------------------------------------------

    train_medians = X_train.median()

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # If an entire training feature is NaN, fall back to zero.
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_medians,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
    y_prob,
):

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }

    # ROC-AUC and PR-AUC require both classes.
    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )

        metrics["pr_auc"] = average_precision_score(
            y_true,
            y_prob
        )

    else:

        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


# ============================================================
# ONE CROSS-REGION EXPERIMENT
# ============================================================

def run_experiment(
    train_region,
    test_region,
    train_df,
    test_df,
):

    print("\n" + "=" * 70)
    print(
        f"TRAIN: {train_region.upper()} "
        f"-> TEST: {test_region.upper()}"
    )
    print("=" * 70)

    X_train, X_test, y_train, y_test, train_medians = (
        prepare_data(
            train_df,
            test_df
        )
    )

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        f"Training events  : {int(y_train.sum())}"
    )

    print(
        f"Testing events   : {int(y_test.sum())}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = evaluate_predictions(
        y_test,
        y_pred,
        y_prob
    )

    print("\nMetrics")
    print("-" * 40)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    if not np.isnan(metrics["roc_auc"]):

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC   : {metrics['pr_auc']:.4f}"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix")
    print("-" * 40)

    print(
        f"TN={tn}  FP={fp}"
    )

    print(
        f"FN={fn}  TP={tp}"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop Feature Importances")
    print("-" * 40)

    for _, row in importance.head(10).iterrows():

        print(
            f"{row['feature']:25s} "
            f"{row['importance']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = test_df[
        [
            c for c in [
                "region",
                "year_month",
                "date",
                TARGET,
                "firms_pre_event_3m",
                "firms_reference_type",
                "firms_event_id",
            ]
            if c in test_df.columns
        ]
    ].copy()

    prediction_df["predicted_event"] = y_pred
    prediction_df["event_probability"] = y_prob

    prediction_path = (
        RESULTS_DIR
        / (
            f"regional_cv_"
            f"{train_region.lower()}_to_"
            f"{test_region.lower()}_predictions.csv"
        )
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_path = (
        RESULTS_DIR
        / (
            f"regional_cv_"
            f"{train_region.lower()}_to_"
            f"{test_region.lower()}_feature_importance.csv"
        )
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    print("\nSaved:")
    print(prediction_path)
    print(importance_path)

    result = {
        "train_region": train_region,
        "test_region": test_region,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_events": int(y_train.sum()),
        "test_events": int(y_test.sum()),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SLOWWATCH REGIONAL CROSS-REGION VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD BOTH REGIONS
    # --------------------------------------------------------

    wayanad = load_region(
        "Wayanad",
        DATASETS["Wayanad"]
    )

    nilgiris = load_region(
        "Nilgiris",
        DATASETS["Nilgiris"]
    )

    # --------------------------------------------------------
    # RUN BOTH DIRECTIONS
    # --------------------------------------------------------

    results = []

    results.append(
        run_experiment(
            "Wayanad",
            "Nilgiris",
            wayanad,
            nilgiris,
        )
    )

    results.append(
        run_experiment(
            "Nilgiris",
            "Wayanad",
            nilgiris,
            wayanad,
        )
    )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        RESULTS_DIR
        / "regional_cross_region_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT FINAL TABLE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL CROSS-REGION RESULTS")
    print("=" * 70)

    display_columns = [
        "train_region",
        "test_region",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSaved:")
    print(results_path)

    print("\n" + "=" * 70)
    print("REGIONAL CROSS-REGION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()