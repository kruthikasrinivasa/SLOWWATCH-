import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# SLOWWATCH - ABLATION STUDY
# ============================================================

INPUT = r"data/splits/slowwatch_ml_targets.csv"

OUT_CSV = r"results/metrics/slowwatch_ablation_results.csv"

os.makedirs("results/metrics", exist_ok=True)

print("=" * 75)
print("SLOWWATCH - ABLATION STUDY")
print("=" * 75)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"\nDataset shape: {df.shape}")
print(
    f"Date range: {df['date'].min().date()} "
    f"-> {df['date'].max().date()}"
)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

TARGET = "target_event_3m"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

# ------------------------------------------------------------
# Chronological split
# Same split used by previous experiments:
# Train = 48
# Validation = 10
# Test = 19
# ------------------------------------------------------------

train = df.iloc[:48].copy()
validation = df.iloc[48:58].copy()
test = df.iloc[58:].copy()

print("\nChronological split:")
print("Train      :", len(train))
print("Validation :", len(validation))
print("Test       :", len(test))

# ------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------

NDVI_FEATURES = [
    "ndvi_mean",
    "ndvi_median"
]

TEMPORAL_FEATURES = [
    "ndvi_rolling_mean_3",
    "ndvi_rolling_std_3",
    "ndvi_change",
    "slope_6m",
    "ndvi_anomaly",
    "anomaly_zscore"
]

ENVIRONMENT_FEATURES = [
    "temperature_c",
    "precipitation_raw",
    "temperature_change",
    "precipitation_change"
]

QUALITY_FEATURES = [
    "quality_fraction",
    "observation_count",
    "valid_pixels"
]

# ------------------------------------------------------------
# Ablation configurations
# ------------------------------------------------------------

experiments = {

    "A_NDVI_ONLY": (
        NDVI_FEATURES
    ),

    "B_NDVI_PLUS_TEMPORAL": (
        NDVI_FEATURES
        + TEMPORAL_FEATURES
    ),

    "C_PLUS_ENVIRONMENT": (
        NDVI_FEATURES
        + TEMPORAL_FEATURES
        + ENVIRONMENT_FEATURES
    ),

    "D_FULL_SLOWWATCH_FEATURES": (
        NDVI_FEATURES
        + TEMPORAL_FEATURES
        + ENVIRONMENT_FEATURES
        + QUALITY_FEATURES
    )
}

# ------------------------------------------------------------
# Verify available features
# ------------------------------------------------------------

for name, features in experiments.items():

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing features: {missing}"
        )

# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def evaluate_model(name, features):

    X_train = train[features].copy()
    y_train = train[TARGET].astype(int)

    X_test = test[features].copy()
    y_test = test[TARGET].astype(int)

    # Handle missing values using TRAIN medians only
    train_medians = X_train.median()

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_test,
            probabilities
        )
    except ValueError:
        pr_auc = np.nan

    return {
        "experiment": name,
        "feature_count": len(features),
        "features": ", ".join(features),
        "test_samples": len(test),
        "positive_test_samples": int(y_test.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------

results = []

for name, features in experiments.items():

    print("\n" + "-" * 75)

    print(name)

    print("-" * 75)

    print("Features:")

    for feature in features:
        print("  -", feature)

    result = evaluate_model(
        name,
        features
    )

    results.append(result)

    print(
        f"\nAccuracy  : {result['accuracy']:.4f}"
    )

    print(
        f"Precision : {result['precision']:.4f}"
    )

    print(
        f"Recall    : {result['recall']:.4f}"
    )

    print(
        f"F1        : {result['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : {result['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : {result['pr_auc']:.4f}"
    )

# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT_CSV,
    index=False
)

# ------------------------------------------------------------
# Display final comparison
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ABLATION RESULTS")
print("=" * 75)

display_columns = [
    "experiment",
    "feature_count",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 75)
print("SLOWWATCH ABLATION COMPLETED")
print("=" * 75)

print(
    f"\nResults saved to:\n{OUT_CSV}"
)

print("\nInterpretation structure:")

print(
    "A = vegetation state only"
)

print(
    "B = vegetation + temporal trajectory"
)

print(
    "C = vegetation + temporal + environmental context"
)

print(
    "D = full feature configuration including quality information"
)

print(
    "\nAll experiments use the same chronological test period."
)

print("\nDone.")