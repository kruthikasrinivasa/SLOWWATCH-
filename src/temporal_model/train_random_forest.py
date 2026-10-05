import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# SLOWWATCH - RANDOM FOREST ML BASELINE
# ============================================================

INPUT = r"data/splits/slowwatch_ml_targets.csv"
METRICS_DIR = r"results/metrics"
FIGURES_DIR = r"results/figures"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("SLOWWATCH - RANDOM FOREST BASELINE")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Leakage-safe predictor set
# ------------------------------------------------------------

FEATURES = [
    "ndvi_mean",
    "ndvi_median",
    "quality_fraction",
    "observation_count",
    "valid_pixels",
    "ndvi_rolling_mean_3",
    "ndvi_rolling_std_3",
    "ndvi_change",
    "slope_6m",
    "month",
    "seasonal_expected_ndvi",
    "ndvi_anomaly",
    "anomaly_zscore",
    "baseline_warning",
    "temperature_c",
    "precipitation_raw",
    "temperature_change",
    "precipitation_change",
]

TARGET = "target_event_3m"

# ------------------------------------------------------------
# 3. Chronological split
# ------------------------------------------------------------

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

print("\nChronological split:")
print(f"Train      : {len(train)}")
print(f"Validation : {len(val)}")
print(f"Test       : {len(test)}")

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_val = val[FEATURES].copy()
y_val = val[TARGET].astype(int)

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)

# ------------------------------------------------------------
# 4. Train-only imputation
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=FEATURES,
    index=X_train.index
)

X_val = pd.DataFrame(
    imputer.transform(X_val),
    columns=FEATURES,
    index=X_val.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=FEATURES,
    index=X_test.index
)

# ------------------------------------------------------------
# 5. Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 6. Evaluation function
# ------------------------------------------------------------

def evaluate(split_name, X, y):

    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    metrics = {
        "model": "RandomForest",
        "split": split_name,
        "n_samples": len(y),
        "positive_samples": int(y.sum()),
        "negative_samples": int((y == 0).sum()),
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability)
            if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, probability)
            if len(np.unique(y)) == 2 else np.nan,
    }

    print(f"\n--- {split_name.upper()} ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, prediction))

    print("\nClassification report:")
    print(classification_report(y, prediction, zero_division=0))

    return metrics, probability, prediction


# ------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------

train_metrics, train_prob, train_pred = evaluate(
    "train", X_train, y_train
)

val_metrics, val_prob, val_pred = evaluate(
    "validation", X_val, y_val
)

test_metrics, test_prob, test_pred = evaluate(
    "test", X_test, y_test
)

# ------------------------------------------------------------
# 8. Save metrics
# ------------------------------------------------------------

metrics_df = pd.DataFrame([
    train_metrics,
    val_metrics,
    test_metrics
])

metrics_path = os.path.join(
    METRICS_DIR,
    "random_forest_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)

# ------------------------------------------------------------
# 9. Save test predictions
# ------------------------------------------------------------

predictions = test[
    ["year_month", "date", TARGET]
].copy()

predictions["predicted_probability"] = test_prob
predictions["predicted_class"] = test_pred

predictions_path = os.path.join(
    METRICS_DIR,
    "random_forest_test_predictions.csv"
)

predictions.to_csv(predictions_path, index=False)

# ------------------------------------------------------------
# 10. Feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importance_path = os.path.join(
    METRICS_DIR,
    "random_forest_feature_importance.csv"
)

importance.to_csv(importance_path, index=False)

# ------------------------------------------------------------
# 11. Feature importance figure
# ------------------------------------------------------------

top = importance.head(12).sort_values("importance")

plt.figure(figsize=(9, 6))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Random Forest importance")
plt.ylabel("Feature")
plt.title("SLOWWATCH Random Forest Feature Importance")
plt.tight_layout()

importance_fig = os.path.join(
    FIGURES_DIR,
    "random_forest_feature_importance.png"
)

plt.savefig(importance_fig, dpi=200)
plt.close()

# ------------------------------------------------------------
# 12. Probability timeline
# ------------------------------------------------------------

plot_df = predictions.copy()
plot_df["date"] = pd.to_datetime(plot_df["date"])
plot_df = plot_df.sort_values("date")

plt.figure(figsize=(11, 5))

plt.plot(
    plot_df["date"],
    plot_df["predicted_probability"],
    marker="o",
    label="Predicted event probability"
)

plt.axhline(
    0.50,
    linestyle="--",
    label="Decision threshold"
)

positive = plot_df[plot_df[TARGET] == 1]

plt.scatter(
    positive["date"],
    positive["predicted_probability"],
    marker="x",
    s=70,
    label="Reference event"
)

plt.xlabel("Date")
plt.ylabel("Probability")
plt.title("SLOWWATCH Random Forest — Chronological Test Predictions")
plt.legend()
plt.tight_layout()

prob_fig = os.path.join(
    FIGURES_DIR,
    "random_forest_test_probability.png"
)

plt.savefig(prob_fig, dpi=200)
plt.close()

# ------------------------------------------------------------
# 13. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST COMPLETED")
print("=" * 70)

print(f"\nMetrics       : {metrics_path}")
print(f"Predictions   : {predictions_path}")
print(f"Importance    : {importance_path}")
print(f"Importance fig: {importance_fig}")
print(f"Probability fig: {prob_fig}")

print("\nTEST PERFORMANCE")
print(f"ROC-AUC : {test_metrics['roc_auc']:.4f}")
print(f"PR-AUC  : {test_metrics['pr_auc']:.4f}")
print(f"F1      : {test_metrics['f1']:.4f}")
print(f"Recall  : {test_metrics['recall']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")

print("\nDone.")