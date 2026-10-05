import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

# ============================================================
# SLOWWATCH CORE
# Temporal Risk Score + Uncertainty + Explainable Evidence
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INPUT = r"data/splits/slowwatch_ml_targets.csv"
MODEL_PATH = r"models/slowwatch_lstm.pt"

METRICS_DIR = r"results/metrics"
FIGURES_DIR = r"results/figures"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEQUENCE_LENGTH = 6

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
    "precipitation_raw",
    "temperature_change",
    "precipitation_change",
]


# ============================================================
# MODEL
# ============================================================

class SlowWatchLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last = output[:, -1, :]

        return self.fc(last).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("SLOWWATCH - TEMPORAL RISK CORE")
print("=" * 75)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"\nObservations: {len(df)}")
print(
    f"Period: {df['date'].min().strftime('%Y-%m')} "
    f"to {df['date'].max().strftime('%Y-%m')}"
)


# ============================================================
# TRAIN-ONLY NORMALIZATION
# ============================================================

train_df = df[df["split"] == "train"].copy()

scaler = StandardScaler()

scaler.fit(train_df[FEATURES])

scaled = scaler.transform(df[FEATURES])

scaled = pd.DataFrame(
    scaled,
    columns=FEATURES
).replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


# ============================================================
# LOAD LSTM
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SlowWatchLSTM(
    input_size=len(FEATURES),
    hidden_size=32
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(f"Device: {device}")
print("LSTM model loaded.")


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

X = []
sequence_indices = []

for i in range(SEQUENCE_LENGTH - 1, len(df)):

    start = i - SEQUENCE_LENGTH + 1

    X.append(
        scaled.iloc[start:i + 1].values
    )

    sequence_indices.append(i)

X = np.asarray(
    X,
    dtype=np.float32
)

X_tensor = torch.tensor(X).to(device)


# ============================================================
# LSTM PROBABILITY
# ============================================================

with torch.no_grad():

    logits = model(X_tensor)

    lstm_probability = (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
    )


# Map probabilities back to observations

df["lstm_probability"] = np.nan

for idx, prob in zip(
    sequence_indices,
    lstm_probability
):
    df.loc[idx, "lstm_probability"] = prob


# ============================================================
# TRAJECTORY EVIDENCE
# ============================================================

# Negative anomaly -> increasing degradation evidence

anomaly = df["anomaly_zscore"].fillna(0)

negative_anomaly = np.clip(
    -anomaly / 2.5,
    0,
    1
)

# Negative slope -> increasing degradation evidence

slope = df["slope_6m"].fillna(0)

negative_slope = np.clip(
    -slope / 0.05,
    0,
    1
)

# NDVI change evidence

ndvi_change = df["ndvi_change"].fillna(0)

negative_change = np.clip(
    -ndvi_change / 0.10,
    0,
    1
)


# ============================================================
# ENVIRONMENTAL CONTEXT
# ============================================================

precip_change = df[
    "precipitation_change"
].fillna(0)

temperature_change = df[
    "temperature_change"
].fillna(0)

# Reduced precipitation = stress evidence

precip_stress = np.clip(
    -precip_change / 0.50,
    0,
    1
)

# Increased temperature = stress evidence

temperature_stress = np.clip(
    temperature_change / 2.0,
    0,
    1
)

context_evidence = (
    0.60 * precip_stress +
    0.40 * temperature_stress
)


# ============================================================
# LSTM EVIDENCE
# ============================================================

lstm_evidence = df[
    "lstm_probability"
].fillna(0)


# ============================================================
# TEMPORAL RISK SCORE
# ============================================================

# Proposed decision-level fusion:
#
# 45% temporal model
# 25% NDVI trajectory
# 20% anomaly/change
# 10% environmental context
#
# This is an interpretable risk fusion layer,
# not a separately trained neural network.

trs = (
    0.45 * lstm_evidence +
    0.25 * negative_slope +
    0.15 * negative_anomaly +
    0.05 * negative_change +
    0.10 * context_evidence
)

trs = np.clip(
    trs,
    0,
    1
)

df["temporal_risk_score"] = trs


# ============================================================
# EVIDENCE DISAGREEMENT / UNCERTAINTY
# ============================================================

evidence_matrix = np.column_stack([
    lstm_evidence.values,
    negative_slope.values,
    negative_anomaly.values,
    context_evidence.values
])

evidence_mean = evidence_matrix.mean(axis=1)

evidence_std = evidence_matrix.std(axis=1)

# High disagreement -> higher uncertainty

uncertainty = np.clip(
    evidence_std * 2.0,
    0,
    1
)

df["uncertainty"] = uncertainty


# ============================================================
# CONFIDENCE
# ============================================================

df["confidence"] = 1.0 - df["uncertainty"]


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(score):

    if score >= 0.75:
        return "HIGH"

    elif score >= 0.50:
        return "MODERATE"

    elif score >= 0.25:
        return "LOW"

    return "MINIMAL"


df["risk_category"] = (
    df["temporal_risk_score"]
    .apply(risk_category)
)


# ============================================================
# WARNING DECISION
# ============================================================

df["slowwatch_warning"] = (
    (df["temporal_risk_score"] >= 0.50) &
    (df["confidence"] >= 0.50)
).astype(int)


# ============================================================
# EXPLAINABLE COMPONENTS
# ============================================================

df["contribution_temporal_model"] = (
    0.45 * lstm_evidence
)

df["contribution_slope"] = (
    0.25 * negative_slope
)

df["contribution_anomaly"] = (
    0.15 * negative_anomaly
)

df["contribution_ndvi_change"] = (
    0.05 * negative_change
)

df["contribution_environment"] = (
    0.10 * context_evidence
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_columns = [
    "date",
    "ndvi_mean",
    "ndvi_change",
    "slope_6m",
    "anomaly_zscore",
    "temperature_c",
    "precipitation_raw",
    "lstm_probability",
    "temporal_risk_score",
    "uncertainty",
    "confidence",
    "risk_category",
    "slowwatch_warning",
    "contribution_temporal_model",
    "contribution_slope",
    "contribution_anomaly",
    "contribution_ndvi_change",
    "contribution_environment"
]

output = df[output_columns].copy()

output_path = os.path.join(
    METRICS_DIR,
    "slowwatch_core_results.csv"
)

output.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SLOWWATCH CORE RESULTS")
print("=" * 75)

print(
    "\nRisk category counts:"
)

print(
    output["risk_category"]
    .value_counts()
    .sort_index()
)

print(
    "\nSLOWWATCH warnings:",
    int(output["slowwatch_warning"].sum())
)

print(
    "Mean TRS:",
    round(
        output["temporal_risk_score"].mean(),
        4
    )
)

print(
    "Mean uncertainty:",
    round(
        output["uncertainty"].mean(),
        4
    )
)

print(
    "Mean confidence:",
    round(
        output["confidence"].mean(),
        4
    )
)


# ============================================================
# FIGURE 1 — TEMPORAL RISK
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    output["date"],
    output["temporal_risk_score"],
    marker="o",
    linewidth=2,
    label="Temporal Risk Score"
)

plt.axhline(
    0.50,
    linestyle="--",
    label="Warning threshold"
)

plt.fill_between(
    output["date"],
    np.clip(
        output["temporal_risk_score"]
        - output["uncertainty"],
        0,
        1
    ),
    np.clip(
        output["temporal_risk_score"]
        + output["uncertainty"],
        0,
        1
    ),
    alpha=0.20,
    label="Uncertainty band"
)

plt.xlabel("Date")
plt.ylabel("Risk score")
plt.title(
    "SLOWWATCH Temporal Risk Score and Uncertainty"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "slowwatch_temporal_risk_uncertainty.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# FIGURE 2 — EVIDENCE COMPONENTS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    output["date"],
    output["contribution_temporal_model"],
    label="Temporal model"
)

plt.plot(
    output["date"],
    output["contribution_slope"],
    label="Trajectory slope"
)

plt.plot(
    output["date"],
    output["contribution_anomaly"],
    label="NDVI anomaly"
)

plt.plot(
    output["date"],
    output["contribution_environment"],
    label="Environmental context"
)

plt.xlabel("Date")
plt.ylabel("Risk contribution")

plt.title(
    "SLOWWATCH Explainable Evidence Contributions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "slowwatch_evidence_contributions.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\nFiles created:")

print(
    output_path
)

print(
    os.path.join(
        FIGURES_DIR,
        "slowwatch_temporal_risk_uncertainty.png"
    )
)

print(
    os.path.join(
        FIGURES_DIR,
        "slowwatch_evidence_contributions.png"
    )
)

print("\nSLOWWATCH CORE COMPLETED.")