import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# SLOWWATCH - LSTM TEMPORAL BASELINE
# ============================================================

INPUT = r"data/splits/slowwatch_ml_targets.csv"
METRICS_DIR = r"results/metrics"
FIGURES_DIR = r"results/figures"
MODEL_DIR = r"models"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("SLOWWATCH - LSTM TEMPORAL BASELINE")
print("=" * 70)
print(f"Device: {DEVICE}")

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

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

TARGET = "target_event_3m"

SEQUENCE_LENGTH = 6


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"\nDataset: {df.shape}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")


# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ------------------------------------------------------------

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nChronological split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ------------------------------------------------------------
# TRAIN-ONLY SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

train_values = train_df[FEATURES].copy()
scaler.fit(train_values)

all_values = pd.DataFrame(
    scaler.transform(df[FEATURES]),
    columns=FEATURES
)

# Fill any remaining numerical gaps
all_values = all_values.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)


# ------------------------------------------------------------
# SEQUENCE CONSTRUCTION
# ------------------------------------------------------------

dates = df["date"].reset_index(drop=True)
targets = df[TARGET].astype(int).reset_index(drop=True)
splits = df["split"].reset_index(drop=True)

X_sequences = []
y_sequences = []
sequence_dates = []
sequence_splits = []

for i in range(SEQUENCE_LENGTH - 1, len(df)):

    start = i - SEQUENCE_LENGTH + 1
    end = i + 1

    # Sequence uses only current and previous observations
    X_seq = all_values.iloc[start:end].values

    y = targets.iloc[i]

    X_sequences.append(X_seq)
    y_sequences.append(y)
    sequence_dates.append(dates.iloc[i])
    sequence_splits.append(splits.iloc[i])

X_sequences = np.asarray(X_sequences, dtype=np.float32)
y_sequences = np.asarray(y_sequences, dtype=np.float32)

print("\nSequence dataset:")
print("Shape:", X_sequences.shape)
print("Sequence length:", SEQUENCE_LENGTH)
print("Features:", len(FEATURES))


# ------------------------------------------------------------
# SPLIT SEQUENCES
# ------------------------------------------------------------

train_mask = np.array(sequence_splits) == "train"
val_mask = np.array(sequence_splits) == "validation"
test_mask = np.array(sequence_splits) == "test"

X_train = X_sequences[train_mask]
y_train = y_sequences[train_mask]

X_val = X_sequences[val_mask]
y_val = y_sequences[val_mask]

X_test = X_sequences[test_mask]
y_test = y_sequences[test_mask]

print("\nSequence split:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ------------------------------------------------------------
# TORCH DATA
# ------------------------------------------------------------

X_train_t = torch.tensor(X_train).to(DEVICE)
y_train_t = torch.tensor(y_train).float().to(DEVICE)

X_val_t = torch.tensor(X_val).to(DEVICE)
y_val_t = torch.tensor(y_val).float().to(DEVICE)

X_test_t = torch.tensor(X_test).to(DEVICE)
y_test_t = torch.tensor(y_test).float().to(DEVICE)


# ------------------------------------------------------------
# LSTM MODEL
# ------------------------------------------------------------

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

        # Last timestep
        last = output[:, -1, :]

        return self.fc(last).squeeze(1)


model = SlowWatchLSTM(
    input_size=len(FEATURES),
    hidden_size=32
).to(DEVICE)


# ------------------------------------------------------------
# CLASS WEIGHT
# ------------------------------------------------------------

positive = max(float(y_train.sum()), 1.0)
negative = max(float(len(y_train) - y_train.sum()), 1.0)

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

EPOCHS = 150
PATIENCE = 20

best_val_loss = float("inf")
best_state = None
patience_count = 0

history = []

print("\nTraining LSTM...")

for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train_t)

    loss = criterion(
        logits,
        y_train_t
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():

        val_logits = model(X_val_t)

        val_loss = criterion(
            val_logits,
            y_val_t
        ).item()

    history.append([
        epoch,
        float(loss.item()),
        float(val_loss)
    ])

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        patience_count = 0

    else:
        patience_count += 1

    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {loss.item():.4f} | "
            f"Val Loss {val_loss:.4f}"
        )

    if patience_count >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break


# ------------------------------------------------------------
# RESTORE BEST MODEL
# ------------------------------------------------------------

if best_state is not None:
    model.load_state_dict(best_state)

model_path = os.path.join(
    MODEL_DIR,
    "slowwatch_lstm.pt"
)

torch.save(
    model.state_dict(),
    model_path
)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def evaluate(name, X, y):

    model.eval()

    with torch.no_grad():

        logits = model(
            torch.tensor(X).to(DEVICE)
        )

        probability = torch.sigmoid(
            logits
        ).cpu().numpy()

    prediction = (
        probability >= 0.50
    ).astype(int)

    metrics = {
        "model": "LSTM",
        "split": name,
        "n_samples": len(y),
        "positive_samples": int(y.sum()),
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(
            y, prediction, zero_division=0
        ),
        "recall": recall_score(
            y, prediction, zero_division=0
        ),
        "f1": f1_score(
            y, prediction, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y, probability)
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "pr_auc": (
            average_precision_score(y, probability)
            if len(np.unique(y)) == 2
            else np.nan
        )
    }

    print(f"\n--- {name.upper()} ---")

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, prediction))

    return metrics, probability, prediction


train_metrics, _, _ = evaluate(
    "train",
    X_train,
    y_train
)

val_metrics, _, _ = evaluate(
    "validation",
    X_val,
    y_val
)

test_metrics, test_prob, test_pred = evaluate(
    "test",
    X_test,
    y_test
)


# ------------------------------------------------------------
# SAVE METRICS
# ------------------------------------------------------------

metrics_df = pd.DataFrame([
    train_metrics,
    val_metrics,
    test_metrics
])

metrics_path = os.path.join(
    METRICS_DIR,
    "lstm_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ------------------------------------------------------------
# SAVE TEST PREDICTIONS
# ------------------------------------------------------------

test_dates = np.array(
    sequence_dates,
    dtype="datetime64[ns]"
)[test_mask]

prediction_df = pd.DataFrame({
    "date": test_dates,
    "target_event_3m": y_test.astype(int),
    "predicted_probability": test_prob,
    "predicted_class": test_pred
})

prediction_path = os.path.join(
    METRICS_DIR,
    "lstm_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ------------------------------------------------------------
# TRAINING CURVE
# ------------------------------------------------------------

history_df = pd.DataFrame(
    history,
    columns=[
        "epoch",
        "train_loss",
        "validation_loss"
    ]
)

history_path = os.path.join(
    METRICS_DIR,
    "lstm_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

plt.figure(figsize=(9, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training loss"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SLOWWATCH LSTM Training History")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_training_history.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# TEST PROBABILITY TIMELINE
# ------------------------------------------------------------

plt.figure(figsize=(11, 5))

plt.plot(
    prediction_df["date"],
    prediction_df["predicted_probability"],
    marker="o",
    label="Predicted probability"
)

plt.axhline(
    0.50,
    linestyle="--",
    label="Decision threshold"
)

event_points = prediction_df[
    prediction_df["target_event_3m"] == 1
]

plt.scatter(
    event_points["date"],
    event_points["predicted_probability"],
    marker="x",
    s=70,
    label="Reference event"
)

plt.xlabel("Date")
plt.ylabel("Event probability")
plt.title(
    "SLOWWATCH LSTM — Chronological Test Predictions"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_test_probability.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LSTM BASELINE COMPLETED")
print("=" * 70)

print(f"Model   : {model_path}")
print(f"Metrics : {metrics_path}")
print(f"Predictions: {prediction_path}")

print("\nTEST PERFORMANCE")
print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
print(f"PR-AUC   : {test_metrics['pr_auc']:.4f}")
print(f"F1       : {test_metrics['f1']:.4f}")
print(f"Recall   : {test_metrics['recall']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")

print("\nDone.")