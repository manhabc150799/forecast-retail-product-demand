"""Global LSTM for multi-series demand forecasting.

Trains a single LSTM model on **all** selected series simultaneously
(Global approach) instead of fitting one model per series.  With only
~400 days per series, a per-series model would overfit immediately.
Pooling ~100 series yields ~35 k training windows, sufficient to learn
shared weekly-seasonal patterns.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LOOKBACK: int = 28
HORIZON: int = 14
HIDDEN: int = 64
LAYERS: int = 1
DROPOUT: float = 0.2
EPOCHS: int = 50
BATCH: int = 64
LR: float = 0.001
PATIENCE: int = 5

PHASE1_FEATURE_COLS: list[str] = [
    "sales",
    "day_of_week", "month", "is_holiday", "is_weekend",
]
PHASE2_FEATURE_COLS: list[str] = PHASE1_FEATURE_COLS + [
    "rolling_7", "rolling_28",
    "lag_7", "lag_14", "lag_28",
    "sell_price", "snap",
]
_SALES_IDX: int = 0


def get_feature_cols(phase: int) -> list[str]:
    """Return the list of input feature column names for a given phase.

    Args:
        phase: Experiment phase (1 or 2).

    Returns:
        List of column name strings (``sales`` is always first).

    Raises:
        ValueError: If *phase* is not 1 or 2.
    """
    if phase == 1:
        return PHASE1_FEATURE_COLS
    if phase == 2:
        return PHASE2_FEATURE_COLS
    raise ValueError(f"phase must be 1 or 2, got {phase}")


@dataclass
class LstmResult:
    """Result for a single series within a fold.

    Attributes:
        y_pred: Predicted values, shape ``(horizon,)``.
        converged: Whether training completed without error.
        error_msg: Empty string on success, descriptive message on failure.
    """

    y_pred: np.ndarray
    converged: bool
    error_msg: str


@dataclass
class LstmFoldResult:
    """Aggregated result for an entire fold (all series).

    Attributes:
        predictions: Mapping ``(item_id, store_id) -> y_pred``.
        skipped: Mapping ``(item_id, store_id) -> reason`` for skipped series.
        converged: ``True`` if global training completed.
        train_time_s: Wall-clock training time in seconds.
    """

    predictions: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    skipped: dict[tuple[str, str], str] = field(default_factory=dict)
    converged: bool = True
    train_time_s: float = 0.0


class GlobalLSTM(nn.Module):
    """Single-layer LSTM that predicts *horizon* steps directly.

    Architecture: LSTM(n_features -> 64) -> Dropout(0.2) -> Linear(64 -> 14).
    Uses direct multi-step output instead of autoregressive decoding
    to avoid error accumulation over the 14-day horizon.

    Args:
        n_features: Number of input features per time step.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN, HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode lookback window, decode to horizon.

        Args:
            x: Input tensor of shape ``(batch, lookback, n_features)``.

        Returns:
            Predictions of shape ``(batch, horizon)``.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))


def _build_windows(
    series_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window (X, y) pairs from a single series.

    Uses stride-tricks for zero-copy windowing.  Stride = 1 to maximise
    the number of training samples from limited data.

    Args:
        series_data: 2-D array of shape ``(n_steps, n_features)``.

    Returns:
        Tuple ``(X, y)`` where X has shape ``(n_windows, lookback, n_features)``
        and y has shape ``(n_windows, horizon)``.
    """
    n_steps = series_data.shape[0]
    n_feat = series_data.shape[1]
    total = LOOKBACK + HORIZON
    if n_steps < total:
        return np.empty((0, LOOKBACK, n_feat)), np.empty((0, HORIZON))

    n_windows = n_steps - total + 1

    from numpy.lib.stride_tricks import as_strided
    item_size = series_data.strides
    all_windows = as_strided(
        series_data,
        shape=(n_windows, total, n_feat),
        strides=(item_size[0], item_size[0], item_size[1]),
    )

    X = np.array(all_windows[:, :LOOKBACK, :], dtype=np.float32)
    y = np.array(all_windows[:, LOOKBACK:, _SALES_IDX], dtype=np.float32)

    return X, y


def prepare_fold_data(
    sc_indexed: pd.DataFrame,
    series_list: list[tuple[str, str]],
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    phase: int = 1,
) -> dict[str, Any]:
    """Prepare training windows and test inputs for one fold.

    Fits a MinMaxScaler on the training portion only (anti-leakage),
    then builds sliding windows from all series and stores the last
    ``LOOKBACK`` rows of each series as test input.

    Args:
        sc_indexed: Sales DataFrame indexed by ``(item_id, store_id, date)``.
        series_list: List of ``(item_id, store_id)`` tuples.
        train_end: Last date of the training period.
        test_start: First date of the test period.
        test_end: Last date of the test period.
        phase: Experiment phase (determines feature set).

    Returns:
        Dictionary with keys ``X``, ``y``, ``scaler``, ``test_inputs``,
        ``skipped``.  ``X``/``y`` are ``None`` when no valid windows exist.
    """
    train_feats_list: list[np.ndarray] = []
    test_inputs: dict[tuple[str, str], np.ndarray] = {}
    skipped: dict[tuple[str, str], str] = {}

    for item_id, store_id in series_list:
        try:
            s = sc_indexed.loc[(item_id, store_id)].sort_index()
        except KeyError:
            skipped[(item_id, store_id)] = "series_not_found"
            continue

        train_s = s.loc[:train_end]

        if len(train_s) < LOOKBACK + HORIZON:
            skipped[(item_id, store_id)] = "insufficient_training_data"
            continue

        feature_cols = get_feature_cols(phase)
        feats = train_s[feature_cols].to_numpy(dtype=np.float32)
        mask = ~np.isnan(feats).any(axis=1)
        feats = feats[mask]

        if len(feats) < LOOKBACK + HORIZON:
            skipped[(item_id, store_id)] = "insufficient_training_data"
            continue

        train_feats_list.append(feats)

        test_inputs[(item_id, store_id)] = feats[-LOOKBACK:]

    if not train_feats_list:
        return {"X": None, "y": None, "scaler": None,
                "test_inputs": {}, "skipped": skipped}

    all_train = np.concatenate(train_feats_list, axis=0)
    scaler = MinMaxScaler()
    scaler.fit(all_train)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for feats in train_feats_list:
        scaled = scaler.transform(feats)
        X_w, y_w = _build_windows(scaled)
        if X_w.shape[0] > 0:
            X_list.append(X_w)
            y_list.append(y_w)

    if not X_list:
        return {"X": None, "y": None, "scaler": scaler,
                "test_inputs": test_inputs, "skipped": skipped}

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return {
        "X": X_all,
        "y": y_all,
        "scaler": scaler,
        "test_inputs": test_inputs,
        "skipped": skipped,
    }


def train_global_lstm(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device | None = None,
) -> tuple[GlobalLSTM, bool]:
    """Train a Global LSTM on pooled training windows.

    Uses 90/10 train/validation split with early stopping
    (patience = 5 epochs).  Adam optimizer with MSE loss.

    Args:
        X: Input windows, shape ``(n_samples, lookback, n_features)``.
        y: Target windows, shape ``(n_samples, horizon)``.
        device: PyTorch device (auto-detected if ``None``).

    Returns:
        Tuple ``(model, converged)`` where *converged* is ``True``
        when training completed (even if early-stopped).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = X.shape[2]
    model = GlobalLSTM(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n = X.shape[0]
    indices = np.arange(n)

    val_size = max(1, int(n * 0.1))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]

    X_train_t = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict[str, Any] = {}

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(X_train_t.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, X_train_t.shape[0], BATCH):
            idx = perm[start : start + BATCH]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    return model, True


def predict_fold(
    model: GlobalLSTM,
    scaler: MinMaxScaler,
    test_inputs: dict[tuple[str, str], np.ndarray],
    device: torch.device | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate predictions for every series in a fold.

    Takes the last ``LOOKBACK`` training rows, scales them, runs the
    model, inverse-transforms, and clamps to >= 0.

    Args:
        model: Trained :class:`GlobalLSTM`.
        scaler: Fitted MinMaxScaler (from training data only).
        test_inputs: Mapping ``(item_id, store_id) -> raw_input``.
        device: PyTorch device (auto-detected if ``None``).

    Returns:
        Mapping ``(item_id, store_id) -> y_pred`` (float64, >= 0).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    results: dict[tuple[str, str], np.ndarray] = {}

    for key, raw_input in test_inputs.items():
        scaled = scaler.transform(raw_input)
        x_t = torch.tensor(
            scaled[np.newaxis, :, :], dtype=torch.float32, device=device,
        )

        with torch.no_grad():
            y_scaled = model(x_t).cpu().numpy().flatten()

        n_features = scaler.n_features_in_
        dummy = np.zeros((HORIZON, n_features), dtype=np.float32)
        dummy[:, _SALES_IDX] = y_scaled
        y_inv = scaler.inverse_transform(dummy)[:, _SALES_IDX]

        results[key] = np.maximum(y_inv, 0.0).astype(np.float64)

    return results


def run_lstm_fold(
    sc_indexed: pd.DataFrame,
    series_list: list[tuple[str, str]],
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    phase: int = 1,
) -> LstmFoldResult:
    """Run the full LSTM pipeline for a single fold.

    Prepares data, trains the Global LSTM, and generates predictions
    for every series that has sufficient data.

    Args:
        sc_indexed: Sales DataFrame indexed by ``(item_id, store_id, date)``.
        series_list: List of ``(item_id, store_id)`` tuples.
        train_end: Last date of the training period.
        test_start: First date of the test period.
        test_end: Last date of the test period.
        phase: Experiment phase (determines feature set).

    Returns:
        :class:`LstmFoldResult` with predictions and skip info.
    """
    fold_data = prepare_fold_data(
        sc_indexed, series_list, train_end, test_start, test_end,
        phase=phase,
    )

    result = LstmFoldResult()
    result.skipped = fold_data["skipped"]

    if fold_data["X"] is None:
        result.converged = False
        return result

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, converged = train_global_lstm(fold_data["X"], fold_data["y"], device)
    result.converged = converged

    preds = predict_fold(model, fold_data["scaler"], fold_data["test_inputs"], device)
    result.predictions = preds

    return result


if __name__ == "__main__":
    from src.data.mock_factory import generate_mock_m5_data

    data = generate_mock_m5_data(n_items=5, n_stores=1)
    sc = data["sales_clean"]
    cfg = data["split_config"]
    fold_1 = cfg["folds"][0]

    series_list = (
        sc[["item_id", "store_id"]]
        .drop_duplicates()
        .apply(lambda r: (r["item_id"], r["store_id"]), axis=1)
        .tolist()
    )

    sc_indexed = sc.set_index(["item_id", "store_id", "date"]).sort_index()

    print(f"Series: {len(series_list)}")
    print(f"Fold 1: train -> {fold_1['train_end']}, test {fold_1['test_start']} -> {fold_1['test_end']}")

    result = run_lstm_fold(
        sc_indexed,
        series_list,
        pd.Timestamp(fold_1["train_end"]),
        pd.Timestamp(fold_1["test_start"]),
        pd.Timestamp(fold_1["test_end"]),
        phase=1,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Skipped: {len(result.skipped)}")
    for key, msg in result.skipped.items():
        print(f"  {key}: {msg}")

    print(f"Predictions: {len(result.predictions)} series")
    for key, y_pred in result.predictions.items():
        assert y_pred.shape == (HORIZON,), f"Wrong shape: {y_pred.shape}"
        assert np.all(y_pred >= 0), f"Negative predictions found for {key}"
        print(f"  {key}: shape={y_pred.shape}, min={y_pred.min():.2f}, max={y_pred.max():.2f}")

    print("\n[PASS] All y_pred shapes correct and >= 0")
