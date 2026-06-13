# global LSTM: train 1 model chung cho tat ca 100 series
# vi moi series chi co ~400 ngay, 1 model rieng se overfit ngay
# gop chung 100 series x ~350 windows = ~35k samples, du de hoc pattern chung

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

# features dung cho LSTM, khong bao gom item_id/store_id/date vi la categorical
# phase 1 chi dung calendar features co ban
PHASE1_FEATURE_COLS: list[str] = [
    "sales",
    "day_of_week", "month", "is_holiday", "is_weekend",
]
# phase 2 them lag/rolling/price/snap (enhanced features)
PHASE2_FEATURE_COLS: list[str] = PHASE1_FEATURE_COLS + [
    "rolling_7", "rolling_28",
    "lag_7", "lag_14", "lag_28",
    "sell_price", "snap",
]
# vi tri cot sales trong FEATURE_COLS, dung de tach y khi build windows
# sales luon la cot dau tien trong ca 2 phase
_SALES_IDX: int = 0


def get_feature_cols(phase: int) -> list[str]:
    """Tra ve danh sach feature tuong ung voi phase (1 hoac 2)."""
    if phase == 1:
        return PHASE1_FEATURE_COLS
    if phase == 2:
        return PHASE2_FEATURE_COLS
    raise ValueError(f"phase must be 1 or 2, got {phase}")


@dataclass
class LstmResult:
    y_pred: np.ndarray
    converged: bool
    error_msg: str


@dataclass
class LstmFoldResult:
    # gom ket qua cua toan bo fold lai 1 cho
    # predictions la dict[tuple[item_id, store_id], np.ndarray]
    predictions: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    skipped: dict[tuple[str, str], str] = field(default_factory=dict)
    converged: bool = True
    train_time_s: float = 0.0


class GlobalLSTM(nn.Module):
    # hidden=64, layers=1, dropout=0.2 theo yeu cau guideline
    # output 14 ngay truc tiep (multi-step direct) thay vi autoregressive
    # vi autoregressive loi tich luy error theo thoi gian

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
            batch_first=True,
        )
        # dropout rieng sau LSTM vi khi layers=1 thi LSTM khong tu dropout
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN, HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chi lay hidden state cua timestep cuoi cung
        # vi du bao la "nhin 28 ngay qua, du doan 14 ngay toi"
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))


def _build_windows(
    series_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # sliding window stride=1, moi window: X = 28 timesteps x n_features, y = 14 ngay sales
    # stride=1 cho nhieu sample nhat co the tu data it
    n_steps = series_data.shape[0]
    n_feat = series_data.shape[1]
    total = LOOKBACK + HORIZON
    if n_steps < total:
        return np.empty((0, LOOKBACK, n_feat)), np.empty((0, HORIZON))

    n_windows = n_steps - total + 1

    # dung as_strided de tao view cua tat ca windows cung luc, khong copy memory
    # shape = (n_windows, total, n_feat), strides buoc theo dong goc
    from numpy.lib.stride_tricks import as_strided
    item_size = series_data.strides
    all_windows = as_strided(
        series_data,
        shape=(n_windows, total, n_feat),
        strides=(item_size[0], item_size[0], item_size[1]),
    )

    X = np.array(all_windows[:, :LOOKBACK, :], dtype=np.float32)
    # y chi la cot sales (idx 0) cua 14 ngay cuoi moi window
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
    # fit scaler CHI tren train de chong data leakage
    # sau do transform ca train lan test bang cung 1 scaler
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
        # dropna vi lag/rolling cot dau tien cua series luon NaN
        mask = ~np.isnan(feats).any(axis=1)
        feats = feats[mask]

        if len(feats) < LOOKBACK + HORIZON:
            skipped[(item_id, store_id)] = "insufficient_training_data"
            continue

        train_feats_list.append(feats)

        # test input = 28 ngay cuoi cua train, dung de predict 14 ngay toi
        test_inputs[(item_id, store_id)] = feats[-LOOKBACK:]

    if not train_feats_list:
        return {"X": None, "y": None, "scaler": None,
                "test_inputs": {}, "skipped": skipped}

    # fit scaler tren toan bo train data cua moi series trong fold
    all_train = np.concatenate(train_feats_list, axis=0)
    scaler = MinMaxScaler()
    scaler.fit(all_train)

    # build windows tu tung series roi gop lai
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

    # y cua target la sales goc (chua scale) de loss co y nghia vat ly
    # nhung X da scale nen y cung phai scale tuong ung
    # -> inverse transform y sau khi predict

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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = X.shape[2]
    model = GlobalLSTM(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # shuffle truoc khi train vi cac windows tu cung 1 series nam ke nhau
    # neu khong shuffle, model hoc theo thu tu thay vi hoc pattern
    n = X.shape[0]
    indices = np.arange(n)

    # tach 10% cuoi lam val de early stopping, khong shuffle phan val
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
        # shuffle lai train moi epoch de tranh hoc thu tu
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

    # load lai best weights, ke ca khi early stop van la converged
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

        # inverse transform chi cot sales (idx 0)
        # tao dummy array voi shape (horizon, n_features) de dung scaler
        n_features = scaler.n_features_in_
        dummy = np.zeros((HORIZON, n_features), dtype=np.float32)
        dummy[:, _SALES_IDX] = y_scaled
        y_inv = scaler.inverse_transform(dummy)[:, _SALES_IDX]

        # clamp >= 0 vi sales khong the am
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
