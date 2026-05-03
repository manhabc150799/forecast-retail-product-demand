# baseline don gian nhat de lam moc so sanh
# neu LSTM thua ca naive thi do cung la 1 finding co gia tri
# dung np.repeat va np.tile thay for-loop vi nhanh hon gap nhieu lan

from __future__ import annotations

import numpy as np


def predict_naive(y_train: np.ndarray, horizon: int = 14) -> np.ndarray:
    # lay gia tri cuoi cung lap lai horizon lan
    # logic: gia dinh tuong lai giong hiet hom nay
    if y_train.size == 0:
        raise ValueError("y_train must not be empty.")

    last_value: float = float(y_train[-1])
    y_pred: np.ndarray = np.repeat(last_value, horizon)

    # sales khong the am
    return np.maximum(y_pred, 0.0)


def predict_snaive(
    y_train: np.ndarray,
    horizon: int = 14,
    season: int = 7,
) -> np.ndarray:
    # lap lai pattern 7 ngay cuoi theo chu ky
    # vi du horizon=14 thi lap dung 2 tuan, phu hop weekly seasonality cua M5
    if y_train.size < season:
        raise ValueError(
            f"y_train length ({y_train.size}) must be >= season ({season})."
        )

    # tile roi cat, nhanh hon viet loop
    last_cycle: np.ndarray = y_train[-season:]
    n_repeats: int = (horizon // season) + 1
    y_pred: np.ndarray = np.tile(last_cycle, n_repeats)[:horizon]

    return np.maximum(y_pred, 0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy_train = rng.integers(0, 30, size=60).astype(np.int16)

    print("=== Naive ===")
    pred_n = predict_naive(dummy_train, horizon=14)
    print(f"  last value : {dummy_train[-1]}")
    print(f"  prediction : {pred_n}")
    print(f"  shape      : {pred_n.shape}")

    print("\n=== Seasonal Naive (season=7) ===")
    pred_s = predict_snaive(dummy_train, horizon=14, season=7)
    print(f"  last 7 vals: {dummy_train[-7:]}")
    print(f"  prediction : {pred_s}")
    print(f"  shape      : {pred_s.shape}")

    try:
        predict_snaive(np.array([1, 2]), horizon=14, season=7)
    except ValueError as e:
        print(f"\n[OK] Expected error: {e}")
