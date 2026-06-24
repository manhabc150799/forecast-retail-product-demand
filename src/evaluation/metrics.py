"""Single source of truth for forecast evaluation metrics.

All models (Naive, SNaive, SARIMAX, Prophet, LSTM) use this module
so that metric definitions are consistent across the project.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 7,
) -> dict[str, float]:
    """Compute MAE, RMSE, MASE, and MAPE for a single forecast horizon.

    Args:
        y_true: Actual values, shape ``(horizon,)``.
        y_pred: Predicted values, shape ``(horizon,)``.
        y_train: Historical training data, shape ``(n_train,)``.
                 Used to compute the seasonal-naive scale for MASE.
        season: Seasonal period (7 = weekly for M5 daily data).

    Returns:
        Dictionary with lowercase keys ``mae``, ``rmse``, ``mase``, ``mape``.
        ``mase`` is ``NaN`` when the training series is near-constant.
        ``mape`` is ``NaN`` when any ``y_true`` value is below 1.0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))

    rmse = float(np.sqrt(np.mean(errors ** 2)))

    naive_errors = np.abs(y_train[season:] - y_train[:-season])
    scale = np.mean(naive_errors)

    if scale < 1e-9:
        mase = float("nan")
    else:
        mase = float(mae / scale)

    if np.any(y_true < 1.0):
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs(errors / y_true)) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "mape": mape,
    }


if __name__ == "__main__":
    y_true = np.array([3.0, 5.0, 2.0])
    y_pred = np.array([4.0, 4.0, 3.0])
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    m = compute_metrics(y_true, y_pred, y_train, season=1)
    print("=== Known-value test ===")
    print(f"  mae  = {m['mae']:.4f}  (expect 1.0)")
    print(f"  rmse = {m['rmse']:.4f}  (expect ~1.0)")
    print(f"  mase = {m['mase']:.4f}")
    print(f"  mape = {m['mape']:.2f}%")

    assert abs(m["mae"] - 1.0) < 1e-9, f"MAE failed: {m['mae']}"
    assert abs(m["rmse"] - 1.0) < 1e-9, f"RMSE failed: {m['rmse']}"
    print("  [PASS] MAE & RMSE match expected values")

    y_train_flat = np.full(20, 5.0)
    m2 = compute_metrics(y_true, y_pred, y_train_flat, season=7)
    assert np.isnan(m2["mase"]), f"MASE should be NaN for flat series, got {m2['mase']}"
    print("\n=== Flat series test ===")
    print(f"  mase = {m2['mase']}  (expect NaN)")
    print("  [PASS] MASE is NaN for constant train series")

    y_true_zero = np.array([0.0, 5.0, 2.0])
    m3 = compute_metrics(y_true_zero, y_pred, y_train, season=1)
    assert np.isnan(m3["mape"]), f"MAPE should be NaN when y_true has zeros, got {m3['mape']}"
    print("\n=== Intermittent demand test ===")
    print(f"  mape = {m3['mape']}  (expect NaN)")
    print("  [PASS] MAPE is NaN when y_true < 1.0")

    print("\nAll tests passed.")
