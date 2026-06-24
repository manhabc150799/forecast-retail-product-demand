"""SARIMAX wrapper with auto_arima, timeout, and SNaive fallback.

Uses ``pmdarima.auto_arima`` with per-series timeout (default 60 s).
When SARIMAX fails to converge or times out, falls back to Seasonal Naive
and marks ``fallback=True`` transparently for downstream reporting.
"""

from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.naive import predict_snaive  # noqa: E402

TIMEOUT_SECONDS: int = 60
HORIZON: int = 14
SEASON: int = 7

_AUTO_ARIMA_KWARGS: dict[str, Any] = {
    "max_p": 3,
    "max_q": 3,
    "max_P": 1,
    "max_Q": 1,
    "seasonal": True,
    "m": SEASON,
    "stepwise": True,
    "suppress_warnings": True,
    "error_action": "ignore",
    "trace": False,
}

PHASE1_EXOG_COLS: list[str] = [
    "day_of_week", "month", "is_holiday", "is_weekend",
]
PHASE2_EXOG_COLS: list[str] = [
    "day_of_week", "month", "is_holiday", "is_weekend",
    "rolling_7", "rolling_28",
    "lag_7", "lag_14", "lag_28",
    "sell_price", "snap",
]


@dataclass
class SarimaxResult:
    """Container for a single SARIMAX forecast result.

    Attributes:
        y_pred: Predicted values, shape ``(horizon,)``.
        fallback: ``True`` when SNaive was used instead of SARIMAX.
        converged: ``True`` when SARIMAX fitted successfully.
        error_msg: Empty string on success, descriptive message on failure.
    """

    y_pred: np.ndarray
    fallback: bool
    converged: bool
    error_msg: str


def _fit_and_predict(
    y_train: np.ndarray,
    exog_train: np.ndarray | None,
    exog_future: np.ndarray | None,
    horizon: int,
) -> np.ndarray:
    """Fit auto_arima and return point forecasts (runs inside a worker thread).

    Args:
        y_train: Training target values.
        exog_train: Exogenous matrix for training (or ``None``).
        exog_future: Exogenous matrix for the forecast horizon (or ``None``).
        horizon: Number of steps to forecast.

    Returns:
        Predicted values as a 1-D float64 array.
    """
    import pmdarima as pm

    model = pm.auto_arima(
        y_train,
        exogenous=exog_train,
        **_AUTO_ARIMA_KWARGS,
    )

    y_pred: np.ndarray = model.predict(
        n_periods=horizon,
        exogenous=exog_future,
    )
    return np.asarray(y_pred, dtype=np.float64)


def get_exog_columns(phase: int) -> list[str]:
    """Return the list of exogenous column names for a given phase.

    Args:
        phase: Experiment phase (1 or 2).

    Returns:
        List of column name strings.

    Raises:
        ValueError: If *phase* is not 1 or 2.
    """
    if phase == 1:
        return PHASE1_EXOG_COLS
    if phase == 2:
        return PHASE2_EXOG_COLS
    raise ValueError(f"phase must be 1 or 2, got {phase}")


def prepare_exog(
    df: pd.DataFrame,
    phase: int,
) -> np.ndarray:
    """Extract exogenous columns from a DataFrame as a float32 array.

    Args:
        df: Source DataFrame containing all required columns.
        phase: Experiment phase (determines which columns to select).

    Returns:
        2-D float32 numpy array of shape ``(len(df), n_exog)``.
    """
    cols = get_exog_columns(phase)
    return df[cols].to_numpy(dtype=np.float32)


def run_sarimax_robust(
    y_train: np.ndarray,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
    horizon: int = HORIZON,
    season: int = SEASON,
    timeout: int = TIMEOUT_SECONDS,
) -> SarimaxResult:
    """Fit SARIMAX with timeout and automatic SNaive fallback.

    Catches all exceptions (timeout, LinAlgError, etc.) and substitutes
    a Seasonal Naive forecast, recording the failure transparently.

    Args:
        y_train: Training target values.
        exog_train: Exogenous training matrix (or ``None``).
        exog_future: Exogenous future matrix, shape ``(horizon, n_exog)``
                     (or ``None``).
        horizon: Forecast horizon in days.
        season: Seasonal period for SNaive fallback.
        timeout: Maximum seconds to wait for auto_arima.

    Returns:
        A :class:`SarimaxResult` with predictions and metadata.

    Raises:
        ValueError: If ``exog_future`` row count does not match *horizon*,
                    or if exactly one of the exog arguments is ``None``.
    """
    if exog_future is not None and exog_future.shape[0] != horizon:
        raise ValueError(
            f"exog_future rows ({exog_future.shape[0]}) != horizon ({horizon})"
        )
    if (exog_train is None) != (exog_future is None):
        raise ValueError(
            "exog_train and exog_future must both be None or both provided."
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _fit_and_predict,
                y_train,
                exog_train,
                exog_future,
                horizon,
            )
            y_pred = future.result(timeout=timeout)

        y_pred = np.asarray(y_pred, dtype=np.float64)[:horizon]
        y_pred = np.maximum(y_pred, 0.0)

        return SarimaxResult(
            y_pred=y_pred,
            fallback=False,
            converged=True,
            error_msg="",
        )

    except FuturesTimeout:
        error_msg = f"timeout ({timeout}s)"
    except np.linalg.LinAlgError as exc:
        error_msg = f"LinAlgError: {exc}"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    y_pred = predict_snaive(y_train, horizon=horizon, season=season)

    return SarimaxResult(
        y_pred=y_pred,
        fallback=True,
        converged=False,
        error_msg=error_msg,
    )


def forecast_series(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    phase: int = 1,
    horizon: int = HORIZON,
    season: int = SEASON,
    timeout: int = TIMEOUT_SECONDS,
) -> SarimaxResult:
    """High-level convenience wrapper: DataFrame in, SarimaxResult out.

    Automatically selects exogenous columns for the requested phase,
    handles NaN values in exogenous features via forward-fill, and
    delegates to :func:`run_sarimax_robust`.

    Args:
        train_df: Training DataFrame with ``sales`` and exog columns.
        future_df: Future DataFrame (test dates) with exog columns.
        phase: Experiment phase (1 = basic exog, 2 = enhanced features).
        horizon: Forecast horizon in days.
        season: Seasonal period for SNaive fallback.
        timeout: Maximum seconds for auto_arima.

    Returns:
        A :class:`SarimaxResult` with predictions and metadata.
    """
    exog_cols = get_exog_columns(phase)

    train_clean = train_df.copy()
    train_clean[exog_cols] = train_clean[exog_cols].ffill().bfill()

    y_train = train_clean["sales"].to_numpy(dtype=np.float64)
    exog_train = train_clean[exog_cols].to_numpy(dtype=np.float32)

    future_slice = future_df.head(horizon).copy()
    future_slice[exog_cols] = future_slice[exog_cols].ffill().bfill()
    exog_future = future_slice[exog_cols].to_numpy(dtype=np.float32)

    return run_sarimax_robust(
        y_train=y_train,
        exog_train=exog_train,
        exog_future=exog_future,
        horizon=horizon,
        season=season,
        timeout=timeout,
    )


if __name__ == "__main__":
    from src.data.mock_factory import generate_mock_m5_data

    data = generate_mock_m5_data()
    sc = data["sales_clean"]
    cfg = data["split_config"]
    fold_1 = cfg["folds"][0]

    series = sc[
        (sc["item_id"] == "ITEM_001") & (sc["store_id"] == "STORE_01")
    ].sort_values("date").reset_index(drop=True)

    train_mask = series["date"] <= pd.Timestamp(fold_1["train_end"])
    test_mask = (
        (series["date"] >= pd.Timestamp(fold_1["test_start"]))
        & (series["date"] <= pd.Timestamp(fold_1["test_end"]))
    )

    train_df = series[train_mask]
    future_df = series[test_mask]

    print(f"Train rows: {len(train_df)}, Future rows: {len(future_df)}")
    print(f"Phase 1 exog cols: {get_exog_columns(1)}")
    print(f"Phase 2 exog cols: {get_exog_columns(2)}")

    print("\n=== Phase 1 SARIMAX ===")
    result = forecast_series(train_df, future_df, phase=1, timeout=60)
    print(f"  fallback : {result.fallback}")
    print(f"  converged: {result.converged}")
    print(f"  error    : {result.error_msg!r}")
    print(f"  y_pred   : {result.y_pred}")
    print(f"  shape    : {result.y_pred.shape}")
