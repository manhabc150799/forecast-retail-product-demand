"""Prophet model wrapper for M5 Forecasting.

Uses Facebook Prophet with ``add_regressor`` for exogenous features.
Falls back to Seasonal Naive when Prophet fails or times out,
mirroring the SARIMAX fallback pattern.
"""

from __future__ import annotations

import logging
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

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

TIMEOUT_SECONDS: int = 120
HORIZON: int = 14
SEASON: int = 7

PHASE1_REGRESSOR_COLS: list[str] = [
    "day_of_week", "month", "is_holiday", "is_weekend",
]
PHASE2_REGRESSOR_COLS: list[str] = [
    "day_of_week", "month", "is_holiday", "is_weekend",
    "rolling_7", "rolling_28",
    "lag_7", "lag_14", "lag_28",
    "sell_price", "snap",
]


@dataclass
class ProphetResult:
    """Container for a single Prophet forecast result.

    Attributes:
        y_pred: Predicted values, shape ``(horizon,)``.
        fallback: ``True`` when SNaive was used instead of Prophet.
        converged: ``True`` when Prophet fitted successfully.
        error_msg: Empty string on success, descriptive message on failure.
    """

    y_pred: np.ndarray
    fallback: bool
    converged: bool
    error_msg: str


def get_regressor_cols(phase: int) -> list[str]:
    """Return the list of regressor column names for a given phase.

    Args:
        phase: Experiment phase (1 or 2).

    Returns:
        List of column name strings.

    Raises:
        ValueError: If *phase* is not 1 or 2.
    """
    if phase == 1:
        return PHASE1_REGRESSOR_COLS
    if phase == 2:
        return PHASE2_REGRESSOR_COLS
    raise ValueError(f"phase must be 1 or 2, got {phase}")


def _prepare_prophet_df(
    df: pd.DataFrame,
    regressor_cols: list[str],
) -> pd.DataFrame:
    """Convert a DataFrame to Prophet's required format (ds, y, regressors).

    Args:
        df: Source DataFrame with ``date``, ``sales``, and regressor columns.
        regressor_cols: List of regressor column names to include.

    Returns:
        New DataFrame with ``ds`` (datetime), ``y`` (target), and regressors.
    """
    pdf = pd.DataFrame()
    pdf["ds"] = pd.to_datetime(df["date"])
    pdf["y"] = df["sales"].to_numpy(dtype=np.float64)

    for col in regressor_cols:
        if col in df.columns:
            pdf[col] = df[col].to_numpy(dtype=np.float64)
        else:
            pdf[col] = 0.0

    return pdf


def _prepare_future_df(
    df: pd.DataFrame,
    regressor_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Convert a future DataFrame to Prophet's format (ds + regressors).

    Args:
        df: Future DataFrame with ``date`` and regressor columns.
        regressor_cols: List of regressor column names to include.
        horizon: Number of rows to keep.

    Returns:
        New DataFrame with ``ds`` and regressor columns.
    """
    future_slice = df.head(horizon)
    fdf = pd.DataFrame()
    fdf["ds"] = pd.to_datetime(future_slice["date"])

    for col in regressor_cols:
        if col in future_slice.columns:
            fdf[col] = future_slice[col].to_numpy(dtype=np.float64)
        else:
            fdf[col] = 0.0

    return fdf


def _fit_and_predict(
    train_pdf: pd.DataFrame,
    future_pdf: pd.DataFrame,
    regressor_cols: list[str],
    horizon: int,
) -> np.ndarray:
    """Fit Prophet and return point forecasts (runs inside a worker thread).

    Args:
        train_pdf: Training data in Prophet format.
        future_pdf: Future data in Prophet format.
        regressor_cols: Regressor column names added to the model.
        horizon: Forecast horizon (for output slicing).

    Returns:
        Predicted values as a 1-D float64 array.
    """
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    )

    for col in regressor_cols:
        model.add_regressor(col)

    model.fit(train_pdf)

    forecast = model.predict(future_pdf)

    y_pred = forecast["yhat"].to_numpy(dtype=np.float64)[:horizon]

    return y_pred


def forecast_series(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    phase: int = 1,
    horizon: int = HORIZON,
    season: int = SEASON,
    timeout: int = TIMEOUT_SECONDS,
) -> ProphetResult:
    """High-level convenience wrapper: DataFrame in, ProphetResult out.

    Handles NaN in regressors via forward-fill, adds regressors for the
    requested phase, and falls back to SNaive on any failure.

    Args:
        train_df: Training DataFrame with ``date``, ``sales``, and regressor columns.
        future_df: Future DataFrame (test dates) with regressor columns.
        phase: Experiment phase (1 = basic regressors, 2 = enhanced features).
        horizon: Forecast horizon in days.
        season: Seasonal period for SNaive fallback.
        timeout: Maximum seconds for Prophet fitting.

    Returns:
        A :class:`ProphetResult` with predictions and metadata.
    """
    regressor_cols = get_regressor_cols(phase)

    train_clean = train_df.copy()
    train_clean[regressor_cols] = train_clean[regressor_cols].ffill().bfill()

    train_pdf = _prepare_prophet_df(train_clean, regressor_cols)

    future_clean = future_df.head(horizon).copy()
    future_clean[regressor_cols] = future_clean[regressor_cols].ffill().bfill()
    future_pdf = _prepare_future_df(future_clean, regressor_cols, horizon)

    train_pdf = train_pdf.dropna().reset_index(drop=True)

    if len(train_pdf) < 2 * season:
        y_train = train_df["sales"].to_numpy(dtype=np.float64)
        y_pred = predict_snaive(y_train, horizon=horizon, season=season)
        return ProphetResult(
            y_pred=y_pred,
            fallback=True,
            converged=False,
            error_msg="insufficient_training_data",
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _fit_and_predict,
                train_pdf,
                future_pdf,
                regressor_cols,
                horizon,
            )
            y_pred = future.result(timeout=timeout)

        y_pred = np.maximum(y_pred, 0.0).astype(np.float64)[:horizon]

        return ProphetResult(
            y_pred=y_pred,
            fallback=False,
            converged=True,
            error_msg="",
        )

    except FuturesTimeout:
        error_msg = f"timeout ({timeout}s)"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    y_train = train_df["sales"].to_numpy(dtype=np.float64)
    y_pred = predict_snaive(y_train, horizon=horizon, season=season)

    return ProphetResult(
        y_pred=y_pred,
        fallback=True,
        converged=False,
        error_msg=error_msg,
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

    print("\n=== Phase 1 Prophet ===")
    result = forecast_series(train_df, future_df, phase=1, timeout=120)
    print(f"  fallback : {result.fallback}")
    print(f"  converged: {result.converged}")
    print(f"  y_pred   : {result.y_pred}")
    print(f"  shape    : {result.y_pred.shape}")
