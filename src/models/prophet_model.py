# Prophet model cho M5 Forecasting
# dung Facebook Prophet voi add_regressor de them exogenous features
# fallback ve SNaive giong SARIMAX khi Prophet fail
# tat uncertainty_samples vi khong can confidence interval, chi can point forecast

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

# them project root vao sys.path de import src.* duoc tu bat ky dau
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.naive import predict_snaive  # noqa: E402

# tat log verbose cua Prophet va cmdstanpy, chi giu WARNING tro len
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# timeout 120s, Prophet cham hon SARIMAX nhung van can gioi han
TIMEOUT_SECONDS: int = 120
HORIZON: int = 14
SEASON: int = 7

# phase 1 chi dung calendar features, phase 2 them lag/rolling/price
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
    """Ket qua du bao cua Prophet, co co fallback giong SarimaxResult."""
    y_pred: np.ndarray
    fallback: bool
    converged: bool
    error_msg: str


def get_regressor_cols(phase: int) -> list[str]:
    """Tra ve danh sach regressor tuong ung voi phase (1 hoac 2)."""
    if phase == 1:
        return PHASE1_REGRESSOR_COLS
    if phase == 2:
        return PHASE2_REGRESSOR_COLS
    raise ValueError(f"phase must be 1 or 2, got {phase}")


def _prepare_prophet_df(
    df: pd.DataFrame,
    regressor_cols: list[str],
) -> pd.DataFrame:
    """Chuyen doi DataFrame thanh format Prophet: ds, y + regressors."""
    # Prophet bat buoc phai co cot 'ds' (datetime) va 'y' (target)
    pdf = pd.DataFrame()
    pdf["ds"] = pd.to_datetime(df["date"])
    pdf["y"] = df["sales"].to_numpy(dtype=np.float64)

    # them cac cot regressor, ep float64 de tranh loi kieu
    for col in regressor_cols:
        if col in df.columns:
            pdf[col] = df[col].to_numpy(dtype=np.float64)
        else:
            # neu cot khong ton tai, dien 0 de tranh crash
            pdf[col] = 0.0

    return pdf


def _prepare_future_df(
    df: pd.DataFrame,
    regressor_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Chuyen doi future DataFrame thanh format Prophet: ds + regressors."""
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
    """Fit Prophet va predict, chay trong thread rieng de co the timeout."""
    # import o day vi Prophet nang, chi load khi can
    from prophet import Prophet

    # cau hinh Prophet
    # - uncertainty_samples=0: tat sampling, chi can point forecast cho nhanh
    # - yearly_seasonality=True: M5 co seasonality theo nam
    # - weekly_seasonality=True: M5 co seasonality theo tuan (7 ngay)
    # - daily_seasonality=False: data la daily, khong co pattern intra-day
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    )

    # them tung regressor vao model truoc khi fit
    for col in regressor_cols:
        model.add_regressor(col)

    # fit model tren training data
    model.fit(train_pdf)

    # predict
    forecast = model.predict(future_pdf)

    # lay gia tri du bao (yhat) cho dung horizon dong
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
    """Ham tien loi: truyen DataFrame vao la xong, tu add regressors + fit.

    Pattern input/output giong sarimax_model.forecast_series de de tich hop.
    Neu Prophet fail hoac timeout, fallback ve SNaive.
    """
    regressor_cols = get_regressor_cols(phase)

    # chuan bi data theo format Prophet
    train_pdf = _prepare_prophet_df(train_df, regressor_cols)
    future_pdf = _prepare_future_df(future_df, regressor_cols, horizon)

    # drop rows co NaN trong train (lag/rolling dau series)
    train_pdf = train_pdf.dropna().reset_index(drop=True)

    if len(train_pdf) < 2 * season:
        # khong du data de fit Prophet, fallback ngay
        y_train = train_df["sales"].to_numpy(dtype=np.float64)
        y_pred = predict_snaive(y_train, horizon=horizon, season=season)
        return ProphetResult(
            y_pred=y_pred,
            fallback=True,
            converged=False,
            error_msg="insufficient_training_data",
        )

    try:
        # chay trong thread rieng de co the timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _fit_and_predict,
                train_pdf,
                future_pdf,
                regressor_cols,
                horizon,
            )
            y_pred = future.result(timeout=timeout)

        # clamp >= 0 vi sales khong the am
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

    # Prophet fail thi dung SNaive, ghi ro fallback=True
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
    print(f"Phase 1 regressor cols: {get_regressor_cols(1)}")
    print(f"Phase 2 regressor cols: {get_regressor_cols(2)}")

    print("\n=== Phase 1 Prophet ===")
    result = forecast_series(train_df, future_df, phase=1, timeout=120)
    print(f"  fallback : {result.fallback}")
    print(f"  converged: {result.converged}")
    print(f"  error    : {result.error_msg!r}")
    print(f"  y_pred   : {result.y_pred}")
    print(f"  shape    : {result.y_pred.shape}")
    assert result.y_pred.shape == (HORIZON,), f"Wrong shape: {result.y_pred.shape}"
    assert np.all(result.y_pred >= 0), "Negative predictions found"
    print("  [PASS] Shape correct and >= 0")

    print("\n=== Phase 2 Prophet ===")
    result2 = forecast_series(train_df, future_df, phase=2, timeout=120)
    print(f"  fallback : {result2.fallback}")
    print(f"  converged: {result2.converged}")
    print(f"  y_pred   : {result2.y_pred}")
    print(f"  shape    : {result2.y_pred.shape}")
    assert result2.y_pred.shape == (HORIZON,), f"Wrong shape: {result2.y_pred.shape}"
    assert np.all(result2.y_pred >= 0), "Negative predictions found"
    print("  [PASS] Shape correct and >= 0")
