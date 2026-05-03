# dung ThreadPoolExecutor thay ProcessPoolExecutor vi Windows ko pickle duoc function
# khi chay qua runpy. signal.alarm cung chi co tren Linux.
# thread-based timeout ko kill duoc thread that su nhung du tot cho use case nay
# vi auto_arima hiem khi treo vinh vien, chi can cat khi qua cham

from __future__ import annotations

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


# timeout 60s theo yeu cau guideline, qua lau thi fallback cho nhanh
TIMEOUT_SECONDS: int = 60
HORIZON: int = 14
SEASON: int = 7

# config auto_arima dung chinh xac theo guideline, khong duoc thay doi
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

# phase 1 chi dung calendar features, phase 2 them lag/rolling/price
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
    # gom ket qua lai 1 cho, de caller biet la SARIMAX that hay fallback SNaive
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
    # chay trong thread rieng de main thread co the timeout duoc
    # import pmdarima o day vi no nang, chi load khi can
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
    # tra ve danh sach cot exog tuong ung voi phase
    if phase == 1:
        return PHASE1_EXOG_COLS
    if phase == 2:
        return PHASE2_EXOG_COLS
    raise ValueError(f"phase must be 1 or 2, got {phase}")


def prepare_exog(
    df: pd.DataFrame,
    phase: int,
) -> np.ndarray:
    # cat dung cac cot exog tu dataframe, ep float32 tiet kiem RAM
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
    # bat moi loi co the xay ra (timeout, LinAlgError, bat ky gi)
    # roi fallback ve SNaive thay vi crash ca pipeline
    # phai validate shape exog truoc khi truyen vao model
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

    # SARIMAX fail thi dung SNaive, ghi ro fallback=True de bao cao sau
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
    # ham tien loi: truyen DataFrame vao la xong, tu cat exog + goi robust
    y_train = train_df["sales"].to_numpy(dtype=np.float64)
    exog_train = prepare_exog(train_df, phase)

    # chi lay dung horizon dong, tranh sai shape khi predict
    future_slice = future_df.head(horizon)
    exog_future = prepare_exog(future_slice, phase)

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
