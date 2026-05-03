# sinh data gia lap M5 de test pipeline truoc khi co data that
# ep kieu hung han (int16, int8, float32, category) de giam ~70% RAM
# vi data that co 30k+ series, khong downcast la tran RAM ngay

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_RNG_SEED = 42
_N_ITEMS = 10
_N_STORES = 3
_START_DATE = "2015-01-01"
_END_DATE = "2016-06-19"
_HORIZON = 14
_N_FOLDS = 7
_TEST_START = "2016-03-01"
_SEASON = 7

# chi lay cac ngay le chinh de mock, khong can day du nhu M5 that
_HOLIDAY_DATES: list[str] = [
    "2015-01-01", "2015-01-19", "2015-02-14", "2015-02-16",
    "2015-05-25", "2015-07-04", "2015-09-07", "2015-10-12",
    "2015-11-11", "2015-11-26", "2015-12-25",
    "2016-01-01", "2016-01-18", "2016-02-14", "2016-02-15",
    "2016-05-30", "2016-07-04",
]


def generate_mock_m5_data(
    n_items: int = _N_ITEMS,
    n_stores: int = _N_STORES,
    start_date: str = _START_DATE,
    end_date: str = _END_DATE,
    seed: int = _RNG_SEED,
) -> dict[str, Any]:
    # tra ve 3 key: calendar, sales_clean, split_config
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")

    calendar = _build_calendar(dates)
    sales_clean = _build_sales_clean(dates, calendar, n_items, n_stores, rng)
    split_config = _build_split_config(dates)

    return {
        "calendar": calendar,
        "sales_clean": sales_clean,
        "split_config": split_config,
    }


def _build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    # mo phong calendar.csv cua M5, ep int8 vi chi la flag 0/1 hoac range nho
    holiday_set = set(pd.to_datetime(_HOLIDAY_DATES))

    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = cal["date"].dt.dayofweek.astype(np.int8)
    cal["month"] = cal["date"].dt.month.astype(np.int8)
    cal["is_holiday"] = cal["date"].isin(holiday_set).astype(np.int8)
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(np.int8)

    # snap that co chu ky phuc tap, o day random cho don gian
    cal["snap"] = np.random.default_rng(123).integers(
        0, 2, size=len(cal)
    ).astype(np.int8)

    return cal


def _generate_seasonal_sales(
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # tao sales co chu ky 7 ngay giong M5 that (cuoi tuan ban nhieu hon)
    # dung poisson noise vi sales la count data, normal noise de them bien dong
    # clip [0, 500] roi cast int16 vi M5 max sales ~ 760, int16 max 32767 du xai
    base = rng.integers(5, 31)
    weekly = np.array([1.0, 0.9, 0.85, 0.8, 0.95, 1.3, 1.2])
    trend = np.linspace(0, rng.uniform(0, 0.005) * n_days, n_days)

    sales = np.empty(n_days, dtype=np.float64)
    for t in range(n_days):
        raw = (base + trend[t]) * weekly[t % 7]
        raw += rng.poisson(2)
        raw += rng.normal(0, 1.5)
        sales[t] = max(raw, 0)

    return np.clip(np.round(sales), 0, 500).astype(np.int16)


def _build_sales_clean(
    dates: pd.DatetimeIndex,
    calendar: pd.DataFrame,
    n_items: int,
    n_stores: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # dataframe chinh chua sales + tat ca features cho ca Phase 1 va Phase 2
    n_days = len(dates)
    rows: list[pd.DataFrame] = []

    for i in range(n_items):
        for s in range(n_stores):
            sales = _generate_seasonal_sales(n_days, rng)

            df = pd.DataFrame({
                "date": dates,
                "item_id": f"ITEM_{i + 1:03d}",
                "store_id": f"STORE_{s + 1:02d}",
                "sales": sales,
            })
            rows.append(df)

    sales_clean = pd.concat(rows, ignore_index=True)

    sales_clean = sales_clean.merge(calendar, on="date", how="left")

    sales_clean.sort_values(["item_id", "store_id", "date"], inplace=True)
    grp = sales_clean.groupby(["item_id", "store_id"])["sales"]

    # shift(1) bat buoc de chong data leakage, khong duoc dung gia tri ngay hien tai
    sales_clean["rolling_7"] = (
        grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )
    sales_clean["rolling_28"] = (
        grp.transform(lambda x: x.shift(1).rolling(28, min_periods=1).mean())
        .astype(np.float32)
    )

    # float32 thay vi float64 vi lag/rolling khong can do chinh xac cao
    for lag in (7, 14, 28):
        col = f"lag_{lag}"
        sales_clean[col] = (
            grp.transform(lambda x, _lag=lag: x.shift(_lag))
            .astype(np.float32)
        )

    # gia ban co dinh theo item-store, thuc te se thay doi nhung mock don gian
    rng_price = np.random.default_rng(99)
    sales_clean["sell_price"] = np.float32(0.0)
    for (item, store), idx in sales_clean.groupby(
        ["item_id", "store_id"]
    ).groups.items():
        price = round(rng_price.uniform(1.0, 15.0), 2)
        sales_clean.loc[idx, "sell_price"] = np.float32(price)

    # category tiet kiem rat nhieu RAM khi co nhieu dong lap lai cung gia tri string
    sales_clean["item_id"] = sales_clean["item_id"].astype("category")
    sales_clean["store_id"] = sales_clean["store_id"].astype("category")
    sales_clean["sales"] = sales_clean["sales"].astype(np.int16)

    sales_clean.reset_index(drop=True, inplace=True)
    return sales_clean


def _build_split_config(dates: pd.DatetimeIndex) -> dict[str, Any]:
    # expanding window: train luon bat dau tu ngay dau, chi test truot di
    # khac sliding window o cho ko cat bot data cu, phu hop time series ngan
    test_start = pd.Timestamp(_TEST_START)
    train_start = dates[0]

    folds: list[dict[str, str | int]] = []
    for fold_idx in range(_N_FOLDS):
        fold_test_start = test_start + pd.Timedelta(days=fold_idx * _HORIZON)
        fold_test_end = fold_test_start + pd.Timedelta(days=_HORIZON - 1)
        fold_train_end = fold_test_start - pd.Timedelta(days=1)

        folds.append({
            "fold": fold_idx + 1,
            "train_start": str(train_start.date()),
            "train_end": str(fold_train_end.date()),
            "test_start": str(fold_test_start.date()),
            "test_end": str(fold_test_end.date()),
        })

    return {
        "n_folds": _N_FOLDS,
        "horizon": _HORIZON,
        "season": _SEASON,
        "strategy": "expanding_window",
        "test_period_start": _TEST_START,
        "test_period_end": str(dates[-1].date()),
        "folds": folds,
    }


if __name__ == "__main__":
    data = generate_mock_m5_data()

    cal = data["calendar"]
    print("=== Calendar ===")
    print(f"Shape : {cal.shape}")
    print(f"Dtypes:\n{cal.dtypes}\n")
    print(f"Memory: {cal.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(cal.head(10))

    sc = data["sales_clean"]
    print("\n=== Sales Clean ===")
    print(f"Shape : {sc.shape}")
    print(f"Dtypes:\n{sc.dtypes}\n")
    print(f"Memory: {sc.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(sc.head(10))

    cfg = data["split_config"]
    print("\n=== Split Config ===")
    print(f"Folds : {cfg['n_folds']}")
    print(f"Horizon: {cfg['horizon']} days")
    print(f"Strategy: {cfg['strategy']}")
    for f in cfg["folds"]:
        print(
            f"  Fold {f['fold']}: train {f['train_start']} -> {f['train_end']}  "
            f"|  test {f['test_start']} -> {f['test_end']}"
        )
