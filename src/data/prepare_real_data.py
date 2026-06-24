"""Build the 3 required input files from raw M5 competition data.

Outputs (saved to ``data/``):
    - ``selected_series.csv``  : 100 (item_id, store_id) pairs
    - ``split_config.json``    : expanding-window fold definitions
    - ``sales_clean.parquet``  : daily sales + all features for the 100 series

Raw inputs expected in ``data/``:
    - ``sales_train_evaluation.csv``
    - ``calendar.csv``
    - ``sell_prices.csv``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

N_SERIES = 100
HORIZON = 14
N_FOLDS = 7
TEST_START = "2016-03-01"
DATA_END = "2016-06-19"
SEASON = 7

_STATE_MAP: dict[str, str] = {
    "CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
    "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
    "WI_1": "WI", "WI_2": "WI", "WI_3": "WI",
}


def _select_top_series(sales_wide: pd.DataFrame, n: int = N_SERIES) -> pd.DataFrame:
    """Pick the *n* (item_id, store_id) pairs with the highest total sales.

    Args:
        sales_wide: Wide-format M5 sales data with columns d_1 … d_1941.
        n: Number of series to keep.

    Returns:
        DataFrame with columns ``item_id`` and ``store_id`` (n rows).
    """
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    sales_wide["_total"] = sales_wide[d_cols].sum(axis=1)
    top = sales_wide.nlargest(n, "_total")[["item_id", "store_id"]].reset_index(drop=True)
    sales_wide.drop(columns="_total", inplace=True)
    return top


def _build_split_config() -> dict:
    """Create expanding-window fold definitions.

    Returns:
        Dictionary with fold metadata suitable for JSON serialisation.
    """
    test_start = pd.Timestamp(TEST_START)
    train_start_str = "2011-01-29"

    folds = []
    for i in range(N_FOLDS):
        fold_test_start = test_start + pd.Timedelta(days=i * HORIZON)
        fold_test_end = fold_test_start + pd.Timedelta(days=HORIZON - 1)
        fold_train_end = fold_test_start - pd.Timedelta(days=1)

        folds.append({
            "fold": i + 1,
            "train_start": train_start_str,
            "train_end": str(fold_train_end.date()),
            "test_start": str(fold_test_start.date()),
            "test_end": str(fold_test_end.date()),
        })

    return {
        "n_folds": N_FOLDS,
        "horizon": HORIZON,
        "season": SEASON,
        "strategy": "expanding_window",
        "test_period_start": TEST_START,
        "test_period_end": DATA_END,
        "folds": folds,
    }


def _melt_sales(
    sales_wide: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Convert selected series from wide to long format.

    Args:
        sales_wide: Full wide-format sales data.
        selected: DataFrame with ``item_id`` and ``store_id`` columns.

    Returns:
        Long DataFrame with columns ``item_id``, ``store_id``, ``d``, ``sales``.
    """
    merged = sales_wide.merge(selected, on=["item_id", "store_id"], how="inner")
    d_cols = [c for c in merged.columns if c.startswith("d_")]
    id_cols = ["item_id", "store_id"]

    long = merged[id_cols + d_cols].melt(
        id_vars=id_cols,
        var_name="d",
        value_name="sales",
    )
    long["sales"] = long["sales"].astype(np.float64)
    return long


def _merge_calendar(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Join calendar features onto the long sales DataFrame.

    Derives ``date``, ``day_of_week``, ``month``, ``is_holiday``, ``is_weekend``
    and per-state ``snap`` from the raw M5 calendar.

    Args:
        long: Long-format sales data with column ``d``.
        calendar: Raw ``calendar.csv`` contents.

    Returns:
        DataFrame with calendar columns appended.
    """
    cal = calendar[
        ["d", "date", "wday", "month", "wm_yr_wk",
         "event_name_1", "event_name_2",
         "snap_CA", "snap_TX", "snap_WI"]
    ].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["day_of_week"] = (cal["wday"] - 1).astype(np.int8)
    cal["is_holiday"] = (
        cal["event_name_1"].notna() | cal["event_name_2"].notna()
    ).astype(np.int8)
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(np.int8)

    merged = long.merge(cal, on="d", how="left")

    merged["state_id"] = merged["store_id"].map(
        lambda sid: _STATE_MAP.get(sid, sid.split("_")[0])
    )
    merged["snap"] = np.int8(0)
    for state in ("CA", "TX", "WI"):
        mask = merged["state_id"] == state
        merged.loc[mask, "snap"] = merged.loc[mask, f"snap_{state}"].astype(np.int8)

    drop_cols = [
        "d", "wday", "event_name_1", "event_name_2",
        "snap_CA", "snap_TX", "snap_WI", "state_id",
    ]
    merged.drop(columns=drop_cols, inplace=True)
    return merged


def _merge_prices(
    df: pd.DataFrame,
    sell_prices: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Join sell prices onto the sales DataFrame.

    Args:
        df: Sales + calendar DataFrame (must contain ``wm_yr_wk``).
        sell_prices: Raw ``sell_prices.csv`` contents.
        selected: DataFrame with ``item_id``, ``store_id`` to filter prices.

    Returns:
        DataFrame with ``sell_price`` column; ``wm_yr_wk`` dropped.
    """
    sp = sell_prices.merge(selected, on=["item_id", "store_id"], how="inner")
    merged = df.merge(
        sp[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    merged["sell_price"] = merged["sell_price"].astype(np.float64).fillna(0.0)
    merged.drop(columns="wm_yr_wk", inplace=True)
    return merged


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling-mean features with 1-day shift to prevent leakage.

    Features added: ``lag_7``, ``lag_14``, ``lag_28``, ``rolling_7``, ``rolling_28``.
    All features use ``shift(1)`` so only past data is visible.

    Args:
        df: DataFrame sorted by ``item_id``, ``store_id``, ``date``.

    Returns:
        DataFrame with new feature columns appended.
    """
    df = df.sort_values(["item_id", "store_id", "date"]).copy()
    grp = df.groupby(["item_id", "store_id"])["sales"]

    df["lag_7"] = grp.transform(lambda x: x.shift(7)).astype(np.float64)
    df["lag_14"] = grp.transform(lambda x: x.shift(14)).astype(np.float64)
    df["lag_28"] = grp.transform(lambda x: x.shift(28)).astype(np.float64)

    df["rolling_7"] = (
        grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        .astype(np.float64)
    )
    df["rolling_28"] = (
        grp.transform(lambda x: x.shift(1).rolling(28, min_periods=1).mean())
        .astype(np.float64)
    )

    return df


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the output schema required by the guideline.

    Columns (in order): date, item_id, store_id, sales, day_of_week, month,
    is_holiday, is_weekend, sell_price, snap, lag_7, lag_14, lag_28,
    rolling_7, rolling_28.

    Args:
        df: Raw assembled DataFrame.

    Returns:
        DataFrame with correct column order and dtypes.
    """
    col_order = [
        "date", "item_id", "store_id", "sales",
        "day_of_week", "month", "is_holiday", "is_weekend",
        "sell_price", "snap",
        "lag_7", "lag_14", "lag_28", "rolling_7", "rolling_28",
    ]

    dtype_map = {
        "sales": np.float64,
        "day_of_week": np.int32,
        "month": np.int32,
        "is_holiday": np.int32,
        "is_weekend": np.int32,
        "sell_price": np.float64,
        "snap": np.int32,
        "lag_7": np.float64,
        "lag_14": np.float64,
        "lag_28": np.float64,
        "rolling_7": np.float64,
        "rolling_28": np.float64,
    }

    out = df[col_order].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["item_id"] = out["item_id"].astype(str)
    out["store_id"] = out["store_id"].astype(str)
    for col, dt in dtype_map.items():
        out[col] = out[col].astype(dt)

    return out.reset_index(drop=True)


def prepare(data_dir: Path | None = None) -> None:
    """Run the full data preparation pipeline.

    Reads raw M5 CSV files, selects top-100 series, engineers features,
    and writes the 3 required output files.

    Args:
        data_dir: Directory containing raw CSVs and where outputs are written.
                  Defaults to ``<project_root>/data/``.
    """
    if data_dir is None:
        data_dir = _DATA_DIR

    print("Loading raw M5 files ...")
    sales_wide = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    sell_prices = pd.read_csv(data_dir / "sell_prices.csv")
    print(f"  sales_wide : {sales_wide.shape}")
    print(f"  calendar   : {calendar.shape}")
    print(f"  sell_prices: {sell_prices.shape}")

    print(f"\nSelecting top {N_SERIES} series by total sales ...")
    selected = _select_top_series(sales_wide, N_SERIES)
    selected.to_csv(data_dir / "selected_series.csv", index=False)
    print(f"  -> {data_dir / 'selected_series.csv'}  ({len(selected)} rows)")

    print("\nBuilding split config ...")
    cfg = _build_split_config()
    with open(data_dir / "split_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  -> {data_dir / 'split_config.json'}  ({cfg['n_folds']} folds)")
    for fold in cfg["folds"]:
        print(f"     Fold {fold['fold']}: train -> {fold['train_end']}  "
              f"|  test {fold['test_start']} -> {fold['test_end']}")

    print("\nMelting wide -> long ...")
    long = _melt_sales(sales_wide, selected)
    print(f"  long shape: {long.shape}")

    print("Merging calendar ...")
    long = _merge_calendar(long, calendar)

    print("Merging sell prices ...")
    long = _merge_prices(long, sell_prices, selected)

    print("Adding lag / rolling features ...")
    long = _add_lag_rolling_features(long)

    print("Enforcing output schema ...")
    sc = _enforce_schema(long)

    sc.to_parquet(data_dir / "sales_clean.parquet", index=False)
    print(f"\n  -> {data_dir / 'sales_clean.parquet'}  ({sc.shape})")
    print(f"     Columns : {list(sc.columns)}")
    print(f"     Dtypes  :\n{sc.dtypes.to_string()}")
    print(f"     Memory  : {sc.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"     NaN counts:\n{sc.isna().sum().to_string()}")

    print("\nDone.")


if __name__ == "__main__":
    prepare()
