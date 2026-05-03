# chay experiment cho Naive, SNaive, SARIMAX
# dung append mode ghi thang xuong CSV, tranh OOM khi loop qua nhieu series
# psutil do RAM/CPU thuc te cua process, khong phai system-wide

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

# them project root de import src.* duoc
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.naive import predict_naive, predict_snaive          # noqa: E402
from src.models.sarimax_model import (                               # noqa: E402
    SarimaxResult,
    forecast_series as sarimax_forecast_series,
    run_sarimax_robust,
    prepare_exog,
)
from src.data.mock_factory import generate_mock_m5_data              # noqa: E402


HORIZON: int = 14
SEASON: int = 7
MODELS: list[str] = ["naive", "snaive", "sarimax"]

# header phai khop dung schema trong guideline, thay doi la sai format output
PRED_HEADER: list[str] = [
    "date", "item_id", "store_id", "y_true", "y_pred", "fold", "fallback",
]
LOG_HEADER: list[str] = [
    "fold", "item_id", "store_id",
    "train_time_s", "peak_ram_mb", "cpu_percent",
    "converged", "error_msg",
]


class CsvAppender:
    # dung append mode de ghi thang xuong o cung, tranh OOM khi loop qua 100 series
    # flush() sau moi row de khong mat data neu crash giua chung

    def __init__(
        self,
        path: Path,
        header: list[str],
        overwrite: bool = False,
    ) -> None:
        self.path = path
        self.header = header
        self.overwrite = overwrite
        self._file = None
        self._writer: csv.writer | None = None

    def __enter__(self) -> "CsvAppender":
        write_header = self.overwrite or not self.path.exists()
        mode = "w" if self.overwrite else "a"
        self._file = open(  # noqa: SIM115
            self.path, mode, newline="", encoding="utf-8",
        )
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self.header)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._file:
            self._file.close()

    def write_row(self, row: list[Any]) -> None:
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()  # type: ignore[union-attr]

    def write_rows(self, rows: list[list[Any]]) -> None:
        assert self._writer is not None
        self._writer.writerows(rows)
        self._file.flush()  # type: ignore[union-attr]


_PROCESS = psutil.Process(os.getpid())


def _snapshot_resources() -> tuple[float, float, float]:
    # chup nhanh RAM + CPU tai thoi diem goi, dung de tinh delta sau
    rss_mb = _PROCESS.memory_info().rss / (1024 ** 2)
    cpu_pct = _PROCESS.cpu_percent(interval=None)
    return time.time(), rss_mb, cpu_pct


def _run_naive(
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_dates: pd.Series,
    item_id: str,
    store_id: str,
    fold: int,
    pred_csv: CsvAppender,
    log_csv: CsvAppender,
) -> None:
    t0, ram0, _ = _snapshot_resources()
    _PROCESS.cpu_percent(interval=None)  # reset counter truoc khi do

    y_pred = predict_naive(y_train, horizon=HORIZON)

    t1 = time.time()
    ram1 = _PROCESS.memory_info().rss / (1024 ** 2)
    cpu = _PROCESS.cpu_percent(interval=None)

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(y_pred[i]),
            fold, False,
        ])

    log_csv.write_row([
        fold, item_id, store_id,
        round(t1 - t0, 6), round(max(ram1 - ram0, 0), 2), round(cpu, 1),
        True, "",
    ])


def _run_snaive(
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_dates: pd.Series,
    item_id: str,
    store_id: str,
    fold: int,
    pred_csv: CsvAppender,
    log_csv: CsvAppender,
) -> None:
    t0, ram0, _ = _snapshot_resources()
    _PROCESS.cpu_percent(interval=None)

    y_pred = predict_snaive(y_train, horizon=HORIZON, season=SEASON)

    t1 = time.time()
    ram1 = _PROCESS.memory_info().rss / (1024 ** 2)
    cpu = _PROCESS.cpu_percent(interval=None)

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(y_pred[i]),
            fold, False,
        ])

    log_csv.write_row([
        fold, item_id, store_id,
        round(t1 - t0, 6), round(max(ram1 - ram0, 0), 2), round(cpu, 1),
        True, "",
    ])


def _run_sarimax(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    y_test: np.ndarray,
    test_dates: pd.Series,
    item_id: str,
    store_id: str,
    fold: int,
    phase: int,
    pred_csv: CsvAppender,
    log_csv: CsvAppender,
) -> None:
    t0, ram0, _ = _snapshot_resources()
    _PROCESS.cpu_percent(interval=None)

    result: SarimaxResult = sarimax_forecast_series(
        train_df, future_df, phase=phase,
        horizon=HORIZON, season=SEASON, timeout=60,
    )

    t1 = time.time()
    ram1 = _PROCESS.memory_info().rss / (1024 ** 2)
    cpu = _PROCESS.cpu_percent(interval=None)

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(result.y_pred[i]),
            fold, result.fallback,
        ])

    log_csv.write_row([
        fold, item_id, store_id,
        round(t1 - t0, 6), round(max(ram1 - ram0, 0), 2), round(cpu, 1),
        result.converged, result.error_msg,
    ])


def load_data(use_real: bool = False) -> dict[str, Any]:
    # khi chua co data that thi dung mock, sau nay doi --real la chay that
    if use_real:
        data_dir = _PROJECT_ROOT / "data"
        sc = pd.read_parquet(data_dir / "sales_clean.parquet")
        import json
        with open(data_dir / "split_config.json") as f:
            cfg = json.load(f)
        series_csv = pd.read_csv(data_dir / "selected_series.csv")
        series_list = list(zip(series_csv["item_id"], series_csv["store_id"]))
    else:
        mock = generate_mock_m5_data()
        sc = mock["sales_clean"]
        cfg = mock["split_config"]
        series_list = (
            sc[["item_id", "store_id"]]
            .drop_duplicates()
            .apply(lambda r: (r["item_id"], r["store_id"]), axis=1)
            .tolist()
        )

    return {
        "sales_clean": sc,
        "split_config": cfg,
        "series_list": series_list,
    }


def run_all_experiments(
    phase: int = 1,
    use_real: bool = False,
    models: list[str] | None = None,
) -> None:
    if models is None:
        models = list(MODELS)

    print(f"\n{'=' * 60}")
    print(f"  M5 Experiment Runner -- Phase {phase}")
    print(f"  Data source: {'REAL' if use_real else 'MOCK'}")
    print(f"{'=' * 60}\n")

    data = load_data(use_real)
    sc: pd.DataFrame = data["sales_clean"]
    cfg: dict = data["split_config"]
    series_list: list[tuple[str, str]] = data["series_list"]
    folds: list[dict] = cfg["folds"]

    n_series = len(series_list)
    n_folds = len(folds)
    print(f"  Series : {n_series}")
    print(f"  Folds  : {n_folds}")
    print(f"  Models : {models}\n")

    # set_index truoc de lookup O(1) thay vi filter O(n) moi lan
    sc_indexed = sc.set_index(["item_id", "store_id", "date"]).sort_index()

    for model_name in models:
        result_dir = _PROJECT_ROOT / "results" / model_name / f"phase{phase}"
        result_dir.mkdir(parents=True, exist_ok=True)

        pred_path = result_dir / "predictions.csv"
        log_path = result_dir / "training_log.csv"

        total_iters = n_folds * n_series
        model_desc = f"[{model_name.upper():>7}] Phase {phase}"

        with (
            CsvAppender(pred_path, PRED_HEADER, overwrite=True) as pred_csv,
            CsvAppender(log_path, LOG_HEADER, overwrite=True) as log_csv,
            tqdm(
                total=total_iters,
                desc=model_desc,
                unit="series",
                ncols=90,
                leave=True,
            ) as pbar,
        ):
            for fold_info in folds:
                fold_num: int = fold_info["fold"]
                train_end = pd.Timestamp(fold_info["train_end"])
                test_start = pd.Timestamp(fold_info["test_start"])
                test_end = pd.Timestamp(fold_info["test_end"])

                for item_id, store_id in series_list:
                    pbar.set_postfix_str(
                        f"F{fold_num} {item_id} {store_id}",
                        refresh=False,
                    )

                    try:
                        series_data = sc_indexed.loc[
                            (item_id, store_id)
                        ].sort_index()
                    except KeyError:
                        pbar.update(1)
                        continue

                    train_slice = series_data.loc[:train_end]
                    test_slice = series_data.loc[test_start:test_end]

                    # bo qua neu khong du 14 ngay test
                    if len(test_slice) < HORIZON:
                        pbar.update(1)
                        continue

                    y_train = train_slice["sales"].to_numpy(dtype=np.float64)
                    y_test = test_slice["sales"].to_numpy(dtype=np.float64)[
                        :HORIZON
                    ]
                    test_dates = test_slice.reset_index()["date"].head(HORIZON)

                    if model_name == "naive":
                        _run_naive(
                            y_train, y_test, test_dates,
                            item_id, store_id, fold_num,
                            pred_csv, log_csv,
                        )
                    elif model_name == "snaive":
                        _run_snaive(
                            y_train, y_test, test_dates,
                            item_id, store_id, fold_num,
                            pred_csv, log_csv,
                        )
                    elif model_name == "sarimax":
                        # can reset_index vi sarimax can cot date trong DataFrame
                        train_df = train_slice.reset_index()
                        future_df = test_slice.reset_index().head(HORIZON)
                        _run_sarimax(
                            train_df, future_df,
                            y_test, test_dates,
                            item_id, store_id, fold_num,
                            phase, pred_csv, log_csv,
                        )

                    pbar.update(1)

        pred_df = pd.read_csv(pred_path)
        log_df = pd.read_csv(log_path)
        print(f"  -> {pred_path}  ({len(pred_df)} rows)")
        print(f"  -> {log_path}  ({len(log_df)} rows)")
        if model_name == "sarimax" and "converged" in log_df.columns:
            n_fail = (~log_df["converged"]).sum()
            print(f"     SARIMAX fallbacks: {n_fail} / {len(log_df)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5 Forecasting - run Naive / SNaive / SARIMAX experiments",
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Experiment phase (1=basic exog, 2=enhanced features)",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Use real parquet data instead of mock",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=MODELS,
        help="Subset of models to run (default: all)",
    )
    args = parser.parse_args()

    models_to_run = args.models if args.models else list(MODELS)

    run_all_experiments(
        phase=args.phase, use_real=args.real, models=models_to_run,
    )
    print("All experiments completed.")


if __name__ == "__main__":
    main()
