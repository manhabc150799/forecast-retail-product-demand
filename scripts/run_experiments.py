"""M5 Experiment Runner — Naive, SNaive, SARIMAX, LSTM, Prophet.

Runs all models for a given phase, saves predictions and training logs
per model, computes per-model ``metrics.csv``, and merges everything
into ``results/metrics_comparison.csv``.

Usage::

    python scripts/run_experiments.py --phase 1 --real
    python scripts/run_experiments.py --phase 2 --real --models sarimax lstm prophet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.naive import predict_naive, predict_snaive          # noqa: E402
from src.models.sarimax_model import (                               # noqa: E402
    SarimaxResult,
    forecast_series as sarimax_forecast_series,
)
from src.models.lstm_model import (                                  # noqa: E402
    HORIZON as LSTM_HORIZON,
    run_lstm_fold,
)
from src.models.prophet_model import (                               # noqa: E402
    ProphetResult,
    forecast_series as prophet_forecast_series,
)
from src.evaluation.profiler import ResourceProfiler                 # noqa: E402
from src.evaluation.metrics import compute_metrics                   # noqa: E402


HORIZON: int = 14
SEASON: int = 7
MODELS: list[str] = ["naive", "snaive", "sarimax", "lstm", "prophet"]

PRED_HEADER: list[str] = [
    "date", "item_id", "store_id", "y_true", "y_pred", "fold", "fallback",
]
LOG_HEADER: list[str] = [
    "fold", "item_id", "store_id",
    "train_time_s", "peak_ram_mb", "cpu_percent",
    "converged", "error_msg",
]
METRICS_HEADER: list[str] = [
    "model", "phase", "item_id", "store_id", "fold",
    "mae", "rmse", "mase", "mape",
]


class CsvAppender:
    """Append-mode CSV writer that flushes after every row to prevent data loss.

    Args:
        path: File path to write to.
        header: Column names for the CSV header.
        overwrite: If ``True``, truncate any existing file.
    """

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

    def __enter__(self) -> CsvAppender:
        """Open the file and optionally write the header row.

        Returns:
            ``self`` for use in a ``with … as csv:`` block.
        """
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
        """Close the underlying file handle."""
        if self._file:
            self._file.close()

    def write_row(self, row: list[Any]) -> None:
        """Write a single row and flush immediately.

        Args:
            row: List of values matching the header length.
        """
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()  # type: ignore[union-attr]

    def write_rows(self, rows: list[list[Any]]) -> None:
        """Write multiple rows and flush.

        Args:
            rows: List of row lists.
        """
        assert self._writer is not None
        self._writer.writerows(rows)
        self._file.flush()  # type: ignore[union-attr]


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
    """Run Naive forecast for a single series/fold and write results.

    Args:
        y_train: Training target values.
        y_test: Actual test values (for y_true column).
        test_dates: Datetime series for the test horizon.
        item_id: Item identifier.
        store_id: Store identifier.
        fold: 1-based fold number.
        pred_csv: Writer for predictions.csv.
        log_csv: Writer for training_log.csv.
    """
    with ResourceProfiler() as prof:
        y_pred = predict_naive(y_train, horizon=HORIZON)

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(y_pred[i]),
            fold, False,
        ])

    stats = prof.to_dict()
    log_csv.write_row([
        fold, item_id, store_id,
        stats["train_time_s"], stats["peak_ram_mb"], stats["cpu_percent"],
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
    """Run Seasonal Naive forecast for a single series/fold and write results.

    Args:
        y_train: Training target values.
        y_test: Actual test values.
        test_dates: Datetime series for the test horizon.
        item_id: Item identifier.
        store_id: Store identifier.
        fold: 1-based fold number.
        pred_csv: Writer for predictions.csv.
        log_csv: Writer for training_log.csv.
    """
    with ResourceProfiler() as prof:
        y_pred = predict_snaive(y_train, horizon=HORIZON, season=SEASON)

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(y_pred[i]),
            fold, False,
        ])

    stats = prof.to_dict()
    log_csv.write_row([
        fold, item_id, store_id,
        stats["train_time_s"], stats["peak_ram_mb"], stats["cpu_percent"],
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
    """Run SARIMAX forecast for a single series/fold and write results.

    Args:
        train_df: Training DataFrame.
        future_df: Future DataFrame with exog columns.
        y_test: Actual test values.
        test_dates: Datetime series for the test horizon.
        item_id: Item identifier.
        store_id: Store identifier.
        fold: 1-based fold number.
        phase: Experiment phase (1 or 2).
        pred_csv: Writer for predictions.csv.
        log_csv: Writer for training_log.csv.
    """
    with ResourceProfiler() as prof:
        result: SarimaxResult = sarimax_forecast_series(
            train_df, future_df, phase=phase,
            horizon=HORIZON, season=SEASON, timeout=60,
        )

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(result.y_pred[i]),
            fold, result.fallback,
        ])

    stats = prof.to_dict()
    log_csv.write_row([
        fold, item_id, store_id,
        stats["train_time_s"], stats["peak_ram_mb"], stats["cpu_percent"],
        result.converged, result.error_msg,
    ])


def _run_prophet(
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
    """Run Prophet forecast for a single series/fold and write results.

    Args:
        train_df: Training DataFrame.
        future_df: Future DataFrame with regressor columns.
        y_test: Actual test values.
        test_dates: Datetime series for the test horizon.
        item_id: Item identifier.
        store_id: Store identifier.
        fold: 1-based fold number.
        phase: Experiment phase (1 or 2).
        pred_csv: Writer for predictions.csv.
        log_csv: Writer for training_log.csv.
    """
    with ResourceProfiler() as prof:
        result: ProphetResult = prophet_forecast_series(
            train_df, future_df, phase=phase,
            horizon=HORIZON, season=SEASON, timeout=120,
        )

    for i in range(HORIZON):
        pred_csv.write_row([
            str(test_dates.iloc[i].date()),
            item_id, store_id,
            float(y_test[i]), float(result.y_pred[i]),
            fold, result.fallback,
        ])

    stats = prof.to_dict()
    log_csv.write_row([
        fold, item_id, store_id,
        stats["train_time_s"], stats["peak_ram_mb"], stats["cpu_percent"],
        result.converged, result.error_msg,
    ])


def _run_lstm_fold(
    sc_indexed: pd.DataFrame,
    series_list: list[tuple[str, str]],
    fold_info: dict,
    pred_csv: CsvAppender,
    log_csv: CsvAppender,
    sc: pd.DataFrame,
    phase: int = 1,
) -> None:
    """Run LSTM for an entire fold and write results.

    LSTM trains one global model per fold, so ``train_time_s`` is the
    same for every series within the fold (repeated per row as required
    by the training_log schema).

    Args:
        sc_indexed: Sales DataFrame indexed by ``(item_id, store_id, date)``.
        series_list: List of ``(item_id, store_id)`` tuples.
        fold_info: Fold definition dict with keys ``fold``, ``train_end``, etc.
        pred_csv: Writer for predictions.csv.
        log_csv: Writer for training_log.csv.
        sc: Original (un-indexed) sales DataFrame.
        phase: Experiment phase.
    """
    fold_num = fold_info["fold"]
    train_end = pd.Timestamp(fold_info["train_end"])
    test_start = pd.Timestamp(fold_info["test_start"])
    test_end = pd.Timestamp(fold_info["test_end"])

    with ResourceProfiler() as prof:
        fold_result = run_lstm_fold(
            sc_indexed, series_list, train_end, test_start, test_end,
            phase=phase,
        )

    stats = prof.to_dict()

    for key, msg in fold_result.skipped.items():
        item_id, store_id = key
        log_csv.write_row([
            fold_num, item_id, store_id,
            stats["train_time_s"], stats["peak_ram_mb"], stats["cpu_percent"],
            False, msg,
        ])

    for key, y_pred in fold_result.predictions.items():
        item_id, store_id = key

        try:
            series_data = sc_indexed.loc[(item_id, store_id)].sort_index()
        except KeyError:
            continue

        test_slice = series_data.loc[test_start:test_end]
        if len(test_slice) < HORIZON:
            continue

        y_test = test_slice["sales"].to_numpy(dtype=np.float64)[:HORIZON]
        test_dates = test_slice.reset_index()["date"].head(HORIZON)

        for i in range(HORIZON):
            pred_csv.write_row([
                str(test_dates.iloc[i].date()),
                item_id, store_id,
                float(y_test[i]), float(y_pred[i]),
                fold_num, False,
            ])

        log_csv.write_row([
            fold_num, item_id, store_id,
            stats["train_time_s"], stats["peak_ram_mb"], stats["cpu_percent"],
            fold_result.converged, "",
        ])


def load_data(use_real: bool = True) -> dict[str, Any]:
    """Load either real M5 data or mock data for testing.

    Args:
        use_real: If ``True``, read from ``data/`` (parquet + json + csv).
                  If ``False``, generate synthetic data via mock_factory.

    Returns:
        Dictionary with keys ``sales_clean``, ``split_config``, ``series_list``.
    """
    if use_real:
        data_dir = _PROJECT_ROOT / "data"
        sc = pd.read_parquet(data_dir / "sales_clean.parquet")
        with open(data_dir / "split_config.json") as f:
            cfg = json.load(f)
        series_csv = pd.read_csv(data_dir / "selected_series.csv")
        series_list = list(zip(series_csv["item_id"], series_csv["store_id"]))
    else:
        from src.data.mock_factory import generate_mock_m5_data
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


def compute_batch_metrics(
    pred_path: Path,
    sc_indexed: pd.DataFrame,
    season: int = SEASON,
) -> pd.DataFrame:
    """Compute evaluation metrics from a predictions CSV.

    Groups predictions by ``(fold, item_id, store_id)`` and computes
    MAE, RMSE, MASE, MAPE for each group.

    Args:
        pred_path: Path to the model's ``predictions.csv``.
        sc_indexed: Sales DataFrame indexed by ``(item_id, store_id, date)``.
        season: Seasonal period for MASE computation.

    Returns:
        DataFrame with one row per (fold, item_id, store_id) and columns
        ``mae``, ``rmse``, ``mase``, ``mape``, ``fold``, ``item_id``,
        ``store_id``.
    """
    pred_df = pd.read_csv(pred_path)
    rows: list[dict[str, Any]] = []

    for (fold, item_id, store_id), grp in pred_df.groupby(
        ["fold", "item_id", "store_id"]
    ):
        y_true = grp["y_true"].to_numpy(dtype=np.float64)
        y_pred = grp["y_pred"].to_numpy(dtype=np.float64)

        try:
            series = sc_indexed.loc[(item_id, store_id)].sort_index()
            test_start = pd.Timestamp(grp["date"].min())
            y_train = series.loc[:test_start - pd.Timedelta(days=1)]["sales"].to_numpy(
                dtype=np.float64,
            )
        except (KeyError, IndexError):
            y_train = np.array([])

        if len(y_train) < season + 1:
            y_train = np.full(season + 2, np.nan)

        m = compute_metrics(y_true, y_pred, y_train, season=season)
        m["fold"] = fold
        m["item_id"] = item_id
        m["store_id"] = store_id
        rows.append(m)

    return pd.DataFrame(rows)


def _get_models_for_phase(phase: int, requested: list[str]) -> list[str]:
    """Determine which models to run for a given phase.

    Naive and SNaive produce identical results in Phase 1 and Phase 2
    (they do not use exogenous features), but Phase 2 results are still
    generated and saved for completeness.

    Args:
        phase: Experiment phase (1 or 2).
        requested: User-requested model list.

    Returns:
        Filtered list of model names to run.
    """
    return requested


def run_all_experiments(
    phase: int = 1,
    use_real: bool = True,
    models: list[str] | None = None,
) -> None:
    """Run experiments for all requested models and save results.

    For each model, saves:
      - ``results/{model}/phase{N}/predictions.csv``
      - ``results/{model}/phase{N}/training_log.csv``
      - ``results/{model}/phase{N}/metrics.csv``

    Finally merges all per-model metrics into
    ``results/metrics_comparison.csv`` with the guideline-mandated schema:
    ``model, phase, item_id, store_id, fold, mae, rmse, mase, mape``.

    Args:
        phase: Experiment phase (1 or 2).
        use_real: Whether to use real M5 data.
        models: Subset of models to run (default: all 5).
    """
    if models is None:
        models = list(MODELS)

    models_to_run = _get_models_for_phase(phase, models)

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
    print(f"  Models : {models_to_run}\n")

    sc_indexed = sc.set_index(["item_id", "store_id", "date"]).sort_index()

    metrics_dir = _PROJECT_ROOT / "results"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []

    for model_name in models_to_run:
        result_dir = _PROJECT_ROOT / "results" / model_name / f"phase{phase}"
        result_dir.mkdir(parents=True, exist_ok=True)

        pred_path = result_dir / "predictions.csv"
        log_path = result_dir / "training_log.csv"

        if model_name == "lstm":
            with (
                CsvAppender(pred_path, PRED_HEADER, overwrite=True) as pred_csv,
                CsvAppender(log_path, LOG_HEADER, overwrite=True) as log_csv,
            ):
                for fold_info in tqdm(
                    folds,
                    desc=f"[   LSTM] Phase {phase}",
                    unit="fold",
                    ncols=90,
                ):
                    _run_lstm_fold(
                        sc_indexed, series_list,
                        fold_info, pred_csv, log_csv, sc,
                        phase=phase,
                    )
        else:
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
                            train_df = train_slice.reset_index()
                            future_df = test_slice.reset_index().head(HORIZON)
                            _run_sarimax(
                                train_df, future_df,
                                y_test, test_dates,
                                item_id, store_id, fold_num,
                                phase, pred_csv, log_csv,
                            )
                        elif model_name == "prophet":
                            train_df = train_slice.reset_index()
                            future_df = test_slice.reset_index().head(HORIZON)
                            _run_prophet(
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

        if len(pred_df) > 0:
            metrics_df = compute_batch_metrics(pred_path, sc_indexed)
            metrics_df["model"] = model_name
            metrics_df["phase"] = phase

            per_model_path = result_dir / "metrics.csv"
            ordered = metrics_df[METRICS_HEADER]
            ordered.to_csv(per_model_path, index=False)
            print(f"  -> {per_model_path}  ({len(ordered)} rows)")

            all_metrics.append(ordered)

            agg = metrics_df[["mae", "rmse", "mase", "mape"]].mean()
            print(f"     mae={agg['mae']:.4f}  rmse={agg['rmse']:.4f}  "
                  f"mase={agg['mase']:.4f}  mape={agg['mape']:.2f}%")
        print()

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)

        out_path = metrics_dir / "metrics_comparison.csv"
        combined.to_csv(out_path, index=False)
        print(f"  -> Metrics comparison: {out_path}  ({len(combined)} rows)")

        summary = (
            combined
            .groupby("model")[["mae", "rmse", "mase", "mape"]]
            .mean()
            .round(4)
        )
        print(f"\n{'=' * 60}")
        print("  Metrics Summary (mean across all folds & series)")
        print(f"{'=' * 60}")
        print(summary.to_string())
        print()


def main() -> None:
    """CLI entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description="M5 Forecasting - run Naive / SNaive / SARIMAX / LSTM / Prophet experiments",
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Experiment phase (1=basic exog, 2=enhanced features)",
    )
    parser.add_argument(
        "--real", action="store_true", default=True,
        help="Use real parquet data (default: True)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock data instead of real data",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=MODELS,
        help="Subset of models to run (default: all)",
    )
    args = parser.parse_args()

    use_real = not args.mock
    models_to_run = args.models if args.models else list(MODELS)

    run_all_experiments(
        phase=args.phase, use_real=use_real, models=models_to_run,
    )
    print("All experiments completed.")


if __name__ == "__main__":
    main()
