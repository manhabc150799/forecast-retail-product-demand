"""Merge per-model predictions into a single metrics_comparison.csv.

Reads ``results/{model}/phase{N}/predictions.csv`` for all models/phases,
computes metrics, saves per-model ``metrics.csv``, and writes the merged
``results/metrics_comparison.csv`` with the guideline-mandated schema.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics


METRICS_HEADER: list[str] = [
    "model", "phase", "item_id", "store_id", "fold",
    "mae", "rmse", "mase", "mape",
]


def main() -> None:
    """Run the merge pipeline."""
    data_dir = _PROJECT_ROOT / "data"
    sc = pd.read_parquet(data_dir / "sales_clean.parquet")
    sc_idx = sc.set_index(["item_id", "store_id", "date"]).sort_index()

    all_dfs: list[pd.DataFrame] = []
    for model in ["naive", "snaive", "sarimax", "lstm", "prophet"]:
        for phase in [1, 2]:
            p = _PROJECT_ROOT / "results" / model / f"phase{phase}" / "predictions.csv"
            if not p.exists():
                print(f"  [SKIP] {p} not found")
                continue
            pred = pd.read_csv(p)
            rows = []
            for (fold, iid, sid), g in pred.groupby(["fold", "item_id", "store_id"]):
                yt = g["y_true"].to_numpy(dtype=np.float64)
                yp = g["y_pred"].to_numpy(dtype=np.float64)
                try:
                    s = sc_idx.loc[(iid, sid)].sort_index()
                    ts = pd.Timestamp(g["date"].min())
                    ytrain = s.loc[:ts - pd.Timedelta(days=1)]["sales"].to_numpy(
                        dtype=np.float64,
                    )
                except Exception:
                    ytrain = np.array([])
                if len(ytrain) < 8:
                    ytrain = np.full(9, np.nan)
                m = compute_metrics(yt, yp, ytrain, season=7)
                m["fold"] = fold
                m["item_id"] = iid
                m["store_id"] = sid
                m["model"] = model
                m["phase"] = phase
                rows.append(m)
            mdf = pd.DataFrame(rows)[METRICS_HEADER]

            per_model_path = (
                _PROJECT_ROOT / "results" / model / f"phase{phase}" / "metrics.csv"
            )
            mdf.to_csv(per_model_path, index=False)

            all_dfs.append(mdf)
            print(f"  {model}/phase{phase}: {len(mdf)} rows -> {per_model_path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    out = _PROJECT_ROOT / "results" / "metrics_comparison.csv"
    combined.to_csv(out, index=False)
    print(f"\nTotal: {len(combined)} rows -> {out}")
    print(f"Models: {sorted(combined['model'].unique())}")
    print(f"Phases: {sorted(combined['phase'].unique())}")

    summary = combined.groupby(["model", "phase"])[
        ["mae", "rmse", "mase", "mape"]
    ].mean().round(4)
    print(f"\n{'=' * 70}")
    print("  FULL METRICS SUMMARY (mean across all folds & series)")
    print(f"{'=' * 70}")
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
