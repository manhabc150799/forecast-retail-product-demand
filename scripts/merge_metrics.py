"""Merge all predictions into a single metrics_comparison.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.evaluation.metrics import compute_metrics
from src.data.mock_factory import generate_mock_m5_data

data = generate_mock_m5_data()
sc = data["sales_clean"]
sc_idx = sc.set_index(["item_id", "store_id", "date"]).sort_index()

all_dfs = []
for model in ["naive", "snaive", "sarimax", "lstm", "prophet"]:
    for phase in [1, 2]:
        p = f"results/{model}/phase{phase}/predictions.csv"
        if not os.path.exists(p):
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
        mdf = pd.DataFrame(rows)
        all_dfs.append(mdf)
        print(f"  {model}/phase{phase}: {len(mdf)} rows")

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv("results/metrics_comparison.csv", index=False)
print(f"\nTotal: {len(combined)} rows")
print(f"Models: {sorted(combined['model'].unique())}")
print(f"Phases: {sorted(combined['phase'].unique())}")

summary = combined.groupby(["model", "phase"])[["MAE", "RMSE", "MASE", "MAPE"]].mean().round(4)
print(f"\n{'=' * 70}")
print("  FULL METRICS SUMMARY (mean across all folds & series)")
print(f"{'=' * 70}")
print(summary.to_string())
print()
