from __future__ import annotations

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 7,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    # MASE: scale = MAE cua seasonal naive 1-step tren tap train
    # cat y_train[season:] va y_train[:-season] de tinh sai so t vs t-season
    # tranh vong lap, dung slicing thang
    naive_errors = np.abs(y_train[season:] - y_train[:-season])
    scale = np.mean(naive_errors)

    # neu scale ~ 0 thi series gan nhu hang so, MASE vo nghia
    # ep NaN de downstream biet bo qua, khong bao gio chia cho 0
    if scale < 1e-9:
        mase = np.nan
    else:
        mase = mae / scale

    # MAPE: ban le co intermittent demand (nhieu ngay ban 0 don vi)
    # chia cho y_true=0 -> inf, vo nghia ve mat thong ke
    # chon nguong 1.0 thay vi 0 vi sales la so nguyen, <1 tuong duong 0
    if np.any(y_true < 1.0):
        mape = np.nan
    else:
        mape = np.mean(np.abs(errors / y_true)) * 100.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MASE": float(mase),
        "MAPE": float(mape),
    }


if __name__ == "__main__":
    # known values tu guideline de verify cong thuc
    y_true = np.array([3.0, 5.0, 2.0])
    y_pred = np.array([4.0, 4.0, 3.0])
    # train du dai de tinh seasonal naive voi season=1 (don gian)
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    m = compute_metrics(y_true, y_pred, y_train, season=1)
    print("=== Known-value test ===")
    print(f"  MAE  = {m['MAE']:.4f}  (expect 1.0)")
    print(f"  RMSE = {m['RMSE']:.4f}  (expect ~1.0)")
    print(f"  MASE = {m['MASE']:.4f}")
    print(f"  MAPE = {m['MAPE']:.2f}%")

    assert abs(m["MAE"] - 1.0) < 1e-9, f"MAE failed: {m['MAE']}"
    assert abs(m["RMSE"] - 1.0) < 1e-9, f"RMSE failed: {m['RMSE']}"
    print("  [PASS] MAE & RMSE match expected values")

    # edge case: series hang so -> MASE = NaN
    y_train_flat = np.full(20, 5.0)
    m2 = compute_metrics(y_true, y_pred, y_train_flat, season=7)
    assert np.isnan(m2["MASE"]), f"MASE should be NaN for flat series, got {m2['MASE']}"
    print("\n=== Flat series test ===")
    print(f"  MASE = {m2['MASE']}  (expect NaN)")
    print("  [PASS] MASE is NaN for constant train series")

    # edge case: y_true co gia tri 0 -> MAPE = NaN
    y_true_zero = np.array([0.0, 5.0, 2.0])
    m3 = compute_metrics(y_true_zero, y_pred, y_train, season=1)
    assert np.isnan(m3["MAPE"]), f"MAPE should be NaN when y_true has zeros, got {m3['MAPE']}"
    print("\n=== Intermittent demand test ===")
    print(f"  MAPE = {m3['MAPE']}  (expect NaN)")
    print("  [PASS] MAPE is NaN when y_true < 1.0")

    print("\nAll tests passed.")
