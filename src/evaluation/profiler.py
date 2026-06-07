from __future__ import annotations

import os
import time
from types import TracebackType

import psutil


class ResourceProfiler:
    # context manager do tai nguyen cua chinh process hien tai
    # dung psutil.Process thay vi system-wide de chinh xac hon
    # peak_ram tinh bang hieu RSS luc ra vs luc vao, clip >= 0

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self.train_time_s: float = 0.0
        self.peak_ram_mb: float = 0.0
        self.cpu_percent: float = 0.0
        self._t0: float = 0.0
        self._ram0: float = 0.0

    def __enter__(self) -> ResourceProfiler:
        # reset cpu counter truoc khi bat dau do
        # neu khong goi truoc thi lan dau cpu_percent() tra ve 0.0 hoac gia tri cu
        self._proc.cpu_percent(interval=None)
        self._ram0 = self._proc.memory_info().rss / (1024 ** 2)
        self._t0 = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.train_time_s = time.time() - self._t0
        ram_now = self._proc.memory_info().rss / (1024 ** 2)
        # clip ve 0 vi RSS co the giam neu GC chay giua chung
        self.peak_ram_mb = max(ram_now - self._ram0, 0.0)
        self.cpu_percent = self._proc.cpu_percent(interval=None)

    def to_dict(self) -> dict[str, float]:
        return {
            "train_time_s": round(self.train_time_s, 6),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "cpu_percent": round(self.cpu_percent, 1),
        }


if __name__ == "__main__":
    import numpy as np

    print("=== ResourceProfiler demo ===")
    with ResourceProfiler() as prof:
        # tao mang lon de RAM tang len ro rang
        _big = np.random.randn(5_000_000)
        _ = np.sort(_big)

    stats = prof.to_dict()
    print(f"  train_time_s : {stats['train_time_s']}")
    print(f"  peak_ram_mb  : {stats['peak_ram_mb']}")
    print(f"  cpu_percent  : {stats['cpu_percent']}")

    assert stats["train_time_s"] > 0, "time should be positive"
    print("\n  [PASS] Profiler works correctly")
