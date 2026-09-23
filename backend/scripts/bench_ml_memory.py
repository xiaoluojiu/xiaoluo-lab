"""ML 链路内存治理基准（可复现入口）。

在真实大数据集上验证「训得动」，并给出**可对照的内存数字**。
默认取 dataset 9（airlines，10,000,000 行 × 10 列，`data/datasets/9/v000001.parquet`）。

为什么必须是**独立子进程 + 自报 PeakWorkingSetSize**：
同进程内跑两条链路时，第二条的峰值永远 ≥ 第一条（内存不归还给 OS），无法归因。
详见 skill `xiaoluo-lab-env` §9.11。

用法（cwd 任意；务必**分开**跑，每条链路一个进程）::

    PY=.venv/Scripts/python.exe
    $PY scripts/bench_ml_memory.py floor                      # 库地基（要减掉）
    $PY scripts/bench_ml_memory.py cluster                    # 聚类：事故路径
    $PY scripts/bench_ml_memory.py regress [target_column]     # 有监督回归
    $PY scripts/bench_ml_memory.py cluster <parquet>           # 指定文件

「数据相关峰值」 = peak_mb − floor_mb。本机 polars + sklearn 地基约 131 MB。

聚类模式严格复刻 `ExperimentService._execute` 的 clustering 分支
（**这正是用户报错的路径**）：cap 抽样 → one-hot 预处理 → kmeans → 轮廓系数。
它还额外报告「不做任何治理时的反事实规模」，用来对齐那条
``Unable to allocate 57.1 GiB for an array with shape (10000000, 767)`` 报错。

本脚本**不调用任何 LLM API**。
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PARQUET = "data/datasets/9/v000001.parquet"
DEFAULT_TARGET = "DepDelay"


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_mb() -> float:
    """本进程生命周期内的峰值工作集（MB）。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # 取 kernel32 的符号：psapi 只是转发层，部分环境下拿不到
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    if not fn(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 1024 / 1024


def report(payload: dict) -> None:
    payload["peak_mb"] = round(peak_mb(), 1)
    print("__RESULT__" + json.dumps(payload, ensure_ascii=False))


def _resolve(src: str) -> Path:
    path = Path(src)
    return path if path.is_absolute() else BACKEND_ROOT / path


def _cluster(source: str) -> int:
    import polars as pl

    from app.core.config import settings
    from app.ml_engine.evaluation import evaluate_clustering
    from app.ml_engine.preprocessing import (
        build_pipeline,
        cap_training_rows,
        default_preprocessing_config,
        estimate_dense_bytes,
        estimate_output_width,
    )
    from app.ml_engine.registry import MODEL_REGISTRY

    src = _resolve(source)
    t0 = time.perf_counter()
    df = pl.read_parquet(src)
    t_load = time.perf_counter() - t0

    rows, cols = df.height, df.width

    # ---- 反事实：不做任何治理时会是多大（用于对齐线上那条 57.1 GiB 报错）----
    nominal_unique = {
        name: int(df[name].n_unique())
        for name, dtype in df.schema.items()
        if isinstance(dtype, (pl.String, pl.Categorical))
    }
    width_unbounded = (cols - len(nominal_unique)) + sum(nominal_unique.values())
    bytes_unbounded = estimate_dense_bytes(rows, width_unbounded)

    # ---- 治理后（实际路径）----
    X, _, sampling = cap_training_rows(df, None, seed=42)
    used_rows = X.height
    del df  # 与 ExperimentService._execute 一致：抽样后立即释放整表

    pp_cfg = default_preprocessing_config(X)
    pipeline = build_pipeline(pp_cfg, X)
    width_planned = estimate_output_width(X, pipeline.encoding)

    t0 = time.perf_counter()
    Xp = pipeline.fit_transform(X)
    t_pp = time.perf_counter() - t0

    model = MODEL_REGISTRY.create("kmeans", {"n_clusters": 8, "random_state": 42})
    t0 = time.perf_counter()
    model.fit(Xp)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    metrics = evaluate_clustering(Xp, model.labels_)
    t_eval = time.perf_counter() - t0

    report(
        {
            "mode": "cluster",
            "dataset": str(src),
            "rows": rows,
            "cols": cols,
            "nominal_columns": nominal_unique,
            "unbounded": {
                "width": width_unbounded,
                "bytes": bytes_unbounded,
                "gib": round(bytes_unbounded / 1024**3, 1),
            },
            "training": {
                "sampling": sampling,
                "used_rows": used_rows,
                "planned_width": width_planned,
                "actual_width": Xp.width,
                "used_matrix_mib": round(estimate_dense_bytes(Xp.height, Xp.width) / 1024**2, 1),
                "onehot_max_categories": settings.ML_ONEHOT_MAX_CATEGORIES,
                "max_train_rows": settings.ML_MAX_TRAIN_ROWS,
                "max_dense_gib": round(settings.ML_MAX_DENSE_BYTES / 1024**3, 2),
            },
            "metrics": metrics,
            "timing_s": {
                "load": round(t_load, 2),
                "preprocess": round(t_pp, 2),
                "fit": round(t_fit, 2),
                "evaluate": round(t_eval, 2),
            },
        }
    )
    return 0


def _regress(source: str, target: str) -> int:
    """有监督回归路径：抽样 → 切分 → 预处理 → 拟合 → 评估。"""
    import polars as pl

    from app.core.config import settings
    from app.ml_engine.evaluation import evaluate_regression
    from app.ml_engine.preprocessing import (
        PreprocessingPipeline,
        build_pipeline,
        cap_training_rows,
        default_preprocessing_config,
    )
    from app.ml_engine.registry import MODEL_REGISTRY

    src = _resolve(source)
    t0 = time.perf_counter()
    df = pl.read_parquet(src)
    t_load = time.perf_counter() - t0

    if target not in df.columns:
        raise SystemExit(f"目标列 {target!r} 不存在；可选：{df.columns}")

    raw_rows, raw_cols = df.height, df.width - 1
    X_full, y_full = df.drop(target), df[target]

    # 与 ExperimentService._execute 同序：先清目标列空值，再 cap，再切分/预处理
    keep = y_full.is_not_null()
    X_clean, y_clean = X_full.filter(keep), y_full.filter(keep)

    X, y, sampling = cap_training_rows(X_clean, y_clean, seed=42)
    del df, X_full, X_clean, y_full, y_clean  # 抽样后立即释放整表

    X_train, X_test, y_train, y_test = PreprocessingPipeline.train_test_split(
        X, y, test_size=0.2, seed=42
    )
    pipeline = build_pipeline(default_preprocessing_config(X), X_train)

    t0 = time.perf_counter()
    Xtr = pipeline.fit_transform(X_train)
    Xte = pipeline.transform(X_test)
    t_pp = time.perf_counter() - t0

    model = MODEL_REGISTRY.create(
        "random_forest_regressor", {"n_estimators": 30, "random_state": 42}
    )
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(Xte)
    metrics = evaluate_regression(y_test, y_pred)
    t_eval = time.perf_counter() - t0

    report(
        {
            "mode": "regress",
            "dataset": str(src),
            "target": target,
            "raw_rows": raw_rows,
            "raw_feature_cols": raw_cols,
            "training": {
                "sampling": sampling,
                "train_rows": Xtr.height,
                "test_rows": Xte.height,
                "width": Xtr.width,
                "max_train_rows": settings.ML_MAX_TRAIN_ROWS,
            },
            "metrics": metrics,
            "timing_s": {
                "load": round(t_load, 2),
                "preprocess": round(t_pp, 2),
                "fit": round(t_fit, 2),
                "predict+eval": round(t_eval, 2),
            },
        }
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    mode = argv[0] if argv else "cluster"
    source = argv[1] if len(argv) > 1 else DEFAULT_PARQUET

    if mode == "floor":
        import polars  # noqa: F401
        import sklearn  # noqa: F401
        report({"mode": "floor"})
        return 0
    if mode == "cluster":
        return _cluster(source)
    if mode == "regress":
        target = argv[2] if len(argv) > 2 else DEFAULT_TARGET
        return _regress(source, target)

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
