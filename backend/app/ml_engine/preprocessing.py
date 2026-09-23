"""机器学习预处理 Pipeline（sklearn 原生组件实现，防数据泄漏，Polars 进出）。"""

from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as _sk_train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

from app.core.config import settings
from app.ml_engine.exceptions import MLEngineException

MISSING_STRATEGIES = {"mean", "median", "mode", "constant"}
ENCODING_METHODS = {"one_hot", "ordinal"}
SCALING_METHODS = {"standard", "min_max"}
_IMPUTE = {"mean": "mean", "median": "median", "mode": "most_frequent", "constant": "constant"}
_TO_FLOAT = FunctionTransformer(np.asarray, kw_args={"dtype": "float64"})

# 每个 float64/int64 元素占 8 字节。所有内存估算都基于它。
_ITEMSIZE = 8


# =========================================================
# 内存治理：配置读取 + 稠密规模估算
# =========================================================
#
# 背景：sklearn 的估计器几乎都要求**稠密** numpy 矩阵，而本模块的对外契约是
# 返回 pl.DataFrame（无法承载稀疏）。于是「one-hot 高基数列 + 大数据集」必然
# 撞内存。实测事故：10,000,000 行 × 767 列 int64 = 57.1 GiB，直接
# numpy._core._exceptions._ArrayMemoryError。
#
# 对策分三层，缺一不可：
#   ① 抽样（cap_training_rows）：让大数据集默认能训得动，且**绝不静默**；
#   ② 基数控制（ML_ONEHOT_MAX_CATEGORIES）：从源头压住列数膨胀；
#   ③ 规模预检（_assert_dense_fits）：真装不下时给出可操作的中文报错，
#      而不是让 numpy 抛一个看不懂的 GiB 数字。
# =========================================================


def _max_dense_bytes() -> int:
    """稠密特征矩阵的内存预算（字节）。<=0 表示不设限。"""
    try:
        return int(settings.ML_MAX_DENSE_BYTES)
    except Exception:  # pragma: no cover - 配置层异常时按默认兜底
        return 2 * 1024 * 1024 * 1024


def _max_train_rows() -> int:
    """单次训练的最大样本数。<=0 表示不抽样。"""
    try:
        return int(settings.ML_MAX_TRAIN_ROWS)
    except Exception:  # pragma: no cover
        return 200_000


def _onehot_max_categories() -> int | None:
    """one-hot 单列最大类别数；<=0 表示不合并（保留旧行为）。"""
    try:
        cap = int(settings.ML_ONEHOT_MAX_CATEGORIES)
    except Exception:  # pragma: no cover
        return 50
    return cap if cap > 0 else None


def _max_silhouette_samples() -> int:
    """轮廓系数的采样上限（该指标复杂度 O(n²)）。<=0 表示不采样。"""
    try:
        return int(settings.ML_MAX_SILHOUETTE_SAMPLES)
    except Exception:  # pragma: no cover
        return 20_000


def estimate_dense_bytes(n_rows: int, n_cols: int) -> int:
    """稠密 float64/int64 矩阵的字节数。"""
    return max(0, int(n_rows)) * max(0, int(n_cols)) * _ITEMSIZE


def _human_bytes(num: int) -> str:
    """字节数的可读表示，自动选单位。

    不要一律输出 GiB —— 预算 512 MB 时 ``{x / 1024**3:.1f}`` 会显示成
    「0.5 GiB」勉强能看，而 64 MB 会显示成「0.1 GiB」、1 MB 会显示成
    「0.0 GiB」。这条提示存在的唯一意义就是让用户知道「差多少」，
    显示成 0.0 等于没提示。
    """
    value = float(max(0, int(num)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover - 上一步已返回


def _assert_dense_fits(n_rows: int, n_cols: int, *, stage: str, hint: str) -> None:
    """规模预检：装不下就**明确拒绝**，并告诉调用方怎么改。"""
    budget = _max_dense_bytes()
    if budget <= 0:
        return
    needed = estimate_dense_bytes(n_rows, n_cols)
    if needed <= budget:
        return
    raise MLEngineException(
        f"{stage}需要稠密矩阵 {n_rows:,} 行 × {n_cols:,} 列 ≈ "
        f"{_human_bytes(needed)}，超过内存预算 {_human_bytes(budget)}。{hint}",
        details={
            "stage": stage,
            "rows": int(n_rows),
            "columns": int(n_cols),
            "needed_bytes": needed,
            "budget_bytes": budget,
        },
    )


def estimate_output_width(X: pl.DataFrame, encoding: dict[str, Any] | None) -> int:
    """估算预处理输出的列数（one-hot 会按类别数展开）。

    用于在**真正物化之前**判断会不会爆内存。刻意取偏大的估计：
    宁可在边界上多拒绝一次，也不要先分配 57 GiB 再崩。
    """
    one_hot_cols: set[str] = set()
    if isinstance(encoding, dict) and encoding.get("method") == "one_hot":
        one_hot_cols = set(encoding.get("columns") or [])

    cap = _onehot_max_categories()
    width = 0
    for name in X.columns:
        if name in one_hot_cols and _kind(X.schema[name]) == "cat":
            unique = int(X[name].n_unique())
            # 合并低频类别后每列最多 cap 个输出（含「其他」列）
            width += min(unique, cap) if cap else unique
        else:
            width += 1
    return width


def cap_training_rows(
    X: pl.DataFrame,
    y: pl.Series | None = None,
    *,
    seed: int | None = None,
) -> tuple[pl.DataFrame, pl.Series | None, dict[str, Any]]:
    """把训练集行数压到 ``ML_MAX_TRAIN_ROWS`` 以内（随机抽样）。

    返回 ``(X, y, info)``，``info`` 形如::

        {"sampled": True, "original_rows": 10_000_000, "used_rows": 200_000}

    **抽样是有损的**，所以调用方必须把 ``info`` 透出到结果里 ——
    静默改变训练集规模会让指标无法解释（用户会以为模型是在全量数据上训的）。
    """
    limit = _max_train_rows()
    total = X.height
    if limit <= 0 or total <= limit:
        return X, y, {"sampled": False, "original_rows": total, "used_rows": total}

    rng = np.random.default_rng(seed)
    picked = rng.choice(total, size=limit, replace=False)
    picked.sort()
    idx = picked.tolist()
    X_small = X[idx]
    y_small = y[idx] if y is not None else None
    return X_small, y_small, {
        "sampled": True,
        "original_rows": total,
        "used_rows": limit,
        "sample_rate": round(limit / total, 6),
    }



def _is_temporal(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.Datetime, pl.Date, pl.Time))


def default_preprocessing_config(X: pl.DataFrame) -> dict[str, Any]:
    """按数据画像给出默认预处理配置（ExperimentService / Workflow 共用）。

    只在确实需要时才生成对应配置：有缺失 -> 填补；有类别列 -> 编码；有数值列 -> 标准化。
    时间列（Date/Datetime/Time）与布尔列一并按数值列处理：它们会被转成时间戳 / 0-1
    后参与填补与标准化。此前它们被判为「原样透传」，会带着 datetime64 直接进模型，
    导致任何含日期列的数据集训练必失败（训练特征必须为数值列）。
    """
    numeric = [
        c
        for c, d in X.schema.items()
        if d.is_numeric() or _is_temporal(d) or d == pl.Boolean
    ]
    categorical = [
        c for c, d in X.schema.items() if isinstance(d, (pl.String, pl.Categorical))
    ]
    missing_cols = [c for c in X.columns if X[c].null_count() > 0]
    cfg: dict[str, Any] = {}
    if missing_cols:
        num_missing = [c for c in missing_cols if X.schema[c].is_numeric()]
        cfg["missing"] = {
            "strategy": "median" if num_missing else "mode",
            "columns": missing_cols,
        }
    if categorical:
        cfg["encoding"] = {"method": "one_hot", "columns": categorical}
    if numeric:
        cfg["scaling"] = {"method": "standard", "columns": numeric}
    return cfg


def build_pipeline(config: dict[str, Any] | None, X: pl.DataFrame) -> PreprocessingPipeline:
    """按配置构造预处理管道；配置为空时自动套用默认配置。

    部分配置兜底（两条，都是「漏配即报错」的高频坑）：

    1. **缺失填补**：用户只给 scaling 却漏配 missing 时，含空值的列会在标准化阶段
       因 NaN 报错。这里检测「未被缺失配置覆盖的空值列」自动补一条填补。
    2. **类别编码**：用户只给 scaling 却漏配 encoding 时，字符串列会被 ColumnTransformer
       按 passthrough 原样透传，最后在 `训练特征必须为数值列` 处失败。这里检测
       「未被编码配置覆盖的类别列」自动补 one_hot。

    两条兜底都只做「补漏」，用户显式给出的策略一律保留。
    """
    cfg = config if isinstance(config, dict) and config else default_preprocessing_config(X)

    # 兜底 1：空值列必须被缺失配置覆盖
    missing_cfg = cfg.get("missing")
    covered = set(missing_cfg.get("columns") or []) if isinstance(missing_cfg, dict) else set()
    uncovered = [c for c in X.columns if X[c].null_count() > 0 and c not in covered]
    if uncovered:
        numeric_uncovered = [c for c in uncovered if _kind(X.schema[c]) == "num"]
        auto = {
            "strategy": "median" if numeric_uncovered else "mode",
            "columns": uncovered,
        }
        if isinstance(missing_cfg, dict):
            # 合并：保留用户原有策略，只把漏掉的列补进去
            missing_cfg = {**missing_cfg, "columns": [*(missing_cfg.get("columns") or []), *uncovered]}
        else:
            missing_cfg = auto
        cfg = {**cfg, "missing": missing_cfg}

    # 兜底 2：类别列必须被编码配置覆盖（否则字符串列透传进模型必失败）
    encoding_cfg = cfg.get("encoding")
    enc_covered = set(encoding_cfg.get("columns") or []) if isinstance(encoding_cfg, dict) else set()
    uncovered_cat = [
        c for c in X.columns
        if _kind(X.schema[c]) == "cat" and c not in enc_covered
    ]
    if uncovered_cat:
        if isinstance(encoding_cfg, dict):
            encoding_cfg = {
                **encoding_cfg,
                "columns": [*(encoding_cfg.get("columns") or []), *uncovered_cat],
            }
        else:
            encoding_cfg = {"method": "one_hot", "columns": uncovered_cat}
        cfg = {**cfg, "encoding": encoding_cfg}

    return PreprocessingPipeline(
        missing=cfg.get("missing"),
        encoding=cfg.get("encoding"),
        scaling=cfg.get("scaling"),
    )


def _kind(dtype: pl.DataType) -> str:
    # 时间列与布尔列同样按数值列处理（见 default_preprocessing_config 的说明）
    if dtype.is_numeric() or _is_temporal(dtype) or dtype == pl.Boolean:
        return "num"
    return "cat" if isinstance(dtype, (pl.String, pl.Categorical)) else "pass"


def _imputer(strategy: str, fill_value: Any) -> SimpleImputer:
    return SimpleImputer(strategy=strategy, fill_value=fill_value, keep_empty_features=True)


def _col_array(s: pl.Series, fill_cat: bool) -> np.ndarray:
    if _kind(s.dtype) == "num":
        # 时间列 -> 时间戳；布尔列 -> 0/1；其余数值列直接取浮点
        if _is_temporal(s.dtype) or s.dtype == pl.Boolean:
            return s.cast(pl.Int64).to_numpy().astype("float64")
        return s.to_numpy().astype("float64")
    if _kind(s.dtype) == "cat":
        values = s.cast(pl.String).to_list()
        values = [np.nan if v is None else v for v in values] if fill_cat else values
        return np.array(values, dtype=object)
    return s.to_numpy()


class PreprocessingPipeline:
    """missing -> encoding -> scaling 预处理管道（配置均可选）。"""

    def __init__(
        self,
        missing: dict[str, Any] | None = None,
        encoding: dict[str, Any] | None = None,
        scaling: dict[str, Any] | None = None,
    ) -> None:
        self._validate_config(missing, encoding, scaling)
        self.missing, self.encoding, self.scaling = missing, encoding, scaling
        self.pipeline_: ColumnTransformer | None = None
        self.feature_names_out_: list[str] = []
        self._ord_cols_: list[str] = []
        self._columns_: list[str] = []
        self.fitted_ = False

    def fit(self, X: pl.DataFrame) -> PreprocessingPipeline:
        self._columns_ = list(X.columns)
        self._guard_dense(X, stage="预处理")
        self.pipeline_ = self._build(X).fit(self._to_matrix(X))
        self._collect_names()
        self.fitted_ = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted_ or self.pipeline_ is None:
            raise MLEngineException("Pipeline 尚未 fit，请先在训练集上调用 fit")
        missing = [c for c in self._columns_ if c not in X.columns]
        if missing:
            raise MLEngineException(f"transform 缺失训练列: {missing}", details={"missing": missing})
        X_aligned = X.select(self._columns_)
        # 预测/转换同样要预检：在 1000 万行上 transform 一样会爆。
        self._guard_dense(X_aligned, stage="转换")
        arr = self.pipeline_.transform(self._to_matrix(X_aligned))
        out = self._matrix_to_frame(arr)
        if self._ord_cols_:
            out = out.with_columns(pl.col(c).fill_nan(None).cast(pl.Int64) for c in self._ord_cols_)
        return out

    def _guard_dense(self, X: pl.DataFrame, *, stage: str) -> None:
        """在物化之前做规模预检（输入 + one-hot 展开后的输出）。"""
        hint = (
            "可采取：① 调小 ML_MAX_TRAIN_ROWS 缩小训练集；"
            "② 把高基数列的 encoding.method 改为 'ordinal'（列数不膨胀）；"
            "③ 调小 ML_ONEHOT_MAX_CATEGORIES 合并低频类别；"
            "④ 若确信内存充足，调大 ML_MAX_DENSE_BYTES。"
        )
        _assert_dense_fits(X.height, X.width, stage=f"{stage}输入", hint=hint)
        _assert_dense_fits(
            X.height,
            estimate_output_width(X, self.encoding),
            stage=f"{stage}输出（含 one-hot 展开）",
            hint=hint,
        )

    def _matrix_to_frame(self, arr: np.ndarray) -> pl.DataFrame:
        """把 ColumnTransformer 的稠密输出转成 DataFrame。

        不要写成 ``{name: arr[:, i].tolist()}`` —— 那是 **O(行数 × 列数) 个
        Python 对象**：767 列 × 1000 万行会造出上百 GB 的临时对象，
        比矩阵本身更致命（矩阵 57 GiB，装箱后是它的数倍）。
        ``pl.from_numpy`` 直接按缓冲区构造，不做逐元素装箱。
        """
        if arr.ndim != 2:  # pragma: no cover - ColumnTransformer 恒返回 2D
            raise MLEngineException(f"预处理输出维度异常：{arr.shape}")
        names = list(self.feature_names_out_)
        if not names:
            return pl.DataFrame()
        if arr.shape[1] != len(names):  # pragma: no cover - 列名与矩阵理应一致
            raise MLEngineException(
                "预处理输出的列数与列名不一致",
                details={"matrix_columns": int(arr.shape[1]), "names": len(names)},
            )
        return pl.from_numpy(np.ascontiguousarray(arr), schema=names)

    def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return self.fit(X).transform(X)

    @staticmethod
    def train_test_split(
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        test_size: float = 0.2,
        seed: int | None = None,
        shuffle: bool = True,
        stratify: pl.Series | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series | None, pl.Series | None]:
        if not 0 < test_size < 1:
            raise MLEngineException("test_size 必须在 (0, 1) 区间")
        if y is not None and y.len() != X.height:
            raise MLEngineException("X 与 y 行数不一致")
        stratify_arr = stratify.to_numpy() if stratify is not None else None
        try:
            train_idx, test_idx = _sk_train_test_split(
                np.arange(X.height),
                test_size=test_size,
                random_state=seed,
                shuffle=shuffle,
                stratify=stratify_arr,
            )
        except ValueError as exc:
            raise MLEngineException(f"train_test_split 失败：{exc}") from exc
        X_train, X_test = X.gather(list(train_idx)), X.gather(list(test_idx))
        y_train = y.gather(list(train_idx)) if y is not None else None
        y_test = y.gather(list(test_idx)) if y is not None else None
        return X_train, X_test, y_train, y_test

    def _build(self, X: pl.DataFrame) -> ColumnTransformer:
        strategy = _IMPUTE.get(self.missing["strategy"]) if self.missing else None
        fill_value = self.missing.get("value", 0) if self.missing else 0
        kinds = [_kind(X.schema[c]) for c in X.columns]
        runs = [(k, [i for i, _ in g]) for k, g in groupby(enumerate(kinds), key=lambda t: t[1])]
        transformers = [
            (
                f"{k}{idxs[0]}",
                "passthrough" if k == "pass" else self._make_pipe(k, strategy, fill_value),
                idxs,
            )
            for k, idxs in runs
        ]
        return ColumnTransformer(transformers, sparse_threshold=0.0)

    def _make_pipe(self, kind: str, strategy: str | None, fill_value: Any) -> Any:
        steps: list[tuple[str, Any]] = [("cast", _TO_FLOAT)] if kind == "num" else []
        if strategy and (kind == "num" or strategy in ("most_frequent", "constant")):
            steps.append(("impute", _imputer(strategy, fill_value)))
        if kind == "num" and self.scaling:
            scaler = StandardScaler() if self.scaling["method"] == "standard" else MinMaxScaler()
            steps.append(("scale", scaler))
        if kind == "cat" and self.encoding:
            if self.encoding["method"] == "one_hot":
                # max_categories：把低频类别合并成一个「其他」列。
                # Origin / Dest 这类 300 量级的高基数列，不合并会让特征数从 10
                # 涨到 767（本次事故的直接原因），既是内存问题也是统计问题
                # （metadata.py 早已把「高基数用 one_hot」标为风险，只是代码没兜住）。
                steps.append(
                    (
                        "encode",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            dtype=np.int64,
                            max_categories=_onehot_max_categories(),
                        ),
                    )
                )
            else:
                steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)))
        return Pipeline(steps) if steps else "passthrough"

    def _to_matrix(self, X: pl.DataFrame) -> np.ndarray:
        fill_cat = self.encoding is not None or (
            self.missing is not None and self.missing["strategy"] in ("mode", "constant")
        )
        return np.column_stack([_col_array(X[c], fill_cat) for c in X.columns])

    def _collect_names(self) -> None:
        """收集 ColumnTransformer 输出字段；修复多分类列共享 OneHotEncoder 时的列名错位。"""
        self.feature_names_out_, self._ord_cols_ = [], []
        if self.pipeline_ is None:
            return
        for name, trans, idxs in self.pipeline_.transformers_:
            cols = [self._columns_[i] for i in idxs]
            enc = None if name.startswith("pass") else getattr(trans, "named_steps", {}).get("encode")
            if enc is None:
                self.feature_names_out_ += cols
            elif isinstance(enc, OneHotEncoder):
                for encoded_name in enc.get_feature_names_out(cols):
                    mapped = encoded_name
                    for col in cols:
                        prefix = f"{col}_"
                        if encoded_name.startswith(prefix):
                            mapped = encoded_name.replace(prefix, f"{col}=", 1)
                            break
                    self.feature_names_out_.append(mapped)
            else:
                self.feature_names_out_ += cols
                self._ord_cols_ += cols

    @staticmethod
    def _validate_config(missing: dict | None, encoding: dict | None, scaling: dict | None) -> None:
        if missing is not None and missing.get("strategy") not in MISSING_STRATEGIES:
            raise MLEngineException(f"missing.strategy 必须是 {sorted(MISSING_STRATEGIES)} 之一")
        if encoding is not None and encoding.get("method") not in ENCODING_METHODS:
            raise MLEngineException(f"encoding.method 必须是 {sorted(ENCODING_METHODS)} 之一")
        if scaling is not None and scaling.get("method") not in SCALING_METHODS:
            raise MLEngineException(f"scaling.method 必须是 {sorted(SCALING_METHODS)} 之一")

    @property
    def report(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted_,
            "features_out": list(self.feature_names_out_),
            "config": {"missing": self.missing, "encoding": self.encoding, "scaling": self.scaling},
        }
