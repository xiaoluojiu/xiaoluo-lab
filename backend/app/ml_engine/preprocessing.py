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

from app.ml_engine.exceptions import MLEngineException

MISSING_STRATEGIES = {"mean", "median", "mode", "constant"}
ENCODING_METHODS = {"one_hot", "ordinal"}
SCALING_METHODS = {"standard", "min_max"}
_IMPUTE = {"mean": "mean", "median": "median", "mode": "most_frequent", "constant": "constant"}
_TO_FLOAT = FunctionTransformer(np.asarray, kw_args={"dtype": "float64"})


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
        arr = self.pipeline_.transform(self._to_matrix(X_aligned))
        out = pl.DataFrame({n: arr[:, i].tolist() for i, n in enumerate(self.feature_names_out_)})
        if self._ord_cols_:
            out = out.with_columns(pl.col(c).fill_nan(None).cast(pl.Int64) for c in self._ord_cols_)
        return out

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
                steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int64)))
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
