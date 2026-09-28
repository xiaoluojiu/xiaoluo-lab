"""交叉验证与超参搜索：预处理管道与模型作为整体在折内拟合，杜绝泄漏。

为什么必须整管进折：
    填补用的均值、标准化用的方差、one-hot 的类别表，全都是**从数据里学出来的**。
    先在整份数据上 fit 预处理、再拿去交叉验证，等于让每一折的验证集提前
    看见了全量数据的统计量 —— 指标会系统性偏乐观，而这份乐观在真实数据上
    一分钱都不值。所以这里把「原始矩阵 + 未拟合 ColumnTransformer + 未拟合
    estimator」打包成一条 sklearn ``Pipeline``，由 ``cross_validate`` 在每折
    的训练部分上重新 fit 一遍（真正的折内拟合）。

对外只有两个入口：
    - :func:`cv_evaluate`：固定参数下的 K 折评估（mean / std / per_fold）；
    - :func:`random_search`：``RandomizedSearchCV`` 走同一套折内拟合流程。

两者都只依赖 sklearn 原生组件，不引入新依赖。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline

from app.ml_engine.exceptions import MLEngineException

# 各任务在交叉验证里同步计算的指标（键为对外口径，值为 sklearn scorer 名）。
# 回归的 mae / mse 在 sklearn 里是「负的」（越大越好），下方统一翻回正号。
_SCORING = {
    "classification": {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    },
    "regression": {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    },
}

# 超参搜索的主指标：分类看 f1_macro（类别不均衡时比 accuracy 诚实），回归看 r2。
_PRIMARY = {"classification": "f1_macro", "regression": "r2"}

#: 参与监督任务的模型才有搜索空间（聚类/降维不在本轮范围）。
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "random_forest_classifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    },
    "hist_gradient_boosting_classifier": {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.5, 1.0],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000, 2000],
    },
    "knn_classifier": {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}


def _splitter(task: str, folds: int, seed: int):
    """分类用分层 K 折（每折类别比例与整体一致），回归用普通 K 折。"""
    if task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _check_task(task: str) -> None:
    if task not in _SCORING:
        raise MLEngineException(
            f"交叉验证仅支持 classification / regression，当前任务 {task!r}"
        )


def cv_evaluate(
    pipeline: Any,
    estimator: Any,
    X_raw: pl.DataFrame,
    y: pl.Series,
    *,
    task: str,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """K 折交叉验证：预处理与模型作为整体在每折训练集内拟合。

    参数
    ----
    pipeline:
        **未拟合**的 :class:`PreprocessingPipeline`（内部只取它的配置搭骨架）；
    estimator:
        **未拟合**的 sklearn 估计器（``ModelAdapter.new_estimator()``）；
    X_raw:
        原始 polars 帧（未被任何预处理加工过）。

    返回 ``{folds, mean, std, per_fold, seed, note}``：``mean`` / ``std``
    是指标在各折上的均值与样本标准差（ddof=1），``per_fold`` 给出每一折的
    原始数值 —— 只给均值会让「某一折特别差」这种信息彻底消失，而它往往
    正是数据里有问题子集的信号。
    """
    _check_task(task)
    if y is None or X_raw.height != y.len():
        raise MLEngineException("X 与 y 行数不一致，无法做交叉验证")
    folds = max(2, min(int(folds), X_raw.height))

    X_mat = pipeline.as_matrix(X_raw)
    y_np = np.asarray(y.to_numpy())
    # 关键：这里的 preprocess 必须是**未拟合**的骨架（pipeline.build），
    # 用训练流程里已经 fit 过的 pipeline_ 就是把泄漏写进每一折。
    pipe = SkPipeline([
        ("preprocess", pipeline.build(X_raw)),
        ("model", clone(estimator)),
    ])
    out = cross_validate(
        pipe,
        X_mat,
        y_np,
        cv=_splitter(task, folds, seed),
        scoring=_SCORING[task],
        n_jobs=-1,
        error_score="raise",
    )

    per_fold: list[dict[str, Any]] = []
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key, values in out.items():
        if not key.startswith("test_"):
            continue
        name = key[len("test_"):]
        arr = np.asarray(values, dtype=float)
        if task == "regression" and name in ("mae", "mse"):
            # sklearn 的误差类 scorer 返回负值（统一成「越大越好」）
            arr = -arr
        per_fold.append({"metric": name, "values": [float(x) for x in arr]})
        means[name] = float(arr.mean())
        stds[name] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if task == "regression" and "mse" in means:
        # RMSE = sqrt(MSE)：量纲回到目标列本身，比 MSE 好解释
        means["rmse"] = float(np.sqrt(max(0.0, means["mse"])))
    return {
        "folds": folds,
        "mean": means,
        "std": stds,
        "per_fold": per_fold,
        "seed": seed,
        "note": (
            f"{folds} 折{'分层' if task == 'classification' else ''}CV，"
            "预处理在每折训练集内拟合"
        ),
    }


def _grid_size(space: dict[str, Any]) -> int | None:
    """候选网格的规模；含连续分布（无法计数）时返回 None。"""
    total = 1
    for values in space.values():
        if isinstance(values, (list, tuple)):
            total *= len(values)
        else:
            return None
    return total


def random_search(
    pipeline: Any,
    model_name: str,
    X_raw: pl.DataFrame,
    y: pl.Series,
    *,
    task: str,
    folds: int = 5,
    seed: int = 42,
    n_iter: int = 20,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """随机搜索超参数：每组候选都在同一套折内拟合流程上评估。

    与 :func:`cv_evaluate` 共用 ``_splitter`` 与「整管进折」的构造方式，
    所以搜出来的分数和手工 CV 的分数是同一个口径 —— 两套口径各自实现
    迟早会出现「搜索说 0.92、跑一遍只有 0.87」这种无法解释的落差。

    ``params`` 为用户的既有参数（含注入的 seed），搜索空间里的键会覆盖它；
    搜索空间之外的参数保持不变。
    """
    _check_task(task)
    from app.ml_engine.registry import MODEL_REGISTRY

    if model_name not in SEARCH_SPACES:
        raise MLEngineException(
            f"模型 {model_name!r} 暂无内置搜索空间",
            details={"available": sorted(SEARCH_SPACES)},
        )
    if y is None or X_raw.height != y.len():
        raise MLEngineException("X 与 y 行数不一致，无法做超参搜索")

    X_mat = pipeline.as_matrix(X_raw)
    adapter_cls = MODEL_REGISTRY.get(model_name)
    # 未拟合的 estimator：RandomizedSearchCV 自己会 clone + fit
    estimator = adapter_cls(dict(params or {})).new_estimator()
    pipe = SkPipeline([
        ("preprocess", pipeline.build(X_raw)),
        ("model", estimator),
    ])
    space = {f"model__{k}": v for k, v in SEARCH_SPACES[model_name].items()}
    # 网格比 n_iter 还小的时候 sklearn 会警告并退化成全网格穷举；
    # 直接收敛 n_iter，把「其实全搜了一遍」这件事写进返回值而不是留在日志里。
    grid = _grid_size(space)
    if grid is not None and grid < n_iter:
        n_iter = grid

    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        cv=_splitter(task, folds, seed),
        scoring=_PRIMARY[task],
        n_jobs=-1,
        random_state=seed,
        return_train_score=True,
        refit=True,
        error_score="raise",
    )
    search.fit(X_mat, np.asarray(y.to_numpy()))

    results = search.cv_results_
    best = {k.replace("model__", ""): _plain(v) for k, v in search.best_params_.items()}
    top: list[dict[str, Any]] = []
    for idx in np.argsort(results["rank_test_score"])[:10]:
        top.append(
            {
                "rank": int(results["rank_test_score"][idx]),
                "mean_score": float(results["mean_test_score"][idx]),
                "std_score": float(results["std_test_score"][idx]),
                "params": {
                    k.replace("model__", ""): _plain(v)
                    for k, v in results["params"][idx].items()
                },
            }
        )
    return {
        "model": model_name,
        "task": task,
        "n_iter": int(n_iter),
        "folds": int(folds),
        "metric": _PRIMARY[task],
        "best_params": best,
        "best_score": float(search.best_score_),
        "top_results": sorted(top, key=lambda x: x["rank"]),
        "note": (
            f"{int(n_iter)} 组候选 × {int(folds)} 折"
            f"{'分层' if task == 'classification' else ''}CV，"
            "预处理在每折训练集内拟合；结果不自动写回实验配置"
        ),
    }


def _plain(value: Any) -> Any:
    """numpy 标量 / None -> JSON 安全的原生类型。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - 非数值标量直接回落
            return value
    return value
