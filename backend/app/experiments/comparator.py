"""Prompt 082：实验比较。

比较多次 ExperimentRun 的 metrics / parameters / runtime，
输出结构化 CompareResult（含每个指标最优 run、参数差异、耗时排名）。

进阶：当两个 run 用的是**同一份切分**（留存的测试集预测逐行对齐）时，
额外给一份配对差异（paired_delta）—— 把「样本本身难易」的影响抵消掉之后，
剩下的才是两个模型的真实差距，并附带 95% 置信区间判断差得显不显著。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.ml_engine.evaluation import (
    bootstrap_confidence_interval,
    class_codes,
    macro_f1_from_codes,
    roc_auc_fast,
)

logger = logging.getLogger(__name__)

# 指标方向：越高越好 / 越低越好
HIGHER_BETTER = {"accuracy", "precision", "recall", "f1", "roc_auc", "r2", "silhouette"}
LOWER_BETTER = {"mae", "mse", "rmse"}

# 需要做配对差异的指标（其余指标要么非数值，要么同样本量下不值得算）
PAIRED_METRICS = ("accuracy", "f1", "roc_auc", "r2", "rmse", "mae")

# 配对 Bootstrap 的次数：比单指标 CI 少一些，因为这里要跑「每个指标 × 每对」
_BOOTSTRAP_REPEATS = 600


@dataclass
class CompareEntry:
    """单个 run 的比较条目。"""

    run_id: int
    experiment_id: int
    status: str
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    runtime: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "runtime": self.runtime,
        }


@dataclass
class CompareResult:
    """结构化比较结果。"""

    entries: list[CompareEntry] = field(default_factory=list)
    # metric -> run_id（仅统计有方向定义的指标）
    best: dict[str, int] = field(default_factory=dict)
    # param_key -> {run_id: value}（只保留各 run 取值不一致的参数）
    parameter_diff: dict[str, dict[int, Any]] = field(default_factory=dict)
    # 按 runtime 升序的 run_id 排名（None 视为最慢）
    runtime_ranking: list[int] = field(default_factory=list)
    # 两个 run 且切分一致时的配对差异；样本不可比时为 None
    paired_delta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "best": self.best,
            "parameter_diff": self.parameter_diff,
            "runtime_ranking": self.runtime_ranking,
            "paired_delta": self.paired_delta,
        }


class ExperimentComparator:
    """比较多次实验运行。"""

    def compare(self, runs: list[Any]) -> CompareResult:
        """runs: ExperimentRun 列表（属性访问 run_id/experiment_id/status/...）。"""
        entries = [
            CompareEntry(
                run_id=run.id,
                experiment_id=run.experiment_id,
                status=run.status,
                metrics=dict(run.metrics or {}),
                parameters=self._params_of(run),
                runtime=run.runtime,
            )
            for run in runs
        ]
        result = CompareResult(entries=entries)
        result.best = self._best_by_metric(entries)
        result.parameter_diff = self._parameter_diff(entries)
        result.runtime_ranking = self._runtime_ranking(entries)
        # 配对差异只在「两个成功 run + 同一份切分」时计算：
        # 两两配对在 3 个以上 run 时既算不完也看不过来，
        # 而不同切分上的 Δ 混着「样本难易」的差异，并不代表模型差距。
        result.paired_delta = self._paired_delta(runs)
        return result

    # ------------------------------------------------------------------
    def _paired_delta(self, runs: list[Any]) -> dict[str, Any] | None:
        """两个 run 在同一份测试集上的配对差异 + 95% 置信区间。

        **配对**是关键：两次重采样抽的是**同一批样本下标**，样本本身的难易
        因此被抵消，剩下的才是模型之间的差距。独立地各抽一次会把噪声叠进去，
        得到一条宽到无法下结论的区间。

        判定口径：区间跨 0 ⇒ 差异不显著（significant=false）。
        这是「能不能说 A 比 B 好」的依据 —— 0.83 与 0.85 在几百行测试集上
        极可能只是抽样噪声，直接下结论就是把噪声当结论。
        """
        if len(runs) != 2:
            return None
        a, b = runs
        if a.status != "success" or b.status != "success":
            return None
        try:
            pa = dict(a.artifacts or {}).get("test_predictions")
            pb = dict(b.artifacts or {}).get("test_predictions")
            if not isinstance(pa, dict) or not isinstance(pb, dict):
                return None
            if not pa.get("rows") or pa.get("rows") != pb.get("rows"):
                return None
            y_true_a = np.asarray(pa.get("y_true"))
            y_true_b = np.asarray(pb.get("y_true"))
            # 同一份切分 ⇒ 真实标签必须逐行相同。不同就是两套测试集，
            # 此时 Δ 里混着「样本不一样」，配对前提不成立。
            if not np.array_equal(y_true_a, y_true_b):
                return None
            rows = int(y_true_a.size)
            if rows < 30:  # 样本太少，区间宽到没有参考价值
                return None
            task = str(dict(a.artifacts or {}).get("task") or "")
            metrics = self._paired_metric_functions(task, pa, pb, rows)
            if not metrics:
                return None

            seed = int(dict(a.artifacts or {}).get("seed") or 42)
            out: list[dict[str, Any]] = []
            full = np.arange(rows)
            for name, (fa, fb) in metrics.items():
                point = float(fa(full)) - float(fb(full))
                ci = bootstrap_confidence_interval(
                    values_for_indices=y_true_a,
                    metric_fn=lambda idx, _fa=fa, _fb=fb: float(_fa(idx)) - float(_fb(idx)),
                    n_repeats=_BOOTSTRAP_REPEATS,
                    seed=seed,
                )
                lower, upper = ci.get("lower"), ci.get("upper")
                if lower is None or upper is None:
                    continue
                out.append(
                    {
                        "metric": name,
                        "delta": round(point, 6),
                        "lower": round(float(lower), 6),
                        "upper": round(float(upper), 6),
                        # 区间跨 0 ⇒ 不能排除「其实一样好」
                        "significant": not (float(lower) <= 0 <= float(upper)),
                    }
                )
            if not out:
                return None
            return {
                "run_a": a.id,
                "run_b": b.id,
                "rows": rows,
                "n_bootstrap": _BOOTSTRAP_REPEATS,
                "note": (
                    "同一份测试集上同步重采样（配对），"
                    "区间跨 0 表示差异不显著"
                ),
                "metrics": out,
            }
        except Exception as exc:  # noqa: BLE001 - 诊断失败不该让对比整体失败
            logger.info("配对差异不可用: %s", exc)
            return None

    @staticmethod
    def _paired_metric_functions(
        task: str, pa: dict[str, Any], pb: dict[str, Any], rows: int
    ) -> dict[str, tuple[Any, Any]]:
        """给出 {指标名: (run_a 的打分函数, run_b 的打分函数)}（输入都是重采样下标）。"""
        y_true = np.asarray(pa.get("y_true"))
        if task == "classification":
            yt_c, pa_c, k = class_codes(y_true, np.asarray(pa.get("y_pred")))
            yt_c2, pb_c, _ = class_codes(y_true, np.asarray(pb.get("y_pred")))
            if not np.array_equal(yt_c, yt_c2):  # 类别编码不一致 ⇒ 无法配对
                return {}
            fns: dict[str, tuple[Any, Any]] = {
                "accuracy": (
                    lambda i: float((yt_c[i] == pa_c[i]).mean()),
                    lambda i: float((yt_c[i] == pb_c[i]).mean()),
                ),
                "f1": (
                    lambda i: macro_f1_from_codes(yt_c[i], pa_c[i], k),
                    lambda i: macro_f1_from_codes(yt_c[i], pb_c[i], k),
                ),
            }
            proba_a, proba_b = pa.get("y_pos_proba"), pb.get("y_pos_proba")
            if k == 2 and proba_a and proba_b and len(proba_a) == rows == len(proba_b):
                p_a = np.asarray(proba_a, dtype="float64")
                p_b = np.asarray(proba_b, dtype="float64")
                fns["roc_auc"] = (
                    lambda i: roc_auc_fast(yt_c[i], p_a[i]),
                    lambda i: roc_auc_fast(yt_c[i], p_b[i]),
                )
            return fns
        if task == "regression":
            yt = np.asarray(y_true, dtype="float64")
            ya = np.asarray(pa.get("y_pred"), dtype="float64")
            yb = np.asarray(pb.get("y_pred"), dtype="float64")

            def r2(idx: np.ndarray, pred: np.ndarray) -> float:
                yt_i = yt[idx]
                ss_tot = float(((yt_i - yt_i.mean()) ** 2).sum())
                if ss_tot <= 0:
                    return float("nan")
                return 1.0 - float(((yt_i - pred[idx]) ** 2).sum()) / ss_tot

            def rmse(idx: np.ndarray, pred: np.ndarray) -> float:
                return float(np.sqrt(float(((yt[idx] - pred[idx]) ** 2).mean())))

            def mae(idx: np.ndarray, pred: np.ndarray) -> float:
                return float(np.abs(yt[idx] - pred[idx]).mean())

            return {
                "r2": (lambda i: r2(i, ya), lambda i: r2(i, yb)),
                "rmse": (lambda i: rmse(i, ya), lambda i: rmse(i, yb)),
                "mae": (lambda i: mae(i, ya), lambda i: mae(i, yb)),
            }
        return {}

    # ------------------------------------------------------------------
    def _params_of(self, run: Any) -> dict[str, Any]:
        exp = getattr(run, "experiment", None)
        if exp is None:
            # 未加载关系时退回 run 自身字段（服务层保证已加载）
            return dict(getattr(run, "parameters", {}) or {})
        return dict(exp.parameters or {})

    def _best_by_metric(self, entries: list[CompareEntry]) -> dict[str, int]:
        """对有方向定义的指标，求最优 run（success 状态才参与）。"""
        best: dict[str, int] = {}
        best_value: dict[str, float] = {}
        for entry in entries:
            if entry.status != "success":
                continue
            for metric, value in entry.metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if metric not in HIGHER_BETTER and metric not in LOWER_BETTER:
                    continue
                if metric not in best_value:
                    best_value[metric] = float(value)
                    best[metric] = entry.run_id
                    continue
                higher = metric in HIGHER_BETTER
                if (higher and value > best_value[metric]) or (
                    not higher and value < best_value[metric]
                ):
                    best_value[metric] = float(value)
                    best[metric] = entry.run_id
        return best

    def _parameter_diff(self, entries: list[CompareEntry]) -> dict[str, dict[int, Any]]:
        all_keys: set[str] = set()
        for entry in entries:
            all_keys.update(entry.parameters.keys())
        diff: dict[str, dict[int, Any]] = {}
        for key in sorted(all_keys):
            values = {
                entry.run_id: entry.parameters.get(key, "__missing__")
                for entry in entries
            }
            if len({repr(v) for v in values.values()}) > 1:
                diff[key] = values
        return diff

    def _runtime_ranking(self, entries: list[CompareEntry]) -> list[int]:
        timed = [e for e in entries if e.runtime is not None]
        timed.sort(key=lambda e: e.runtime)
        ranked = [e.run_id for e in timed]
        ranked.extend(e.run_id for e in entries if e.runtime is None)
        return ranked
