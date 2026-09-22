"""Prompt 082：实验比较。

比较多次 ExperimentRun 的 metrics / parameters / runtime，
输出结构化 CompareResult（含每个指标最优 run、参数差异、耗时排名）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 指标方向：越高越好 / 越低越好
HIGHER_BETTER = {"accuracy", "precision", "recall", "f1", "roc_auc", "r2", "silhouette"}
LOWER_BETTER = {"mae", "mse", "rmse"}


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "best": self.best,
            "parameter_diff": self.parameter_diff,
            "runtime_ranking": self.runtime_ranking,
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
        return result

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
