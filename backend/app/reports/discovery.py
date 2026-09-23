"""报告的建模/实验上下文发现（供 API 与 Agent 工具共用）。

真实缺陷：``POST /reports/generate`` 会自动发现该数据集下最新的 Experiment/Run
并写入报告的「四、建模与评估」章，但 Agent 工具 ``report.generate`` 没有这一步。
于是用户在 AI 实验室里说「选择适合的机器学习模型进行处理，并给出数据分析报告」，
拿到的报告里**完全没有建模章节** —— 章节号一路排到「一 / 二 / 三 / 六」，
恰好暴露了这件事（详见 ``app.reports.numbering``）。

两处入口必须共用同一份发现逻辑，否则会再次分叉。本模块即该唯一实现。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MlContext:
    """报告的建模上下文。

    :param ml: 「建模与评估」章的输入（task / model / target_column / metrics / …），
        无可用运行时为 ``None``。
    :param experiments: 「关联实验」章的列表项。
    :param reason: ``ml`` 为 ``None`` 时的**可操作**原因，会写进报告的未生成章节说明。
    """

    ml: dict[str, Any] | None = None
    experiments: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


def discover_ml_context(
    db: Any,
    dataset_service: Any,
    dataset_id: int,
    *,
    max_experiments: int = 5,
    max_runs: int = 3,
) -> MlContext:
    """发现该数据集下最近的实验与运行，组装报告建模章节所需的事实。

    选取口径（比「取最新一条」更贴近报告读者的期待）：
    - 「关联实验」列出最近 ``max_experiments`` 个实验、每个实验最近 ``max_runs`` 次运行；
    - 「建模与评估」优先取**最近一次成功运行**（``status == "success"`` 或 metrics 非空），
      只有全部失败时才回退到最近一次运行（并把 ``error`` 带进报告，让读者看到失败原因）。
    """
    experiments: list[dict[str, Any]] = []
    try:
        from sqlalchemy import select

        from app.models.experiment import Experiment
        from app.models.experiment_run import ExperimentRun
    except Exception:  # pragma: no cover - 模型缺失属环境问题
        logger.exception("实验模型导入失败，报告将不含建模章节")
        return MlContext(ml=None, experiments=[], reason="实验模块不可用，无法读取建模结果。")

    exp_rows = list(
        db.scalars(
            select(Experiment)
            .where(Experiment.dataset_id == dataset_id)
            .order_by(Experiment.id.desc())
            .limit(max_experiments)
        )
    )
    if not exp_rows:
        return MlContext(
            ml=None,
            experiments=[],
            reason="该数据集下没有任何 Experiment 记录（未执行 ml.train），无法给出建模与评估。",
        )

    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for exp in exp_rows:
        run_rows = list(
            db.scalars(
                select(ExperimentRun)
                .where(ExperimentRun.experiment_id == exp.id)
                .order_by(ExperimentRun.id.desc())
                .limit(max_runs)
            )
        )
        for run in run_rows:
            metrics = dict(run.metrics or {})
            item = {
                "experiment_id": exp.id,
                "run_id": run.id,
                "task": exp.task,
                "model": exp.model,
                "target_column": exp.target_column,
                "status": run.status,
                "runtime": run.runtime,
                "metrics": metrics,
                "error": run.error,
            }
            experiments.append(item)
            candidate = {
                "task": exp.task,
                "model": exp.model,
                "target_column": exp.target_column,
                "metrics": metrics,
                "error": run.error,
                "status": run.status,
                "runtime": run.runtime,
                "experiment_id": exp.id,
                "run_id": run.id,
                "sampling": (run.artifacts or {}).get("sampling"),
            }
            if fallback is None:
                fallback = candidate
            if best is None and (run.status == "success" or metrics) and not run.error:
                best = candidate

    if best is None:
        best = fallback
    if best is None:
        return MlContext(
            ml=None,
            experiments=experiments,
            reason="该数据集下的 Experiment 尚无任何运行记录（Run），没有可写入报告的建模结果。",
        )
    if not best.get("metrics") and best.get("error"):
        best["metrics"] = {}
    return MlContext(ml=best, experiments=experiments, reason=None)
