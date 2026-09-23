"""Quality 层 · 标准化子模块。

放这里而不是 ``app/analysis.py`` 里的原因：``analysis.py`` 已是一个 68 KB 的单体模块，
再往里塞「字段语义」与「方法适配」两套机制只会让它更难维护。本包承载 Quality 层
**判定策略** 部分（字段语义 + 方法适配），``app/analysis.py`` 继续负责具体的
统计计算与 QualityIssue 产出，两者通过本包的标准类型对接。

* :mod:`app.quality.semantics` —— 字段语义标准（第二层）
* :mod:`app.quality.outlier_strategy` —— 异常值方法适配（第二层）
"""

from __future__ import annotations

from app.quality.semantics import (
    BusinessRule,
    ColumnRole,
    ColumnSemantics,
    DistributionShape,
    ValueDomain,
    column_tokens,
    infer_semantics,
)

__all__ = [
    "BusinessRule",
    "ColumnRole",
    "ColumnSemantics",
    "DistributionShape",
    "ValueDomain",
    "column_tokens",
    "infer_semantics",
]
