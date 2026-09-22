"""Data Engine 统一接口。

DataEngineBase 只定义契约（load / preview / schema / profile），
具体实现见 service.DataEngineService。

约束：本模块（及整个 data_engine 包）不得依赖 FastAPI，
必须可以脱离 Web API 单独运行（脚本 / 测试 / 未来 Worker 均可直接使用）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from app.data_engine.loaders import LoadedTable


class DataEngineBase(ABC):
    """Data Engine 抽象接口。

    - load:     从文件 / 字节流加载为 DataFrame
    - preview:  分页预览（绝不把整个数据集发给前端）
    - schema:   输出 JSON 可序列化的 Schema 分析
    - profile:  输出 JSON 可序列化的统计画像
    """

    @abstractmethod
    def load(self, source: Any, **options: Any) -> LoadedTable:
        """加载一个数据源（路径 / 字节流 / 其他引擎约定形式）。"""

    @abstractmethod
    def preview(
        self, df: pl.DataFrame | pl.LazyFrame, **options: Any
    ) -> dict[str, Any]:
        """分页预览数据（page / page_size / columns / sort / filter）。"""

    @abstractmethod
    def schema(self, df: pl.DataFrame) -> dict[str, Any]:
        """Schema 分析，结果必须 JSON 可序列化。"""

    @abstractmethod
    def profile(self, df: pl.DataFrame) -> dict[str, Any]:
        """数据画像，结果必须 JSON 可序列化。"""
