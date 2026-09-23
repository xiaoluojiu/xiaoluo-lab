"""数据库连接器模块（拓展功能 · 大数据流接入）。

模块职责
--------
把「外部数据库」变成与小洛实验室内部数据集等价的一等公民：

    DbConnector(元数据)  →  Engine(连接)  →  iter_batches(分块抽取)
                                              ↓
                                      Parquet 快照 → DatasetVersion

文件组成
--------
- ``crypto``    口令加密与掩码（绝不明文落库）
- ``dialects``  方言注册表：URL 构造、驱动依赖、表结构内省 SQL、标识符引用
- ``extract``   分块抽取引擎（服务端游标 / keyset / limit-offset 三级策略）
- ``service``    业务服务：CRUD、试连、列表、预览、导入为数据集
- ``api``        REST 路由（app/api/v1/connectors.py）

为什么这条路能突破文件体积限制
------------------------------
文件上传的瓶颈在「一次请求内搬运完整个文件」；而数据库抽取天然可分页——
``extract`` 分批拉取、逐批写成 Parquet row group，中间不做任何全量物化，
因此单次导入规模只受目标磁盘容量约束，与进程内存无关。
"""

from __future__ import annotations

from app.connectors.crypto import get_cipher, mask_secret
from app.connectors.dialects import (
    DIALECTS,
    SUPPORTED_DIALECTS,
    DialectSpec,
    build_url,
    get_dialect,
)
from app.connectors.extract import (
    ExtractResult,
    close_engine,
    extract_table_to_parquet,
    iter_batches,
    open_engine,
)

__all__ = [
    "DIALECTS",
    "SUPPORTED_DIALECTS",
    "DialectSpec",
    "ExtractResult",
    "build_url",
    "close_engine",
    "extract_table_to_parquet",
    "get_cipher",
    "get_dialect",
    "iter_batches",
    "mask_secret",
    "open_engine",
]
