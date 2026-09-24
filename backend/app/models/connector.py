"""DbConnector：外部数据库连接配置。

设计取舍
--------
为什么单独一张表，而不是塞进 Dataset：
    连接器是**数据来源的配置**，数据集是**数据本身**。一个连接器可以反复导入
    （每次导入产生一个新的 Dataset + 版本），而连接器自己需要长期持有 host /
    账号 / 口令。混在一起会让「改一次口令」变成「改一次数据集」，也会让
    数据集的删除语义变得含糊（删数据集要不要删连接配置？）。

为什么口令单独一列且只存密文：
    见 app/connectors/crypto.py。此列永远不出现在任何 Response Schema 里，
    只通过 ``ConnectorService`` 的掩码字段对外。

为什么保存 last_status / last_error：
    「配好的连接器哪天开始连不上了」是这类功能最常见的真实故障。把最近一次
    试连结论落在表上，用户回到页面就能看到，而不用反复手点「测试连接」。

为什么记 last_import_json：
    导入是长事务，失败原因（keyset 列有空值 / 驱动缺失 / 权限不足）比「失败」
    两个字有用得多。把最近一次抽取的 strategy / 批次 / 告警存下来，
    配合 DatasetVersion 就能复盘「这份数据是怎么进来的」。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel

# 连接状态机
CONNECTOR_STATUSES = ("unknown", "ok", "error")


class DbConnector(BaseModel):
    """一个外部数据库连接。"""

    __tablename__ = "db_connectors"

    # 展示名（唯一，便于在列表中区分同库不同账号）
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    # 方言名，取值见 app/connectors/dialects.py 的 DIALECTS
    dialect: Mapped[str] = mapped_column(String(32), index=True)

    host: Mapped[str | None] = mapped_column(String(255), nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # SQLite 为文件路径；其余为库名
    database: Mapped[str] = mapped_column(String(512), default="")
    # Postgres 的 public / SQL Server 的 dbo / MySQL 留空
    schema_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 密文（enc:v1:...）。绝不对外输出原文。
    password_enc: Mapped[str] = mapped_column(Text, default="")

    # 额外连接参数（如 sslmode / connect_timeout 覆盖）
    options_json: Mapped[dict] = mapped_column(JSON, default=dict)

    # ---- 健康状态 ----
    last_status: Mapped[str] = mapped_column(String(16), default="unknown")
    last_error: Mapped[str] = mapped_column(Text, default="")
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ---- 最近一次导入 ----
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    last_import_json: Mapped[dict] = mapped_column(JSON, default=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DbConnector id={self.id} name={self.name!r} dialect={self.dialect!r} "
            f"status={self.last_status!r}>"
        )
