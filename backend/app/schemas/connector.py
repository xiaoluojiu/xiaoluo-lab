"""数据库连接器相关 Schema。

安全约定：``password`` 只入不出。所有 Response Schema 都不含该字段，
只暴露 ``has_password: bool``——前端据此显示「已配置」，但拿不回原文。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConnectorBase(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    dialect: str = Field(min_length=1, max_length=32)
    host: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str = Field(default="", max_length=512)
    schema_name: str | None = Field(default=None, max_length=128)
    username: str | None = Field(default=None, max_length=255)
    options: dict = Field(default_factory=dict)

    @field_validator("name", "database", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ConnectorCreate(ConnectorBase):
    """新建连接器。``password`` 可为空（SQLite / 免密连接）。"""

    password: str | None = Field(default=None, max_length=512)
    # 可选：创建后立即试连
    test_on_create: bool = False


class ConnectorUpdate(BaseModel):
    """更新连接器（部分更新）。

    ``password`` 的语义：``None`` = 不改动；``""`` = 清空；
    非空字符串 = 覆盖。
    """

    name: str | None = Field(default=None, min_length=1, max_length=128)
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = Field(default=None, max_length=512)
    schema_name: str | None = Field(default=None, max_length=128)
    username: str | None = Field(default=None, max_length=255)
    options: dict | None = None
    password: str | None = Field(default=None, max_length=512)


class ConnectorResponse(BaseModel):
    """连接器详情（绝不含口令明文）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    dialect: str
    host: str | None
    port: int | None
    database: str
    schema_name: str | None
    username: str | None
    options: dict = Field(default_factory=dict)
    # 只暴露「有没有口令」，不暴露口令本身
    has_password: bool = False
    last_status: str = "unknown"
    last_error: str = ""
    last_checked_at: datetime | None = None
    dataset_id: int | None = None
    last_import: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ConnectorTestRequest(BaseModel):
    """不落库的试连（用于「填完表单先测一下」）。"""

    dialect: str
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str = ""
    schema_name: str | None = None
    username: str | None = None
    password: str | None = None
    options: dict = Field(default_factory=dict)
    # 已保存的连接器：只覆盖局部字段做试连（password 留空则用已存的密文）
    use_saved: bool = False
    connector_id: int | None = None


class ConnectorTestResponse(BaseModel):
    ok: bool
    latency_ms: float = 0.0
    server_version: str | None = None
    message: str = ""
    dialect: str = ""
    driver_installed: bool = True


class ConnectorTableInfo(BaseModel):
    name: str
    type: str = "table"
    schema_name: str | None = None


class ConnectorColumnInfo(BaseModel):
    name: str
    type: str = ""
    nullable: bool = True
    primary_key: bool = False


class ConnectorPreviewRequest(BaseModel):
    table: str = Field(min_length=1)
    schema_name: str | None = None
    columns: list[str] | None = None
    where: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class ConnectorPreviewResponse(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    row_count: int = 0
    limit: int = 0
    sql: str = ""


class ConnectorImportRequest(BaseModel):
    """把一张表 / 一段查询导入为数据集。"""

    table: str = Field(min_length=1)
    schema_name: str | None = None
    columns: list[str] | None = None
    where: str | None = None
    order_by: str | None = None
    # keyset 列：给一个单调唯一列可让分页代价与页深无关
    keyset_column: str | None = None
    batch_size: int | None = Field(default=None, ge=1, le=1_000_000)
    max_rows: int | None = Field(default=None, ge=1)
    # 目标数据集：给 dataset_id 追加版本；否则新建
    dataset_id: int | None = None
    dataset_name: str | None = Field(default=None, max_length=255)
    dataset_description: str = ""


class ConnectorImportResponse(BaseModel):
    dataset_id: int
    dataset_version: int
    row_count: int
    column_count: int
    strategy: str
    batches: int = 0
    elapsed_seconds: float = 0.0
    rows_per_second: float = 0.0
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
