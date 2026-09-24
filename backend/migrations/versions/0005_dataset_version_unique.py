"""dataset_versions: 增加 (dataset_id, version) 唯一约束

Revision ID: 0005_dataset_version_unique
Revises: 0004_db_connectors
Create Date: 2026-09-24

为什么必须加这条约束
--------------------
版本号的生成方式是 ``max(version) + 1``（应用层读后写）。两个并发请求会同时
读到同一个 max，算出同一个 version：

    请求 A -> latest = v7 -> 写 v8
    请求 B -> latest = v7 -> 写 v8

已有的 ``storage_path UNIQUE`` 拦不住这个场景：storage key 是由
``dataset_id`` + ``version`` 派生的，version 相同则 key 也相同，两个请求写的是
**同一个文件**，唯一约束在「谁先写谁赢」的覆盖中完全失效，最终得到两份指向
同一路径的 v8 记录，且后写者覆盖了先写者的快照内容。

只有把唯一性交给数据库裁决才能终止这个竞态：预留版本号时对
``(dataset_id, version)`` 做 INSERT，由 IntegrityError 决定谁重试下一个号。

为什么可以直接加约束（不会撞上历史脏数据）
------------------------------------------
``storage_path`` 由 ``dataset_id``/``version`` 确定性派生，且**已经**是 UNIQUE。
因此历史库中「同 dataset 同 version 的两行」必然拥有相同的 storage_path，
会被既有约束挡在建表之后——也就是说脏数据在这条迁移之前根本不可能产生。
这里无需去重即可安全建约束。

SQLite 注意事项
---------------
SQLite 不支持 ``ALTER TABLE ADD CONSTRAINT``，因此走 ``batch_alter_table``
（Alembic 会以「建新表 + 拷数据 + 改名」的方式重建）。
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_dataset_version_unique"
down_revision: Union[str, None] = "0004_db_connectors"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONSTRAINT_NAME = "uq_dataset_versions_dataset_id_version"


def upgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.create_unique_constraint(
            CONSTRAINT_NAME,
            ["dataset_id", "version"],
        )


def downgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.drop_constraint(
            CONSTRAINT_NAME,
            type_="unique",
        )
