"""Prompt 114：Permission 模型。

最小权限集合：所有敏感操作都必须声明所需权限，
PermissionManager 据此决定放行 / 拒绝 / 需要确认。
"""

from __future__ import annotations

from enum import StrEnum


class Permission(StrEnum):
    """系统权限。"""

    READ_DATA = "read_data"
    ANALYZE_DATA = "analyze_data"
    MODIFY_DATA = "modify_data"
    CREATE_VERSION = "create_version"
    TRAIN_MODEL = "train_model"
    EXPORT_DATA = "export_data"
    EXPORT_REPORT = "export_report"
    EXECUTE_WORKFLOW = "execute_workflow"
    DELETE_DATA = "delete_data"


class Decision(StrEnum):
    """权限判定结果。"""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"


# 角色预设（便于测试与初始化；生产可替换为 DB 权限表）
ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    "admin": set(Permission),
    "analyst": {
        Permission.READ_DATA,
        Permission.ANALYZE_DATA,
        Permission.MODIFY_DATA,
        Permission.CREATE_VERSION,
        Permission.TRAIN_MODEL,
        Permission.EXPORT_REPORT,
        Permission.EXECUTE_WORKFLOW,
    },
    "viewer": {Permission.READ_DATA, Permission.ANALYZE_DATA},
}
