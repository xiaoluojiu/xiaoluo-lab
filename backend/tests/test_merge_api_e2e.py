"""T0-Mx：Merge API E2E 黑盒测试。

覆盖 mapping / keys / preview / validate / execute / 版本固定 / 各种 join 类型 /
冲突 / null key / duplicate / composite key。
"""

from __future__ import annotations

import polars as pl
import pytest
from app.api.deps import get_data_engine_service, get_dataset_service
from app.core.database import get_db
from app.data_engine.service import DataEngineService
from app.main import app
from app.services.dataset_service import DatasetService
from fastapi.testclient import TestClient


@pytest.fixture()
def api(db, storage) -> TestClient:
    ds_service = DatasetService(db, storage)
    engine = DataEngineService(ds_service)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_dataset_service] = lambda: ds_service
    app.dependency_overrides[get_data_engine_service] = lambda: engine
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture()
def service(db, storage) -> DatasetService:
    return DatasetService(db, storage)


def _mk(service: DatasetService, name: str, df: pl.DataFrame) -> tuple[int, int]:
    """创建数据集 + 版本，返回 (dataset_id, version)。"""
    ds = service.create(name)
    v = service.create_version(ds.id, df)
    return ds.id, v.version


# ----------------------------------------------------------------------
# 1. mapping 基础
# ----------------------------------------------------------------------
def test_mapping_returns_candidates(api, service):
    l_id, _ = _mk(service, "ml", pl.DataFrame({"id": [1, 2], "v": [10, 20]}))
    r_id, _ = _mk(service, "mr", pl.DataFrame({"id": [1, 2], "w": [1, 2]}))
    resp = api.post(
        "/api/v1/merge/mapping",
        json={"left_dataset_id": l_id, "right_dataset_id": r_id},
    )
    body = resp.json()
    assert body["success"] is True
    candidates = body["data"]
    assert any(c["source_column"] == "id" and c["target_column"] == "id" for c in candidates)


# ----------------------------------------------------------------------
# 2. keys 基础（单 key）
# ----------------------------------------------------------------------
def test_keys_single_basic(api, service):
    l_id, _ = _mk(service, "kl", pl.DataFrame({"k": [1, 2, 3]}))
    r_id, _ = _mk(service, "kr", pl.DataFrame({"k": [1, 2, 3]}))
    resp = api.post(
        "/api/v1/merge/keys",
        json={
            "left_dataset_id": l_id,
            "right_dataset_id": r_id,
            "left_key": "k",
            "right_key": "k",
        },
    )
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["cardinality"] == "one-to-one"


# ----------------------------------------------------------------------
# 3. preview 不创建版本
# ----------------------------------------------------------------------
def test_preview_does_not_create_version(api, service):
    l_id, l_ver = _mk(service, "pl", pl.DataFrame({"id": [1, 2, 3]}))
    r_id, r_ver = _mk(service, "pr", pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]}))
    resp = api.post(
        "/api/v1/merge/preview",
        json={
            "left_dataset_id": l_id,
            "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "id", "right": "id"}], "join_type": "inner"},
        },
    )
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["output_rows"] == 3
    assert data["output_columns"] == 2
    assert data["left_version"] == l_ver
    assert data["right_version"] == r_ver
    assert len(data["preview"]["items"]) == 3
    assert data["preview"]["columns"] == ["id", "v"]


# ----------------------------------------------------------------------
# 4. validate 通过 / 失败
# ----------------------------------------------------------------------
def test_validate_ok_and_fail(api, service):
    l_id, _ = _mk(service, "vl", pl.DataFrame({"id": [1, 2, 3]}))
    r_id, _ = _mk(service, "vr", pl.DataFrame({"id": [1, 2, 3]}))
    # 通过
    ok = api.post(
        "/api/v1/merge/validate",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "id", "right": "id"}]},
        },
    ).json()["data"]
    assert ok["ok"] is True
    # 失败：key 不存在
    bad = api.post(
        "/api/v1/merge/validate",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "nope", "right": "id"}]},
        },
    ).json()["data"]
    assert bad["ok"] is False
    assert any("not found" in e for e in bad["errors"])


# ----------------------------------------------------------------------
# 5. execute inner / left / outer / right
# ----------------------------------------------------------------------
@pytest.mark.parametrize("join_type,expected_rows", [
    ("inner", 2),  # {2,3} 交集
    ("left", 3),   # 全部左表
    ("right", 3),  # 全部右表
    ("outer", 4),  # 并集
])
def test_execute_all_join_types(api, service, join_type, expected_rows):
    l_id, l_ver = _mk(service, f"jl-{join_type}", pl.DataFrame({"id": [1, 2, 3]}))
    r_id, r_ver = _mk(service, f"jr-{join_type}", pl.DataFrame({"id": [2, 3, 4]}))
    resp = api.post(
        "/api/v1/merge/execute",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "left_version": l_ver, "right_version": r_ver,
            "plan": {"keys": [{"left": "id", "right": "id"}], "join_type": join_type},
        },
    )
    body = resp.json()
    assert body["success"] is True, f"{join_type} 失败: {body}"
    assert body["data"]["report"]["output_rows"] == expected_rows
    assert body["data"]["version"]["version"] == 2  # 新版本


# ----------------------------------------------------------------------
# 6. T0-M4：execute 缺省版本号 → 422/400
# ----------------------------------------------------------------------
def test_execute_without_version_rejected(api, service):
    l_id, _ = _mk(service, "nv-l", pl.DataFrame({"id": [1, 2]}))
    r_id, _ = _mk(service, "nv-r", pl.DataFrame({"id": [1, 2]}))
    resp = api.post(
        "/api/v1/merge/execute",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "id", "right": "id"}]},
        },
    )
    assert resp.status_code in (400, 422)


# ----------------------------------------------------------------------
# 7. T0-M6：列名冲突唯一（amount 与 amount_right 共存）
# ----------------------------------------------------------------------
def test_column_conflict_naming_unique(api, service):
    """左表已有 amount 与 amount_right，右表也有 amount。
    右表的 amount 必须重命名为 amount_right_2（不能与左表 amount_right 冲突）。
    """
    l_id, l_ver = _mk(
        service,
        "conf-l",
        pl.DataFrame({"id": [1, 2], "amount": [10.0, 20.0], "amount_right": [1.0, 2.0]}),
    )
    r_id, r_ver = _mk(
        service,
        "conf-r",
        pl.DataFrame({"id": [1, 2], "amount": [99.0, 88.0]}),
    )
    resp = api.post(
        "/api/v1/merge/preview",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "id", "right": "id"}], "join_type": "inner"},
        },
    )
    body = resp.json()
    assert body["success"] is True
    cols = body["data"]["output_columns_list"]
    # 必须没有重复列名
    assert len(cols) == len(set(cols)), f"出现重复列名: {cols}"
    # 左表 amount 与 amount_right 都保留，右表 amount 被重命名为 amount_right_2
    assert "amount" in cols
    assert "amount_right" in cols
    assert "amount_right_2" in cols


# ----------------------------------------------------------------------
# 8. null key 警告
# ----------------------------------------------------------------------
def test_null_key_warning(api, service):
    l_id, _ = _mk(service, "nl-l", pl.DataFrame({"id": [1, None, 3]}))
    r_id, _ = _mk(service, "nl-r", pl.DataFrame({"id": [1, 2, 3]}))
    resp = api.post(
        "/api/v1/merge/validate",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "id", "right": "id"}]},
        },
    )
    body = resp.json()
    assert body["success"] is True
    assert any("null" in w.lower() for w in body["data"]["warnings"])


# ----------------------------------------------------------------------
# 9. duplicate key → many-to-many 拒绝
# ----------------------------------------------------------------------
def test_duplicate_key_many_to_many_rejected(api, service):
    l_id, _ = _mk(service, "dup-l", pl.DataFrame({"k": [1, 1, 2]}))
    r_id, _ = _mk(service, "dup-r", pl.DataFrame({"k": [1, 1, 2]}))
    resp = api.post(
        "/api/v1/merge/validate",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "k", "right": "k"}]},
        },
    )
    body = resp.json()
    assert body["data"]["ok"] is False
    assert any("many-to-many" in e for e in body["data"]["errors"])


# ----------------------------------------------------------------------
# 10. T0-M5：composite key 正确处理
# ----------------------------------------------------------------------
def test_composite_key_cardinality(api, service):
    """组合 key (a, b)：单独 a 重复，但 (a, b) 唯一 → one-to-one。"""
    l_id, _ = _mk(
        service,
        "ck-l",
        pl.DataFrame({"a": [1, 1, 2], "b": [10, 20, 10], "v": [100, 200, 300]}),
    )
    r_id, _ = _mk(
        service,
        "ck-r",
        pl.DataFrame({"a": [1, 1, 2], "b": [10, 20, 10], "w": ["x", "y", "z"]}),
    )
    resp = api.post(
        "/api/v1/merge/keys",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "left_keys": ["a", "b"],
            "right_keys": ["a", "b"],
        },
    )
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["cardinality"] == "one-to-one"

    # execute composite key 也应工作
    l_ver = 1
    r_ver = 1
    exec_resp = api.post(
        "/api/v1/merge/execute",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "left_version": l_ver, "right_version": r_ver,
            "plan": {
                "keys": [{"left": "a", "right": "a"}, {"left": "b", "right": "b"}],
                "join_type": "inner",
            },
        },
    )
    exec_body = exec_resp.json()
    assert exec_body["success"] is True, f"composite key execute 失败: {exec_body}"
    assert exec_body["data"]["report"]["output_rows"] == 3


# ----------------------------------------------------------------------
# 11. T0-M5：composite key 重复被检测
# ----------------------------------------------------------------------
def test_composite_key_duplicate_detected(api, service):
    """(a, b) 中 (1, 10) 在两侧都重复 → many-to-many。"""
    l_id, _ = _mk(
        service,
        "ckd-l",
        pl.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]}),
    )
    r_id, _ = _mk(
        service,
        "ckd-r",
        pl.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]}),
    )
    resp = api.post(
        "/api/v1/merge/validate",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {
                "keys": [{"left": "a", "right": "a"}, {"left": "b", "right": "b"}],
            },
        },
    )
    body = resp.json()
    assert body["data"]["ok"] is False
    assert any("many-to-many" in e for e in body["data"]["errors"])


# ----------------------------------------------------------------------
# 12. T0-M4：版本固定 - 原版本不被覆盖
# ----------------------------------------------------------------------
def test_original_version_unchanged_after_merge(api, service):
    """merge 后查询原版本数据，应保持不变。"""
    l_id, l_ver = _mk(service, "orig-l", pl.DataFrame({"id": [1, 2, 3]}))
    r_id, r_ver = _mk(service, "orig-r", pl.DataFrame({"id": [1, 2, 3], "w": [10, 20, 30]}))
    # 执行 merge
    api.post(
        "/api/v1/merge/execute",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "left_version": l_ver, "right_version": r_ver,
            "plan": {"keys": [{"left": "id", "right": "id"}], "join_type": "inner"},
        },
    )
    # 查询原左版本 preview：仍应是 3 行 1 列
    preview = api.get(f"/api/v1/datasets/{l_id}/preview", params={"version": l_ver}).json()["data"]
    assert preview["total"] == 3
    assert preview["columns"] == ["id"]


# ----------------------------------------------------------------------
# 13. T0-M4：execute 使用错误版本号被拒绝
# ----------------------------------------------------------------------
def test_execute_wrong_version_rejected(api, service):
    l_id, _ = _mk(service, "wv-l", pl.DataFrame({"id": [1, 2, 3]}))
    r_id, _ = _mk(service, "wv-r", pl.DataFrame({"id": [1, 2, 3]}))
    resp = api.post(
        "/api/v1/merge/execute",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "left_version": 999,  # 不存在的版本
            "right_version": 1,
            "plan": {"keys": [{"left": "id", "right": "id"}]},
        },
    )
    # 应返回 404 / 400 / 422
    assert resp.status_code in (400, 404, 422)


# ----------------------------------------------------------------------
# 14. preview 失败时返回明确错误
# ----------------------------------------------------------------------
def test_preview_bad_plan_returns_error(api, service):
    l_id, _ = _mk(service, "bp-l", pl.DataFrame({"id": [1, 2]}))
    r_id, _ = _mk(service, "bp-r", pl.DataFrame({"id": [1, 2]}))
    resp = api.post(
        "/api/v1/merge/preview",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "plan": {"keys": [{"left": "nope", "right": "id"}]},  # key 不存在
        },
    )
    # 失败时状态码 >= 400，且响应体含错误信息（middleware 包装为 {code/message/details}）
    assert resp.status_code >= 400
    body = resp.json()
    # body 可能是 ApiResponse(success=False) 或 middleware 包装的 {code/message}
    msg = body.get("error", {}).get("message") or body.get("message") or ""
    assert msg, f"应返回明确错误信息，实际: {body}"


# ----------------------------------------------------------------------
# 15. mapping 列重命名（right_column -> output_column）
# ----------------------------------------------------------------------
def test_execute_with_mapping_rename(api, service):
    l_id, l_ver = _mk(service, "mr-l", pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}))
    r_id, r_ver = _mk(service, "mr-r", pl.DataFrame({"id": [1, 2, 3], "tag": ["x", "y", "z"]}))
    resp = api.post(
        "/api/v1/merge/execute",
        json={
            "left_dataset_id": l_id, "right_dataset_id": r_id,
            "left_version": l_ver, "right_version": r_ver,
            "plan": {
                "keys": [{"left": "id", "right": "id"}],
                "mapping": [{"right_column": "tag", "output_column": "label"}],
                "join_type": "inner",
            },
        },
    )
    body = resp.json()
    assert body["success"] is True
    # 验证 label 列存在（通过 preview 新版本）
    new_ver = body["data"]["version"]["version"]
    preview = api.get(
        f"/api/v1/datasets/{l_id}/preview", params={"version": new_ver}
    ).json()["data"]
    assert "label" in preview["columns"]
    assert "tag" not in preview["columns"]
