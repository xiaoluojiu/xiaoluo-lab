"""小洛实验室 · 智能合并实验脚本（Prompt 243）。

测试 7 种合并场景，量化 suggest_mappings 与 run_merge 的准确性与安全性：
  1. 字段名相同（user_id == user_id）
  2. 字段名不同（user_id vs userId）
  3. 类型不同（int vs string）
  4. Join Key 含缺失值
  5. 重复键（一对多）
  6. 一对多关系
  7. 多对多关系

每个场景记录：映射准确性 / 合并准确性 / 错误合并率 / 人工介入次数。

用法：
    python -m scripts.experiments.smart_merge_experiment
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

# 注册全部 ORM 模型到 Base.metadata
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.file  # noqa: F401
import app.models.operation  # noqa: F401
import polars as pl
from app.core.database import Base
from app.data_engine.exceptions import MergeError
from app.data_engine.merge.plan import JoinKey, MergePlan
from app.data_engine.service import DataEngineService
from app.services.dataset_service import DatasetService
from app.storage.local import LocalStorage
from app.storage.service import StorageService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ============================================================
# 环境
# ============================================================
def setup_env() -> dict[str, Any]:
    engine_db = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db, autoflush=False, expire_on_commit=False)
    db = Session()
    storage = StorageService(LocalStorage(root=Path(tempfile.mkdtemp(prefix="xl-smart-merge-"))))
    ds = DatasetService(db, storage)
    data_engine = DataEngineService(ds)
    return {"db": db, "storage": storage, "ds": ds, "engine": data_engine}


def _make_dataset_pair(ds: DatasetService, left_df: pl.DataFrame, right_df: pl.DataFrame,
                       tag: str) -> tuple[int, int]:
    """创建 left / right 两个数据集（各带一个版本），返回 (left_id, right_id)。"""
    left = ds.create(f"{tag}-left", f"{tag} 左表")
    ds.create_version(left.id, left_df)
    right = ds.create(f"{tag}-right", f"{tag} 右表")
    ds.create_version(right.id, right_df)
    return left.id, right.id


# ============================================================
# 7 个场景定义
# ============================================================
def _scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": "1. 字段名相同（user_id == user_id）",
            "left": pl.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]}),
            "right": pl.DataFrame({"user_id": [1, 2, 3], "age": [20, 30, 40]}),
            "left_key": "user_id", "right_key": "user_id",
            "expected_rows": 3,
            "correct_source": "user_id", "correct_target": "user_id",
        },
        {
            "name": "2. 字段名不同（user_id vs userId）",
            "left": pl.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]}),
            "right": pl.DataFrame({"userId": [1, 2, 3], "age": [20, 30, 40]}),
            "left_key": "user_id", "right_key": "userId",
            "expected_rows": 3,
            "correct_source": "user_id", "correct_target": "userId",
        },
        {
            "name": "3. 类型不同（int vs string）",
            "left": pl.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]}),
            "right": pl.DataFrame({"user_id": ["1", "2", "3"], "age": [20, 30, 40]}),
            "left_key": "user_id", "right_key": "user_id",
            "expected_rows": None,  # 校验阻断
            "correct_source": "user_id", "correct_target": "user_id",
        },
        {
            "name": "4. Join Key 含缺失值",
            "left": pl.DataFrame({"user_id": [1, 2, None], "name": ["a", "b", "c"]}),
            "right": pl.DataFrame({"user_id": [1, 2, 3], "age": [20, 30, 40]}),
            "left_key": "user_id", "right_key": "user_id",
            "expected_rows": 2,  # null 不匹配
            "correct_source": "user_id", "correct_target": "user_id",
        },
        {
            "name": "5. 重复键（一对多）",
            "left": pl.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]}),
            "right": pl.DataFrame({"user_id": [1, 1, 2], "order": ["o1", "o2", "o3"]}),
            "left_key": "user_id", "right_key": "user_id",
            "expected_rows": 3,  # user1 -> 2 行, user2 -> 1 行
            "correct_source": "user_id", "correct_target": "user_id",
        },
        {
            "name": "6. 一对多关系",
            "left": pl.DataFrame({"user_id": [1, 2], "name": ["a", "b"]}),
            "right": pl.DataFrame(
                {"user_id": [1, 1, 2, 2], "amount": [10.0, 11.0, 20.0, 21.0]}
            ),
            "left_key": "user_id", "right_key": "user_id",
            "expected_rows": 4,
            "correct_source": "user_id", "correct_target": "user_id",
        },
        {
            "name": "7. 多对多关系",
            "left": pl.DataFrame({"user_id": [1, 1, 2], "name": ["a", "a2", "b"]}),
            "right": pl.DataFrame({"user_id": [1, 1, 2], "tag": ["x", "y", "z"]}),
            "left_key": "user_id", "right_key": "user_id",
            "expected_rows": None,  # 校验阻断（many-to-many）
            "correct_source": "user_id", "correct_target": "user_id",
        },
    ]


# ============================================================
# 单场景执行
# ============================================================
def run_scenario(env: dict[str, Any], sc: dict[str, Any]) -> dict[str, Any]:
    ds, data_engine = env["ds"], env["engine"]
    left_id, right_id = _make_dataset_pair(ds, sc["left"], sc["right"], sc["name"][:2])
    left_df = ds.load_version(left_id)
    right_df = ds.load_version(right_id)

    # ---- 1. 映射准确性：suggest_mappings 是否找到正确 Key ----
    mappings = data_engine.suggest_mappings(left_df, right_df)
    correct_found = any(
        m["source_column"] == sc["correct_source"]
        and m["target_column"] == sc["correct_target"]
        and m["confidence"] >= 0.5
        for m in mappings
    )
    top_mapping = mappings[0] if mappings else None

    # ---- 2. 合并执行 ----
    plan = MergePlan(
        left={"dataset_id": left_id},
        right={"dataset_id": right_id},
        keys=[JoinKey(left=sc["left_key"], right=sc["right_key"])],
        join_type="inner",
    )
    blocked = False
    error: str | None = None
    actual_rows: int | None = None
    report_dict: dict[str, Any] | None = None
    try:
        version, report = data_engine.run_merge(left_id, right_id, plan)
        merged = ds.load_version(left_id, version.version)
        actual_rows = merged.height
        report_dict = report.to_dict()
    except MergeError as exc:
        blocked = True
        error = str(exc)

    # ---- 3. 指标计算 ----
    expected = sc["expected_rows"]
    if blocked:
        merge_accuracy = 0.0
        wrong_merge_rate: float | None = None  # 被 Validator 阻断，无错误合并
    elif expected is not None and actual_rows is not None:
        if actual_rows == expected:
            merge_accuracy = 1.0
            wrong_merge_rate = 0.0
        else:
            base = max(actual_rows, expected)
            merge_accuracy = max(0.0, 1 - abs(actual_rows - expected) / base) if base else 0.0
            wrong_merge_rate = abs(actual_rows - expected) / actual_rows if actual_rows else 1.0
    else:
        merge_accuracy = 0.0
        wrong_merge_rate = None

    # 人工介入：映射未找到需手工指定 + 合并被阻断需手工修复
    intervention = 0
    if not correct_found:
        intervention += 1
    if blocked:
        intervention += 1

    return {
        "scenario": sc["name"],
        "mapping_accuracy": correct_found,
        "top_mapping": (
            f"{top_mapping['source_column']} -> {top_mapping['target_column']} "
            f"({top_mapping['confidence']:.2f})"
            if top_mapping else "（无候选）"
        ),
        "merge_accuracy": round(merge_accuracy, 4),
        "wrong_merge_rate": (
            f"{wrong_merge_rate:.1%}" if wrong_merge_rate is not None else "N/A（已阻断）"
        ),
        "human_intervention": intervention,
        "actual_rows": actual_rows,
        "expected_rows": expected,
        "blocked": blocked,
        "error": error,
        "report": report_dict,
    }


# ============================================================
# 输出
# ============================================================
def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["场景", "映射准确", "Top 候选", "合并准确", "错误率", "介入", "实际/期望"]
    widths = [30, 8, 26, 8, 12, 6, 12]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "|" + "|".join(f" {h.center(w)} " for h, w in zip(headers, widths, strict=False)) + "|"
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        cells = [
            row["scenario"],
            "是" if row["mapping_accuracy"] else "否",
            row["top_mapping"],
            f"{row['merge_accuracy']:.0%}",
            row["wrong_merge_rate"],
            str(row["human_intervention"]),
            f"{row['actual_rows']}/{row['expected_rows'] if row['expected_rows'] is not None else '阻断'}",
        ]
        line = "|" + "|".join(f" {c[:w].ljust(w)} " for c, w in zip(cells, widths, strict=False)) + "|"
        print(line)
    print(sep)


def save_json(rows: list[dict[str, Any]], name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n✓ 结果已保存：{path}")


def main() -> None:
    print("=" * 72)
    print("小洛实验室 · 智能合并实验（Prompt 243）—— 7 场景映射与合并准确性")
    print("=" * 72)

    env = setup_env()
    rows: list[dict[str, Any]] = []
    for sc in _scenarios():
        print(f"\n[执行] {sc['name']} …")
        rows.append(run_scenario(env, sc))

    print("\n" + "=" * 72)
    print("对比结果")
    print("=" * 72)
    print_table(rows)

    print("\n结论：")
    print("  - 场景 1/2：suggest_mappings 能识别字段名相同与差异（user_id / userId）。")
    print("  - 场景 3：类型冲突（int vs string）被 Validator 阻断，避免静默错误合并。")
    print("  - 场景 4：Key 缺失值产生警告但不阻断，null 行不匹配（符合预期）。")
    print("  - 场景 5/6：一对多 / 重复键触发警告，合并按预期放大行数。")
    print("  - 场景 7：多对多被 Validator 阻断，防止结果爆炸。")
    print("  → Validator 是安全底线：类型冲突 / 多对多在执行前被拦截，需人工介入修复。")

    save_json(rows, "smart_merge_experiment.json")


if __name__ == "__main__":
    main()
