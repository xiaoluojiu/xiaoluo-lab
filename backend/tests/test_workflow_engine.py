"""Workflow 执行器 / 服务 / 默认 runner 的集成测试。

依赖：仅 polars 构造 ~200 行小数据，不走数据库、不加载 dataset、不发 LLM。

覆盖三个层次：
1. WorkflowExecutor + 假 runners：成功链、失败传递、取消边界。
2. WorkflowService：草稿语义（create 不拦缺参）、run 预检（缺 target_column 报错）。
3. 真实 default runners 直调（upstream 直接给 _df），验证 ml.train / ml.predict /
   ml.cluster 的平铺 config 兼容路径与产物闭合。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import polars as pl
import pytest

from app.core.exceptions import ValidationException
from app.workflow.executor import WorkflowExecutor
from app.workflow.models import Workflow
from app.workflow.runners import NODE_REQUIRED_CONFIG, build_default_runners
from app.workflow.service import WorkflowService
from app.workflow.state import NodeStatus


def _node(node_id: str, type_: str = "noop", config: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=node_id, type=type_, config=config or {})


def _edge(source: str, target: str) -> SimpleNamespace:
    return SimpleNamespace(source=source, target=target)


def _workflow(nodes: list, edges: list) -> Workflow:
    return Workflow(name="test", nodes=nodes, edges=edges)


# ----------------------------------------------------------------------
# 1. Executor + 假 runners
# ----------------------------------------------------------------------

def test_executor_all_success_logs_match_node_count() -> None:
    ex = WorkflowExecutor()
    runners = {"noop": lambda node, upstream, ctx: {"ok": True}}
    wf = _workflow(
        [_node("a"), _node("b"), _node("c")],
        [_edge("a", "b"), _edge("b", "c")],
    )
    result = ex.execute(wf, runners)
    assert result.status == NodeStatus.SUCCESS
    # 不变式：logs 与节点一一对应，len(logs) 恒等于节点数。
    assert len(result.logs) == 3
    assert result.node_states["a"] == NodeStatus.SUCCESS.value
    assert result.node_states["c"] == NodeStatus.SUCCESS.value


def test_executor_failure_skips_descendants_but_isolates_other_branch() -> None:
    ex = WorkflowExecutor()

    def runner(node, upstream, ctx):
        if node.id == "b":
            raise ValueError("boom")
        return {"ok": True}

    runners = {"noop": runner}
    # a -> b -> c 为一条链，d 独立分支。
    wf = _workflow(
        [_node("a"), _node("b"), _node("c"), _node("d")],
        [_edge("a", "b"), _edge("b", "c")],
    )
    result = ex.execute(wf, runners)
    assert result.status == NodeStatus.FAILED
    assert result.node_states["b"] == NodeStatus.FAILED.value
    assert result.node_states["c"] == NodeStatus.SKIPPED.value
    # 另一独立分支不受失败影响，照常成功。
    assert result.node_states["d"] == NodeStatus.SUCCESS.value
    assert result.node_states["a"] == NodeStatus.SUCCESS.value


def test_executor_cancel_at_first_boundary_cancels_remaining() -> None:
    ex = WorkflowExecutor()
    runners = {"noop": lambda node, upstream, ctx: {"ok": True}}
    wf = _workflow(
        [_node("a"), _node("b"), _node("c")],
        [_edge("a", "b"), _edge("b", "c")],
    )
    result = ex.execute(wf, runners, is_cancelled=lambda: True)
    assert result.status == NodeStatus.CANCELLED
    assert all(v == NodeStatus.CANCELLED.value for v in result.node_states.values())
    assert len(result.logs) == 3


# ----------------------------------------------------------------------
# 2. WorkflowService（假 runners + NODE_REQUIRED_CONFIG）
# ----------------------------------------------------------------------

def test_service_create_missing_params_is_a_valid_draft() -> None:
    svc = WorkflowService(
        node_runners={"ml.train": lambda node, upstream, ctx: {}},
        required_config_keys=NODE_REQUIRED_CONFIG,
    )
    # 草稿语义：create 只做结构校验，缺 target_column 也能保存。
    wf = svc.create(
        name="draft",
        nodes=[{"id": "t", "type": "ml.train", "config": {"model": "rf"}}],
        edges=[],
    )
    assert wf.metadata["id"] == 1
    assert svc.list()[0]["nodes"] == 1


def test_service_run_missing_target_column_raises_with_node_id() -> None:
    svc = WorkflowService(
        node_runners={"ml.train": lambda node, upstream, ctx: {}},
        required_config_keys=NODE_REQUIRED_CONFIG,
    )
    wf = svc.create(
        name="x",
        nodes=[{"id": "train", "type": "ml.train", "config": {"model": "rf"}}],
        edges=[],
    )
    with pytest.raises(ValidationException) as exc_info:
        svc.run(wf.metadata["id"])
    # 预检报错要带出「该去改哪个节点」的 id。
    assert any("train" in err for err in exc_info.value.details["errors"])


# ----------------------------------------------------------------------
# 3. 真实 default runners 直调（绕过 data.load）
# ----------------------------------------------------------------------

def _ml_df(n: int = 200) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [float(i % 7) for i in range(n)],
            "b": [float((i * 3) % 11) for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


def test_ml_train_classifier_metrics_and_sampling_closed() -> None:
    runners = build_default_runners()
    df = _ml_df()
    node = _node("train", "ml.train", {"target_column": "label", "model": "decision_tree_classifier"})
    result = runners["ml.train"](node, {"load": {"_df": df}}, {})
    assert result["task"] == "classification"
    assert "accuracy" in result["metrics"]
    assert math.isfinite(result["metrics"]["accuracy"])
    # train + test 行数必须与抽样信息闭合（200 行未触发抽样上限）。
    assert result["train_rows"] + result["test_rows"] == result["sampling"]["used_rows"]
    assert result["sampling"]["used_rows"] == df.height


def test_ml_predict_after_train_preserves_rows_and_adds_prediction() -> None:
    runners = build_default_runners()
    df = _ml_df()
    train_node = _node("train", "ml.train", {"target_column": "label", "model": "decision_tree_classifier"})
    train_result = runners["ml.train"](train_node, {"load": {"_df": df}}, {})
    predict_node = _node("predict", "ml.predict", {})
    result = runners["ml.predict"](predict_node, {"train": train_result}, {})
    assert result["_df"].height == df.height
    assert "prediction" in result["_df"].columns


def test_ml_cluster_kmeans_emits_cluster_column_and_count() -> None:
    runners = build_default_runners()
    df = _ml_df()
    node = _node("cluster", "ml.cluster", {"model": "kmeans", "n_clusters": 3})
    result = runners["ml.cluster"](node, {"load": {"_df": df}}, {})
    assert "cluster" in result["_df"].columns
    assert result["_df"].height == df.height
    assert result["metrics"]["cluster_count"] == 3
