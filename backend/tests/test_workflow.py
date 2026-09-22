"""Phase 5（Prompt 083-087）Workflow 测试。"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from app.core.exceptions import NotFoundException, ValidationException
from app.workflow.executor import WorkflowExecutor
from app.workflow.models import Edge, Node, Workflow
from app.workflow.service import WorkflowRunHandle, WorkflowService
from app.workflow.state import NodeStatus
from app.workflow.validator import validate_workflow


# ----------------------------------------------------------------------
# Prompt 083: 数据结构
# ----------------------------------------------------------------------
def test_workflow_models_serialization_roundtrip():
    wf = Workflow(
        name="demo",
        nodes=[Node("load", "load_data", {"dataset_id": 1}), Node("train", "train_model")],
        edges=[Edge("load", "train")],
        metadata={"owner": "xiaoluo"},
    )
    data = wf.to_dict()
    restored = Workflow.from_dict(data)
    assert restored == wf
    assert wf.predecessors("train") == ["load"]


# ----------------------------------------------------------------------
# Prompt 084: Validator
# ----------------------------------------------------------------------
def _wf(nodes, edges, name="wf"):
    return Workflow(
        name=name,
        nodes=[Node(n[0], n[1], n[2] if len(n) > 2 else {}) for n in nodes],
        edges=[Edge(*e) for e in edges],
    )


def test_validator_ok():
    result = validate_workflow(
        _wf([("a", "load"), ("b", "train")], [("a", "b")]),
        known_types={"load", "train"},
    )
    assert result.ok and result.errors == []


def test_validator_edge_endpoint_missing():
    result = validate_workflow(_wf([("a", "load")], [("a", "ghost")]))
    assert not result.ok
    assert any("ghost" in e for e in result.errors)


def test_validator_detects_cycle():
    result = validate_workflow(
        _wf([("a", "t"), ("b", "t"), ("c", "t")], [("a", "b"), ("b", "c"), ("c", "a")])
    )
    assert not result.ok
    assert any("循环" in e for e in result.errors)


def test_validator_self_loop_and_duplicates():
    result = validate_workflow(
        _wf([("a", "t"), ("b", "t")], [("a", "a"), ("a", "b"), ("a", "b")])
    )
    assert any("自环" in e for e in result.errors)
    assert any("重复边" in w for w in result.warnings)


def test_validator_duplicate_node_id_and_missing_type():
    result = validate_workflow(_wf([("a", "load"), ("a", "train")], []))
    assert any("重复" in e for e in result.errors)
    result2 = validate_workflow(_wf([("a", "")], []))
    assert any("type" in e for e in result2.errors)


def test_validator_required_config_keys():
    result = validate_workflow(
        _wf([("a", "train", {"model": "lr"})], []),
        required_config_keys={"train": {"model", "target"}},
    )
    assert not result.ok
    assert any("target" in e for e in result.errors)


def test_validator_required_config_accepts_nested_params_shape():
    """config 写成 {params: {...}} 时不能把合法配置误报为缺失。

    平台里 ml.* / data.* 操作节点通行 {params: {...}} 写法，data.load 用平铺写法。
    必需参数检查若不先做形状归一化，同一份合法配置换个节点就会被判为缺失。
    """
    result = validate_workflow(
        _wf([("a", "load", {"params": {"dataset_id": 3}})], []),
        required_config_keys={"load": {"dataset_id"}},
    )
    assert result.ok and result.errors == []
    flat = validate_workflow(
        _wf([("a", "load", {"dataset_id": 3})], []),
        required_config_keys={"load": {"dataset_id"}},
    )
    assert flat.ok and flat.errors == []


def test_validator_required_config_treats_null_and_ui_only_as_missing():
    """显式 null 与「只有画布坐标 __ui」都算缺失（对齐前端 == null 口径）。"""
    null_value = validate_workflow(
        _wf([("a", "load", {"dataset_id": None})], []),
        required_config_keys={"load": {"dataset_id"}},
    )
    assert not null_value.ok
    ui_only = validate_workflow(
        _wf([("a", "load", {"__ui": {"x": 1, "y": 2}})], []),
        required_config_keys={"load": {"dataset_id"}},
    )
    assert not ui_only.ok


def test_validator_unknown_type_dependency_incomplete():
    result = validate_workflow(
        _wf([("a", "quantum_jump")], []), known_types={"load", "train"}
    )
    assert not result.ok
    assert any("未注册" in e for e in result.errors)


# ----------------------------------------------------------------------
# Prompt 085: Executor（DAG 执行）
# ----------------------------------------------------------------------
def _runners(record: dict[str, Any] | None = None):
    def load(node, upstream, ctx):
        return {"rows": 10}

    def train(node, upstream, ctx):
        assert upstream and all(v == {"rows": 10} for v in upstream.values())
        if record is not None:
            record["trained"] = True
        return {"model": "lr"}

    def boom(node, upstream, ctx):
        raise ValueError("训练爆炸")

    return {"load": load, "train": train, "boom": boom}


def test_executor_diamond_dag_success():
    """菱形 DAG：a -> (b, c) -> d，d 汇聚两个分支输出。"""
    wf = Workflow(
        name="diamond",
        nodes=[
            Node("a", "load"),
            Node("b", "scale"),
            Node("c", "enrich"),
            Node("d", "train"),
        ],
        edges=[Edge("a", "b"), Edge("a", "c"), Edge("b", "d"), Edge("c", "d")],
    )
    runners = {
        "load": lambda n, u, c: {"rows": 10},
        "scale": lambda n, u, c: {"rows": 8},
        "enrich": lambda n, u, c: {"rows": 5},
        "train": lambda n, u, c: {"sum": sum(v["rows"] for v in u.values())},
    }
    result = WorkflowExecutor().execute(wf, runners)
    assert result.success
    assert result.node_states == {
        "a": "success",
        "b": "success",
        "c": "success",
        "d": "success",
    }
    assert result.outputs["d"] == {"sum": 13}
    assert all(
        log.duration_ms is not None
        for log in result.logs
        if log.status == "success" and log.finished_at
    )


def test_executor_failure_skips_downstream_but_other_branch_runs():
    wf = Workflow(
        name="branch",
        nodes=[
            Node("src", "load"),
            Node("bad", "boom"),
            Node("ok", "train"),
            Node("child_of_bad", "train"),
        ],
        edges=[Edge("src", "bad"), Edge("src", "ok"), Edge("bad", "child_of_bad")],
    )
    result = WorkflowExecutor().execute(wf, _runners())
    assert result.status == NodeStatus.FAILED
    assert result.node_states["bad"] == "failed"
    assert result.node_states["child_of_bad"] == "skipped"
    assert result.node_states["ok"] == "success"
    # 日志包含失败原因
    bad_log = next(log for log in result.logs if log.node_id == "bad")
    assert "训练爆炸" in bad_log.error


def test_executor_invalid_workflow_rejected():
    wf = _wf([("a", "load")], [("a", "ghost")])
    with pytest.raises(Exception, match="校验失败"):
        WorkflowExecutor().execute(wf, _runners())


def test_executor_cancel_midway():
    wf = Workflow(
        name="cancel",
        nodes=[Node("a", "load"), Node("b", "train"), Node("c", "train")],
        edges=[Edge("a", "b"), Edge("b", "c")],
    )
    executed: list[str] = []

    def slow(node, upstream, ctx):
        executed.append(node.id)
        time.sleep(0.15)
        return {"ok": True}

    def cancel_probe():
        return "a" in executed  # a 完成后请求取消

    runners = {"load": slow, "train": slow}
    result = WorkflowExecutor().execute(wf, runners, is_cancelled=cancel_probe)
    assert result.status == NodeStatus.CANCELLED
    assert result.node_states["a"] == "success"
    assert result.node_states["b"] == "cancelled"
    assert result.node_states["c"] == "cancelled"


def test_executor_context_shared_between_nodes():
    seen: dict[str, Any] = {}

    def writer(node, upstream, ctx):
        ctx["value"] = 42
        return None

    def reader(node, upstream, ctx):
        seen["value"] = ctx.get("value")
        return None

    wf = Workflow(
        name="ctx",
        nodes=[Node("w", "load"), Node("r", "train")],
        edges=[Edge("w", "r")],
    )
    WorkflowExecutor().execute(wf, {"load": writer, "train": reader}, context={})
    assert seen["value"] == 42


def test_executor_logs_one_entry_per_node_and_keeps_precise_skip_reason():
    """失败节点的传递后继只记一条日志，且保留「是哪个上游失败」的精确定位。

    回归：_skip_descendants 会先写一条精确日志，主循环走到该节点时又以泛化的
    「上游节点未全部成功」再写一条；后者在 errors 字典里后来居上，把精确原因
    覆盖掉，同时让 len(logs) > 节点数（工具摘要的分母随之失真）。
    """
    wf = Workflow(
        name="dup-log",
        nodes=[
            Node("src", "load"),
            Node("bad", "boom"),
            Node("ok", "train"),
            Node("child_of_bad", "train"),
        ],
        edges=[Edge("src", "bad"), Edge("src", "ok"), Edge("bad", "child_of_bad")],
    )
    result = WorkflowExecutor().execute(wf, _runners())
    logged = [log.node_id for log in result.logs]
    assert len(logged) == len(set(logged)), f"同一节点出现多条日志：{logged}"
    assert len(result.logs) == len(wf.nodes)
    child_log = next(log for log in result.logs if log.node_id == "child_of_bad")
    assert "'bad'" in child_log.error


# ----------------------------------------------------------------------
# Prompt 087: WorkflowService
# ----------------------------------------------------------------------
@pytest.fixture()
def service() -> WorkflowService:
    return WorkflowService(node_runners=_runners())


def test_service_create_get_list(service: WorkflowService):
    wf = service.create(
        name="pipeline",
        nodes=[{"id": "a", "type": "load"}, {"id": "b", "type": "train"}],
        edges=[{"source": "a", "target": "b"}],
        metadata={"env": "test"},
    )
    got = service.get(wf.metadata["id"])
    assert got.name == "pipeline"
    assert got.metadata["id"] == wf.metadata["id"]
    items = service.list()
    assert any(i["name"] == "pipeline" for i in items)
    with pytest.raises(NotFoundException):
        service.get(99999)


def test_service_create_rejects_invalid(service: WorkflowService):
    with pytest.raises(ValidationException):
        service.create(
            name="bad",
            nodes=[{"id": "a", "type": "load"}],
            edges=[{"source": "a", "target": "ghost"}],
        )
    # 未注册节点类型（依赖完整校验）
    with pytest.raises(ValidationException):
        service.create(name="bad2", nodes=[{"id": "a", "type": "nope"}], edges=[])


def test_service_run_success(service: WorkflowService):
    wf = service.create(
        name="run-ok",
        nodes=[{"id": "a", "type": "load"}, {"id": "b", "type": "train"}],
        edges=[{"source": "a", "target": "b"}],
    )
    handle = service.run(wf.metadata["id"])
    assert handle.status == NodeStatus.SUCCESS
    assert handle.result.outputs["b"] == {"model": "lr"}
    got = service.get_run(handle.run_id)
    assert got.result.success


def test_service_run_failure_recorded(service: WorkflowService):
    wf = service.create(
        name="run-bad",
        nodes=[{"id": "a", "type": "load"}, {"id": "b", "type": "boom"}],
        edges=[{"source": "a", "target": "b"}],
    )
    handle = service.run(wf.metadata["id"])
    assert handle.status == NodeStatus.FAILED
    assert handle.result.node_states["b"] == "failed"


def test_service_update_revalidates(service: WorkflowService):
    wf = service.create(name="v1", nodes=[{"id": "a", "type": "load"}], edges=[])
    wid = wf.metadata["id"]
    updated = service.update(
        wid,
        nodes=[{"id": "a", "type": "load"}, {"id": "b", "type": "train"}],
        edges=[{"source": "a", "target": "b"}],
    )
    assert [n.id for n in updated.nodes] == ["a", "b"]
    with pytest.raises(ValidationException):
        service.update(wid, edges=[{"source": "a", "target": "ghost"}])


def test_service_clone(service: WorkflowService):
    wf = service.create(
        name="orig",
        nodes=[{"id": "a", "type": "load"}, {"id": "b", "type": "train"}],
        edges=[{"source": "a", "target": "b"}],
        metadata={"id": None, "team": "lab"},
    )
    clone = service.clone(wf.metadata["id"])
    assert clone.name == "orig (copy)"
    assert clone.metadata["id"] != wf.metadata["id"]
    # 克隆不共享节点对象（深拷贝）
    clone.nodes[0].config["x"] = 1
    assert "x" not in wf.nodes[0].config
    # 克隆可独立运行
    handle = service.run(clone.metadata["id"])
    assert handle.status == NodeStatus.SUCCESS


def test_service_cancel_run(service: WorkflowService):
    def slow(node, upstream, ctx):
        time.sleep(0.3)
        return {"ok": True}

    svc = WorkflowService(node_runners={"load": slow, "train": slow})
    wf = svc.create(
        name="long",
        nodes=[{"id": "a", "type": "load"}, {"id": "b", "type": "train"}],
        edges=[{"source": "a", "target": "b"}],
    )
    results: dict[str, Any] = {}

    def worker():
        results["handle"] = svc.run(wf.metadata["id"])

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.1)
    handle_id = next(iter(svc._runs))
    assert svc.cancel(handle_id) is True
    thread.join(timeout=5)
    assert results["handle"].status == NodeStatus.CANCELLED


def test_service_run_preflight_blocks_incomplete_config_but_allows_draft():
    """缺必需参数：可以保存草稿，但运行时被预检拦下且不执行任何节点。"""
    executed: list[str] = []

    def load(node, upstream, ctx):
        executed.append(node.id)
        return {"rows": 1}

    svc = WorkflowService(
        node_runners={"load": load},
        required_config_keys={"load": {"dataset_id"}},
    )
    # 画布式草稿：先搭图、后填参数 ⇒ create 必须放行（前端「流程体检」也只警告）
    wf = svc.create(name="draft", nodes=[{"id": "a", "type": "load"}], edges=[])
    with pytest.raises(ValidationException, match="配置不完整"):
        svc.run(wf.metadata["id"])
    assert executed == [], "预检未通过时不应执行任何节点"
    assert svc._runs == {}, "预检被拦下的运行不应留下句柄"

    # 补齐参数后即可运行
    svc.update(wf.metadata["id"], nodes=[{"id": "a", "type": "load", "config": {"dataset_id": 7}}])
    handle = svc.run(wf.metadata["id"])
    assert handle.status == NodeStatus.SUCCESS
    assert executed == ["a"]


def test_service_without_spec_keeps_old_behaviour():
    """未配置规格时不做必需参数检查（保持既有调用方行为不变）。"""
    svc = WorkflowService(node_runners=_runners())
    wf = svc.create(name="no-spec", nodes=[{"id": "a", "type": "load"}], edges=[])
    assert svc.run(wf.metadata["id"]).status == NodeStatus.SUCCESS


def test_service_run_ids_never_repeat_and_runs_are_bounded():
    """run_id 全局唯一；运行句柄按上限淘汰最旧的已结束运行。

    回归：run_id 曾由 f"wf_{id}_{len(self._runs) + 1}" 推导，一旦引入淘汰
    （_runs 里的句柄持有 _df/_model，必须设上限）就会重号并覆盖活句柄，
    使 cancel(run_id) 打到错误的运行上。
    """
    svc = WorkflowService(node_runners=_runners())
    svc._MAX_RUNS = 3
    wf = svc.create(name="bounded", nodes=[{"id": "a", "type": "load"}], edges=[])
    seen: list[str] = []
    for _ in range(8):
        seen.append(svc.run(wf.metadata["id"]).run_id)
    assert len(seen) == len(set(seen)), f"run_id 重号：{seen}"
    assert len(svc._runs) <= svc._MAX_RUNS


def test_service_eviction_never_drops_running_handle():
    """淘汰只能发生在终态句柄上，否则用户将永久无法取消正在跑的运行。"""
    svc = WorkflowService(node_runners=_runners())
    svc._MAX_RUNS = 1
    wf = svc.create(name="inflight", nodes=[{"id": "a", "type": "load"}], edges=[])
    first = svc.run(wf.metadata["id"])
    # 人为把新句柄置为 RUNNING 后再触发一次淘汰，验证它不会被移除。
    with svc._run_lock:
        svc._run_seq += 1
        running = WorkflowRunHandle(run_id=f"wf_{wf.metadata['id']}_{svc._run_seq}", workflow_id=wf.metadata["id"])
        running.status = NodeStatus.RUNNING
        svc._runs[running.run_id] = running
        svc._evict_runs_locked()
    assert running.run_id in svc._runs
    assert first.run_id not in svc._runs
