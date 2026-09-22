from __future__ import annotations

import polars as pl

from app.workflow.models import Node
from app.workflow.runners import NODE_REQUIRED_CONFIG, build_default_runners, sanitize_output
from app.workflow.service import WorkflowService


def test_default_runners_register_ml_and_ai_nodes():
    runners = build_default_runners()
    for node_type in ("ml.train", "ml.predict", "ml.evaluate", "ml.cluster", "ml.pca", "ai.analyze"):
        assert node_type in runners


def test_ml_train_and_predict_runner():
    runners = build_default_runners()
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]})
    train_node = Node("train", "ml.train", {"params": {"model": "linear_regression", "target_column": "y", "test_size": 0.25, "random_state": 42}})
    trained = runners["ml.train"](train_node, {"load": {"_df": df}}, {})
    assert trained["task"] == "regression"
    assert trained["metrics"]
    predict_node = Node("predict", "ml.predict", {"params": {"output_column": "prediction"}})
    predicted = runners["ml.predict"](predict_node, {"train": trained}, {})
    assert "prediction" in predicted["_df"].columns


def test_ml_pca_runner_uses_transform():
    runners = build_default_runners()
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})
    node = Node("pca", "ml.pca", {"params": {"n_components": 1}})
    output = runners["ml.pca"](node, {"load": {"_df": df}}, {})
    assert output["_df"].columns == ["pc_1"]


def test_sanitize_output_removes_runtime_objects():
    cleaned = sanitize_output({"_df": object(), "_model": object(), "metrics": {"r2": 0.9}})
    assert cleaned == {"metrics": {"r2": 0.9}}


def test_required_config_spec_only_names_registered_node_types():
    """必需参数规格不能指向不存在的节点类型，否则永远查不到、白白多一份死配置。"""
    runners = build_default_runners()
    unknown = sorted(set(NODE_REQUIRED_CONFIG) - set(runners))
    assert unknown == [], f"规格里出现了未注册的节点类型：{unknown}"


def test_load_dataset_accepts_nested_params_shape():
    """data.load 过去只认平铺 dataset_id，{params: {...}} 会被误判成「没给」。"""
    class _Row:
        dataset_id = 5
        version = 2

    class _DatasetService:
        def get_version_row(self, dataset_id, version):
            assert dataset_id == 5
            return _Row()

        def load_version(self, dataset_id, version):
            return pl.DataFrame({"x": [1, 2, 3]})

    node = Node("load", "data.load", {"params": {"dataset_id": 5}})
    output = build_default_runners()["data.load"](
        node, {}, {"dataset_service": _DatasetService()}
    )
    assert output["dataset_id"] == 5
    assert output["row_count"] == 3


def test_run_preflight_does_not_reject_nested_params_as_missing():
    """★ 回归：{params: {...}} 写法必须能通过运行前预检。

    这是本次启用必需参数校验的最大风险点 —— 平台里 ml.* / data.* 操作节点通行
    {params: {...}} 写法，若预检只看顶层键，会把**全部**这类合法工作流拦死。
    这里用真实 runner 组合验证：预检放行，失败发生在节点执行阶段（缺 dataset_service）。
    """
    svc = WorkflowService(
        node_runners=build_default_runners(),
        required_config_keys=NODE_REQUIRED_CONFIG,
    )
    wf = svc.create(
        name="nested",
        nodes=[{"id": "load", "type": "data.load", "config": {"params": {"dataset_id": 5}}}],
        edges=[],
    )
    handle = svc.run(wf.metadata["id"])
    # 若被预检拦下会直接抛 ValidationException；能拿到 handle 说明校验已放行
    assert handle.status.value == "failed"
    assert "dataset_service" in (handle.result.logs[0].error or "")


def test_blocked_result_is_structured_and_self_correctable():
    """预检失败要变成带明细的失败结果，而不是裸异常 —— Agent 才能补齐参数重试。"""
    from app.core.exceptions import ValidationException
    from app.tools.workflow_tools import _blocked_result

    detail = "节点 'load'（type='data.load'）缺少必需参数：['dataset_id']"
    result = _blocked_result(3, ValidationException("工作流无法执行：配置不完整", details={"errors": [detail]}))
    assert result.success is False
    assert result.data["executed"] is False
    assert result.data["errors"] == [detail]
    assert result.metadata["preflight"] is True
    assert "dataset_id" in result.errors[0]
