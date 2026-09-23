"""Phase 6（Prompt 088-113）Tools 测试。"""

from __future__ import annotations

import polars as pl
import pytest
from app.agent.permission.models import Permission
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.tools.base import ToolConfirmationRequired, ToolPermissionError, ToolServices
from app.tools.builtin import TOOL_REGISTRY
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry


@pytest.fixture()
def dataset_service(db, storage) -> DatasetService:
    return DatasetService(db, storage)


@pytest.fixture()
def data_engine(dataset_service) -> DataEngineService:
    return DataEngineService(dataset_service)


@pytest.fixture()
def experiment_service(db, dataset_service) -> ExperimentService:
    return ExperimentService(db, dataset_service)


@pytest.fixture()
def services(dataset_service, data_engine, experiment_service, db) -> ToolServices:
    return ToolServices(
        dataset_service=dataset_service,
        data_engine_service=data_engine,
        experiment_service=experiment_service,
        db=db,
    )


@pytest.fixture()
def seeded(dataset_service: DatasetService) -> int:
    """创建分类数据集，返回 dataset_id。"""
    ds = dataset_service.create("tools-demo")
    df = pl.DataFrame(
        {
            "a": [1.0, 1.5, 2.0, 8.0, 8.5, 9.0, 1.2, 8.2],
            "b": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5, 1.6, 8.6],
            "city": ["北京", "上海", "北京", "上海", "北京", "上海", "北京", "上海"],
            "label": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )
    dataset_service.create_version(ds.id, df)
    return ds.id


def full_ctx(dataset_id: int | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id="tester",
        session_id="s1",
        permissions=set(Permission),
        dataset_ids={dataset_id} if dataset_id else set(),
    )


# ----------------------------------------------------------------------
# Prompt 088/089: Tool 抽象与注册表
# ----------------------------------------------------------------------
def test_builtin_tools_registered():
    names = TOOL_REGISTRY.names()
    # 关键能力必须注册；数量会随业务工具增加而变，这里只断言关键集合存在，
    # 再断言「不低于基线」，避免每次新增工具都要改这个魔法数字。
    assert {
        "dataset.list", "data.filter", "data.merge", "eda.describe", "ml.train",
        "ml.predict", "workflow.create", "workflow.run", "workflow.build_and_run",
        "report.generate",
    } <= set(names)
    assert len(names) >= 29

    # 报告链路与工作流链路的工具必须带描述与 input_schema，否则 LLM 无法正确调用
    for name in ("workflow.build_and_run", "report.generate", "eda.visualize"):
        tool = TOOL_REGISTRY.get(name)
        assert tool.description.strip(), f"{name} 缺少 description"
        assert isinstance(tool.input_schema, dict) and tool.input_schema, f"{name} 缺少 input_schema"


def test_tool_describe_fields():
    tool = TOOL_REGISTRY.get("data.merge")
    desc = tool.describe()
    assert desc["name"] == "data.merge"
    assert desc["permission"] == "create_version"
    assert desc["risk_level"] == "high"
    assert "keys" in desc["input_schema"]["properties"]


def test_registry_duplicate_and_unknown():
    registry = ToolRegistry()
    tool = TOOL_REGISTRY.get("dataset.list")
    registry.register(tool)
    with pytest.raises(Exception, match="已注册"):
        registry.register(tool)
    with pytest.raises(Exception, match="Tool not found"):
        registry.get("nope.tool")
    registry.unregister("dataset.list")
    with pytest.raises(Exception, match="Tool not found"):
        registry.unregister("dataset.list")


# ----------------------------------------------------------------------
# 权限集成（Prompt 089 + 116）
# ----------------------------------------------------------------------
def test_execute_denied_without_permission(services, seeded):
    ctx = ToolExecutionContext(user_id="u1", permissions=set())  # 无任何权限
    with pytest.raises(ToolPermissionError):
        TOOL_REGISTRY.execute(
            "dataset.list", {}, ctx, services
        )


def test_execute_denied_outside_dataset_scope(services, seeded):
    ctx = full_ctx(dataset_id=999)  # 只允许 999
    with pytest.raises(ToolPermissionError, match="允许范围"):
        TOOL_REGISTRY.execute(
            "dataset.preview", {"dataset_id": seeded}, ctx, services
        )


def test_execute_high_risk_requires_confirmation(services, seeded):
    ctx = full_ctx()
    params = {
        "dataset_id": seeded,
        "missing": {"strategy": "constant", "value": 0, "columns": ["a"]},
    }
    # data.clean 为高风险：未确认时拒绝执行
    with pytest.raises(ToolConfirmationRequired):
        TOOL_REGISTRY.execute("data.clean", params, ctx, services)
    # 确认后执行成功
    result = TOOL_REGISTRY.execute("data.clean", params, ctx, services, confirmed=True)
    assert result.success
    assert result.data["versions"] == [2]

    # data.filter 为中风险：无需确认即可执行
    filtered = TOOL_REGISTRY.execute(
        "data.filter",
        {"dataset_id": seeded, "conditions": [{"column": "a", "op": "gt", "value": 5}]},
        ctx,
        services,
    )
    assert filtered.success
    assert filtered.data["new_version"] == 3


# ----------------------------------------------------------------------
# Prompt 092-097: Dataset 工具
# ----------------------------------------------------------------------
def test_dataset_list_and_inspect(services, seeded):
    result = TOOL_REGISTRY.execute("dataset.list", {}, full_ctx(), services)
    assert result.success and result.data["total"] >= 1
    inspect = TOOL_REGISTRY.execute(
        "dataset.inspect", {"dataset_id": seeded}, full_ctx(), services
    )
    assert inspect.data["latest_version"]["rows"] == 8
    assert inspect.data["latest_version"]["columns"] == 4


def test_dataset_preview_schema(services, seeded):
    ctx = full_ctx()
    preview = TOOL_REGISTRY.execute(
        "dataset.preview",
        {"dataset_id": seeded, "page_size": 3, "columns": ["a", "label"]},
        ctx,
        services,
    )
    assert preview.success
    assert preview.data["total"] == 8
    names = [c["name"] if isinstance(c, dict) else c for c in preview.data["columns"]]
    assert names == ["a", "label"]

    schema = TOOL_REGISTRY.execute(
        "dataset.schema", {"dataset_id": seeded}, ctx, services
    )
    assert {c["column"] for c in schema.data["columns"]} == {"a", "b", "city", "label"}


def test_dataset_profile_quality(services, seeded):
    ctx = full_ctx()
    profile = TOOL_REGISTRY.execute(
        "dataset.profile", {"dataset_id": seeded}, ctx, services
    )
    assert profile.success
    quality = TOOL_REGISTRY.execute(
        "dataset.quality", {"dataset_id": seeded}, ctx, services
    )
    assert "issues" in quality.data


# ----------------------------------------------------------------------
# Prompt 098-102: Data 工具
# ----------------------------------------------------------------------
def test_data_filter_transform_aggregate(services, seeded):
    ctx = full_ctx()
    # filter（高风险需确认）
    filtered = TOOL_REGISTRY.execute(
        "data.filter",
        {"dataset_id": seeded, "conditions": [{"column": "a", "op": "gte", "value": 8}]},
        ctx, services, confirmed=True,
    )
    assert filtered.data["rows"] == 4

    # transform：新增数值列（结构化表达式）
    transformed = TOOL_REGISTRY.execute(
        "data.transform",
        {
            "dataset_id": seeded,
            "name": "a2",
            "expression": {
                "type": "math",
                "op": "mul",
                "left": {"type": "column", "column": "a"},
                "right": {"type": "value", "value": 2},
            },
        },
        ctx, services, confirmed=True,
    )
    assert transformed.success and transformed.data["columns"] == 5

    # aggregate：显式指定在 v1 上执行（v2 已被过滤为单一类别）
    agg = TOOL_REGISTRY.execute(
        "data.aggregate",
        {
            "dataset_id": seeded,
            "version": 1,
            "group_by": ["label"],
            "aggregations": [{"column": "a", "func": "mean"}],
        },
        ctx, services, confirmed=True,
    )
    assert agg.success and agg.data["rows"] == 2


def test_data_clean_generates_versions(services, seeded):
    ctx = full_ctx()
    result = TOOL_REGISTRY.execute(
        "data.clean",
        {
            "dataset_id": seeded,
            "missing": {"strategy": "constant", "value": 0, "columns": ["a"]},
            "deduplicate": {"subset": ["a", "b"]},
        },
        ctx, services, confirmed=True,
    )
    assert result.success
    assert result.data["versions"] == [2, 3]


def test_data_clean_requires_steps(services, seeded):
    result = TOOL_REGISTRY.execute(
        "data.clean", {"dataset_id": seeded}, full_ctx(), services, confirmed=True
    )
    assert not result.success


def test_data_merge_plan_validate_execute(services, dataset_service):
    """合并双数据集：自动建议 mapping + 校验 + 执行。"""
    left = dataset_service.create("left")
    dataset_service.create_version(
        left.id,
        pl.DataFrame({"uid": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}),
    )
    right = dataset_service.create("right")
    dataset_service.create_version(
        right.id,
        pl.DataFrame({"uid": [1, 2, 3], "city": ["北京", "上海", "广州"]}),
    )
    ctx = full_ctx()
    result = TOOL_REGISTRY.execute(
        "data.merge",
        {
            "left_dataset_id": left.id,
            "right_dataset_id": right.id,
            "keys": [{"left": "uid", "right": "uid"}],
        },
        ctx, services, confirmed=True,
    )
    assert result.success, result.errors
    assert result.data["report"]["matched_rows"] == 3
    assert result.data["new_version"] == 2


def test_data_merge_auto_infer_locationid_key(services, dataset_service):
    """不传 keys 时按后缀包含关系推断 Join Key（NYC Taxi 场景）。

    事实表外键 PULocationID / DOLocationID 指向维度表主键 LocationID，
    两者列名不同、归一化后也不同（pulocationid vs locationid），
    旧实现只看同名列 + <name>_id 约定，必然推断失败（回归）。

    多个外键命中同一维度主键时**只保留一个**（data.merge 是单次 join，
    同时返回两个 right=LocationID 会触发执行层重复列名崩溃——回归）。
    """
    from app.tools.data_tools import _infer_join_keys

    class _DF:
        def __init__(self, cols):
            self.columns = cols

    left = _DF([
        "VendorID", "tpep_pickup_datetime", "PULocationID", "DOLocationID",
        "passenger_count", "fare_amount", "total_amount",
    ])
    right = _DF(["LocationID", "Borough", "Zone", "service_zone"])
    keys = _infer_join_keys(left, right)
    assert keys, "应推断出 LocationID 相关的 Join Key"
    # 只保留一个外键，且 right 唯一指向 LocationID
    assert len(keys) == 1
    assert keys[0]["right"] == "LocationID"
    assert keys[0]["left"] in ("PULocationID", "DOLocationID")


def test_data_merge_auto_infer_same_name(services, dataset_service):
    """同名列仍优先于后缀包含。"""
    from app.tools.data_tools import _infer_join_keys

    class _DF:
        def __init__(self, cols):
            self.columns = cols

    keys = _infer_join_keys(_DF(["user_id", "name"]), _DF(["user_id", "age"]))
    assert keys == [{"left": "user_id", "right": "user_id"}]


def test_data_merge_auto_infer_skips_redundant_dimension_cols(services, dataset_service):
    """同名列在右表不唯一时让位给唯一主键（回归：many-to-many）。

    NYC Taxi 场景：左表（yellow 出租车）历史上已 join 过一次 zone 维度表，
    带入了 Borough / Zone / service_zone 三个同名维度列；右表是 265 行的
    zone lookup，真正唯一主键是 LocationID（Zone 有 3 个重名、Borough/service_zone
    高度冗余）。旧实现贪心堆叠这三个同名列做 composite key，右表组合不唯一
    → many-to-many 被 validator 拒绝。

    修复后应跳过非唯一同名列，走后缀包含锁定唯一主键 LocationID。
    """
    import polars as pl

    from app.tools.data_tools import _infer_join_keys

    right = pl.DataFrame({
        "LocationID": [1, 2, 3, 4],
        "Borough": ["EWR", "Queens", "Bronx", "Queens"],
        "Zone": ["Newark Airport", "Jamaica Bay", "Allerton", "Jamaica Bay"],  # Zone 有重复
        "service_zone": ["EWR", "Boro Zone", "Boro Zone", "Boro Zone"],
    })
    left = pl.DataFrame({
        "VendorID": [1, 2],
        "PULocationID": [1, 2],
        "DOLocationID": [2, 3],
        "fare_amount": [10.0, 20.0],
        "Borough": ["EWR", "Queens"],
        "Zone": ["Newark Airport", "Jamaica Bay"],
        "service_zone": ["EWR", "Boro Zone"],
    })
    keys = _infer_join_keys(left, right)
    assert len(keys) == 1
    assert keys[0]["right"] == "LocationID"
    assert keys[0]["left"] in ("PULocationID", "DOLocationID")



def test_data_merge_auto_infer_no_common_key(services, dataset_service):
    """两表完全无关联列时返回空（交由调用方报明确的列清单错误）。"""
    from app.tools.data_tools import _infer_join_keys
    class _DF:
        def __init__(self, cols):
            self.columns = cols

    keys = _infer_join_keys(_DF(["foo", "bar"]), _DF(["baz", "qux"]))
    assert keys == []


def test_data_merge_validation_failure(services, dataset_service):
    left = dataset_service.create("l2")
    dataset_service.create_version(left.id, pl.DataFrame({"uid": [1]}))
    right = dataset_service.create("r2")
    dataset_service.create_version(right.id, pl.DataFrame({"uid": [1]}))
    result = TOOL_REGISTRY.execute(
        "data.merge",
        {
            "left_dataset_id": left.id,
            "right_dataset_id": right.id,
            "keys": [{"left": "ghost", "right": "ghost"}],
        },
        full_ctx(), services, confirmed=True,
    )
    assert not result.success
    assert result.metadata.get("stage") == "validate"


# ----------------------------------------------------------------------
# Prompt 103-107: EDA 工具
# ----------------------------------------------------------------------
def test_eda_tools(services, seeded):
    ctx = full_ctx()
    describe = TOOL_REGISTRY.execute(
        "eda.describe", {"dataset_id": seeded}, ctx, services
    )
    assert describe.success and "a" in str(describe.data)

    dist = TOOL_REGISTRY.execute(
        "eda.distribution",
        {"dataset_id": seeded, "column": "city"},
        ctx, services,
    )
    assert dist.success

    corr = TOOL_REGISTRY.execute(
        "eda.correlation", {"dataset_id": seeded}, ctx, services
    )
    assert corr.success

    outlier = TOOL_REGISTRY.execute(
        "eda.outlier", {"dataset_id": seeded}, ctx, services
    )
    assert outlier.success

    viz = TOOL_REGISTRY.execute(
        "eda.visualize",
        {"dataset_id": seeded, "chart": "histogram", "column": "a"},
        ctx, services,
    )
    assert viz.success


# ----------------------------------------------------------------------
# Prompt 108-113: ML 工具
# ----------------------------------------------------------------------
def test_ml_detect_task(services, seeded):
    ctx = full_ctx()
    cls = TOOL_REGISTRY.execute(
        "ml.detect_task", {"dataset_id": seeded, "target": "label"}, ctx, services
    )
    assert cls.data["task"] == "classification"
    assert cls.data["reasons"]

    reg = TOOL_REGISTRY.execute(
        "ml.detect_task", {"dataset_id": seeded, "target": "a"}, ctx, services
    )
    assert reg.data["task"] == "regression"

    clu = TOOL_REGISTRY.execute(
        "ml.detect_task", {"dataset_id": seeded}, ctx, services
    )
    assert clu.data["task"] == "clustering"


def test_ml_prepare(services, seeded):
    result = TOOL_REGISTRY.execute(
        "ml.prepare", {"dataset_id": seeded, "target": "label"}, full_ctx(), services
    )
    assert result.success
    config = result.data["config"]
    assert config["encoding"]["method"] == "one_hot"
    assert set(config["encoding"]["columns"]) == {"city"}
    assert config["scaling"]["method"] == "standard"


def test_ml_train_evaluate_compare_explain(services, seeded):
    ctx = full_ctx()
    train = TOOL_REGISTRY.execute(
        "ml.train",
        {
            "dataset_id": seeded,
            "target": "label",
            "model": "logistic_regression",
            "seed": 42,
            "preprocessing": {"encoding": {"method": "one_hot"}},
            "description": "tools 训练",
        },
        ctx, services, confirmed=True,
    )
    assert train.success, train.errors
    assert train.data["status"] == "success"
    assert "accuracy" in train.data["metrics"]
    run_id = train.data["run_id"]

    evaluate = TOOL_REGISTRY.execute("ml.evaluate", {"run_id": run_id}, ctx, services)
    assert evaluate.data["metrics"] == train.data["metrics"]

    # 再训练一次不同模型用于比较（同样需要编码 city 列）
    train2 = TOOL_REGISTRY.execute(
        "ml.train",
        {
            "dataset_id": seeded,
            "target": "label",
            "model": "decision_tree_classifier",
            "seed": 42,
            "preprocessing": {"encoding": {"method": "one_hot"}},
        },
        ctx, services, confirmed=True,
    )
    compare = TOOL_REGISTRY.execute(
        "ml.compare",
        {"run_ids": [run_id, train2.data["run_id"]]},
        ctx, services,
    )
    assert len(compare.data["entries"]) == 2

    explain = TOOL_REGISTRY.execute("ml.explain", {"run_id": run_id}, ctx, services)
    assert explain.success, explain.errors
    features = {item["feature"] for item in explain.data["importances"]}
    assert features == {"a", "b", "city=北京", "city=上海"}


def test_ml_train_infers_convention_target_column(services, seeded):
    """未指定 target 时按命名约定推断出 `label`，任务随之判为分类。

    这正是过去失效的地方：ml.train 调用 ml.detect_task 时漏传 infer_target=True
    （注释声称会传、代码没传），于是「训练一个模型」链路静默退化成聚类。
    """
    result = TOOL_REGISTRY.execute(
        "ml.train",
        {
            "dataset_id": seeded,
            "model": "kmeans",
            "params": {"n_clusters": 2},
            "seed": 0,
            "preprocessing": {"encoding": {"method": "one_hot"}},
        },
        full_ctx(), services, confirmed=True,
    )
    assert result.success, result.errors
    # 推断出 label → 分类；用户给的 kmeans 被换成该任务的默认模型
    assert result.metadata["task"] == "classification"
    assert result.metadata["model"] == "logistic_regression"
    # ★ 换模型时必须丢掉不属于新模型的参数，否则 sklearn 报
    #   "LogisticRegression.__init__() got an unexpected keyword argument 'n_clusters'"
    assert result.data["model_adjusted"]["dropped_params"] == ["n_clusters"]


def test_ml_train_without_target_or_convention_falls_back_to_clustering(services, dataset_service):
    """既没有 target 也没有 target/label/y/class 列，且名称无任务提示 → 聚类。"""
    ds = dataset_service.create("tools-demo-plain")
    dataset_service.create_version(
        ds.id,
        pl.DataFrame(
            {
                "a": [float(i % 5) for i in range(30)],
                "b": [float(i % 7) for i in range(30)],
                "city": ["北京", "上海", "广州"] * 10,
            }
        ),
    )
    result = TOOL_REGISTRY.execute(
        "ml.train",
        {
            "dataset_id": ds.id,
            "model": "kmeans",
            "params": {"n_clusters": 2},
            "seed": 0,
            "preprocessing": {"encoding": {"method": "one_hot"}},
        },
        full_ctx(ds.id), services, confirmed=True,
    )
    assert result.success, result.errors
    assert result.metadata["task"] == "clustering"
    assert result.data["status"] == "success"


def test_ml_train_fails_instead_of_clustering_when_dataset_name_says_regression(
    services, dataset_service
):
    """数据集名称写明是回归任务、又**无法唯一确定**目标列时，必须失败并给出候选目标列。

    真实事故：数据集「航空公司出发延误预测（回归）」因为没有 target/label/y/class
    字段被判成无监督聚类，kmeans 跑「成功」，用户要的延误预测整条链路跑偏。

    这里刻意造成「两列都符合回归且无法区分」的歧义：推断器必须如实报告
    「无法唯一确定」，而不是随手挑一个（也不许退回聚类）。
    """
    ds = dataset_service.create("延误预测（回归）")
    dataset_service.create_version(
        ds.id,
        pl.DataFrame(
            {
                "x1": [float(i % 80) for i in range(200)],
                "x2": [float(i % 90) for i in range(200)],
                "city": ["北京", "上海", "广州", "深圳"] * 50,
            }
        ),
    )
    result = TOOL_REGISTRY.execute(
        "ml.train",
        {"dataset_id": ds.id, "model": "auto", "seed": 0},
        full_ctx(ds.id), services, confirmed=True,
    )
    assert not result.success
    assert result.data["needs_target"] is True
    assert result.data["dataset_hint"] == "regression"
    assert {"x1", "x2"} <= set(result.data["target_candidates"]["regression"])
    assert "请显式指定 target" in result.errors[0]


def test_ml_train_auto_infers_target_from_dataset_name(services, experiment_service, dataset_service):
    """事故的正解：数据集名写着「出发延误预测」，列名 `DepDelay` 就是那个目标 ——
    无需用户指定即可选中、训练，并把推断依据回执出来。

    数据刻意做成「70 档取整后的延误分钟」：这是一个**取值不多的回归目标**。
    过去分层切分判定只看唯一值数，会把这种目标按类别分层，然后以
    「测试集样本不足以覆盖 70 个类别」失败（把回归说成分类，排查方向完全错）。
    """
    ds = dataset_service.create("航空公司出发延误预测（回归）")
    dataset_service.create_version(
        ds.id,
        pl.DataFrame(
            {
                "DepDelay": [float(i % 70) for i in range(200)],
                "x1": [float(i % 80) for i in range(200)],
                "Month": [float(i % 12 + 1) for i in range(200)],
                "Distance": [float(i % 90) for i in range(200)],
            }
        ),
    )
    result = TOOL_REGISTRY.execute(
        "ml.train",
        {"dataset_id": ds.id, "model": "auto", "seed": 0},
        full_ctx(ds.id), services, confirmed=True,
    )
    assert result.success, result.errors
    assert result.metadata["task"] == "regression"
    inferred = result.data["target_inferred"]
    assert inferred["target"] == "DepDelay"
    assert inferred["source"] == "goal_match"
    assert inferred["reasons"], "推断必须给出理由"
    assert any("delay" in r for r in inferred["reasons"])
    # ★ 回归任务不得被分层切分（否则低基数回归目标会被误判成多分类）
    run = experiment_service.get_run(result.data["run_id"])
    assert (run.artifacts or {}).get("stratified") is False
