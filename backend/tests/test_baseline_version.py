"""会话基线版本：只读分析不能被上一次写操作的中间结果顶掉。

真实事故（实机第六轮）：在 1220 行 × 14 列的信贷样本上跑「按 purpose 分组求
credit_amount 的平均值」→ ``data.aggregate`` 把 7 行 × 2 列的聚合结果存成了
数据集的新版本，而新版本即 latest。于是紧接着的每一步分析都跑在这张废表上：

- ``eda.correlation`` 报「相关性分析至少需要 2 个数值字段」（明明有 7 个数值列）；
- ``dataset.quality`` 说「共 7 行、2 列，两个字段 purpose 与 credit_amount_mean」；
- 用户点名「age 的分布」，槽位抽取器从 latest 的列名里只能挑到 ``purpose``，
  于是回答「实际分析的是 purpose 列」。

修法是把「用户当前认为的数据状态」显式建模成**会话基线版本**：只读分析一律
锚在基线上；只有真正改写数据状态的写操作（clean/filter/transform）才推进基线；
aggregate/merge 只把结果另存，不动基线。
"""

from __future__ import annotations

import pytest

from app.agent.engine import AgentEngine, _advances_baseline, _produced_version
from app.agent.models import AgentRun
from app.agent.playbooks import PlaybookStep
from app.agent.store import AgentStore, _session_from_dict
from app.tools.base import Tool
from app.tools.result import ToolResult


# ---------------------------------------------------------------- 假基础设施


class _VersionRow:
    def __init__(self, version: int) -> None:
        self.version = version


class _FakeSchema:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def names(self) -> list[str]:
        return self._names


class _FakeFrame:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def collect_schema(self) -> _FakeSchema:
        return _FakeSchema(self._names)


class _FakeDatasetService:
    """只实现引擎读版本 / 读列名用到的两个方法，并记录被问到的版本号。"""

    def __init__(self, latest: int, columns: list[str] | None = None) -> None:
        self.latest = latest
        self.columns = columns or ["age", "credit_amount", "duration"]
        self.scanned_versions: list[int | None] = []

    def get_version_row(self, dataset_id: int, version: int | None):
        return _VersionRow(self.latest)

    def scan_version(self, dataset_id: int, version: int | None = None, *, columns=None):
        self.scanned_versions.append(version)
        return _FakeFrame(self.columns)


class _ReadTool(Tool):
    """模拟 eda.distribute / dataset.quality 这类带 version 入参的只读工具。"""

    name = "fake.read"
    description = "只读分析"
    category = "eda"
    permission = "analyze_data"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "column": {"type": "string"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"ok": True}, summary="ok")


def _engine(store: AgentStore, service: _FakeDatasetService) -> AgentEngine:
    return AgentEngine(store, llm=None, dataset_service=service)


def _step() -> PlaybookStep:
    return PlaybookStep(tool="fake.read", title="读数据", required_slots=("column",))


# ---------------------------------------------------------------- 锚定基线


def test_read_tools_anchor_version_to_baseline(store: AgentStore):
    service = _FakeDatasetService(latest=3)
    engine = _engine(store, service)
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="看看 age 的分布")

    params = engine._build_params(run, session, _step(), _ReadTool(), {})

    assert params["dataset_id"] == 1
    # 首次用到数据集时把当前版本固化成基线
    assert params["version"] == 3
    # 列名也必须按基线版本读：拿 latest 的列名会让槽位抽取器只看见聚合结果的列
    assert service.scanned_versions == [3]


def test_analysis_still_reads_baseline_after_another_write_appends_version(store: AgentStore):
    """聚合把 latest 顶到 v4 之后，下一次分析仍然锚在 v3。这是本次修复的核心。"""
    service = _FakeDatasetService(latest=3)
    engine = _engine(store, service)
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="按 purpose 分组求 credit_amount 的平均值")
    engine._build_params(run, session, _step(), _ReadTool(), {})

    # 模拟：聚合产出 v4，latest 随之被顶到 4
    service.latest = 4
    run2 = AgentRun(id="r-2", session_id=session.id, user_request="看看 age 的分布")
    params = engine._build_params(run2, session, _step(), _ReadTool(), {})

    assert params["version"] == 3, "只读分析必须锚在基线上，不能读 latest"


def test_baseline_is_persisted_on_the_session(store: AgentStore):
    service = _FakeDatasetService(latest=3)
    engine = _engine(store, service)
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="看看数据质量")

    engine._build_params(run, session, _step(), _ReadTool(), {})

    assert store.get_session(session.id).baseline_versions == {1: 3}


def test_baseline_survives_json_roundtrip(store: AgentStore):
    """基线必须跨重启可用：JSON 的键是字符串，读回时得转回 int。"""
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    session.baseline_versions[1] = 3
    store.update_session(session)

    restored = _session_from_dict(session.to_dict())

    assert restored.baseline_versions == {1: 3}
    assert all(isinstance(k, int) for k in restored.baseline_versions)


# ---------------------------------------------------------------- 谁推进基线


def test_aggregate_does_not_advance_baseline(store: AgentStore):
    """聚合是「产出统计结果」，不是「数据变成这样了」—— 绝不能推进基线。"""
    engine = _engine(store, _FakeDatasetService(latest=3))
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    aggregate = type("Agg", (), {"name": "data.aggregate", "op_type": "aggregate"})()

    engine._advance_baseline(
        session, aggregate, {"dataset_id": 1}, ToolResult.ok({"new_version": 4})
    )

    assert session.baseline_versions.get(1) is None


def test_filter_and_transform_advance_baseline(store: AgentStore):
    """筛选/派生列确实改变了数据状态，后续分析应基于新版本。"""
    engine = _engine(store, _FakeDatasetService(latest=3))
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    filt = type("F", (), {"name": "data.filter", "op_type": "filter"})()
    engine._advance_baseline(session, filt, {"dataset_id": 1}, ToolResult.ok({"new_version": 5}))

    transform = type("T", (), {"name": "data.transform", "op_type": "transform"})()
    engine._advance_baseline(session, transform, {"dataset_id": 1}, ToolResult.ok({"new_version": 6}))

    assert session.baseline_versions[1] == 6


def test_clean_advances_baseline_to_its_last_step(store: AgentStore):
    """data.clean 一路可能产出多个版本，基线必须落在最后一个。"""
    engine = _engine(store, _FakeDatasetService(latest=3))
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    clean = type("C", (), {"name": "data.clean", "op_type": ""})()

    engine._advance_baseline(
        session, clean, {"dataset_id": 1}, ToolResult.ok({"versions": [4, 5]})
    )

    assert session.baseline_versions[1] == 5


def test_no_dataset_id_means_no_baseline_change(store: AgentStore):
    engine = _engine(store, _FakeDatasetService(latest=3))
    session = store.create_session(user_id="tester", title="基线", dataset_ids=[1])
    filt = type("F", (), {"name": "data.filter", "op_type": "filter"})()

    engine._advance_baseline(session, filt, {}, ToolResult.ok({"new_version": 5}))

    assert session.baseline_versions == {}


# ---------------------------------------------------------------- 判定辅助


def test_advances_baseline_matches_tool_declaration():
    assert _advances_baseline(type("A", (), {"name": "data.aggregate", "op_type": "aggregate"})()) is False
    assert _advances_baseline(type("M", (), {"name": "data.merge", "op_type": "merge"})()) is False
    assert _advances_baseline(type("C", (), {"name": "data.clean", "op_type": ""})()) is True
    assert _advances_baseline(type("F", (), {"name": "data.filter", "op_type": "filter"})()) is True
    assert _advances_baseline(type("T", (), {"name": "data.transform", "op_type": "transform"})()) is True


def test_produced_version_reads_both_shapes():
    assert _produced_version(ToolResult.ok({"new_version": 4})) == 4
    assert _produced_version(ToolResult.ok({"versions": [4, 5]})) == 5
    assert _produced_version(ToolResult.ok({"rows": 10})) is None
    assert _produced_version(ToolResult.fail("x")) is None
    assert _produced_version(None) is None
