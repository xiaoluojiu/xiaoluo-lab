"""Agent 后端测试公共夹具。

三条硬约束
----------
1. **绝不发起真实 LLM 调用。** 旧测试套件里有 6 个文件会打真实 API，烧掉过
   额度。这里在导入 app 之前就把 ``LLM_REMOTE_ENABLED`` 关掉，
   ``build_default_provider()`` 因此恒为 None，全链路走零 Token 路径。
2. **绝不写仓库目录。** 数据库、数据根目录、模型目录全部落在临时目录。
3. **每个用例独享 AgentStore。** Agent 存储是进程级单例，测试里替换掉
   ``app.api.deps.AGENT_STORE`` 即可（依赖函数在调用时才读这个全局）。
"""

from __future__ import annotations

import os
import pathlib
import tempfile

# ---- 必须在 import app 之前设置环境变量 ----------------------------------
_BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TMP_HOME = pathlib.Path(tempfile.mkdtemp(prefix="xiaoluo-agent-test-"))

os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_HOME / 'app.db').as_posix()}"
os.environ["DATA_ROOT"] = str(_TMP_HOME / "data")
os.environ["MODEL_ROOT"] = str(_TMP_HOME / "models")
os.environ["LLM_REMOTE_ENABLED"] = "false"
os.environ["RATE_LIMIT_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.agent.engine import AgentEngine  # noqa: E402
from app.agent.store import AgentStore  # noqa: E402
from app.api import deps as api_deps  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.database import Base, engine as db_engine  # noqa: E402
from app.main import app  # noqa: E402

Base.metadata.create_all(db_engine)


@pytest.fixture(autouse=True)
def _no_remote_llm():
    """兜底断言：任何用例都不许拿到真实 provider。"""
    assert settings.remote_llm_available() is False
    yield


@pytest.fixture()
def store(tmp_path: pathlib.Path) -> AgentStore:
    """用例级 AgentStore，落盘在 tmp_path。"""
    return AgentStore(tmp_path / "agent_store.json")


@pytest.fixture()
def api_store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> AgentStore:
    """替换掉 API 层的进程级单例，让 HTTP 用例互不污染。"""
    isolated = AgentStore(tmp_path / "api_agent_store.json")
    monkeypatch.setattr(api_deps, "AGENT_STORE", isolated)
    return isolated


@pytest.fixture()
def client(api_store: AgentStore) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def engine(store: AgentStore) -> AgentEngine:
    """无 LLM、无 db 的纯引擎。工具链只认 store 与注册表。"""
    return AgentEngine(store, llm=None)


@pytest.fixture()
def session(store: AgentStore):
    return store.create_session(user_id="tester", title="单测会话")


@pytest.fixture()
def bound_session(store: AgentStore):
    """绑定了数据集的会话。

    大多数分析工具都需要 dataset_id；不绑定的话引擎会先反问「哪个数据集」，
    那些用例就没法走到权限裁决与工具执行。
    """
    return store.create_session(user_id="tester", title="已绑定数据集", dataset_ids=[1])


@pytest.fixture()
def api_session(client: TestClient) -> str:
    resp = client.post("/api/v1/agent/sessions", json={"title": "API 单测", "dataset_ids": []})
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]["id"]
