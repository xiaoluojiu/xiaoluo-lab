"""FastAPI 依赖注入。"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.agent.engine import AgentEngine
from app.agent.llm import LLMProvider, build_default_provider
from app.agent.store import AgentStore
from app.connectors.service import ConnectorService
from app.core.database import SessionLocal, get_db
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.services.file_service import FileService
from app.storage.service import StorageService, get_storage
from app.workflow.runners import NODE_REQUIRED_CONFIG, build_default_runners
from app.workflow.service import WorkflowService

#: Agent 会话与运行的存储。进程级单例：SSE 通道需要跨请求复用同一份运行记录
#: （确认后要从同一条流补发 completed），所以不能按请求新建。
AGENT_STORE = AgentStore()

WORKFLOW_SERVICE = WorkflowService(
    node_runners=build_default_runners(),
    # 必需的节点参数规格：只在 run() 前预检生效，create/update 仍允许保存未配置的草稿。
    required_config_keys=NODE_REQUIRED_CONFIG,
)


def get_storage_service() -> StorageService:
    return get_storage()


def get_dataset_service(
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> DatasetService:
    return DatasetService(db, storage)


def get_data_engine_service(dataset_service: DatasetService = Depends(get_dataset_service)) -> DataEngineService:
    return DataEngineService(dataset_service)


def get_connector_service(
    db: Session = Depends(get_db),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ConnectorService:
    """数据库连接器服务（拓展功能）。

    依赖 DatasetService 是刻意的：连接器导入的产物必须是与其他数据集完全等价的
    DatasetVersion，因此抽取落盘走的是同一条 ``stage_version`` 链路，
    而不是连接器自己另搞一套存储。
    """
    return ConnectorService(db, dataset_service)


def get_experiment_service(
    db: Session = Depends(get_db),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ExperimentService:
    return ExperimentService(db, dataset_service)


def get_llm_provider() -> LLMProvider | None:
    """当前生效的 LLM Provider；不可用返回 None。

    「设置页关掉远程大模型」与「未配置 API Key」都返回 None —— 两者的处置
    完全相同：Agent 走规则路由 + 模板渲染，链路照常完成，只是不用模型。
    """
    return build_default_provider()


def get_agent_store() -> AgentStore:
    return AGENT_STORE


def build_agent_engine(llm: LLMProvider | None = None) -> AgentEngine:
    """构造一个**自包含**的引擎，供后台线程使用。

    为什么不用 Depends(get_agent_engine)：
    Agent 在 SSE 场景下跑在后台线程，而请求级 db session 会在请求结束时被
    关闭，后台线程再访问就会炸。引擎因此自带会话工厂、自建依赖链、用完自关。
    """
    return AgentEngine(AGENT_STORE, llm=llm if llm is not None else build_default_provider())


def get_file_service(db: Session = Depends(get_db), storage: StorageService = Depends(get_storage_service)) -> FileService:
    return FileService(db, storage)


def get_learning_service(db: Session = Depends(get_db)) -> LearningService:  # noqa: F821
    from app.learning.service import LearningService

    return LearningService(db)


def get_workflow_service() -> WorkflowService:
    return WORKFLOW_SERVICE
