"""FastAPI 依赖注入。"""
from __future__ import annotations
from fastapi import Depends
from sqlalchemy.orm import Session
from app.agent.llm.base import LLMProvider
from app.agent.llm.capabilities import LLMCapabilities
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.agent.runtime.models import AgentStore
from app.agent.runtime.agent_runtime import AgentRuntime
from app.core.config import settings
from app.core.database import get_db
from app.connectors.service import ConnectorService
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.learning.service import LearningService
from app.services.dataset_service import DatasetService
from app.services.file_service import FileService
from app.storage.service import StorageService, get_storage
from app.workflow.runners import NODE_REQUIRED_CONFIG, build_default_runners
from app.workflow.service import WorkflowService

AGENT_STORE = AgentStore()
WORKFLOW_SERVICE = WorkflowService(
    node_runners=build_default_runners(),
    # 必需的节点参数规格：只在 run() 前预检生效，create/update 仍允许保存未配置的草稿。
    required_config_keys=NODE_REQUIRED_CONFIG,
)

def get_storage_service() -> StorageService:
    return get_storage()

def get_dataset_service(db: Session = Depends(get_db), storage: StorageService = Depends(get_storage_service)) -> DatasetService:
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

def get_experiment_service(db: Session = Depends(get_db), dataset_service: DatasetService = Depends(get_dataset_service)) -> ExperimentService:
    return ExperimentService(db, dataset_service)

def get_llm_provider() -> LLMProvider | None:
    """默认 Agent Provider；具体厂商由 Base URL / OpenAI-compatible 协议决定。

    返回 None ⇒ 上层退回平台自带的规则规划器。两种情况：
    ① 设置页把「启用远程 API 大模型」关掉了（用于测试平台自带小模型）；
    ② 从未配置过 API Key。
    """
    if not settings.remote_llm_available():
        return None
    capabilities = LLMCapabilities(
        chat=True,
        context_window=settings.LLM_CONTEXT_WINDOW,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
    )
    return OpenAICompatibleProvider(settings.LLM_BASE_URL, settings.LLM_MODEL, settings.LLM_API_KEY, capabilities=capabilities)

def get_agent_runtime(
    data_engine: DataEngineService = Depends(get_data_engine_service),
    experiment_service: ExperimentService = Depends(get_experiment_service),
    db: Session = Depends(get_db),
    llm: LLMProvider | None = Depends(get_llm_provider),
) -> AgentRuntime:
    # 统一 Loop：决策/执行/收尾全部在 AgentLoop 内，这里只注入基础设施与共享 store。
    return AgentRuntime(data_engine, experiment_service=experiment_service, db=db, llm=llm, store=AGENT_STORE)

def get_file_service(db: Session = Depends(get_db), storage: StorageService = Depends(get_storage_service)) -> FileService:
    return FileService(db, storage)

def get_learning_service(db: Session = Depends(get_db)) -> LearningService:
    return LearningService(db)

def get_workflow_service() -> WorkflowService:
    return WORKFLOW_SERVICE
