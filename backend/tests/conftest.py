"""Phase 3（Prompt 011-027）共享 fixtures：临时存储 + 内存数据库 + API 测试客户端。"""

from __future__ import annotations

# 导入全部模型，确保表注册到 Base.metadata（必须在 from app.main import app 之前，
# 否则 `import app.models.*` 会把名字 app 重绑定为包模块）
import app.models.connector  # noqa: F401
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.file  # noqa: F401
import app.models.learning  # noqa: F401
import app.models.operation  # noqa: F401
import pytest
from app.core.database import Base
from app.data_engine.cache import VersionFrameCache
from app.main import app
from app.storage.local import LocalStorage
from app.storage.service import StorageService
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(autouse=True)
def _isolate_version_frame_cache():
    """每个测试使用干净的进程级版本缓存。

    背景（为什么必须隔离）：
    ``VersionFrameCache`` 是进程级单例，key 只有 ``(dataset_id, version)``，
    因为生产环境里一个进程只服务一个存储后端，版本快照又不可变。
    但测试每个用例都新建**内存库 + 临时 storage**，dataset_id 却总是从 1 开始，
    于是上一个用例的 ``(1, 1)`` 会被下一个用例命中，读到列名完全不同的旧 frame，
    表现为莫名奇妙的「目标列不存在于数据版本中」——与被测代码无关。

    这里在用例前后清空单例，使每个用例只看到自己写入的版本。
    """
    from app.data_engine.cache import get_version_frame_cache

    cache = get_version_frame_cache()
    cache.clear()
    yield cache
    cache.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def storage(tmp_path) -> StorageService:
    return StorageService(LocalStorage(root=tmp_path / "storage_root"))


@pytest.fixture()
def db() -> Session:
    # StaticPool：所有连接共享同一个内存 SQLite，
    # 避免 TestClient 线程池线程拿到全新的空库
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = factory()
    yield session
    session.close()
