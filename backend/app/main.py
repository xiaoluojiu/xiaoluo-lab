"""小洛实验室后端入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.v1 import api_router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期。"""
    setup_logging(settings.LOG_LEVEL)

    logger.info(
        "startup | app=%s env=%s version=%s",
        settings.APP_NAME,
        settings.APP_ENV,
        __version__,
    )

    yield

    logger.info("shutdown")


app = FastAPI(
    title="XiaoLuo Lab API",
    description="小洛实验室：智能数据科学实验平台",
    version=__version__,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)

app.include_router(api_router)


@app.get("/api/v1/health", tags=["system"])
async def health() -> dict[str, str]:
    """基础健康检查。"""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "version": __version__,
    }
