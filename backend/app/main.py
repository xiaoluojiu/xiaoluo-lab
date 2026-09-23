"""小洛实验室后端入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1 import api_router
from app.core.config import settings
from app.core.database import check_runtime_configuration
from app.core.logging import get_logger, setup_logging
from app.core.middleware import GZipMiddleware, RateLimitMiddleware, RequestContextMiddleware

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
    # 启动体检：把「能跑起来但跑不好」的配置问题在日志里说清楚，
    # 而不是等用户点了某个按钮才发现调不动。
    for message in check_runtime_configuration():
        logger.warning("startup check | %s", message)

    yield

    logger.info("shutdown")


app = FastAPI(
    title="XiaoLuo Lab API",
    description="小洛实验室：智能数据科学实验平台",
    version=__version__,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# 中间件顺序：Starlette 里 **后 add 的在外层**。
# 期望的内 -> 外：GZip -> RateLimit -> CORS -> RequestContext。
# 这样 OPTIONS 预检在到达限流/压缩之前就被消化，而异常兜底仍包住全部下游。
if settings.GZIP_ENABLED:
    app.add_middleware(GZipMiddleware, minimum_size=settings.GZIP_MINIMUM_SIZE)
if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)
origins = settings.cors_allowed_origins
if origins:
    logger.info("startup | CORS enabled for %s", origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        # 只放开本项目实际用到的动词与请求头：
        # 所有平台接口都套 ApiResponse 信封，前端只发 JSON 表单，
        # X-Request-ID 用于把浏览器请求与后端日志对上。
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
        max_age=600,
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
