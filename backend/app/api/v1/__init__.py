"""Prompt 205：API Router 统一注册入口。

main.py 只 include 本模块的 api_router，不在入口堆积 router 代码。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    agent,
    connectors,
    dataset_analysis,
    datasets,
    eda,
    experiments,
    files,
    learning,
    merge,
    notifications,
    processing,
    reports,
    settings,
    workflow,
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(datasets.router)
api_router.include_router(dataset_analysis.router)
api_router.include_router(files.router)
api_router.include_router(processing.router)
api_router.include_router(merge.router)
api_router.include_router(eda.router)
api_router.include_router(experiments.router)
api_router.include_router(experiments.ml_router)
api_router.include_router(workflow.router)
api_router.include_router(agent.router)
api_router.include_router(reports.router)
api_router.include_router(settings.router)
api_router.include_router(learning.router)
api_router.include_router(notifications.router)
api_router.include_router(connectors.router)
