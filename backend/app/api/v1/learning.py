"""学习中心 API。

端点一览：
    GET    /learning/tracks                  两条主线的定义
    GET    /learning/experiments             实验目录（含该学习者进度）
    GET    /learning/experiments/{key}       单个实验详情（含草稿与历史）
    POST   /learning/experiments/{key}/check 检查一次提交（可持久化）
    POST   /learning/experiments/{key}/draft 保存草稿
    DELETE /learning/experiments/{key}/progress  重置进度
    GET    /learning/overview                学习总览

设计要点：
- ``learner_id`` 是查询/表单参数，默认 anonymous。项目无鉴权（既定取舍），
  因此这里只做「学习者分区」而非「身份认证」，安全讨论写在论文局限里。
- 检查同时做 AST 静态分析与真实数据集探针，**不执行提交的代码**。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import get_dataset_service, get_learning_service
from app.core.exceptions import ValidationException
from app.learning.service import LearningService
from app.schemas.common import ApiResponse
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/learning", tags=["learning"])

# 学习场景下预览用的行列上限：够看清结构，又不至于把大表搬进内存
_CANDIDATE_COLUMNS = 40


class CheckRequest(BaseModel):
    code: str = Field(default="", description="学习者提交的代码")
    dataset_id: int | None = Field(default=None, description="用于数据侧校验与上下文")
    version: int | None = Field(default=None, description="数据版本（缺省最新）")
    learner_id: str = Field(default="anonymous")
    persist: bool = Field(
        default=True, description="是否把这次检查记入进度（临时预览可传 false）"
    )


class DraftRequest(BaseModel):
    code: str = Field(default="")
    dataset_id: int | None = None
    learner_id: str = Field(default="anonymous")


class CustomCardCreate(BaseModel):
    title: str = Field(default="", description="卡片标题")
    kind: str = Field(default="practice", description="practice / checklist")
    goal: str = Field(default="", description="目标描述（练习卡用于 AI 点评上下文）")
    code: str = Field(default="", description="起始代码（练习卡）")
    items: list[dict] = Field(default_factory=list, description="要点列表（清单卡）")
    learner_id: str = Field(default="anonymous")


class CustomCardUpdate(BaseModel):
    title: str | None = None
    goal: str | None = None
    code: str | None = None
    items: list[dict] | None = None
    learner_id: str = Field(default="anonymous")


@router.get("/tracks", response_model=ApiResponse[list])
def list_tracks() -> ApiResponse[list]:
    return ApiResponse[list](data=LearningService().list_tracks())


@router.get("/overview", response_model=ApiResponse[dict])
def overview(
    learner_id: str = Query("anonymous"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    """学习总览：完成度、主线分布、最近活动、推荐下一个实验。"""
    return ApiResponse[dict](data=service.overview(learner_id))


@router.get("/experiments", response_model=ApiResponse[list])
def list_experiments(
    learner_id: str = Query("anonymous"),
    track: str | None = Query(None, description="ml / llm，缺省返回全部"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[list]:
    return ApiResponse[list](
        data=service.list_experiments(learner_id, track=track)
    )


@router.get("/experiments/{key}", response_model=ApiResponse[dict])
def get_experiment(
    key: str,
    learner_id: str = Query("anonymous"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=service.get_experiment(learner_id, key))


@router.post("/experiments/{key}/check", response_model=ApiResponse[dict])
def check_submission(
    key: str,
    body: CheckRequest,
    service: LearningService = Depends(get_learning_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[dict]:
    """检查提交：AST 结构 + 真实数据探针，返回逐条要求的达成情况。

    这里加载真实数据集（而非让学习者上传），保证数据侧校验用的就是
    他界面上看到的那份，判定与所见一致。
    """
    if not body.code.strip():
        raise ValidationException("提交的代码为空")

    df = None
    dataset_error: str | None = None
    if body.dataset_id is not None:
        try:
            version_row = dataset_service.get_version_row(body.dataset_id, body.version)
            df = dataset_service.load_version(body.dataset_id, version_row.version)
        except Exception as exc:  # noqa: BLE001 - 数据不可用不应阻断结构检查
            dataset_error = str(exc)

    result = service.check(
        body.learner_id,
        key,
        body.code,
        df,
        dataset_id=body.dataset_id,
        persist=body.persist,
    )
    if dataset_error:
        result["dataset_warning"] = f"数据集读取失败，仅完成结构检查：{dataset_error}"
    return ApiResponse[dict](data=result)


@router.post("/experiments/{key}/draft", response_model=ApiResponse[dict])
def save_draft(
    key: str,
    body: DraftRequest,
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    """保存草稿：落代码与数据集，不改完成状态。"""
    return ApiResponse[dict](
        data=service.save_draft(
            body.learner_id, key, body.code, dataset_id=body.dataset_id
        )
    )


@router.delete("/experiments/{key}/progress", response_model=ApiResponse[dict])
def reset_progress(
    key: str,
    learner_id: str = Query("anonymous"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=service.reset(learner_id, key))


@router.get("/datasets/{dataset_id}/learning-context", response_model=ApiResponse[dict])
def learning_context(
    dataset_id: int,
    version: int | None = Query(None),
    service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[dict]:
    """给学习台的「数据体检」面板：真实行数、列清单、缺失情况。

    学习场景需要先看懂数据再动手，这个接口把「看数据」从 AI 对话里
    拆出来，变成确定性的、可复核的展示。
    """
    version_row = service.get_version_row(dataset_id, version)
    dataset = service.get(dataset_id)
    df = service.load_version(dataset_id, version_row.version)

    columns: list[dict[str, Any]] = []
    for name, dtype in list(df.schema.items())[:_CANDIDATE_COLUMNS]:
        series = df[name]
        nulls = int(series.null_count())
        columns.append(
            {
                "name": name,
                "dtype": str(dtype),
                "is_numeric": bool(dtype.is_numeric()),
                "null_count": nulls,
                "null_rate": round(nulls / df.height, 4) if df.height else 0.0,
                "unique_count": int(series.n_unique()),
            }
        )

    numeric_columns = [c["name"] for c in columns if c["is_numeric"]]
    # 可作为分类标签的候选：基数低且非浮点
    label_candidates = [
        c["name"]
        for c in columns
        if not c["is_numeric"] and 1 < c["unique_count"] <= max(50, int(df.height * 0.05))
    ]
    # 可作为回归目标的候选：连续数值
    target_candidates = [
        c["name"]
        for c in columns
        if c["is_numeric"] and c["unique_count"] > max(10, int(df.height * 0.1))
    ]

    return ApiResponse[dict](
        data={
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "version": version_row.version,
            "row_count": df.height,
            "column_count": df.width,
            "columns": columns,
            "numeric_columns": numeric_columns,
            "label_candidates": label_candidates,
            "target_candidates": target_candidates,
            "missing_cells": int(df.null_count().sum_horizontal().item())
            if df.width
            else 0,
        }
    )


# ----------------------------------------------------------------------
# 自建学习卡片（practice / checklist）
# ----------------------------------------------------------------------


@router.get("/custom-cards", response_model=ApiResponse[list])
def list_custom_cards(
    learner_id: str = Query("anonymous"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[list]:
    return ApiResponse[list](data=service.list_custom_cards(learner_id))


@router.post("/custom-cards", response_model=ApiResponse[dict])
def create_custom_card(
    body: CustomCardCreate,
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](
        data=service.create_custom_card(
            body.learner_id,
            title=body.title,
            kind=body.kind,
            goal=body.goal,
            code=body.code,
            items=body.items,
        )
    )


@router.get("/custom-cards/{card_id}", response_model=ApiResponse[dict])
def get_custom_card(
    card_id: int,
    learner_id: str = Query("anonymous"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=service.get_custom_card(learner_id, card_id))


@router.patch("/custom-cards/{card_id}", response_model=ApiResponse[dict])
def update_custom_card(
    card_id: int,
    body: CustomCardUpdate,
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](
        data=service.update_custom_card(
            body.learner_id,
            card_id,
            title=body.title,
            goal=body.goal,
            code=body.code,
            items=body.items,
        )
    )


@router.delete("/custom-cards/{card_id}", response_model=ApiResponse[dict])
def delete_custom_card(
    card_id: int,
    learner_id: str = Query("anonymous"),
    service: LearningService = Depends(get_learning_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=service.delete_custom_card(learner_id, card_id))
