"""Experiments API（Prompt 200 / 202 合并）。

- GET  /ml/models                       模型目录（Model Registry）
- POST /ml/train                        一键训练：创建实验 + 运行 + 返回指标
- POST /ml/predict                      加载训练产物对新数据推理
- GET  /ml/explain/{run_id}             模型解释（特征重要性）
- POST /experiments                     创建实验
- GET  /experiments                     实验列表（分页）
- GET  /experiments/{id}                实验详情
- POST /experiments/{id}/run            运行实验
- GET  /experiments/{id}/runs           运行历史
- GET  /experiments/runs/{run_id}       单次运行详情
- POST /experiments/compare             运行/实验对比
"""

from __future__ import annotations

import json
import logging
import queue
import threading

from fastapi import APIRouter, Depends, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_dataset_service, get_experiment_service, get_storage_service
from app.api.v1._serializers import experiment_dict, run_dict
from app.core.database import SessionLocal
from app.experiments.service import ExperimentService
from app.ml_engine.metadata import build_catalog
from app.ml_engine.registry import MODEL_REGISTRY
from app.notifications import notify
from app.schemas.common import ApiResponse, PageInfo, Pagination
from app.services.dataset_service import DatasetService
from app.storage.service import StorageService

router = APIRouter(prefix="/experiments", tags=["experiments"])

ml_router = APIRouter(prefix="/ml", tags=["ml"])


class ExperimentCreate(BaseModel):
    dataset_id: int
    version: int | None = Field(default=None, description="数据版本号（缺省最新）")
    task: str
    model: str
    target_column: str | None = None
    parameters: dict = Field(default_factory=dict)
    preprocessing: dict = Field(default_factory=dict)
    excluded_columns: list[str] = Field(default_factory=list, description="训练前排除的特征列")
    test_size: float | None = Field(
        default=None, gt=0, lt=1, description="测试集比例，缺省 0.2"
    )
    seed: int | None = None
    description: str = ""


class CompareRequest(BaseModel):
    run_ids: list[int] = Field(default_factory=list)
    experiment_ids: list[int] = Field(default_factory=list)


class PredictRequest(BaseModel):
    run_id: int = Field(description="用于推理的成功运行 ID")
    dataset_id: int | None = Field(default=None, description="推理数据源数据集")
    version: int | None = Field(default=None, description="数据版本号（缺省最新）")
    limit: int | None = Field(default=200, ge=1, le=5000, description="预览返回行数")
    # 训练后参数：改它不需要重新训练。只对二分类有效，其余场景会显式报错而非静默忽略。
    # 边界取开区间 (0, 1)：0 与 1 会把所有样本判成同一类，没有选择价值。
    threshold: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="决策阈值（仅二分类；缺省用 sklearn 默认的 0.5）",
    )


class TrainRequest(BaseModel):
    dataset_id: int
    version: int | None = Field(default=None, description="数据版本号（缺省最新）")
    task: str = "classification"
    model: str = "logistic_regression"
    target_column: str | None = None
    parameters: dict = Field(default_factory=dict)
    preprocessing: dict = Field(default_factory=dict)
    excluded_columns: list[str] = Field(default_factory=list, description="训练前排除的特征列")
    test_size: float | None = Field(
        default=None, gt=0, lt=1, description="测试集比例，缺省 0.2"
    )
    seed: int | None = None
    description: str = ""


@ml_router.get("/catalog", response_model=ApiResponse[dict])
def ml_catalog() -> ApiResponse[dict]:
    """ML 教学目录：流程步骤 / 参数说明 / 模型参数 / 指标解读。

    单一事实源在 `app/ml_engine/metadata.py`，本端点只做透出，
    供「机器学习」页与「学习中心」直接渲染，避免文档与代码两套口径。
    """
    catalog = build_catalog(MODEL_REGISTRY.list())
    return ApiResponse[dict](data=catalog)


@ml_router.get("/models", response_model=ApiResponse[list])
def list_models() -> ApiResponse[list]:
    """模型目录：name / task / 是否支持 random_state（隐藏 dimensionality 任务）。"""
    data = []
    for m in MODEL_REGISTRY.list():
        if m.get("task") == "dimensionality":
            continue
        try:
            meta = MODEL_REGISTRY.metadata(m["name"])
        except Exception:  # noqa: BLE001 - 模型目录不应因单个模型失败而整体不可用
            meta = {}
        data.append({**m, **meta})
    return ApiResponse[list](data=data)


@ml_router.post("/predict", response_model=ApiResponse[dict])
def predict(
    body: PredictRequest,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """用已训练模型推理：加载 model.pkl + pipeline.pkl，输出预测与概率预览。

    可传 threshold 覆盖决策阈值（二分类专用）——它作用在概率上、不动模型，
    因此改阈值不需要重新训练。
    """
    data = experiment_service.predict(
        body.run_id,
        dataset_id=body.dataset_id,
        version=body.version,
        limit=body.limit,
        threshold=body.threshold,
    )
    return ApiResponse[dict](data=data)


@ml_router.get("/explain/{run_id}", response_model=ApiResponse[dict])
def explain(
    run_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """解释一次成功运行的模型（特征重要性）。"""
    return ApiResponse[dict](data=experiment_service.explain(run_id))


@ml_router.post("/train", response_model=ApiResponse[dict])
def train(
    body: TrainRequest,
    experiment_service: ExperimentService = Depends(get_experiment_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[dict]:
    """创建实验并立即运行（数据版本绑定，防泄漏预处理）。"""
    version_row = dataset_service.get_version_row(body.dataset_id, body.version)
    exp = experiment_service.create(
        dataset_id=body.dataset_id,
        dataset_version_id=version_row.id,
        task=body.task,
        model=body.model,
        target_column=body.target_column,
        parameters=body.parameters or None,
        preprocessing=body.preprocessing or None,
        excluded_columns=body.excluded_columns,
        seed=body.seed,
        description=body.description,
        test_size=body.test_size,
    )
    run = experiment_service.run(exp.id)
    _notify_training_done(run, exp)
    return ApiResponse[dict](data={"experiment": experiment_dict(exp), "run": run_dict(run)})


@ml_router.post("/train/stream")
def train_stream(
    body: TrainRequest,
    storage: StorageService = Depends(get_storage_service),
) -> StreamingResponse:
    """流式训练：实时推送进度事件，训练完成后推送完整结果。

    大表训练可能 30s+，同步接口会撞上前端 15s 超时。这里把训练放到后台线程
    （独立 DB session），进度经进程内队列桥接到 SSE，前端边收边渲染进度条；
    训练结束（无论成败）都会收到 result / error 事件。

    SSE 事件：
    - event: progress   data: {stage,label,done,total,detail}
    - event: result     data: {experiment, run}
    - event: error      data: {error}
    - event: done       data: {}
    """
    progress_q: queue.Queue = queue.Queue()

    def worker() -> None:
        db = SessionLocal()
        try:
            dataset_service = DatasetService(db, storage)
            experiment_service = ExperimentService(db, dataset_service)
            version_row = dataset_service.get_version_row(body.dataset_id, body.version)
            exp = experiment_service.create(
                dataset_id=body.dataset_id,
                dataset_version_id=version_row.id,
                task=body.task,
                model=body.model,
                target_column=body.target_column,
                parameters=body.parameters or None,
                preprocessing=body.preprocessing or None,
                excluded_columns=body.excluded_columns,
                seed=body.seed,
                description=body.description,
                test_size=body.test_size,
            )
            run = experiment_service.run(exp.id, on_progress=progress_q.put)
            progress_q.put(
                {
                    "type": "result",
                    "data": {"experiment": experiment_dict(exp), "run": run_dict(run)},
                }
            )
            _notify_training_done(run, exp)
        except Exception as exc:  # noqa: BLE001 - 训练失败也要把错误送回前端
            logging.getLogger(__name__).exception("流式训练后台任务失败")
            progress_q.put({"type": "error", "error": str(exc)})
            # 创建/运行阶段失败时无法拿到 run，退化为一条系统级失败通知。
            notify(
                "system",
                "训练未完成",
                f"模型 {body.model} 的训练在启动阶段失败：{str(exc)[:200]}",
                link="/ml",
            )
        finally:
            db.close()
            progress_q.put(None)  # 流结束标记

    threading.Thread(target=worker, daemon=True, name="ml-train").start()

    def generate():
        while True:
            item = progress_q.get()
            if item is None:
                break
            try:
                kind = item.get("type", "progress")
                if kind == "result":
                    # jsonable_encoder 与同步端点同一套转换，正确处理 numpy 等类型
                    payload = json.dumps(jsonable_encoder(item["data"]), ensure_ascii=False)
                    yield f"event: result\ndata: {payload}\n\n"
                elif kind == "error":
                    payload = json.dumps({"error": item["error"]}, ensure_ascii=False)
                    yield f"event: error\ndata: {payload}\n\n"
                else:
                    payload = json.dumps(item, ensure_ascii=False, default=str)
                    yield f"event: progress\ndata: {payload}\n\n"
            except Exception as exc:  # noqa: BLE001 - 任何异常都不应中断 SSE
                logging.getLogger(__name__).exception("训练流生成事件失败")
                payload = json.dumps({"error": f"训练流生成失败：{exc}"}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"
                break
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("", response_model=ApiResponse[dict])
def create_experiment(
    body: ExperimentCreate,
    experiment_service: ExperimentService = Depends(get_experiment_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[dict]:
    version_row = dataset_service.get_version_row(body.dataset_id, body.version)
    exp = experiment_service.create(
        dataset_id=body.dataset_id,
        dataset_version_id=version_row.id,
        task=body.task,
        model=body.model,
        target_column=body.target_column,
        parameters=body.parameters or None,
        preprocessing=body.preprocessing or None,
        excluded_columns=body.excluded_columns,
        seed=body.seed,
        description=body.description,
        test_size=body.test_size,
    )
    return ApiResponse[dict](data=experiment_dict(exp))


@router.get("", response_model=ApiResponse[Pagination[dict]])
def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[Pagination[dict]]:
    items, total = experiment_service.list(page=page, page_size=page_size)
    result = Pagination[dict](
        items=[experiment_dict(e) for e in items],
        page_info=PageInfo.build(page, page_size, total),
    )
    return ApiResponse[Pagination[dict]](data=result)


@router.get("/runs/{run_id}", response_model=ApiResponse[dict])
def get_run(
    run_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=run_dict(experiment_service.get_run(run_id)))


@router.get("/{experiment_id}", response_model=ApiResponse[dict])
def get_experiment(
    experiment_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=experiment_dict(experiment_service.get(experiment_id)))


@router.get("/{experiment_id}/narrative", response_model=ApiResponse[dict])
def get_experiment_narrative(
    experiment_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """实验的结构化解读（第五层改造）。

    把「一次实验到底验证了什么、结果说明什么、失败为什么、下一步做什么」
    组装成固定四字段，避免实验列表变成一堆没有解释的状态与数字。
    """
    experiment = experiment_service.get(experiment_id)
    runs = experiment_service.list_runs(experiment_id)
    compare = None
    if len(runs) > 1:
        from app.experiments.comparator import ExperimentComparator

        compare = ExperimentComparator().compare(runs)
    from app.experiments.structured import build_narrative

    narrative = build_narrative(experiment, runs, compare=compare)
    return ApiResponse[dict](data=narrative.to_dict())


@router.get("/{experiment_id}/runs", response_model=ApiResponse[list])
def list_runs(
    experiment_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[list]:
    experiment_service.get(experiment_id)
    return ApiResponse[list](
        data=[run_dict(r) for r in experiment_service.list_runs(experiment_id)]
    )


@router.post("/{experiment_id}/run", response_model=ApiResponse[dict])
def run_experiment(
    experiment_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    run = experiment_service.run(experiment_id)
    exp = experiment_service.get(experiment_id)
    _notify_training_done(run, exp)
    return ApiResponse[dict](data=run_dict(run))


@router.delete("/{experiment_id}", response_model=ApiResponse[dict])
def delete_experiment(
    experiment_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    experiment_service.delete(experiment_id)
    return ApiResponse[dict](
        data={"deleted": True, "experiment_id": experiment_id}
    )


@router.delete("", response_model=ApiResponse[dict])
def delete_all_experiments(
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    count = experiment_service.delete_all()
    return ApiResponse[dict](data={"deleted": True, "count": count})


@router.post("/compare", response_model=ApiResponse[dict])
def compare(
    body: CompareRequest,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """运行对比（run_ids 优先）或实验对比（experiment_ids）。"""
    if body.run_ids:
        result = experiment_service.compare_runs(body.run_ids)
    elif body.experiment_ids:
        result = experiment_service.compare_experiments(body.experiment_ids)
    else:
        from app.core.exceptions import ValidationException

        raise ValidationException("run_ids 与 experiment_ids 至少提供一个")
    return ApiResponse[dict](data=result.to_dict())


def _notify_training_done(run, exp) -> None:
    """训练结束后写一条通知（成功或失败）。"""
    if run.status == "success":
        notify(
            "training",
            "训练完成",
            f"模型 {exp.model} 训练成功（Run #{run.id}），可查看指标或直接推理。",
            link="/ml",
        )
    else:
        notify(
            "training",
            "训练失败",
            f"模型 {exp.model} 训练失败（Run #{run.id}）：{str(run.error or '未知错误')[:200]}",
            link="/ml",
        )
