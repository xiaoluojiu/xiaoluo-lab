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
- GET  /experiments/runs/{run_id}/report         详细报告导出（HTML/Markdown/PDF）
- GET  /experiments/runs/{run_id}/predict-export 全量推理结果导出（CSV/Parquet）
- POST /experiments/compare             运行/实验对比
- DELETE /experiments/{id}              删除实验
- DELETE /experiments                   删除全部实验
- POST /experiments/batch-delete        批量删除实验（删除选中）
- DELETE /experiments/runs/{run_id}     删除单次运行（实验保留）
- POST /experiments/runs/batch-delete   批量删除运行
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app.api.deps import get_dataset_service, get_experiment_service, get_storage_service
from app.api.v1._serializers import experiment_dict, run_dict
from app.api.v1.files import _content_disposition
from app.core.database import SessionLocal
from app.experiments.service import ExperimentService
from app.ml_engine.metadata import build_catalog
from app.ml_engine.registry import MODEL_REGISTRY
from app.models.dataset_version import DatasetVersion
from app.notifications import notify
from app.reports.html import export_html
from app.reports.markdown import export_markdown
from app.reports.pdf import export_pdf
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


class ExperimentIdsRequest(BaseModel):
    """批量删除实验的入参。「删除选中」与「一键删除全部」共用这一个端点：
    后者只是把当前列表里所有 id 传进来，不必再维护第二条清空路径。"""

    experiment_ids: list[int] = Field(default_factory=list)


class RunIdsRequest(BaseModel):
    run_ids: list[int] = Field(default_factory=list)


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


class OptimizeRequest(BaseModel):
    """自动超参搜索请求。"""

    n_iter: int = Field(
        default=20, ge=1, le=200, description="随机搜索的候选组数（每组都要跑 folds 折）"
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
    # 训练数据预算：二者互斥；均缺省时由后端按 ML_MAX_TRAIN_ROWS 自动治理
    max_rows: int | None = Field(
        default=None, gt=0, description="参与训练的数据行数上限（随机抽样）"
    )
    train_fraction: float | None = Field(
        default=None, gt=0, le=1, description="参与训练的数据占比（0~1），1 表示全量"
    )
    # 学习曲线需要额外多次拟合，默认关闭；概率校准无需开关（二分类自动计算）
    enable_learning_curve: bool = False
    # 交叉验证：5 折 × 每折重新拟合，训练耗时约为普通的 5 倍，默认关闭
    enable_cv: bool = False
    seed: int | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check_budget_exclusive(self) -> "TrainRequest":
        if self.max_rows is not None and self.train_fraction is not None:
            raise ValueError("max_rows 与 train_fraction 只能二选一，不能同时指定")
        return self


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
        max_rows=body.max_rows,
        train_fraction=body.train_fraction,
        enable_learning_curve=body.enable_learning_curve,
        enable_cv=body.enable_cv,
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
                max_rows=body.max_rows,
                train_fraction=body.train_fraction,
                enable_learning_curve=body.enable_learning_curve,
                enable_cv=body.enable_cv,
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
    with_metrics: bool = Query(
        False,
        description=(
            "是否附带每个实验「最近一次成功运行」的指标。"
            "列表页要回答「哪个实验效果更好」时需要它"
        ),
    ),
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[Pagination[dict]]:
    if not with_metrics:
        items, total = experiment_service.list(page=page, page_size=page_size)
        payload = [experiment_dict(e) for e in items]
    else:
        items, latest, total = experiment_service.list_with_latest_runs(
            page=page, page_size=page_size
        )
        payload = []
        for exp in items:
            row = experiment_dict(exp)
            run = latest.get(exp.id)
            row["latest_run"] = (
                {
                    "run_id": run.id,
                    "status": run.status,
                    "metrics": dict(run.metrics or {}),
                    "runtime": run.runtime,
                    "created_at": str(run.created_at) if run.created_at else None,
                }
                if run is not None
                else None
            )
            payload.append(row)

    result = Pagination[dict](
        items=payload,
        page_info=PageInfo.build(page, page_size, total),
    )
    return ApiResponse[Pagination[dict]](data=result)


@router.get("/runs/{run_id}", response_model=ApiResponse[dict])
def get_run(
    run_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=run_dict(experiment_service.get_run(run_id)))


@router.get("/runs/{run_id}/report")
def export_run_report(
    run_id: int,
    format: str = Query("html", pattern="^(html|markdown|pdf)$"),
    experiment_service: ExperimentService = Depends(get_experiment_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Response:
    """一键导出单次运行的详细报告（HTML / Markdown / PDF）。

    报告体由 `experiments/run_report.py` 装配成既有的 `Report` 模型，
    渲染完全复用 `app/reports` 的三个导出器 —— 不另写一套模板，
    这样报告的外观与数据集分析报告天然一致。
    """
    run = experiment_service.get_run(run_id)
    if run.status != "success":
        raise HTTPException(status_code=400, detail="仅成功运行可导出详细报告")

    dataset_name = ""
    try:
        exp = experiment_service.get(run.experiment_id)
        version = experiment_service.db.get(DatasetVersion, exp.dataset_version_id)
        if version is not None:
            dataset_name = dataset_service.get(version.dataset_id).name
    except Exception:  # noqa: BLE001 - 数据集名称查不到不影响报告主体
        dataset_name = ""

    from app.experiments.run_report import build_run_report

    report = build_run_report(run, dataset_name=dataset_name)
    filename = f"ML-Run{run_id}-{datetime.now().strftime('%Y%m%d')}"
    if format == "markdown":
        return Response(
            content=export_markdown(report).encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": _content_disposition(filename + ".md")},
        )
    if format == "pdf":
        return Response(
            content=export_pdf(report),
            media_type="application/pdf",
            headers={"Content-Disposition": _content_disposition(filename + ".pdf")},
        )
    return Response(
        content=export_html(report).encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(filename + ".html")},
    )


@router.get("/runs/{run_id}/predict-export")
def export_predictions(
    run_id: int,
    dataset_id: int | None = Query(None, description="推理数据源数据集（缺省用训练时的版本）"),
    version: int | None = Query(None, description="数据版本号（缺省最新）"),
    threshold: float | None = Query(
        None, gt=0.0, lt=1.0, description="决策阈值（仅二分类；缺省 0.5）"
    ),
    format: str = Query("csv", pattern="^(csv|parquet)$"),
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> Response:
    """导出一次推理的**全量**结果（原始列 + 预测列 + 概率列）。

    与 `POST /ml/predict` 的区别只在于出口：那边返回前 N 行给界面看，
    这里把每一行都写成文件 —— 批量推理的意义本来就是「每行都要有结果」，
    只给预览等于让用户拿着 200 行去汇报全量结论。
    """
    content, ext = experiment_service.export_predictions(
        run_id,
        dataset_id=dataset_id,
        version=version,
        threshold=threshold,
        format=format,
    )
    filename = f"ML-Run{run_id}-predictions-{datetime.now().strftime('%Y%m%d')}.{ext}"
    media = (
        "application/vnd.apache.parquet" if ext == "parquet" else "text/csv; charset=utf-8"
    )
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": _content_disposition(filename)},
    )


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


@router.post("/{experiment_id}/optimize", response_model=ApiResponse[dict])
def optimize_experiment(
    experiment_id: int,
    body: OptimizeRequest,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """自动搜索超参数：RandomizedSearchCV over 5 折 CV，预处理在折内拟合。

    **只返回最佳参数与 Top10，不改动实验配置** —— 采纳与否由用户在界面上决定：
    一次搜索的训练量是普通训练的 n_iter × folds 倍，静默改配置再重训
    既昂贵又无法解释「这些参数是哪来的」。
    """
    return ApiResponse[dict](
        data=experiment_service.optimize(experiment_id, n_iter=body.n_iter)
    )


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


@router.post("/runs/batch-delete", response_model=ApiResponse[dict])
def delete_runs(
    body: RunIdsRequest,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """批量删除运行记录（含各自的模型产物）；实验定义本身保留。

    路由必须声明在 ``/{experiment_id}`` 之前不是必需的（段数不同不会冲突），
    但它与 ``/runs/{run_id}`` 同属「运行」语义，放一起便于日后维护。
    """
    count = experiment_service.delete_runs(body.run_ids)
    return ApiResponse[dict](
        data={"deleted": True, "count": count, "requested": len(body.run_ids)}
    )


@router.delete("/runs/{run_id}", response_model=ApiResponse[dict])
def delete_run(
    run_id: int,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """删除单次运行：实验配置保留，只清掉这一次的运行记录与模型产物。"""
    experiment_service.delete_run(run_id)
    return ApiResponse[dict](data={"deleted": True, "run_id": run_id})


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


@router.post("/batch-delete", response_model=ApiResponse[dict])
def delete_experiments(
    body: ExperimentIdsRequest,
    experiment_service: ExperimentService = Depends(get_experiment_service),
) -> ApiResponse[dict]:
    """批量删除实验（「删除选中」与「一键删除全部」共用）。"""
    count = experiment_service.delete_experiments(body.experiment_ids)
    return ApiResponse[dict](
        data={"deleted": True, "count": count, "requested": len(body.experiment_ids)}
    )


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
