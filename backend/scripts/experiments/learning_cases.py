"""小洛实验室 · 学习系统实验脚本（Prompt 244）。

演示 8 个学习领域的「知识 / 数据 / 实验 / 结果 / AI 解释」闭环：
  1. 数据清洗（缺失值、重复行、异常值）
  2. 数据合并（两表 Join）
  3. EDA（描述统计、相关性）
  4. 分类（逻辑回归）
  5. 回归（线性回归）
  6. 聚类（KMeans）
  7. LLM 工具调用（Agent 用工具）
  8. Agent（完整运行时：规划 + 执行）

用法：
    python -m scripts.experiments.learning_cases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

# 注册全部 ORM 模型到 Base.metadata
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.file  # noqa: F401
import app.models.operation  # noqa: F401
import polars as pl
from app.agent.executor.executor import AgentExecutor
from app.agent.llm.mock import MockLLM
from app.agent.permission.models import ROLE_PERMISSIONS
from app.agent.runtime.runtime import AgentRuntime
from app.analysis import CorrelationAnalyzer, DescriptiveAnalyzer
from app.core.database import Base
from app.data_engine.merge.plan import JoinKey, MergePlan
from app.data_engine.operations import drop_duplicates, handle_missing
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.storage.local import LocalStorage
from app.storage.service import StorageService
from app.tools.base import ToolServices
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401 - 导入即注册
from app.tools.context import ToolExecutionContext
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEP = "-" * 72
SECTION = "=" * 72


# ============================================================
# 环境
# ============================================================
def setup_env() -> dict[str, Any]:
    engine_db = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db, autoflush=False, expire_on_commit=False)
    db = Session()
    storage = StorageService(LocalStorage(root=Path(tempfile.mkdtemp(prefix="xl-learning-"))))
    ds = DatasetService(db, storage)
    data_engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    return {"db": db, "storage": storage, "ds": ds, "engine": data_engine, "exp": exp}


def _make_dataset(env: dict[str, Any], name: str, df: pl.DataFrame) -> int:
    ds = env["ds"]
    dataset = ds.create(name, f"学习用数据集 {name}")
    ds.create_version(dataset.id, df)
    return dataset.id


def _print_block(title: str, body: str) -> None:
    print(f"【{title}】")
    print(body)


def _ai_explain(topic: str, fact: str) -> str:
    """规则化 AI 解释（确定性，锚定当前实验结果）。

    MockLLM 在领域 7/8 单独演示 LLM 工具调用与 Agent 编排；
    此处用规则解释保证每次输出都对应真实实验事实。
    """
    return f"[AI 解释 · {topic}] {fact}"


# ============================================================
# 领域 1：数据清洗
# ============================================================
def domain_1_cleaning(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 1：数据清洗（缺失值 / 重复行 / 异常值）")
    print(SECTION)
    _print_block("知识", "数据清洗是数据分析的第一步：处理缺失值（drop/mean/median/mode/constant）、\n"
                       "去除重复行、识别异常值。清洗后数据才能进入建模。")

    df = pl.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "age": [20, None, 22, 200, 25],  # 缺失 + 离群 200
            "city": ["北京", "上海", "上海", None, "成都"],
        }
    )
    _print_block("数据", f"原始数据（{df.height} 行）：\n{df}")

    # 1) 中位数填充 age 缺失
    filled = handle_missing(df, strategy="median", columns=["age"])
    # 2) 常量填充 city 缺失
    filled = handle_missing(filled, strategy="constant", columns=["city"], value="未知")
    # 3) 去重（按 id）
    deduped = drop_duplicates(filled, subset=["id"], keep="first")
    _print_block("实验", "handle_missing(median) -> handle_missing(constant) -> drop_duplicates(subset=id)")

    missing_before = df.null_count().sum()
    missing_after = deduped.null_count().sum()
    dups_before = df.height - df.unique(subset=["id"]).height
    dups_after = deduped.height - deduped.unique(subset=["id"]).height
    _print_block("结果", f"清洗后（{deduped.height} 行）：\n{deduped}\n"
                       f"缺失值：{missing_before} -> {missing_after}；"
                       f"重复 id：{dups_before} -> {dups_after}")
    fact = f"缺失值 {missing_before}→{missing_after}，重复 id {dups_before}→{dups_after}，age=200 是离群值。"
    _print_block("AI 解释", _ai_explain("数据清洗", fact))
    return {"domain": "数据清洗", "rows_before": df.height, "rows_after": deduped.height,
            "missing_before": missing_before, "missing_after": missing_after}


# ============================================================
# 领域 2：数据合并
# ============================================================
def domain_2_merge(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 2：数据合并（两表 Join）")
    print(SECTION)
    _print_block("知识", "合并把两张表按 Key 关联：MergePlan 声明 Key/连接类型/映射，\n"
                       "Validator 先校验（类型/基数），Executor 再执行，最后生成 MergeReport。")

    users = pl.DataFrame({"user_id": [1, 2, 3], "name": ["小洛", "小爱", "小数"]})
    orders = pl.DataFrame({"user_id": [1, 2, 1], "product": ["A", "B", "C"]})
    _print_block("数据", f"左表 users：\n{users}\n右表 orders：\n{orders}")

    left_id = _make_dataset(env, "users", users)
    right_id = _make_dataset(env, "orders", orders)
    data_engine = env["engine"]
    plan = MergePlan(
        left={"dataset_id": left_id}, right={"dataset_id": right_id},
        keys=[JoinKey(left="user_id", right="user_id")], join_type="inner",
    )
    version, report = data_engine.run_merge(left_id, right_id, plan)
    merged = env["ds"].load_version(left_id, version.version)
    _print_block("实验", "MergePlan(keys=[user_id↔user_id], inner) -> run_merge")

    _print_block("结果", f"合并结果（{merged.height} 行）：\n{merged}\n"
                       f"匹配 {report.matched_rows} 行，输出列 {report.output_columns}")
    fact = f"两表按 user_id 内连接，输出 {merged.height} 行，匹配 {report.matched_rows} 行。"
    _print_block("AI 解释", _ai_explain("数据合并", fact))
    return {"domain": "数据合并", "output_rows": merged.height, "matched": report.matched_rows}


# ============================================================
# 领域 3：EDA
# ============================================================
def domain_3_eda(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 3：EDA（描述统计 / 相关性）")
    print(SECTION)
    _print_block("知识", "探索性数据分析（EDA）：描述统计揭示分布与集中趋势，\n"
                       "相关性矩阵衡量变量间线性/单调关系强度（Pearson/Spearman）。")

    df = pl.DataFrame(
        {
            "height": [160, 165, 170, 175, 180, 185, 190],
            "weight": [55, 60, 65, 70, 75, 80, 85],
            "age": [20, 22, 25, 28, 30, 35, 40],
        }
    )
    _print_block("数据", f"样本（{df.height} 行）：\n{df}")

    desc = DescriptiveAnalyzer().analyze(df)
    corr = CorrelationAnalyzer().analyze(df)
    _print_block("实验", "DescriptiveAnalyzer().analyze(df) + CorrelationAnalyzer().analyze(df)")

    height_desc = next(c for c in desc["columns"] if c["column"] == "height")
    hw_corr = corr["matrix"]["height"]["weight"]
    _print_block("结果", f"height 描述：mean={height_desc['mean']}, std={height_desc['std']}, "
                       f"min={height_desc['min']}, max={height_desc['max']}\n"
                       f"height-weight 相关性：{hw_corr:.3f}")
    fact = f"height 与 weight 相关性 {hw_corr:.3f}（接近 1，强正相关）。"
    _print_block("AI 解释", _ai_explain("EDA", fact))
    return {"domain": "EDA", "height_weight_corr": round(hw_corr, 3)}


# ============================================================
# 领域 4：分类
# ============================================================
def domain_4_classification(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 4：分类（逻辑回归）")
    print(SECTION)
    _print_block("知识", "分类预测离散标签。逻辑回归用 sigmoid 把线性组合映射到概率，\n"
                       "评估指标：accuracy / precision / recall / f1。")

    df = pl.DataFrame(
        {
            "x1": [1.0, 1.2, 1.1, 8.0, 8.2, 8.1, 7.9, 1.05, 8.05, 1.5],
            "x2": [1.0, 1.1, 1.2, 8.0, 8.1, 7.9, 8.2, 1.15, 7.95, 1.6],
            "label": [0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
        }
    )
    _print_block("数据", f"分类样本（{df.height} 行，2 类）：\n{df}")

    dataset_id = _make_dataset(env, "cls", df)
    exp, ds = env["exp"], env["ds"]
    version_row = ds.get_version_row(dataset_id)
    experiment = exp.create(
        dataset_id=dataset_id, dataset_version_id=version_row.id,
        task="classification", model="logistic_regression",
        target_column="label", seed=42, description="学习用分类实验",
    )
    run = exp.run(experiment.id)
    _print_block("实验", "ExperimentService.create(classification, logistic_regression) -> run()")

    metrics = run.metrics or {}
    _print_block("结果", f"运行状态：{run.status}，耗时 {run.runtime}s\n"
                       f"指标：accuracy={metrics.get('accuracy')}, "
                       f"precision={metrics.get('precision')}, f1={metrics.get('f1')}")
    fact = f"逻辑回归准确率 {metrics.get('accuracy')}，f1={metrics.get('f1')}。"
    _print_block("AI 解释", _ai_explain("分类", fact))
    return {"domain": "分类", "accuracy": metrics.get("accuracy"), "f1": metrics.get("f1")}


# ============================================================
# 领域 5：回归
# ============================================================
def domain_5_regression(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 5：回归（线性回归）")
    print(SECTION)
    _print_block("知识", "回归预测连续值。线性回归最小化预测值与真实值的均方误差，\n"
                       "评估指标：MAE / MSE / RMSE / R²。R² 越接近 1 拟合越好。")

    # y ≈ 2*x + 1（线性可分）
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [3.0, 5.1, 7.0, 9.1, 11.0, 13.0, 15.1, 17.0],
        }
    )
    _print_block("数据", f"回归样本（{df.height} 行，y≈2x+1）：\n{df}")

    dataset_id = _make_dataset(env, "reg", df)
    exp, ds = env["exp"], env["ds"]
    version_row = ds.get_version_row(dataset_id)
    experiment = exp.create(
        dataset_id=dataset_id, dataset_version_id=version_row.id,
        task="regression", model="linear_regression",
        target_column="y", seed=42, description="学习用回归实验",
    )
    run = exp.run(experiment.id)
    _print_block("实验", "ExperimentService.create(regression, linear_regression) -> run()")

    metrics = run.metrics or {}
    _print_block("结果", f"运行状态：{run.status}，耗时 {run.runtime}s\n"
                       f"指标：MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}, "
                       f"R²={metrics.get('r2')}")
    fact = f"线性回归 R²={metrics.get('r2')}（接近 1 表示拟合好），RMSE={metrics.get('rmse')}。"
    _print_block("AI 解释", _ai_explain("回归", fact))
    return {"domain": "回归", "r2": metrics.get("r2"), "rmse": metrics.get("rmse")}


# ============================================================
# 领域 6：聚类
# ============================================================
def domain_6_clustering(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 6：聚类（KMeans）")
    print(SECTION)
    _print_block("知识", "聚类是无监督学习：不依赖标签，按相似性把样本分组。\n"
                       "KMeans 迭代更新簇中心；评估用轮廓系数（silhouette，越接近 1 越好）。")

    # 3 个明显的簇
    df = pl.DataFrame(
        {
            "x": [1.0, 1.1, 0.9, 5.0, 5.1, 4.9, 9.0, 9.1, 8.9],
            "y": [1.0, 0.9, 1.1, 5.0, 4.9, 5.1, 9.0, 8.9, 9.1],
        }
    )
    _print_block("数据", f"聚类样本（{df.height} 行，3 个簇）：\n{df}")

    dataset_id = _make_dataset(env, "clu", df)
    exp, ds = env["exp"], env["ds"]
    version_row = ds.get_version_row(dataset_id)
    experiment = exp.create(
        dataset_id=dataset_id, dataset_version_id=version_row.id,
        task="clustering", model="kmeans",
        parameters={"n_clusters": 3}, seed=42, description="学习用聚类实验",
    )
    run = exp.run(experiment.id)
    _print_block("实验", "ExperimentService.create(clustering, kmeans, n_clusters=3) -> run()")

    metrics = run.metrics or {}
    _print_block("结果", f"运行状态：{run.status}，耗时 {run.runtime}s\n"
                       f"指标：cluster_count={metrics.get('cluster_count')}, "
                       f"silhouette={metrics.get('silhouette')}")
    fact = f"KMeans 分 {metrics.get('cluster_count')} 簇，轮廓系数 {metrics.get('silhouette')}（接近 1 表示簇分离清晰）。"
    _print_block("AI 解释", _ai_explain("聚类", fact))
    return {"domain": "聚类", "silhouette": metrics.get("silhouette")}


# ============================================================
# 领域 7：LLM 工具调用
# ============================================================
def domain_7_llm_tool_calling(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 7：LLM 工具调用（Agent 用工具）")
    print(SECTION)
    _print_block("知识", "LLM 通过 function calling 选择工具：Planner 把用户请求转为\n"
                       "结构化 AgentPlan（tool + arguments），Executor 经 ToolRegistry 执行。")

    df = pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    dataset_id = _make_dataset(env, "tool-demo", df)
    _print_block("数据", f"样本数据集 {dataset_id}（{df.height} 行）：\n{df}")

    # 用 PendingAction 构造结构化步骤（模拟 LLM 选工具：dataset.quality）
    from app.agent.state import PendingAction
    step = PendingAction(tool="dataset.quality", arguments={"dataset_id": dataset_id}, source="llm_tool_calling")
    _print_block("实验", "PendingAction(dataset.quality) -> AgentExecutor.execute_step(dataset.quality)")

    # 执行步骤
    executor = AgentExecutor()
    tool_ctx = ToolExecutionContext(
        user_id="student", dataset_ids={dataset_id},
        permissions=set(ROLE_PERMISSIONS["analyst"]),
    )
    services = ToolServices(dataset_service=env["ds"], data_engine_service=env["engine"])
    record = executor.execute_step(step, tool_ctx, services)
    _print_block("结果", f"执行 {record.tool}：status={record.status}, 耗时 {record.elapsed_ms}ms\n"
                       f"摘要：{record.result.summary if record.result else record.error}")
    fact = f"LLM 选择了 dataset.quality 工具，执行{record.status}。"
    _print_block("AI 解释", _ai_explain("LLM 工具调用", fact))
    return {"domain": "LLM 工具调用", "tool": record.tool, "status": record.status,
            "plan_goal": "检查数据集质量"}


# ============================================================
# 领域 8：Agent（完整运行时）
# ============================================================
def domain_8_agent(env: dict[str, Any]) -> dict[str, Any]:
    print("\n" + SECTION)
    print("领域 8：Agent（完整运行时：规划 + 执行）")
    print(SECTION)
    _print_block("知识", "完整 Agent 闭环：Context → Plan → Permission → Tool → Validation\n"
                       "→ Replan → Final Answer。AgentRuntime 统一编排，安全机制全程介入。")

    df = pl.DataFrame(
        {
            "x1": [1.0, 8.0, 1.2, 8.2, 1.1, 7.9],
            "x2": [1.0, 8.0, 1.1, 8.1, 1.2, 8.2],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    dataset_id = _make_dataset(env, "agent-demo", df)
    _print_block("数据", f"样本数据集 {dataset_id}（{df.height} 行）：\n{df}")

    runtime = AgentRuntime(
        env["engine"], experiment_service=env["exp"], db=env["db"], llm=MockLLM()
    )
    session = runtime.create_session(dataset_ids=[dataset_id])
    _print_block("实验", "AgentRuntime.run('检查一下数据质量')（规则规划器 + 默认安全机制）")

    run = runtime.run(session, "检查一下数据质量")
    plan_tools = [s["tool"] for s in (run.plan or {}).get("steps", [])]
    _print_block("结果", f"状态：{run.status}\n"
                       f"计划工具：{plan_tools}\n"
                       f"工具调用次数：{run.tool_call_count}\n"
                       f"事件数：{len(run.events)}\n"
                       f"最终答案：{run.final_answer}")
    fact = f"Agent 自动规划 {len(plan_tools)} 步，调用 {run.tool_call_count} 次工具，状态 {run.status}。"
    _print_block("AI 解释", _ai_explain("Agent", fact))
    return {"domain": "Agent", "status": str(run.status),
            "tool_calls": run.tool_call_count, "plan_tools": plan_tools}


# ============================================================
# 主入口
# ============================================================
def save_json(rows: list[dict[str, Any]], name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n✓ 学习报告已保存：{path}")


def main() -> None:
    print(SECTION)
    print("小洛实验室 · 学习系统实验（Prompt 244）—— 8 领域学习报告")
    print(SECTION)

    env = setup_env()

    summaries: list[dict[str, Any]] = []
    summaries.append(domain_1_cleaning(env))
    summaries.append(domain_2_merge(env))
    summaries.append(domain_3_eda(env))
    summaries.append(domain_4_classification(env))
    summaries.append(domain_5_regression(env))
    summaries.append(domain_6_clustering(env))
    summaries.append(domain_7_llm_tool_calling(env))
    summaries.append(domain_8_agent(env))

    print("\n" + SECTION)
    print("学习报告汇总")
    print(SECTION)
    for i, s in enumerate(summaries, 1):
        print(f"  {i}. {s['domain']}：{s}")
    save_json(summaries, "learning_cases.json")


if __name__ == "__main__":
    main()
