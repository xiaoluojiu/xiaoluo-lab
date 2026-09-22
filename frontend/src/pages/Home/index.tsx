import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { listDatasets } from "../../api/datasets";
import { listExperiments, getExperimentRuns } from "../../api/ml";
import { listWorkflows } from "../../api/workflow";
import type { Dataset } from "../../types/dataset";
import type { Experiment } from "../../types/ml";
import type { WorkflowSummary } from "../../types/workflow";
import { EmptyState, HeroBand, Panel, SectionHeader } from "../../components/viz/Blocks";
import { KpiCard } from "../../components/viz/KpiCard";
import { Icon, type IconName } from "../../components/icons/Icon";

interface HomeData {
  datasets: Dataset[];
  experiments: Experiment[];
  workflows: WorkflowSummary[];
  totals: { datasets: number | null; experiments: number | null; workflows: number | null };
  /** 最近实验的最新一次成功 run 的主指标（按实验创建时间倒序），用于「模型性能趋势」。 */
  performanceSeries: { value: number; metric: string }[];
}

const initialData: HomeData = {
  datasets: [],
  experiments: [],
  workflows: [],
  totals: { datasets: null, experiments: null, workflows: null },
  performanceSeries: [],
};

function timeLabel(value: string | null | undefined) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未知";
  return date.toLocaleString([], { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function timestamp(value: string | null | undefined) {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

/** 从一次 run 的 metrics 里挑出「最能代表性能」的单一数值。
 *  口径：分类取 accuracy、回归取 r²、聚类取 silhouette；都取不到返回 null。
 *  只读真实值，不虚构、不归一化。 */
function primaryMetric(metrics: Record<string, number | string> | undefined): number | null {
  if (!metrics) return null;
  const candidates = ["accuracy", "r2", "silhouette", "f1"];
  for (const key of candidates) {
    const raw = metrics[key];
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  }
  return null;
}

/** 主指标的展示标签与单位（与后端 eval 口径对齐）。 */
function metricLabel(metrics: Record<string, number | string> | undefined): string | null {
  if (!metrics) return null;
  if (typeof metrics.accuracy === "number") return "accuracy";
  if (typeof metrics.r2 === "number") return "r²";
  if (typeof metrics.silhouette === "number") return "轮廓系数";
  if (typeof metrics.f1 === "number") return "f1";
  return null;
}

const QUICK_LINKS: { title: string; desc: string; to: string; icon: IconName }[] = [
  { title: "数据", desc: "上传、预览与管理数据集版本", to: "/datasets", icon: "database" },
  { title: "机器学习", desc: "训练模型并查看评估指标", to: "/ml", icon: "chip" },
  { title: "AI 实验室", desc: "用自然语言驱动分析流程", to: "/ai", icon: "sparkles" },
  { title: "Workflow", desc: "把处理步骤编排成可复用流程", to: "/workflow", icon: "flow" },
];

export default function Home() {
  const [data, setData] = useState<HomeData>(initialData);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([listDatasets(1, 6), listExperiments(1, 6), listWorkflows()]).then(async ([datasets, experiments, workflows]) => {
      if (cancelled) return;
      const datasetItems = datasets.status === "fulfilled" ? datasets.value.items : [];
      const experimentItems = experiments.status === "fulfilled" ? experiments.value.items : [];
      const workflowItems = workflows.status === "fulfilled" ? workflows.value : [];

      // 拉取每个实验的最新 run，提取真实主指标，构建性能趋势序列（不虚构）。
      const performanceSeries: { value: number; metric: string }[] = [];
      if (experimentItems.length) {
        const results = await Promise.allSettled(experimentItems.map((item) => getExperimentRuns(item.id)));
        experimentItems.forEach((_exp, index) => {
          const runsResult = results[index];
          const runs = runsResult && runsResult.status === "fulfilled" ? runsResult.value : [];
          const lastSuccess = [...runs].reverse().find((run) => run.status === "success" && primaryMetric(run.metrics) !== null);
          if (!lastSuccess) return;
          const value = primaryMetric(lastSuccess.metrics);
          const metric = metricLabel(lastSuccess.metrics);
          if (value !== null && metric) performanceSeries.push({ value, metric });
        });
      }

      setData({
        datasets: datasetItems,
        experiments: experimentItems,
        workflows: workflowItems.slice(0, 6),
        totals: {
          datasets: datasets.status === "fulfilled" ? datasets.value.page_info?.total ?? datasetItems.length : null,
          experiments: experiments.status === "fulfilled" ? experiments.value.page_info?.total ?? experimentItems.length : null,
          workflows: workflows.status === "fulfilled" ? workflows.value.length : null,
        },
        performanceSeries,
      });
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const latestDataset = data.datasets[0];
  const totalRows = useMemo(
    () => data.datasets.reduce((sum, item) => sum + (item.latest_version?.row_count ?? 0), 0),
    [data.datasets],
  );
  const totalColumns = useMemo(
    () => data.datasets.reduce((sum, item) => sum + (item.latest_version?.column_count ?? 0), 0),
    [data.datasets],
  );
  const workflowNodes = useMemo(
    () => data.workflows.reduce((sum, item) => sum + (item.nodes ?? 0), 0),
    [data.workflows],
  );
  // 数据集体积的真实分布：取最近 6 个数据集的行数，用于 KPI 卡的趋势线（不虚构任何序列）。
  const rowSeries = useMemo(
    () => data.datasets.map((item) => item.latest_version?.row_count ?? 0),
    [data.datasets],
  );
  // 模型性能趋势（真实主指标序列）；无成功 run 时为空数组。
  const perfValues = useMemo(
    () => data.performanceSeries.map((item) => item.value),
    [data.performanceSeries],
  );
  const latestPerf = data.performanceSeries[0] ?? null;
  const latestUpdateTime = latestDataset?.updated_at ?? null;

  // 三类资源混合成一条按时间倒序的动态流，替代原先「每类只显示一条」。
  const feed = useMemo(() => {
    const items = [
      ...data.datasets.map((item) => ({
        key: `dataset-${item.id}`,
        title: item.name,
        kind: "数据集",
        icon: "database" as IconName,
        to: `/datasets/${item.id}`,
        time: timestamp(item.updated_at),
        meta: item.latest_version
          ? `v${item.latest_version.version} · ${item.latest_version.row_count} 行 × ${item.latest_version.column_count} 列`
          : "暂无版本",
      })),
      ...data.experiments.map((item) => ({
        key: `experiment-${item.id}`,
        title: item.model,
        kind: "实验",
        icon: "beaker" as IconName,
        to: "/experiments",
        time: timestamp(item.created_at),
        meta: item.task ?? "机器学习任务",
      })),
      ...data.workflows.map((item) => ({
        key: `workflow-${item.id}`,
        title: item.name,
        kind: "Workflow",
        icon: "flow" as IconName,
        to: "/workflow",
        time: 0,
        meta: `${item.nodes} 个节点 · ${item.edges} 条连线`,
      })),
    ];
    const sortable = items.filter((item) => item.time > 0).sort((a, b) => b.time - a.time);
    const undated = items.filter((item) => item.time === 0);
    return [...sortable, ...undated].slice(0, 6);
  }, [data]);

  const heroDescription = latestDataset?.latest_version
    ? `当前 v${latestDataset.latest_version.version} · ${latestDataset.latest_version.row_count} 行 × ${latestDataset.latest_version.column_count} 列，最近更新于 ${timeLabel(latestDataset.updated_at)}。`
    : "还没有数据集。上传第一份数据后，就可以在这里处理、分析并生成报告。";

  return (
    <div>
      <HeroBand
        eyebrow="继续工作"
        icon="sparkles"
        title={latestDataset ? latestDataset.name : "从小洛实验室开始"}
        description={heroDescription}
        actions={
          <>
            {latestDataset ? (
              <Link className="btn primary" to={`/datasets/${latestDataset.id}`}>继续数据集</Link>
            ) : (
              <Link className="btn primary" to="/datasets">进入数据中心</Link>
            )}
            {latestDataset ? (
              <>
                <Link className="btn" to={`/processing?dataset=${latestDataset.id}`}>处理</Link>
                <Link className="btn" to={`/analysis?dataset=${latestDataset.id}`}>分析</Link>
                <Link className="btn" to={`/ml?dataset=${latestDataset.id}`}>机器学习</Link>
                <Link className="btn" to={`/workflow?dataset=${latestDataset.id}`}>Workflow</Link>
                <Link className="btn" to={`/ai?dataset=${latestDataset.id}`}>AI 实验室</Link>
              </>
            ) : null}
          </>
        }
        meta={
          <>
            <div className="hero-meta-item">
              <span className="hero-meta-key">已接入行数</span>
              <span className="hero-meta-value">{totalRows.toLocaleString()}</span>
            </div>
            <div className="hero-meta-item">
              <span className="hero-meta-key">流程节点</span>
              <span className="hero-meta-value">{workflowNodes}</span>
            </div>
          </>
        }
      />

      <SectionHeader title="资源概览" description="数据集、实验、流程与关键运行指标的存量情况。" />
      <div className="grid g3">
        <KpiCard
          label="数据集"
          value={data.totals.datasets}
          icon="database"
          tone="primary"
          to="/datasets"
          loading={loading}
          spark={rowSeries}
          hint={rowSeries.length > 1 ? "折线为最近 6 个数据集的行数分布" : "上传数据后即可开始分析"}
        />
        <KpiCard
          label="实验"
          value={data.totals.experiments}
          icon="beaker"
          tone="success"
          to="/experiments"
          loading={loading}
          hint={data.experiments[0] ? `最近一次：${data.experiments[0].model}` : "还没有训练记录"}
        />
        <KpiCard
          label="Workflow"
          value={data.totals.workflows}
          icon="flow"
          tone="info"
          to="/workflow"
          loading={loading}
          hint={workflowNodes > 0 ? `合计 ${workflowNodes} 个节点` : "把处理步骤编排成可复用流程"}
        />
      </div>
      <div className="grid g3" style={{ marginTop: "var(--space-4)" }}>
        <KpiCard
          label="模型性能"
          value={latestPerf ? latestPerf.value : null}
          icon="trend-up"
          tone="success"
          to="/experiments"
          loading={loading}
          precision={3}
          suffix={latestPerf ? latestPerf.metric : undefined}
          spark={perfValues.length > 1 ? perfValues : undefined}
          hint={latestPerf ? "最近成功实验的主指标（按时间倒序）" : "训练模型后这里会显示性能趋势"}
        />
        <KpiCard
          label="数据总行数"
          value={totalRows}
          icon="layers"
          tone="primary"
          to="/datasets"
          loading={loading}
          hint={latestDataset ? `分布在 ${data.datasets.length} 个数据集 · 共 ${totalColumns} 列` : "上传数据后统计行数"}
        />
        <KpiCard
          label="最近更新"
          value={latestUpdateTime ? timeLabel(latestUpdateTime) : null}
          icon="clock"
          tone="info"
          to={latestDataset ? `/datasets/${latestDataset.id}` : "/datasets"}
          loading={loading}
          hint={latestDataset ? latestDataset.name : "还没有数据更新记录"}
        />
      </div>

      <SectionHeader
        title="最近工作"
        description="按最近更新时间排列。"
        actions={
          <Link className="btn btn-sm" to="/datasets">
            查看数据集
            <Icon name="arrow-right" size={14} />
          </Link>
        }
      />
      <Panel flush>
        {loading ? (
          <div style={{ padding: "var(--space-5)" }}>
            <div className="skeleton skeleton-text" style={{ width: "60%" }} />
            <div className="skeleton skeleton-text" style={{ width: "40%" }} />
            <div className="skeleton skeleton-text" style={{ width: "50%", marginBottom: 0 }} />
          </div>
        ) : feed.length === 0 ? (
          <EmptyState
            title="还没有最近工作"
            description="创建或上传一个数据集，之后的处理会显示在这里。"
            action={<Link className="btn primary" to="/datasets">进入数据中心</Link>}
          />
        ) : (
          <div className="activity-list">
            {feed.map((item) => (
              <Link key={item.key} to={item.to} className="activity-item">
                <span className="activity-icon">
                  <Icon name={item.icon} size={17} />
                </span>
                <span className="activity-text">
                  <strong className="activity-title">{item.title}</strong>
                  <span className="activity-meta">{item.kind} · {item.meta}</span>
                </span>
                <span className="activity-time">{item.time > 0 ? timeLabel(new Date(item.time).toISOString()) : "—"}</span>
                <Icon className="activity-arrow" name="arrow-right" size={15} />
              </Link>
            ))}
          </div>
        )}
      </Panel>

      <SectionHeader title="快速开始" />
      <div className="grid g4">
        {QUICK_LINKS.map((item) => (
          <Link key={item.to} to={item.to} className="kpi-card is-link">
            <div className="kpi-head">
              <span className="kpi-icon">
                <Icon name={item.icon} size={18} />
              </span>
              <span className="kpi-label">{item.title}</span>
            </div>
            <span className="kpi-hint">{item.desc}</span>
            <span className="kpi-foot-link">
              进入
              <Icon name="arrow-right" size={13} />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
