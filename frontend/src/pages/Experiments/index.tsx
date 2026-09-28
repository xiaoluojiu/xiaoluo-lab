import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ExperimentDetail } from "../../features/experiment/ExperimentDetail";
import { ExperimentCompare } from "../../features/experiment/ExperimentCompare";
import {
  compareExperiments,
  deleteAllExperiments,
  deleteExperiment,
  deleteExperiments,
  EXPERIMENT_RUN_ESTIMATE,
  EXPERIMENT_RUN_STAGES,
  getExperiment,
  getExperimentRuns,
  listExperiments,
  runExperiment,
} from "../../api/experiments";
import type { CompareResult, Experiment, ExperimentRun } from "../../types/ml";
import { metricSummary } from "../../features/ml/metrics";
import { ModelComparison } from "../../features/ml/ModelComparison";
import { PageHeader } from "../../components/PageHeader";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { EmptyState } from "../../components/viz/Blocks";
import { ErrorState, Skeleton } from "../../components/StateBlock";
import { TaskProgress } from "../../components/TaskProgress";
import { useToast } from "../../components/ToastProvider";

// Prompt 189：实验中心 —— 列表 → 详情 → 运行 → 对比 → 删除。
export default function Experiments() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [current, setCurrent] = useState<Experiment | null>(null);
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pendingRemove, setPendingRemove] = useState<Experiment | null>(null);
  const [pendingRemoveAll, setPendingRemoveAll] = useState(false);
  // 勾选原本只服务「横向对比」，现在同时服务「删除选中」——同一份选择，两个动作。
  const [pendingRemoveSelected, setPendingRemoveSelected] = useState(false);
  // 实验级横向对比：勾选 2 个以上实验，比较逻辑走后端 ExperimentComparator（不另写一套）
  const [selected, setSelected] = useState<number[]>([]);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparing, setComparing] = useState(false);
  // 重跑任务的分阶段反馈状态。rerun 走的是同步接口（无 SSE），所以进度只能是不确定进度，
  // 但「在跑 / 成功 / 失败」三态与失败原因必须显式呈现 —— 否则用户只看到按钮变成「运行中」，
  // 不知道到底跑没跑、跑完没有。
  const [runStatus, setRunStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const toastApi = useToast();

  async function refresh() {
    const list = await listExperiments(1, 50, true);
    setExperiments(list.items);
  }

  function load() {
    setLoading(true);
    setError(null);
    // with_metrics=true：列表要能直接回答「哪个实验效果更好」
    listExperiments(1, 50, true)
      .then((r) => setExperiments(r.items))
      .catch((e) => setError(e instanceof Error ? e.message : "加载实验失败"))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  async function remove(id: number) {
    setPendingRemove(null);
    setError(null);
    try {
      await deleteExperiment(id);
      if (current?.id === id) {
        setCurrent(null);
        setRuns([]);
      }
      await refresh();
      toastApi.success("实验已删除");
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    }
  }

  async function removeAll() {
    setPendingRemoveAll(false);
    setError(null);
    try {
      await deleteAllExperiments();
      setExperiments([]);
      setCurrent(null);
      setRuns([]);
      toastApi.success("全部实验已删除");
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    }
  }

  function toggleSelected(id: number) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  /** 删除勾选的实验。勾选同时服务对比，所以删除后要把选中与对比结果一起清掉。 */
  async function removeSelected() {
    setPendingRemoveSelected(false);
    const targets = [...selected];
    if (!targets.length) return;
    setError(null);
    try {
      const result = await deleteExperiments(targets);
      setSelected([]);
      setComparison(null);
      if (current && targets.includes(current.id)) {
        setCurrent(null);
        setRuns([]);
      }
      await refresh();
      toastApi.success(`已删除 ${result.count} 个实验`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    }
  }

  async function doCompare() {
    if (selected.length < 2) return;
    setComparing(true);
    setError(null);
    try {
      setComparison(await compareExperiments(selected));
    } catch (e) {
      setError(e instanceof Error ? e.message : "对比失败");
    } finally {
      setComparing(false);
    }
  }

  async function select(id: number) {
    setError(null);
    // 换实验后上一次重跑的状态不再适用（成功/失败都是上一个实验的结论）。
    setRunStatus("idle");
    setRunError(null);
    try {
      const [exp, rs] = await Promise.all([getExperiment(id), getExperimentRuns(id)]);
      setCurrent(exp);
      setRuns(rs);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载详情失败");
    }
  }

  async function rerun() {
    if (!current) return;
    setBusy(true);
    setError(null);
    setRunError(null);
    setRunStatus("running");
    try {
      await runExperiment(current.id);
      const rs = await getExperimentRuns(current.id);
      setRuns(rs);
      const list = await listExperiments(1, 50, true);
      setExperiments(list.items);
      setRunStatus("success");
    } catch (e) {
      // 失败原因进进度条（那里紧挨着「重试」按钮）；页面级 error 留给列表加载等其它失败。
      setRunStatus("error");
      setRunError(e instanceof Error ? e.message : "运行失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="实验中心"
        description="查看训练过的实验、运行记录与指标，支持重新运行与横向对比。"
      />

      <div className="card">
        <div className="flex-between">
          <h3 style={{ margin: 0 }}>实验列表</h3>
          {experiments.length > 0 && (
            <div className="inline-actions">
              <button
                className="btn danger btn-sm"
                type="button"
                disabled={!selected.length}
                onClick={() => setPendingRemoveSelected(true)}
              >
                删除选中（{selected.length}）
              </button>
              <button className="btn danger btn-sm" onClick={() => setPendingRemoveAll(true)}>
                一键删除全部（{experiments.length}）
              </button>
            </div>
          )}
        </div>
        {error && <div style={{ marginTop: "var(--space-3)" }}><ErrorState message={error} onRetry={load} /></div>}
        {loading && <Skeleton lines={3} />}
        {!loading && !error && !experiments.length && (
          <div style={{ marginTop: "var(--space-3)" }}>
            <EmptyState
              icon="beaker"
              title="还没有实验记录"
              description="实验记录在训练完成后自动归档。去「机器学习」选好数据集、任务与模型跑一次，结果就会出现在这里。"
              action={
                /* 用 SPA 路由跳转而不是 <a href>：整页刷新会丢掉当前会话里的状态。 */
                <Link className="btn primary btn-sm" to="/ml">
                  创建第一个实验
                </Link>
              }
              secondary={
                <Link className="btn link" to="/learning">
                  查看教程
                </Link>
              }
            />
          </div>
        )}
        {experiments.length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th aria-label="选择用于对比">
                    <input
                      type="checkbox"
                      aria-label="全选实验"
                      checked={selected.length === experiments.length && experiments.length > 0}
                      onChange={(e) => setSelected(e.target.checked ? experiments.map((x) => x.id) : [])}
                    />
                  </th>
                  <th>ID</th>
                  <th>数据集</th>
                  <th>任务</th>
                  <th>模型</th>
                  <th>目标列</th>
                  <th>最近一次成功运行</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((e) => {
                  const summary = e.latest_run ? metricSummary(e.latest_run.metrics) : null;
                  return (
                    <tr key={e.id}>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`选择实验 #${e.id}（用于对比或删除）`}
                          checked={selected.includes(e.id)}
                          onChange={() => toggleSelected(e.id)}
                        />
                      </td>
                      <td>#{e.id}</td>
                      <td>#{e.dataset_id}</td>
                      <td>{e.task}</td>
                      <td>{e.model}</td>
                      <td>{e.target_column ?? "-"}</td>
                      <td>
                        {summary ? (
                          <>
                            <strong>{summary}</strong>
                            <span className="muted" style={{ marginLeft: 6 }}>
                              run #{e.latest_run?.run_id}
                            </span>
                          </>
                        ) : (
                          <span className="muted">还没有成功运行</span>
                        )}
                      </td>
                      <td>
                        <div className="inline-actions">
                          <button className="btn btn-sm" onClick={() => void select(e.id)}>查看详情</button>
                          <button className="btn btn-sm danger" onClick={() => setPendingRemove(e)}>删除</button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {experiments.length > 1 && (
        <div className="card">
          <div className="flex-between">
            <div>
              <h3 style={{ margin: 0 }}>实验横向对比</h3>
              <div className="muted" style={{ marginTop: "var(--space-1)" }}>
                勾选 2 个以上实验；每个实验取最近一次成功运行，指标好坏由后端既有比较器判定。
              </div>
            </div>
            <button
              className="btn btn-primary"
              type="button"
              disabled={comparing || selected.length < 2}
              onClick={() => void doCompare()}
            >
              {comparing ? "对比中..." : `对比选中的 ${selected.length} 个实验`}
            </button>
          </div>
          <div className="mt">
            <ModelComparison result={comparison} />
          </div>
        </div>
      )}

      {/* 重跑是同步接口：只给不确定进度，但「在跑 / 跑完了 / 哪一步挂了」必须说清楚 */}
      {runStatus !== "idle" && (
        <div className="card">
          <TaskProgress
            stages={EXPERIMENT_RUN_STAGES}
            status={runStatus}
            indeterminate={runStatus === "running"}
            title="实验运行中"
            estimate={EXPERIMENT_RUN_ESTIMATE}
            error={runError}
            onViewResult={() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
            onRetry={() => void rerun()}
          />
        </div>
      )}

      <div className="card" ref={detailRef}>
        <h3>实验详情</h3>
        <ExperimentDetail experiment={current} runs={runs} busy={busy} onRun={() => void rerun()} />
      </div>

      {current && (
        <div className="card">
          <h3>运行对比</h3>
          <ExperimentCompare runs={runs} />
        </div>
      )}

      <ConfirmDialog
        open={pendingRemove !== null}
        title="删除这个实验？"
        message={<>实验 #{pendingRemove?.id} 的所有运行记录与模型产物会一并清理，不可恢复。</>}
        confirmText="删除"
        danger
        onConfirm={() => { if (pendingRemove) void remove(pendingRemove.id); }}
        onCancel={() => setPendingRemove(null)}
      />
      <ConfirmDialog
        open={pendingRemoveAll}
        title="删除全部实验？"
        message={<>将删除当前 {experiments.length} 个实验及其全部运行记录与模型产物，不可恢复。</>}
        confirmText="全部删除"
        danger
        onConfirm={() => void removeAll()}
        onCancel={() => setPendingRemoveAll(false)}
      />
      <ConfirmDialog
        open={pendingRemoveSelected}
        title={`删除选中的 ${selected.length} 个实验？`}
        message={<>选中的 {selected.length} 个实验及其全部运行记录与模型产物会一并清理，不可恢复。</>}
        confirmText={`删除 ${selected.length} 个`}
        danger
        onConfirm={() => void removeSelected()}
        onCancel={() => setPendingRemoveSelected(false)}
      />
    </div>
  );
}
