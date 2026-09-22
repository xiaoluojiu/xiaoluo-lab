import { useEffect, useState } from "react";
import { ExperimentDetail } from "../../features/experiment/ExperimentDetail";
import { ExperimentCompare } from "../../features/experiment/ExperimentCompare";
import {
  deleteAllExperiments,
  deleteExperiment,
  getExperiment,
  getExperimentRuns,
  listExperiments,
  runExperiment,
} from "../../api/ml";
import type { Experiment, ExperimentRun } from "../../types/ml";
import { PageHeader } from "../../components/PageHeader";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { EmptyState, ErrorState, Skeleton } from "../../components/StateBlock";
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
  const toastApi = useToast();

  async function refresh() {
    const list = await listExperiments(1, 50);
    setExperiments(list.items);
  }

  function load() {
    setLoading(true);
    setError(null);
    listExperiments(1, 50)
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

  async function select(id: number) {
    setError(null);
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
    try {
      await runExperiment(current.id);
      const rs = await getExperimentRuns(current.id);
      setRuns(rs);
      const list = await listExperiments(1, 50);
      setExperiments(list.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "运行失败");
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
            <button className="btn danger btn-sm" onClick={() => setPendingRemoveAll(true)}>
              一键删除全部（{experiments.length}）
            </button>
          )}
        </div>
        {error && <div style={{ marginTop: "var(--space-3)" }}><ErrorState message={error} onRetry={load} /></div>}
        {loading && <Skeleton lines={3} />}
        {!loading && !error && !experiments.length && (
          <div style={{ marginTop: "var(--space-3)" }}>
            <EmptyState
              title="还没有实验"
              description="先在「机器学习」页选择任务与模型训练一次，结果会自动归档到这里。"
              action={<a className="btn btn-sm btn-primary" href="/ml">前往机器学习</a>}
            />
          </div>
        )}
        {experiments.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>数据集</th>
                <th>任务</th>
                <th>模型</th>
                <th>目标列</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.id}>
                  <td>#{e.id}</td>
                  <td>#{e.dataset_id}</td>
                  <td>{e.task}</td>
                  <td>{e.model}</td>
                  <td>{e.target_column ?? "-"}</td>
                  <td>
                    <div className="inline-actions">
                      <button className="btn btn-sm" onClick={() => void select(e.id)}>查看详情</button>
                      <button className="btn btn-sm danger" onClick={() => setPendingRemove(e)}>删除</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
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
    </div>
  );
}
