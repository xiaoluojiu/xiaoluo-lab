import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createConnector,
  deleteConnector,
  describeConnectorTable,
  dialectLabel,
  getDialectCatalog,
  importConnectorTable,
  listConnectorTables,
  listConnectors,
  previewConnectorTable,
  statusLabel,
  testConnector,
  testSavedConnector,
  type Connector,
  type ConnectorColumn,
  type ConnectorImportResult,
  type ConnectorPayload,
  type ConnectorPreview,
  type ConnectorTable,
  type ConnectorTestResult,
  type DialectInfo,
} from "../../api/connectors";
import { Panel, SectionHeader } from "../../components/viz/Blocks";

/* =============================================================
   数据库连接器面板
   -------------------------------------------------------------
   两个部件：
   1. ConnectorPanel   —— 连接器管理（目录 / 新建 / 试连 / 删除）
   2. ConnectorWorkbench —— 单个连接器的数据接入（选表 → 预览 → 导入）

   与后端的分工：前端不构造 SQL，只传「表名 / 列名 / WHERE 片段」这样的意图参数，
   标识符的引用与白名单校验全部在后端（app/connectors/extract.py）。
   ============================================================= */

const EMPTY_FORM = {
  name: "",
  dialect: "sqlite",
  host: "",
  port: "",
  database: "",
  schema_name: "",
  username: "",
  password: "",
};

type FormState = typeof EMPTY_FORM;

function statusTone(status: string): "success" | "danger" | "warning" {
  if (status === "ok") return "success";
  if (status === "error") return "danger";
  return "warning";
}

export default function ConnectorPanel() {
  const [catalog, setCatalog] = useState<DialectInfo[]>([]);
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [formOpen, setFormOpen] = useState(false);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ConnectorTestResult | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [activeId, setActiveId] = useState<number | null>(null);

  const spec = useMemo(
    () => catalog.find((item) => item.name === form.dialect) ?? null,
    [catalog, form.dialect],
  );

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await listConnectors(1, 100);
      setConnectors(page.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载连接器失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    getDialectCatalog()
      .then(setCatalog)
      .catch(() => setCatalog([]));
    void reload();
  }, [reload]);

  const patchForm = (patch: Partial<FormState>) => {
    setForm((prev) => ({ ...prev, ...patch }));
    setTestResult(null);
  };

  const buildPayload = (): ConnectorPayload => ({
    name: form.name.trim(),
    dialect: form.dialect,
    host: form.host.trim() || null,
    port: form.port ? Number(form.port) : null,
    database: form.database.trim(),
    schema_name: form.schema_name.trim() || null,
    username: form.username.trim() || null,
    password: form.password || null,
  });

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testConnector({
        dialect: form.dialect,
        host: form.host.trim() || null,
        port: form.port ? Number(form.port) : null,
        database: form.database.trim(),
        schema_name: form.schema_name.trim() || null,
        username: form.username.trim() || null,
        password: form.password || null,
      });
      setTestResult(result);
    } catch (e) {
      setTestResult({
        ok: false,
        latency_ms: 0,
        server_version: null,
        message: e instanceof Error ? e.message : "测试失败",
        dialect: form.dialect,
        driver_installed: true,
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    if (!form.name.trim()) {
      setError("请填写连接器名称");
      return;
    }
    if (!form.database.trim()) {
      setError(form.dialect === "sqlite" ? "请填写 SQLite 文件路径" : "请填写数据库名");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await createConnector(buildPayload());
      setForm(EMPTY_FORM);
      setFormOpen(false);
      setTestResult(null);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  const handleTestSaved = async (id: number) => {
    setBusyId(id);
    try {
      const result = await testSavedConnector(id);
      await reload();
      setError(result.ok ? null : `连接「${connectors.find((c) => c.id === id)?.name ?? id}」失败：${result.message}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "测试失败");
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (record: Connector) => {
    setBusyId(record.id);
    try {
      await deleteConnector(record.id);
      if (activeId === record.id) setActiveId(null);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    } finally {
      setBusyId(null);
    }
  };

  const missingDriver = catalog.filter((item) => item.enabled && !item.driver_installed);

  return (
    <>
      <SectionHeader
        title="数据库连接器"
        description="接入 PostgreSQL / MySQL / SQLite 等外部数据库，按批抽取为平台内的数据集版本——不受上传文件体积限制。"
        actions={
          <button className="btn" type="button" onClick={() => setFormOpen((v) => !v)}>
            {formOpen ? "收起" : "新建连接"}
          </button>
        }
      />

      {error ? <p className="connector-alert is-error">{error}</p> : null}

      {missingDriver.length > 0 ? (
        <p className="connector-alert is-warn">
          以下数据库类型缺少驱动，需先安装对应 Python 包：
          {missingDriver.map((item) => (
            <span key={item.name} className="connector-hint-chip">
              {item.label} → <code>{item.driver_hint}</code>
            </span>
          ))}
        </p>
      ) : null}

      {formOpen ? (
        <Panel title="新建连接" description="口令仅以密文保存，接口不会回显原文。">
          <div className="connector-form">
            <label className="field">
              <span className="field-label">连接名称</span>
              <input
                type="text"
                value={form.name}
                placeholder="例如 生产库只读"
                onChange={(e) => patchForm({ name: e.target.value })}
              />
            </label>

            <label className="field">
              <span className="field-label">数据库类型</span>
              <select
                value={form.dialect}
                onChange={(e) => {
                  const next = catalog.find((item) => item.name === e.target.value);
                  patchForm({
                    dialect: e.target.value,
                    port: next?.default_port ? String(next.default_port) : "",
                  });
                }}
              >
                {catalog.map((item) => (
                  <option key={item.name} value={item.name} disabled={!item.enabled}>
                    {item.label}
                    {item.enabled ? "" : "（未启用）"}
                    {item.driver_installed ? "" : "（缺驱动）"}
                  </option>
                ))}
              </select>
            </label>

            {spec?.requires_host ? (
              <>
                <label className="field">
                  <span className="field-label">主机</span>
                  <input
                    type="text"
                    value={form.host}
                    placeholder="127.0.0.1"
                    onChange={(e) => patchForm({ host: e.target.value })}
                  />
                </label>
                <label className="field">
                  <span className="field-label">端口</span>
                  <input
                    type="number"
                    value={form.port}
                    placeholder={spec?.default_port ? String(spec.default_port) : ""}
                    onChange={(e) => patchForm({ port: e.target.value })}
                  />
                </label>
              </>
            ) : null}

            <label className="field connector-form-wide">
              <span className="field-label">
                {form.dialect === "sqlite" ? "数据库文件路径" : "数据库名"}
              </span>
              <input
                type="text"
                value={form.database}
                placeholder={form.dialect === "sqlite" ? "D:/data/app.db" : "analytics"}
                onChange={(e) => patchForm({ database: e.target.value })}
              />
            </label>

            {spec?.requires_host ? (
              <>
                <label className="field">
                  <span className="field-label">用户名</span>
                  <input
                    type="text"
                    value={form.username}
                    onChange={(e) => patchForm({ username: e.target.value })}
                  />
                </label>
                <label className="field">
                  <span className="field-label">口令</span>
                  <input
                    type="password"
                    value={form.password}
                    autoComplete="new-password"
                    onChange={(e) => patchForm({ password: e.target.value })}
                  />
                </label>
                <label className="field">
                  <span className="field-label">Schema（可空）</span>
                  <input
                    type="text"
                    value={form.schema_name}
                    placeholder="public"
                    onChange={(e) => patchForm({ schema_name: e.target.value })}
                  />
                </label>
              </>
            ) : null}
          </div>

          <div className="connector-actions">
            <button className="btn btn-sm" type="button" onClick={handleTest} disabled={testing}>
              {testing ? "测试中…" : "测试连接"}
            </button>
            <button className="btn btn-sm" type="button" onClick={handleSave} disabled={saving}>
              {saving ? "保存中…" : "保存"}
            </button>
          </div>

          {testResult ? (
            <p className={`connector-alert ${testResult.ok ? "is-ok" : "is-error"}`}>
              {testResult.ok ? "✓ " : "✕ "}
              {testResult.message}
              {testResult.server_version ? ` · 服务端版本 ${testResult.server_version}` : ""}
            </p>
          ) : null}
        </Panel>
      ) : null}

      <Panel
        title={`已配置连接（${connectors.length}）`}
        description="点击「数据接入」选择表并导入为数据集。"
      >
        {loading ? (
          <p className="connector-muted">加载中…</p>
        ) : connectors.length === 0 ? (
          <p className="connector-muted">暂无连接器。点击右上角「新建连接」开始。</p>
        ) : (
          <div className="connector-list">
            {connectors.map((record) => (
              <div className="connector-row" key={record.id}>
                <div className="connector-row-main">
                  <div className="connector-row-title">
                    <strong>{record.name}</strong>
                    <span className="tag">{dialectLabel(record.dialect)}</span>
                    <span className={`badge ${statusTone(record.last_status)}`}>
                      {statusLabel(record.last_status)}
                    </span>
                  </div>
                  <p className="connector-desc">
                    {record.host ? `${record.host}${record.port ? `:${record.port}` : ""} / ` : ""}
                    {record.database || "（未填库名）"}
                    {record.schema_name ? ` · ${record.schema_name}` : ""}
                    {record.has_password ? " · 已配置口令" : " · 无口令"}
                    {record.dataset_id ? ` · 最近导入 → 数据集 ${record.dataset_id}` : ""}
                  </p>
                  {record.last_status === "error" && record.last_error ? (
                    <p className="connector-alert is-error is-compact">{record.last_error}</p>
                  ) : null}
                </div>
                <div className="connector-row-actions">
                  <button
                    className="btn btn-sm"
                    type="button"
                    disabled={busyId === record.id}
                    onClick={() => handleTestSaved(record.id)}
                  >
                    {busyId === record.id ? "…" : "测试"}
                  </button>
                  <button
                    className="btn btn-sm"
                    type="button"
                    onClick={() => setActiveId(activeId === record.id ? null : record.id)}
                  >
                    {activeId === record.id ? "收起" : "数据接入"}
                  </button>
                  <button
                    className="btn btn-sm"
                    type="button"
                    disabled={busyId === record.id}
                    onClick={() => handleDelete(record)}
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      {activeId !== null ? (
        <ConnectorWorkbench
          connector={connectors.find((item) => item.id === activeId) ?? null}
        />
      ) : null}
    </>
  );
}

/* =============================================================
   单个连接器的数据接入工作台
   ============================================================= */

function ConnectorWorkbench({ connector }: { connector: Connector | null }) {
  const [tables, setTables] = useState<ConnectorTable[]>([]);
  const [schemas, setSchemas] = useState<string[]>([]);
  const [schema, setSchema] = useState<string>("");
  const [loadingTables, setLoadingTables] = useState(false);
  const [tableError, setTableError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string>("");
  const [columns, setColumns] = useState<ConnectorColumn[]>([]);
  const [pickedColumns, setPickedColumns] = useState<string[]>([]);

  const [where, setWhere] = useState("");
  const [orderBy, setOrderBy] = useState("");
  const [keyset, setKeyset] = useState("");
  const [maxRows, setMaxRows] = useState("");
  const [batchSize, setBatchSize] = useState("");
  const [datasetName, setDatasetName] = useState("");

  const [preview, setPreview] = useState<ConnectorPreview | null>(null);
  const [importing, setImporting] = useState(false);
  const [importResult, setImportResult] = useState<ConnectorImportResult | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const connectorId = connector?.id ?? null;

  const loadTables = useCallback(async () => {
    if (connectorId === null) return;
    setLoadingTables(true);
    setTableError(null);
    try {
      const data = await listConnectorTables(connectorId, schema || undefined);
      setTables(data.tables);
      setSchemas(data.schemas);
    } catch (e) {
      setTableError(e instanceof Error ? e.message : "读取表清单失败");
      setTables([]);
    } finally {
      setLoadingTables(false);
    }
  }, [connectorId, schema]);

  useEffect(() => {
    setSelected("");
    setColumns([]);
    setPickedColumns([]);
    setPreview(null);
    setImportResult(null);
    void loadTables();
  }, [loadTables]);

  const handlePickTable = async (table: string) => {
    setSelected(table);
    setPreview(null);
    setImportResult(null);
    setActionError(null);
    if (connectorId === null) return;
    try {
      const data = await describeConnectorTable(connectorId, table, schema || undefined);
      setColumns(data.columns);
      setPickedColumns([]);
      const pk = data.columns.find((col) => col.primary_key)?.name;
      setKeyset(pk ?? "");
      setOrderBy(pk ?? "");
    } catch (e) {
      setColumns([]);
      setActionError(e instanceof Error ? e.message : "读取表结构失败");
    }
  };

  const handlePreview = async () => {
    if (connectorId === null || !selected) return;
    setActionError(null);
    try {
      const data = await previewConnectorTable(connectorId, {
        table: selected,
        schema_name: schema || null,
        columns: pickedColumns.length ? pickedColumns : null,
        where: where.trim() || null,
        limit: 50,
      });
      setPreview(data);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "预览失败");
    }
  };

  const handleImport = async () => {
    if (connectorId === null || !selected) return;
    setImporting(true);
    setActionError(null);
    setImportResult(null);
    try {
      const result = await importConnectorTable(connectorId, {
        table: selected,
        schema_name: schema || null,
        columns: pickedColumns.length ? pickedColumns : null,
        where: where.trim() || null,
        order_by: orderBy.trim() || null,
        keyset_column: keyset.trim() || null,
        batch_size: batchSize ? Number(batchSize) : null,
        max_rows: maxRows ? Number(maxRows) : null,
        dataset_name: datasetName.trim() || null,
      });
      setImportResult(result);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "导入失败");
    } finally {
      setImporting(false);
    }
  };

  if (!connector) {
    return (
      <Panel title="数据接入">
        <p className="connector-muted">连接器不存在或已被删除。</p>
      </Panel>
    );
  }

  return (
    <Panel
      title={`数据接入 · ${connector.name}`}
      description="选择表 → 预览确认 → 抽取为数据集版本。抽取按批进行，可处理远超上传体积上限的数据量。"
    >
      {tableError ? <p className="connector-alert is-error">{tableError}</p> : null}

      <div className="connector-workbench">
        <div className="connector-pane">
          <div className="connector-pane-head">
            <span className="field-label">表 / 视图（{tables.length}）</span>
            {schemas.length > 0 ? (
              <select value={schema} onChange={(e) => setSchema(e.target.value)}>
                <option value="">默认 schema</option>
                {schemas.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            ) : null}
          </div>
          <div className="connector-table-list">
            {loadingTables ? (
              <p className="connector-muted">加载中…</p>
            ) : tables.length === 0 ? (
              <p className="connector-muted">未发现可用的表。</p>
            ) : (
              tables.map((item) => (
                <button
                  key={`${item.schema ?? ""}.${item.name}`}
                  type="button"
                  className={`connector-table-item${selected === item.name ? " is-active" : ""}`}
                  onClick={() => handlePickTable(item.name)}
                >
                  <span>{item.name}</span>
                  <span className="connector-table-kind">{item.type === "view" ? "视图" : "表"}</span>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="connector-pane connector-pane-wide">
          {!selected ? (
            <p className="connector-muted">请从左侧选择一张表。</p>
          ) : (
            <>
              <div className="connector-summary">
                <strong>{selected}</strong>
                <span className="connector-muted">{columns.length} 列</span>
              </div>

              {columns.length > 0 ? (
                <div className="connector-columns">
                  {columns.map((col) => {
                    const checked = pickedColumns.includes(col.name);
                    return (
                      <label key={col.name} className={`connector-column${checked ? " is-on" : ""}`}>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) =>
                            setPickedColumns((prev) =>
                              e.target.checked
                                ? [...prev, col.name]
                                : prev.filter((name) => name !== col.name),
                            )
                          }
                        />
                        <span className="connector-column-name">{col.name}</span>
                        <span className="connector-column-type">{col.type}</span>
                        {col.primary_key ? <span className="tag">PK</span> : null}
                      </label>
                    );
                  })}
                </div>
              ) : null}

              <p className="connector-muted connector-tip">
                不勾选任何列 = 全列导入。WHERE 为原始 SQL 片段（拒绝分号，防多语句注入）。
              </p>

              <div className="connector-form">
                <label className="field connector-form-wide">
                  <span className="field-label">WHERE 条件（可空）</span>
                  <input
                    type="text"
                    value={where}
                    placeholder="created_at >= '2025-01-01'"
                    onChange={(e) => setWhere(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span className="field-label">排序 / keyset 列</span>
                  <input
                    type="text"
                    value={orderBy}
                    placeholder="id"
                    onChange={(e) => setOrderBy(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span className="field-label">keyset 列（单调唯一）</span>
                  <input
                    type="text"
                    value={keyset}
                    placeholder="id"
                    onChange={(e) => setKeyset(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span className="field-label">批次行数（可空）</span>
                  <input
                    type="number"
                    value={batchSize}
                    placeholder="50000"
                    onChange={(e) => setBatchSize(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span className="field-label">最大行数（可空）</span>
                  <input
                    type="number"
                    value={maxRows}
                    placeholder="不限"
                    onChange={(e) => setMaxRows(e.target.value)}
                  />
                </label>
                <label className="field connector-form-wide">
                  <span className="field-label">目标数据集名（可空）</span>
                  <input
                    type="text"
                    value={datasetName}
                    placeholder={`${connector.name} · ${selected}`}
                    onChange={(e) => setDatasetName(e.target.value)}
                  />
                </label>
              </div>

              <div className="connector-actions">
                <button className="btn btn-sm" type="button" onClick={handlePreview}>
                  预览前 50 行
                </button>
                <button
                  className="btn btn-sm"
                  type="button"
                  onClick={handleImport}
                  disabled={importing}
                >
                  {importing ? "抽取中…" : "导入为数据集"}
                </button>
              </div>

              {actionError ? <p className="connector-alert is-error">{actionError}</p> : null}

              {importResult ? (
                <div className="connector-result">
                  <p className="connector-alert is-ok">
                    ✓ 已导入 {importResult.row_count.toLocaleString()} 行 /{" "}
                    {importResult.column_count} 列 → 数据集 {importResult.dataset_id}（v
                    {importResult.dataset_version}）
                  </p>
                  <div className="connector-metrics">
                    <span>抽取策略：{importResult.strategy}</span>
                    <span>批次：{importResult.batches}</span>
                    <span>耗时：{importResult.elapsed_seconds.toFixed(2)}s</span>
                    <span>
                      吞吐：
                      {importResult.rows_per_second > 0
                        ? `${importResult.rows_per_second.toLocaleString()} 行/秒`
                        : "—"}
                    </span>
                    {importResult.truncated ? <span className="is-warn">已触及行数上限，结果被截断</span> : null}
                  </div>
                  {importResult.warnings.length > 0 ? (
                    <ul className="connector-warnings">
                      {importResult.warnings.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ) : null}

              {preview ? (
                <div className="connector-preview">
                  <p className="connector-muted">
                    预览 {preview.row_count} 行（{preview.sql}）
                  </p>
                  <div className="connector-preview-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {preview.columns.map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((row, index) => (
                          <tr key={index}>
                            {preview.columns.map((col) => (
                              <td key={col}>{formatCell(row[col])}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>
      </div>
    </Panel>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
