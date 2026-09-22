import { PageHeader } from "../../components/PageHeader";
import "./extensions.css";

const SLOTS = [
  { name: "数据库连接器", status: "soon" as const, desc: "接入 Postgres / MySQL / 对象存储等外部数据源。" },
  { name: "LLM 供应商", status: "soon" as const, desc: "接入更多大模型供应商，与默认兼容端点并存。" },
  { name: "通知渠道", status: "soon" as const, desc: "运行完成 / 失败事件推送到邮件、Slack、企业微信与 Webhook。" },
  { name: "自定义算子", status: "soon" as const, desc: "注册业务专属的数据处理与建模算子。" },
  { name: "团队协作", status: "soon" as const, desc: "多租户、成员权限与共享工作区。" },
  { name: "内置工具集", status: "enabled" as const, desc: "已接入 22 个内置工具（数据集 / 数据 / EDA / ML）。" },
];

const DEV_API = [
  { name: "REST API Token", desc: "为第三方与服务账号签发长期有效的 API Token。" },
  { name: "Webhook", desc: "关键事件（运行完成 / 失败）回调外部系统。" },
  { name: "OpenAPI", desc: "自动生成的 /openapi.json 与交互式文档。" },
];

const ROADMAP = [
  { m: "M1", t: "数据库连接器 + 通知渠道 Alpha" },
  { m: "M2", t: "LLM 供应商可插拔" },
  { m: "M3", t: "自定义算子市场" },
  { m: "M4", t: "团队协作（多租户 + 权限）" },
];

/**
 * 扩展中心。
 * 列出已接入与规划中的扩展能力，规划项随后端能力逐步点亮。
 */
export default function Extensions() {
  return (
    <div className="extensions-page">
      <PageHeader
        title="扩展中心"
        breadcrumbs={<>小洛实验室 / 扩展中心</>}
      />

      <section className="card-block">
        <h2 className="section-title">已接入与规划中</h2>
        <div className="grid g3">
          {SLOTS.map((s) => (
            <div className="card ext-card" key={s.name}>
              <div className="ext-head">
                <span className="ext-name">{s.name}</span>
                <span className={`badge ${s.status === "enabled" ? "success" : "warning"}`}>
                  {s.status === "enabled" ? "已接入" : "规划中"}
                </span>
              </div>
              <p className="ext-desc">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card-block">
        <h2 className="section-title">开发者接口</h2>
        <div className="grid g3">
          {DEV_API.map((d) => (
            <div className="card ext-card" key={d.name}>
              <div className="ext-head">
                <span className="ext-name">{d.name}</span>
              </div>
              <p className="ext-desc">{d.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card-block">
        <h2 className="section-title">路线图</h2>
        <div className="roadmap">
          {ROADMAP.map((r) => (
            <div className="roadmap-item" key={r.m}>
              <span className="roadmap-m">{r.m}</span>
              <span className="roadmap-t">{r.t}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
