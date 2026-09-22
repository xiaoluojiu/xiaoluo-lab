import { Link, useLocation } from "react-router-dom";

/** 404：历史上未知路由被静默重定向到首页，用户完全不知道发生了什么。 */
export default function NotFound() {
  const location = useLocation();
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="crumbs">小洛实验室 / <b>页面不存在</b></div>
          <h1>404 · 找不到这个页面</h1>
          <p>地址 <code>{location.pathname}</code> 没有对应的功能页面，可能是链接失效或功能已调整。</p>
        </div>
        <div className="page-actions">
          <Link className="btn btn-primary" to="/">回到工作台</Link>
        </div>
      </div>

      <div className="card">
        <h3>你可能想去</h3>
        <div className="grid g3" style={{ marginTop: "var(--space-3)" }}>
          <Link className="link-card" to="/datasets"><b>数据集列表</b><div className="muted">查看与上传数据集</div></Link>
          <Link className="link-card" to="/processing"><b>数据处理</b><div className="muted">清洗、转换与聚合</div></Link>
          <Link className="link-card" to="/ai"><b>AI 实验室</b><div className="muted">用对话完成分析</div></Link>
          <Link className="link-card" to="/ml"><b>机器学习</b><div className="muted">训练与评估模型</div></Link>
          <Link className="link-card" to="/workflow"><b>Workflow</b><div className="muted">编排数据处理流程</div></Link>
          <Link className="link-card" to="/reports"><b>报告中心</b><div className="muted">生成与导出报告</div></Link>
        </div>
      </div>
    </div>
  );
}
