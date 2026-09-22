import type { ReactNode } from "react";

/**
 * 统一页面头部：面包屑 + 标题 + 一句说明 + 右上角操作区。
 *
 * 历史问题：13 个页面里只有 9 个用了 .page-header，工作台 / AI 实验室 / 设置
 * 各自写了一套，标题字号与操作按钮位置因此不统一；且没有任何页面有面包屑，
 * /learning/workspace、/learning/history 实际上成了「进得去、出不来」的孤儿路由。
 */
export function PageHeader({
  title,
  description,
  breadcrumbs,
  actions,
}: {
  title: string;
  description?: ReactNode;
  breadcrumbs?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div className="page-header-main">
        {breadcrumbs && <div className="crumbs">{breadcrumbs}</div>}
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  );
}
