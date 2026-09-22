/* =============================================================
   Blocks —— 页面骨架级区块
   -------------------------------------------------------------
   针对「像过时后台」的两个根因给出统一解法：

   1. 面板与标题混在一个白框里 → SectionHeader 把分区标题**外提**到卡片之外，
      Panel 只承载内容，两层信息不再压成一坨。
   2. 所有 section 都是同一个白卡 → HeroBand 提供唯一一处品牌色带，
      用于页面首屏的身份区（每页最多一处）。

   这些组件只负责结构与留白，颜色全部来自 tokens.css。
   ============================================================= */

import type { ReactNode } from "react";
import { Icon, type IconName } from "../icons/Icon";

/* ---------- 分区标题（Title 外提，不放在卡片里） ---------- */
export interface SectionHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
}

export function SectionHeader({ title, description, actions, className }: SectionHeaderProps) {
  return (
    <div className={`section-header${className ? ` ${className}` : ""}`}>
      <div className="section-header-text">
        <h2 className="section-title">{title}</h2>
        {description ? <p className="section-desc">{description}</p> : null}
      </div>
      {actions ? <div className="section-actions">{actions}</div> : null}
    </div>
  );
}

/* ---------- 段落卡 ---------- */
export interface PanelProps {
  children: ReactNode;
  /** 卡片内的二级标题（无分区感时用，有 SectionHeader 时不必再传） */
  title?: string;
  description?: string;
  actions?: ReactNode;
  footer?: ReactNode;
  /** 去掉内边距，用于表格等自带边界的内容 */
  flush?: boolean;
  className?: string;
}

export function Panel({ children, title, description, actions, footer, flush, className }: PanelProps) {
  return (
    <section className={`panel${flush ? " is-flush" : ""}${className ? ` ${className}` : ""}`}>
      {(title || actions) && (
        <header className="panel-head">
          <div>
            {title ? <h3 className="panel-title">{title}</h3> : null}
            {description ? <p className="panel-desc">{description}</p> : null}
          </div>
          {actions ? <div className="panel-actions">{actions}</div> : null}
        </header>
      )}
      <div className="panel-body">{children}</div>
      {footer ? <footer className="panel-foot">{footer}</footer> : null}
    </section>
  );
}

/* ---------- 首屏品牌带 ---------- */
export interface HeroBandProps {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
  /** 右侧或下方的补充信息条 */
  meta?: ReactNode;
  icon?: IconName;
}

export function HeroBand({ eyebrow, title, description, actions, meta, icon = "sparkles" }: HeroBandProps) {
  return (
    <section className="hero-band">
      <div className="hero-main">
        {eyebrow ? (
          <span className="hero-eyebrow">
            <Icon name={icon} size={14} />
            {eyebrow}
          </span>
        ) : null}
        <h1 className="hero-title">{title}</h1>
        {description ? <p className="hero-desc">{description}</p> : null}
        {actions ? <div className="hero-actions">{actions}</div> : null}
      </div>
      {meta ? <div className="hero-meta">{meta}</div> : null}
    </section>
  );
}

/* ---------- 插画化空态 ---------- */
export interface EmptyStateProps {
  title: string;
  description?: string;
  action?: ReactNode;
  icon?: IconName;
  compact?: boolean;
}

export function EmptyState({ title, description, action, icon = "database", compact = false }: EmptyStateProps) {
  return (
    <div className={`empty-state${compact ? " is-compact" : ""}`}>
      <span className="empty-illustration" aria-hidden="true">
        <Icon name={icon} size={26} />
      </span>
      <strong className="empty-title">{title}</strong>
      {description ? <span className="empty-desc">{description}</span> : null}
      {action ? <div className="empty-action">{action}</div> : null}
    </div>
  );
}
