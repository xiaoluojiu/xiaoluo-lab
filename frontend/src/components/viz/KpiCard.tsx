/* =============================================================
   KpiCard —— 承载信息的指标卡
   -------------------------------------------------------------
   替代原先各页手写的「一行标签 + 一行裸数字」。
   一块卡承担四类信息：语义图标砖 / 主数值 / 环比徽标 / 迷你趋势。

   - 数值变化用 count-up（700ms 三次缓出），并遵守 prefers-reduced-motion。
   - 趋势色：正向 --success，负向 --danger（与金融红绿无关，此处是业务语义）。
   - to 有值时整卡可点，保留键盘可达性（真实 <a>，不用 div+onClick）。
   ============================================================= */

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Icon, type IconName } from "../icons/Icon";
import { Sparkline, type SparkTone } from "./Sparkline";

export interface KpiTrend {
  /** 环比增量，正负决定颜色与箭头方向 */
  delta: number;
  /** 对比口径说明，如「较上周」 */
  label?: string;
  /** 已带单位的文本（如 "+12.4%"），缺省时按 delta 推导 */
  text?: string;
}

export interface KpiCardProps {
  label: string;
  value: number | string | null;
  icon: IconName;
  tone?: SparkTone;
  trend?: KpiTrend;
  spark?: number[];
  hint?: string;
  to?: string;
  loading?: boolean;
  suffix?: string;
  /** value 为数字时的小数位，默认 0 */
  precision?: number;
}

function prefersReducedMotion() {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** 数字滚动：从上一次的目标值缓动到新值，尊重减弱动效偏好。 */
function useCountUp(target: number, animate: boolean) {
  const [display, setDisplay] = useState(target);
  const prevTarget = useRef(target);

  useEffect(() => {
    const from = prevTarget.current;
    prevTarget.current = target;
    if (!animate || typeof target !== "number" || prefersReducedMotion() || from === target) {
      setDisplay(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const duration = 700;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(from + (target - from) * eased);
      if (progress < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, animate]);

  return display;
}

function formatValue(value: number, precision: number) {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  });
}

export function KpiCard({
  label,
  value,
  icon,
  tone = "primary",
  trend,
  spark,
  hint,
  to,
  loading = false,
  suffix,
  precision = 0,
}: KpiCardProps) {
  const numeric = typeof value === "number" && Number.isFinite(value);
  const animated = useCountUp(numeric ? value : 0, numeric);
  const shown = loading ? "—" : numeric ? formatValue(animated, precision) : (value ?? "—");

  const inner = (
    <>
      <div className="kpi-head">
        <span className={`kpi-icon kpi-tone-${tone}`}>
          <Icon name={icon} size={18} />
        </span>
        <span className="kpi-label">{label}</span>
      </div>
      <div className="kpi-body">
        <span className="kpi-value">
          {shown}
          {suffix ? <span className="kpi-suffix">{suffix}</span> : null}
        </span>
        {trend && !loading ? (
          <span className={`kpi-delta ${trend.delta >= 0 ? "is-up" : "is-down"}`}>
            <Icon name={trend.delta >= 0 ? "trend-up" : "trend-down"} size={13} />
            {trend.text ?? `${trend.delta > 0 ? "+" : ""}${trend.delta}`}
          </span>
        ) : null}
      </div>
      {hint ? <span className="kpi-hint">{hint}</span> : null}
      {spark && spark.length > 1 ? (
        <div className="kpi-spark">
          <Sparkline data={spark} tone={tone} variant="area" height={34} />
        </div>
      ) : null}
    </>
  );

  if (to) {
    return (
      <Link to={to} className="kpi-card is-link">
        {inner}
      </Link>
    );
  }
  return <div className="kpi-card">{inner}</div>;
}
