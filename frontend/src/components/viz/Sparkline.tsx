/* =============================================================
   Sparkline —— 零依赖迷你趋势图
   -------------------------------------------------------------
   为什么不用 recharts：
   - KPI 卡里要的是一条 120×36 的趋势线，引入图表库只为这个是浪费；
   - recharts 的属性（fill/stroke）不解析 var()，而这里直接用 CSS 属性，
     可以天然继承 tokens.css 的主题变量，深浅色自动跟随。

   variant: line（折线 + 末端点）/ area（折线 + 渐变面，实为纯色低透明）/ bar（柱状）
   ============================================================= */

export type SparkTone = "primary" | "success" | "warning" | "danger" | "info";

const TONE_VAR: Record<SparkTone, string> = {
  primary: "var(--primary)",
  success: "var(--success)",
  warning: "var(--warning)",
  danger: "var(--danger)",
  info: "var(--info)",
};

export interface SparklineProps {
  data: number[];
  /** 视觉高度，宽度自适应容器 */
  height?: number;
  variant?: "line" | "area" | "bar";
  tone?: SparkTone;
  className?: string;
  ariaLabel?: string;
}

/** 归一化到画布坐标；全平的序列走中线，避免除零导致 NaN 路径。 */
function toPoints(data: number[], width: number, height: number, pad: number) {
  const safe = data.length ? data : [0];
  const max = Math.max(...safe);
  const min = Math.min(...safe);
  const span = max - min;
  const usable = height - pad * 2;
  const step = safe.length > 1 ? width / (safe.length - 1) : width;
  return safe.map((value, index) => {
    const ratio = span === 0 ? 0.5 : (value - min) / span;
    return { x: index * step, y: pad + (1 - ratio) * usable };
  });
}

export function Sparkline({
  data,
  height = 36,
  variant = "line",
  tone = "primary",
  className,
  ariaLabel,
}: SparklineProps) {
  const width = 120;
  const pad = 3;
  const color = TONE_VAR[tone];
  const points = toPoints(data, width, height, pad);

  if (!data || data.length === 0) {
    return <div className={`spark-empty${className ? ` ${className}` : ""}`} style={{ height }} aria-hidden="true" />;
  }

  const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
  const last = points[points.length - 1];

  return (
    <svg
      className={className}
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      preserveAspectRatio="none"
      role={ariaLabel ? "img" : "presentation"}
      aria-label={ariaLabel}
      aria-hidden={ariaLabel ? undefined : true}
    >
      {variant === "bar" ? (
        points.map((p, i) => {
          const barWidth = Math.max(2, (width / points.length) * 0.62);
          return (
            <rect
              key={i}
              x={p.x - barWidth / 2}
              y={p.y}
              width={barWidth}
              height={Math.max(1, height - pad - p.y)}
              rx={1.5}
              fill={color}
              opacity={0.85}
            />
          );
        })
      ) : (
        <>
          {variant === "area" && (
            <path
              d={`${line} L${width} ${height} L0 ${height} Z`}
              fill={color}
              opacity={0.14}
              stroke="none"
            />
          )}
          <path
            d={line}
            fill="none"
            stroke={color}
            strokeWidth={1.8}
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
          <circle cx={last.x} cy={last.y} r={2.4} fill={color} />
        </>
      )}
    </svg>
  );
}
