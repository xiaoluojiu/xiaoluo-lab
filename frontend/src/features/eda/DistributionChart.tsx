import { CHART_SERIES, CHART_GRID } from "../../lib/chartColors";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface DistributionData {
  type: string;
  column: string;
  bins?: { label: string; count: number }[];
  counts?: { value: string; count: number }[];
  missing: number;
}

const COLORS = CHART_SERIES;

// Prompt 169：分布图（recharts 柱状图，数值/分类分布通用）。
export function DistributionChart({ data }: { data: DistributionData | null }) {
  if (!data) return <div className="muted">请先运行分布分析</div>;
  const rows =
    data.type === "numeric"
      ? (data.bins ?? []).map((b) => ({ name: b.label, count: b.count }))
      : (data.counts ?? []).map((c) => ({ name: c.value, count: c.count }));
  if (!rows.length) return <div className="muted">该列无可用分布数据（可能为常数列或空列）</div>;

  return (
    <div>
      <p className="muted">
        列「{data.column}」· {data.type === "numeric" ? "数值分布" : "分类分布"} · 缺失 {data.missing}
      </p>
      {/* 统一 16:9 比例（aspect），避免容器宽度变化导致的畸形拉伸 */}
      <ResponsiveContainer width="100%" aspect={1.8}>
        <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
          <XAxis dataKey="name" fontSize={11} interval={0} angle={-20} height={52} textAnchor="end" />
          <YAxis fontSize={11} allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {rows.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
