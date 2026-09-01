"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { CandidateEvaluation } from "@/lib/api";

export default function ParetoChart({ candidates }: { candidates: CandidateEvaluation[] }) {
  const optimal = candidates.filter((c) => c.is_pareto_optimal);
  const dominated = candidates.filter((c) => !c.is_pareto_optimal);

  const toPoint = (c: CandidateEvaluation) => ({
    cost: c.metrics.cost_usd,
    quality: c.metrics.quality,
    label: `v${c.version}`,
  });

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="cost"
          type="number"
          name="cost"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(v: number) => v.toFixed(4)}
          tick={{ fontSize: 12 }}
          label={{ value: "cost (USD/request)", position: "insideBottom", offset: -4, fontSize: 12 }}
        />
        <YAxis
          dataKey="quality"
          type="number"
          name="quality"
          domain={[0, 1]}
          tick={{ fontSize: 12 }}
          label={{ value: "quality", angle: -90, position: "insideLeft", fontSize: 12 }}
        />
        <ZAxis range={[60, 60]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(value: number) => value.toFixed(4)}
          labelFormatter={() => ""}
        />
        <Scatter name="dominated" data={dominated.map(toPoint)} fill="#cbd5e1" isAnimationActive={false} />
        <Scatter name="Pareto-optimal" data={optimal.map(toPoint)} fill="#16a34a" isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
