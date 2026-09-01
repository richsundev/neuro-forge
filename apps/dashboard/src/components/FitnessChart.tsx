"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function FitnessChart({ history }: { history: number[] }) {
  const data = history.map((fitness, i) => ({ candidate: i + 1, fitness }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="candidate"
          type="number"
          domain={["dataMin", "dataMax"]}
          tick={{ fontSize: 12 }}
          label={{ value: "candidate #", position: "insideBottom", offset: -2, fontSize: 12 }}
        />
        <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
        <Tooltip formatter={(v: number) => v.toFixed(4)} />
        <Line
          type="monotone"
          dataKey="fitness"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
