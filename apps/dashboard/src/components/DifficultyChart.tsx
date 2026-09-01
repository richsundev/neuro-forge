"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DatasetSummary } from "@/lib/api";

export default function DifficultyChart({ versions }: { versions: DatasetSummary[] }) {
  const data = [...versions]
    .sort((a, b) => a.version - b.version)
    .map((v) => ({ label: `v${v.version}`, difficulty: v.difficulty_score, n: v.n_challenges }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" tick={{ fontSize: 12 }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Bar dataKey="difficulty" fill="#6366f1" radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
