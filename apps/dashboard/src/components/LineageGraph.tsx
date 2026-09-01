"use client";

import { useMemo, useState } from "react";
import type { LineageGraph as LineageGraphData, LineageNode } from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  GENERATED: "#94a3b8",
  VALIDATED: "#38bdf8",
  BENCHMARKED: "#a78bfa",
  HOLDOUT_TESTED: "#f59e0b",
  APPROVED: "#22c55e",
  CANARY: "#06b6d4",
  PROMOTED: "#16a34a",
  REJECTED: "#ef4444",
  ROLLED_BACK: "#b91c1c",
};

const COL_WIDTH = 170;
const ROW_HEIGHT = 90;
const NODE_R = 22;

export default function LineageGraph({ graph }: { graph: LineageGraphData }) {
  const [selected, setSelected] = useState<LineageNode | null>(null);

  const { positioned, width, height } = useMemo(() => {
    const byGeneration = new Map<number, LineageNode[]>();
    for (const node of graph.nodes) {
      const arr = byGeneration.get(node.generation) ?? [];
      arr.push(node);
      byGeneration.set(node.generation, arr);
    }
    const generations = [...byGeneration.keys()].sort((a, b) => a - b);
    const positioned = new Map<string, { x: number; y: number; node: LineageNode }>();
    let maxRow = 0;
    generations.forEach((gen, colIdx) => {
      const nodes = byGeneration.get(gen)!.sort((a, b) => a.version - b.version);
      nodes.forEach((node, rowIdx) => {
        positioned.set(node.hash, {
          x: 60 + colIdx * COL_WIDTH,
          y: 50 + rowIdx * ROW_HEIGHT,
          node,
        });
        maxRow = Math.max(maxRow, rowIdx);
      });
    });
    return {
      positioned,
      width: 120 + generations.length * COL_WIDTH,
      height: 100 + maxRow * ROW_HEIGHT,
    };
  }, [graph]);

  if (graph.nodes.length === 0) {
    return <p className="text-sm text-slate-500">No genomes yet for this system.</p>;
  }

  return (
    <div className="flex gap-6">
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50">
        <svg width={Math.max(width, 400)} height={Math.max(height, 200)}>
          {graph.edges.map((edge) => {
            const from = positioned.get(edge.from);
            const to = positioned.get(edge.to);
            if (!from || !to) return null;
            const midX = (from.x + to.x) / 2;
            return (
              <path
                key={`${edge.from}-${edge.to}`}
                d={`M ${from.x + NODE_R} ${from.y} C ${midX} ${from.y}, ${midX} ${to.y}, ${to.x - NODE_R} ${to.y}`}
                fill="none"
                stroke="#cbd5e1"
                strokeWidth={2}
              />
            );
          })}
          {[...positioned.values()].map(({ x, y, node }) => (
            <g key={node.hash} onClick={() => setSelected(node)} className="cursor-pointer">
              <circle
                cx={x}
                cy={y}
                r={NODE_R}
                fill={STATUS_COLOR[node.status] ?? "#94a3b8"}
                stroke={selected?.hash === node.hash ? "#0f172a" : "white"}
                strokeWidth={selected?.hash === node.hash ? 3 : 2}
              />
              <text x={x} y={y + 4} textAnchor="middle" fontSize={11} fill="white" fontWeight={600}>
                v{node.version}
              </text>
              <text x={x} y={y + NODE_R + 14} textAnchor="middle" fontSize={10} fill="#64748b">
                gen {node.generation}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="card w-72 shrink-0">
        {selected ? (
          <div className="space-y-2 text-sm">
            <p className="stat-label">Selected genome</p>
            <p className="font-mono text-xs">{selected.hash}</p>
            <p>
              version <b>{selected.version}</b> · generation <b>{selected.generation}</b>
            </p>
            <p>
              status:{" "}
              <span
                className="badge text-white"
                style={{ backgroundColor: STATUS_COLOR[selected.status] ?? "#94a3b8" }}
              >
                {selected.status}
              </span>
            </p>
            <p className="stat-label pt-2">Mutations from parent</p>
            {selected.mutations.length === 0 ? (
              <p className="text-slate-500">baseline (no parent)</p>
            ) : (
              <ul className="list-inside list-disc space-y-1">
                {selected.mutations.map((m, i) => (
                  <li key={i} className="font-mono text-xs">
                    {m}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <p className="text-sm text-slate-500">Click a genome to inspect its lineage.</p>
        )}
      </div>
    </div>
  );
}
