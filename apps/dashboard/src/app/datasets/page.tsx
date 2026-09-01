"use client";

import { useEffect, useState } from "react";
import DifficultyChart from "@/components/DifficultyChart";
import { apiGet, type DatasetSummary } from "@/lib/api";

export default function ChallengeEvolutionPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<DatasetSummary[]>("/api/v1/datasets")
      .then(setDatasets)
      .catch((err) => setError((err as Error).message));
  }, []);

  const grouped = datasets.reduce<Record<string, DatasetSummary[]>>((acc, d) => {
    (acc[d.dataset_id] ??= []).push(d);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Challenge Evolution</h1>
        <p className="text-sm text-slate-500">
          The benchmark itself gets harder as candidates saturate it (via <code>dataset evolve</code>),
          so a high score can&apos;t be gamed against a fixed easy dataset.
        </p>
      </div>
      {error && <div className="card text-sm text-bad">{error}</div>}
      {Object.keys(grouped).length === 0 && !error && (
        <p className="text-sm text-slate-500">No datasets seeded yet.</p>
      )}
      {Object.entries(grouped).map(([datasetId, versions]) => (
        <div key={datasetId} className="card">
          <p className="stat-label mb-2">{datasetId}</p>
          <DifficultyChart versions={versions} />
          <table className="mt-2 w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-1">version</th>
                <th className="pb-1">source</th>
                <th className="pb-1">challenges</th>
                <th className="pb-1">difficulty</th>
              </tr>
            </thead>
            <tbody>
              {[...versions]
                .sort((a, b) => a.version - b.version)
                .map((v) => (
                  <tr key={v.version} className="border-t border-slate-100">
                    <td className="py-1">v{v.version}</td>
                    <td className="py-1">{v.source}</td>
                    <td className="py-1">{v.n_challenges}</td>
                    <td className="py-1">{v.difficulty_score.toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
