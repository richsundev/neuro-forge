"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import FitnessChart from "@/components/FitnessChart";
import ParetoChart from "@/components/ParetoChart";
import { apiGet, type CandidateEvaluation, type ExperimentDetail } from "@/lib/api";

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
    </div>
  );
}

function pct(x: number): string {
  return `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`;
}

export default function ExperimentDetailPage() {
  const params = useParams<{ id: string }>();
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [candidates, setCandidates] = useState<CandidateEvaluation[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const d = await apiGet<ExperimentDetail>(`/api/v1/experiments/${params.id}`);
        setDetail(d);
        try {
          const c = await apiGet<{ candidates: CandidateEvaluation[] }>(
            `/api/v1/experiments/${params.id}/candidates`
          );
          setCandidates(c.candidates);
        } catch {
          // no events yet — fine, just skip the Pareto chart
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }
    load();
  }, [params.id]);

  if (error) return <div className="card text-sm text-bad">{error}</div>;
  if (!detail) return <p className="text-sm text-slate-500">Loading…</p>;

  const result = detail.result;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">{detail.experiment_id}</h1>
        <p className="text-sm text-slate-500">
          status: {detail.status} · strategy: {String(detail.config.search_strategy)}
        </p>
      </div>

      {!result ? (
        <div className="card text-sm text-slate-500">
          No result yet — run this experiment via the CLI (<code>neuroforge experiment run</code>) or
          <code> POST /api/v1/experiments/{"{id}"}/run</code>.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard label="Generations" value={String(result.generations_completed)} />
            <StatCard label="Candidates evaluated" value={String(result.candidates_evaluated)} />
            <StatCard label="Quality change" value={pct(result.comparison.relative_diff)} />
            <StatCard label="Conclusion" value={result.comparison.conclusion.replaceAll("_", " ")} />
          </div>

          <div className="card">
            <p className="stat-label mb-1">Recommendation</p>
            <p className="text-base font-medium">{result.recommendation}</p>
            <p className="mt-2 text-sm text-slate-500">
              {result.comparison.summary} (stopped: {result.stop_reason})
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <div className="card">
              <p className="stat-label mb-2">Fitness over candidates evaluated</p>
              <FitnessChart history={result.fitness_history} />
            </div>
            <div className="card">
              <p className="stat-label mb-2">Pareto frontier (quality vs. cost)</p>
              {candidates.length > 0 ? (
                <ParetoChart candidates={candidates} />
              ) : (
                <p className="text-sm text-slate-500">No per-candidate events recorded.</p>
              )}
            </div>
          </div>

          <div className="card overflow-x-auto">
            <p className="stat-label mb-2">Baseline vs. best candidate</p>
            <table className="w-full text-sm">
              <thead className="text-left text-slate-500">
                <tr>
                  <th className="pb-2">metric</th>
                  <th className="pb-2">baseline</th>
                  <th className="pb-2">best candidate</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(result.baseline_metrics).map((key) => (
                  <tr key={key} className="border-t border-slate-100">
                    <td className="py-1.5">{key}</td>
                    <td className="py-1.5">{result.baseline_metrics[key].toFixed(4)}</td>
                    <td className="py-1.5 font-medium">{result.best_metrics[key]?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
