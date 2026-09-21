"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import FitnessChart from "@/components/FitnessChart";
import ParetoChart from "@/components/ParetoChart";
import {
  apiGet,
  apiPost,
  type CandidateEvaluation,
  type ExperimentDetail,
  type ExperimentProgress,
} from "@/lib/api";

const ACTIVE = new Set(["queued", "running"]);

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

function ConfigCard({ config }: { config: ExperimentDetail["config"] }) {
  const sections = [...new Set(Object.keys(config.search_space.parameters).map((p) => p.split(".")[0]))];
  const weights = Object.entries(config.objectives.objectives)
    .filter(([, o]) => o.weight > 0)
    .sort((a, b) => b[1].weight - a[1].weight);
  const g = config.promotion_gates;
  return (
    <div className="card grid gap-4 text-sm md:grid-cols-4">
      <div>
        <p className="stat-label">Searching</p>
        <p>{sections.join(", ")}</p>
        <p className="text-xs text-slate-500">{Object.keys(config.search_space.parameters).length} fields</p>
      </div>
      <div>
        <p className="stat-label">Optimizing for</p>
        <p>{weights.slice(0, 3).map(([n, o]) => `${n} ${Math.round(o.weight * 100)}%`).join(" · ")}</p>
      </div>
      <div>
        <p className="stat-label">Budget</p>
        <p>{config.budget.max_candidates} candidates</p>
        <p className="text-xs text-slate-500">batch {config.batch_size} · seed {config.seed}</p>
      </div>
      <div>
        <p className="stat-label">Gates</p>
        <p>cost ≤ +{Math.round(g.max_cost_increase * 100)}% · latency ≤ +{Math.round(g.max_latency_increase * 100)}%</p>
        <p className="text-xs text-slate-500">
          quality ≥ {g.min_quality} · violations ≤ {config.safety_constraints.max_policy_violation_rate}
        </p>
      </div>
    </div>
  );
}

function LiveProgress({ detail, progress }: { detail: ExperimentDetail; progress: ExperimentProgress | null }) {
  const total = detail.config.budget.max_candidates;
  const done = progress?.candidates_completed ?? 0;
  const pct = Math.min(100, Math.round((done / total) * 100));
  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">
          {detail.status === "queued" ? "Queued…" : "Running"} — {done} / {total} candidates
        </span>
        <span className="text-slate-500">
          {progress ? `generation ${progress.generations_completed}` : "waiting for the first generation"}
          {progress?.best_fitness != null && ` · best fitness ${progress.best_fitness.toFixed(3)}`}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${pct}%` }} />
      </div>
      {progress && progress.fitness_history.length > 1 && <FitnessChart history={progress.fitness_history} />}
    </div>
  );
}

export default function ExperimentDetailPage() {
  const params = useParams<{ id: string }>();
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [progress, setProgress] = useState<ExperimentProgress | null>(null);
  const [candidates, setCandidates] = useState<CandidateEvaluation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const active = detail !== null && ACTIVE.has(detail.status);

  // Load once, then keep polling for as long as the experiment is queued or running, and once more
  // after it stops so the final result (and its candidates) replace the live view.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let loaded = false;
    let wasActive = false;

    async function load() {
      try {
        const d = await apiGet<ExperimentDetail>(`/api/v1/experiments/${params.id}`);
        if (cancelled) return;
        setDetail(d);
        loaded = true;
        const running = ACTIVE.has(d.status);
        wasActive = running;
        try {
          setProgress(await apiGet<ExperimentProgress>(`/api/v1/experiments/${params.id}/checkpoint`));
        } catch {
          // no checkpoint yet — the first generation hasn't finished
        }
        if (!running) {
          try {
            const c = await apiGet<{ candidates: CandidateEvaluation[] }>(`/api/v1/experiments/${params.id}/candidates`);
            if (!cancelled) setCandidates(c.candidates);
          } catch {
            // no events yet — fine, just skip the Pareto chart
          }
        }
        if (running && !cancelled) timer = setTimeout(load, 1000);
      } catch (err) {
        if (cancelled) return;
        // A dropped request mid-run shouldn't replace a live page with an error and stop the updates.
        if (loaded && wasActive) timer = setTimeout(load, 3000);
        else setError((err as Error).message);
      }
    }

    load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [params.id, tick]);

  const act = useCallback(
    async (path: string) => {
      setActionError(null);
      try {
        await apiPost(path, {});
        setTick((t) => t + 1);
      } catch (err) {
        setActionError((err as Error).message);
      }
    },
    []
  );

  if (error) return <div className="card text-sm text-bad">{error}</div>;
  if (!detail) return <p className="text-sm text-slate-500">Loading…</p>;

  const result = detail.result;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">{detail.experiment_id}</h1>
          <p className="text-sm text-slate-500">
            status: <span className="font-medium text-ink">{detail.status}</span> · strategy: {detail.config.search_strategy}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(detail.status === "created" || detail.status === "failed" || detail.status === "cancelled") && (
            <button onClick={() => act(`/api/v1/experiments/${params.id}/start`)} className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90">
              {detail.status === "created" ? "Start" : "Resume"}
            </button>
          )}
          {active && (
            <button onClick={() => act(`/api/v1/experiments/${params.id}/cancel`)} className="rounded-md border border-bad/40 px-3 py-1.5 text-sm font-medium text-bad hover:bg-bad/5">
              Cancel
            </button>
          )}
          {detail.status === "completed" && result && (
            <Link href="/promotions" className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90">
              Review for promotion →
            </Link>
          )}
        </div>
      </div>
      {actionError && <div className="card text-sm text-bad">{actionError}</div>}

      <ConfigCard config={detail.config} />

      {active && <LiveProgress detail={detail} progress={progress} />}

      {!result && !active ? (
        <div className="card text-sm text-slate-500">
          {detail.status === "failed"
            ? "This run failed or was interrupted — see the API logs. Resume continues from its last checkpoint."
            : detail.status === "cancelled"
              ? "Cancelled before finishing. Resume continues from its last checkpoint."
              : "Not started yet."}
        </div>
      ) : !result ? null : (
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
