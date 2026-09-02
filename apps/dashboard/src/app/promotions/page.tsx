"use client";

import { useEffect, useState } from "react";
import {
  apiGet,
  apiPost,
  type CanaryRecord,
  type ExperimentSummary,
  type PromotionDecision,
  type PromotionRecord,
} from "@/lib/api";

function StatusBadge({ ok, okLabel = "approved", badLabel = "rejected" }: { ok: boolean; okLabel?: string; badLabel?: string }) {
  return (
    <span className={`badge ${ok ? "bg-good/10 text-good" : "bg-bad/10 text-bad"}`}>
      {ok ? okLabel : badLabel}
    </span>
  );
}

function CandidateRow({
  experiment,
  onActionComplete,
}: {
  experiment: ExperimentSummary;
  onActionComplete: () => void;
}) {
  const [decision, setDecision] = useState<PromotionDecision | null>(null);
  const [canaryResult, setCanaryResult] = useState<
    { rollback_triggered: boolean; reasons: string[] } | null
  >(null);
  const [showCanaryForm, setShowCanaryForm] = useState(false);
  const [trafficFraction, setTrafficFraction] = useState(0.1);
  const [nRequests, setNRequests] = useState(200);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const datasetId = `${experiment.application_id}-dataset`;
  const baselineIdent = `${experiment.application_id}@v1`;

  async function requestPromotion() {
    if (!experiment.best_genome_hash) return;
    setBusy(true);
    setError(null);
    try {
      const result = await apiPost<PromotionDecision>("/api/v1/promotions", {
        genome_hash: experiment.best_genome_hash,
        experiment_id: experiment.experiment_id,
      });
      setDecision(result);
      onActionComplete();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function runCanary() {
    if (!experiment.best_genome_hash) return;
    setBusy(true);
    setError(null);
    try {
      const result = await apiPost<{ rollback_triggered: boolean; reasons: string[] }>(
        "/api/v1/canaries",
        {
          baseline_hash: baselineIdent,
          candidate_hash: experiment.best_genome_hash,
          dataset_id: datasetId,
          traffic_fraction: trafficFraction,
          n_requests: nRequests,
        }
      );
      setCanaryResult(result);
      onActionComplete();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border-t border-slate-100 py-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="font-medium">{experiment.experiment_id}</span>
        <span className="text-sm text-slate-500">{experiment.comparison_summary}</span>
        <span className="ml-auto flex gap-2">
          <button
            disabled={busy || !experiment.best_genome_hash}
            onClick={requestPromotion}
            className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-40"
          >
            Request promotion
          </button>
          <button
            disabled={busy || !experiment.best_genome_hash}
            onClick={() => setShowCanaryForm((v) => !v)}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium hover:bg-slate-50 disabled:opacity-40"
          >
            Run canary
          </button>
        </span>
      </div>
      <p className="mt-1 text-sm">{experiment.recommendation}</p>

      {showCanaryForm && (
        <div className="mt-3 flex flex-wrap items-end gap-3 rounded-md bg-slate-50 p-3 text-sm">
          <div>
            <label className="stat-label block">baseline</label>
            <span className="font-mono text-xs">{baselineIdent}</span>
          </div>
          <div>
            <label className="stat-label block">dataset</label>
            <span className="font-mono text-xs">{datasetId}</span>
          </div>
          <div>
            <label className="stat-label block">traffic fraction</label>
            <input
              type="number"
              step="0.05"
              min="0.01"
              max="1"
              value={trafficFraction}
              onChange={(e) => setTrafficFraction(Number(e.target.value))}
              className="w-20 rounded border border-slate-300 px-2 py-1"
            />
          </div>
          <div>
            <label className="stat-label block">n requests</label>
            <input
              type="number"
              min="10"
              value={nRequests}
              onChange={(e) => setNRequests(Number(e.target.value))}
              className="w-24 rounded border border-slate-300 px-2 py-1"
            />
          </div>
          <button
            disabled={busy}
            onClick={runCanary}
            className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-40"
          >
            Start
          </button>
        </div>
      )}

      {error && <p className="mt-2 text-sm text-bad">{error}</p>}

      {decision && (
        <div className="mt-2 rounded-md border border-slate-200 p-3 text-sm">
          <StatusBadge ok={decision.approved} /> next status: {decision.next_status}
          <ul className="mt-1 list-inside list-disc text-slate-600">
            {decision.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {canaryResult && (
        <div className="mt-2 rounded-md border border-slate-200 p-3 text-sm">
          <StatusBadge
            ok={!canaryResult.rollback_triggered}
            okLabel="canary passed"
            badLabel="rollback triggered"
          />
          <ul className="mt-1 list-inside list-disc text-slate-600">
            {canaryResult.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function PromotionsPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [promotions, setPromotions] = useState<PromotionRecord[]>([]);
  const [canaries, setCanaries] = useState<CanaryRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const [e, p, c] = await Promise.all([
        apiGet<ExperimentSummary[]>("/api/v1/experiments"),
        apiGet<PromotionRecord[]>("/api/v1/promotions"),
        apiGet<CanaryRecord[]>("/api/v1/canaries"),
      ]);
      setExperiments(e.filter((x) => x.status === "completed" && x.recommendation));
      setPromotions(p);
      setCanaries(c);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  // Refetch only the history tables after a row action — refetching `experiments` too would
  // remount every CandidateRow and drop its in-progress local state (open canary form, etc).
  async function refreshHistory() {
    try {
      const [p, c] = await Promise.all([
        apiGet<PromotionRecord[]>("/api/v1/promotions"),
        apiGet<CanaryRecord[]>("/api/v1/canaries"),
      ]);
      setPromotions(p);
      setCanaries(c);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Promotion</h1>
        <p className="text-sm text-slate-500">
          NeuroForge only recommends — nothing here reaches production without an explicit action
          below. See <code>docs/adr/0013-generation-separate-from-promotion.md</code>.
        </p>
      </div>

      {error && <div className="card text-sm text-bad">{error}</div>}

      <section className="card">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-base font-semibold">Candidates awaiting a decision</h2>
          <button onClick={load} className="text-xs text-accent hover:underline">
            refresh
          </button>
        </div>
        {experiments.length === 0 ? (
          <p className="text-sm text-slate-500">No completed experiments with a recommendation yet.</p>
        ) : (
          experiments.map((e) => (
            <CandidateRow key={e.experiment_id} experiment={e} onActionComplete={refreshHistory} />
          ))
        )}
      </section>

      <section className="card">
        <h2 className="mb-3 text-base font-semibold">Promotion history</h2>
        {promotions.length === 0 ? (
          <p className="text-sm text-slate-500">No promotion decisions recorded yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2">genome</th>
                <th className="pb-2">experiment</th>
                <th className="pb-2">decision</th>
                <th className="pb-2">when</th>
              </tr>
            </thead>
            <tbody>
              {promotions.map((p, i) => (
                <tr key={i} className="border-t border-slate-100">
                  <td className="py-1.5 font-mono text-xs">{p.genome_hash}</td>
                  <td className="py-1.5">{p.experiment_id}</td>
                  <td className="py-1.5">
                    <StatusBadge ok={p.approved} />
                  </td>
                  <td className="py-1.5 text-slate-500">{new Date(p.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h2 className="mb-3 text-base font-semibold">Canary results</h2>
        {canaries.length === 0 ? (
          <p className="text-sm text-slate-500">No canary runs recorded yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2">candidate</th>
                <th className="pb-2">traffic</th>
                <th className="pb-2">outcome</th>
                <th className="pb-2">when</th>
              </tr>
            </thead>
            <tbody>
              {canaries.map((c, i) => (
                <tr key={i} className="border-t border-slate-100">
                  <td className="py-1.5 font-mono text-xs">{c.candidate_hash}</td>
                  <td className="py-1.5">{(c.traffic_split * 100).toFixed(0)}%</td>
                  <td className="py-1.5">
                    <StatusBadge ok={!c.rollback_triggered} okLabel="passed" badLabel="rolled back" />
                  </td>
                  <td className="py-1.5 text-slate-500">{new Date(c.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
