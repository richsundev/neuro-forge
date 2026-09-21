"use client";

import { useEffect, useState } from "react";
import {
  apiGet,
  apiPost,
  type ApplicationSummary,
  type CanaryRecord,
  type ExperimentSummary,
  type PromotionApproval,
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
  promotions,
  canaries,
  production,
  onActionComplete,
}: {
  experiment: ExperimentSummary;
  promotions: PromotionRecord[];
  canaries: CanaryRecord[];
  production: ApplicationSummary["production"];
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

  // Promotions/canaries are sorted newest-first by the API, so the first match is the latest —
  // this reads server truth rather than only this row's own local state, so eligibility survives
  // a page reload instead of resetting whenever `decision`/`canaryResult` go back to null.
  const latestDecision = promotions.find((p) => p.genome_hash === experiment.best_genome_hash);
  const latestCanary = canaries.find((c) => c.candidate_hash === experiment.best_genome_hash);
  const stale = !experiment.baseline_is_current;
  const canPromote =
    !!latestDecision?.approved && !!latestCanary && !latestCanary.rollback_triggered && !stale;
  // Promoted-ness comes from the server (is this genome the application's champion?), not this
  // row's local state: after a reload local state is gone, and the button used to re-enable for an
  // already-promoted genome. A rolled-back genome is no longer the champion, so it can be promoted again.
  const isChampion = !!production && production.genome_hash === experiment.best_genome_hash;
  const promoted = isChampion ? { approved_by: production?.promoted_by ?? "" } : null;

  const datasetId = experiment.dataset_id;
  const baselineIdent = experiment.baseline;

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

  async function finalizePromotion() {
    if (!experiment.best_genome_hash) return;
    setBusy(true);
    setError(null);
    try {
      await apiPost("/api/v1/promotions/finalize", {
        genome_hash: experiment.best_genome_hash,
        experiment_id: experiment.experiment_id,
      });
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
        <span className="badge bg-slate-100 text-slate-600" title={experiment.baseline}>
          {experiment.baseline_version === 1 ? "from original v1" : `from champion v${experiment.baseline_version}`}
        </span>
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
          <button
            disabled={busy || !canPromote || !!promoted}
            onClick={finalizePromotion}
            title={
              stale
                ? "Production has moved on since this experiment ran — re-run it from the current champion"
                : canPromote
                  ? "Requires an admin-role API key"
                  : "Requires an approved promotion decision and a passed (non-rollback) canary run first"
            }
            className="rounded-md bg-good px-3 py-1.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-40"
          >
            Promote to production
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
          {decision.evidence.length > 0 && (
            <div className="mt-2 border-t border-slate-100 pt-2">
              <p className="stat-label">holdout evidence</p>
              <ul className="mt-1 list-inside list-disc text-xs text-slate-500">
                {decision.evidence.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </div>
          )}
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

      {promoted && (
        <div className="mt-2 rounded-md border border-slate-200 p-3 text-sm">
          <StatusBadge ok okLabel="promoted to production" /> approved by{" "}
          <span className="font-mono text-xs">{promoted.approved_by}</span>
        </div>
      )}
      {!promoted && stale && (
        <p className="mt-2 text-sm text-slate-500">
          Superseded &mdash; production has moved on since this experiment ran, so its winner can no longer be
          promoted. Re-run it from the current champion.
        </p>
      )}
      {!promoted && !stale && latestCanary && !latestCanary.rollback_triggered && latestDecision?.approved && (
        <p className="mt-2 text-sm text-slate-500">
          Passed promotion gates and canary — awaiting an admin&rsquo;s explicit &ldquo;Promote to
          production&rdquo;.
        </p>
      )}
    </div>
  );
}

export default function PromotionsPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [promotions, setPromotions] = useState<PromotionRecord[]>([]);
  const [canaries, setCanaries] = useState<CanaryRecord[]>([]);
  const [approvals, setApprovals] = useState<PromotionApproval[]>([]);
  const [applications, setApplications] = useState<ApplicationSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const [e, p, c, a, apps] = await Promise.all([
        apiGet<ExperimentSummary[]>("/api/v1/experiments"),
        apiGet<PromotionRecord[]>("/api/v1/promotions"),
        apiGet<CanaryRecord[]>("/api/v1/canaries"),
        apiGet<PromotionApproval[]>("/api/v1/promotions/approvals"),
        apiGet<ApplicationSummary[]>("/api/v1/applications"),
      ]);
      setExperiments(e.filter((x) => x.status === "completed" && x.recommendation));
      setPromotions(p);
      setCanaries(c);
      setApprovals(a);
      setApplications(apps);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  // Refetch only the history tables after a row action — refetching `experiments` too would
  // remount every CandidateRow and drop its in-progress local state (open canary form, etc).
  async function refreshHistory() {
    try {
      const [p, c, a, apps] = await Promise.all([
        apiGet<PromotionRecord[]>("/api/v1/promotions"),
        apiGet<CanaryRecord[]>("/api/v1/canaries"),
        apiGet<PromotionApproval[]>("/api/v1/promotions/approvals"),
        apiGet<ApplicationSummary[]>("/api/v1/applications"),
      ]);
      setPromotions(p);
      setCanaries(c);
      setApprovals(a);
      setApplications(apps);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function rollback(app: ApplicationSummary) {
    const label = app.production ? `v${app.production.version}` : "the current champion";
    if (!window.confirm(`Take ${label} of "${app.id}" out of production? The previous champion is restored.`)) return;
    try {
      await apiPost("/api/v1/promotions/rollback", { system_id: app.id });
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

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
            <CandidateRow
              key={e.experiment_id}
              experiment={e}
              promotions={promotions}
              canaries={canaries}
              production={applications.find((a) => a.id === e.application_id)?.production ?? null}
              onActionComplete={refreshHistory}
            />
          ))
        )}
      </section>

      <section className="card">
        <h2 className="mb-1 text-base font-semibold">In production</h2>
        <p className="mb-3 text-xs text-slate-500">
          Each application&rsquo;s champion. New experiments evolve from it, and promoting a candidate
          replaces it (the old one is kept and can be restored by a rollback).
        </p>
        {applications.length === 0 ? (
          <p className="text-sm text-slate-500">No applications yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2">application</th>
                <th className="pb-2">champion</th>
                <th className="pb-2">promoted by</th>
                <th className="pb-2" />
              </tr>
            </thead>
            <tbody>
              {applications.map((a) => (
                <tr key={a.id} className="border-t border-slate-100">
                  <td className="py-1.5 font-medium">{a.id}</td>
                  <td className="py-1.5">
                    {a.production ? (
                      <>
                        v{a.production.version}{" "}
                        <span className="font-mono text-xs text-slate-500">{a.production.genome_hash}</span>
                      </>
                    ) : (
                      <span className="text-slate-500">original baseline (v1)</span>
                    )}
                  </td>
                  <td className="py-1.5 text-slate-500">
                    {a.production?.promoted_by ?? "—"}
                    {a.production?.promoted_at && ` · ${new Date(a.production.promoted_at).toLocaleString()}`}
                  </td>
                  <td className="py-1.5 text-right">
                    {a.production && (
                      <button
                        onClick={() => rollback(a)}
                        title="Requires an admin-role API key"
                        className="rounded-md border border-bad/40 px-3 py-1 text-xs font-medium text-bad hover:bg-bad/5"
                      >
                        Roll back
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h2 className="mb-1 text-base font-semibold">Production history</h2>
        <p className="mb-2 text-xs text-slate-500">
          Every explicit admin action that changed what is in production — the human-approval gate
          between a passed canary and PROMOTED, and rollbacks.
        </p>
        {approvals.length === 0 ? (
          <p className="text-sm text-slate-500">Nothing has been promoted to production yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2">action</th>
                <th className="pb-2">application</th>
                <th className="pb-2">genome</th>
                <th className="pb-2">by</th>
                <th className="pb-2">when</th>
              </tr>
            </thead>
            <tbody>
              {approvals.map((a, i) => (
                <tr key={i} className="border-t border-slate-100">
                  <td className="py-1.5">
                    <StatusBadge ok={a.action === "promote"} okLabel="promoted" badLabel="rolled back" />
                  </td>
                  <td className="py-1.5">{a.system_id}</td>
                  <td className="py-1.5">
                    v{a.version} <span className="font-mono text-xs text-slate-500">{a.genome_hash}</span>
                  </td>
                  <td className="py-1.5">{a.approved_by}</td>
                  <td className="py-1.5 text-slate-500">{new Date(a.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
