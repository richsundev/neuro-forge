"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import ApiKeyBar from "@/components/ApiKeyBar";
import { apiGet, type ApplicationSummary, type ExperimentSummary } from "@/lib/api";

const STATUS_BADGE: Record<string, string> = {
  completed: "bg-good/10 text-good",
  running: "bg-accent/10 text-accent",
  created: "bg-slate-200 text-slate-600",
  queued: "bg-warn/10 text-warn",
  failed: "bg-bad/10 text-bad",
  cancelled: "bg-warn/10 text-warn",
};

export default function OverviewPage() {
  const [apps, setApps] = useState<ApplicationSummary[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const [a, e] = await Promise.all([
        apiGet<ApplicationSummary[]>("/api/v1/applications"),
        apiGet<ExperimentSummary[]>("/api/v1/experiments"),
      ]);
      setApps(a);
      setExperiments(e);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-6">
      <ApiKeyBar onSaved={load} />
      {error && (
        <div className="card border-bad/30 bg-bad/5 text-sm text-bad">
          Could not reach the API ({error}). Set NEXT_PUBLIC_API_URL and a valid X-API-Key above, then
          refresh.
        </div>
      )}

      <section className="card">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-base font-semibold">Applications</h2>
          <button onClick={load} className="text-xs text-accent hover:underline">
            refresh
          </button>
        </div>
        {apps.length === 0 ? (
          <p className="text-sm text-slate-500">
            No applications yet — seed one with <code>neuroforge app create</code> or the API.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2">id</th>
                <th className="pb-2">domain</th>
                <th className="pb-2">in production</th>
                <th className="pb-2">evolution graph</th>
              </tr>
            </thead>
            <tbody>
              {apps.map((a) => (
                <tr key={a.id} className="border-t border-slate-100">
                  <td className="py-2 font-medium">{a.id}</td>
                  <td className="py-2">{a.domain}</td>
                  <td className="py-2">
                    {a.production ? (
                      <span title={a.production.genome_hash}>
                        <span className="badge bg-good/10 text-good">v{a.production.version}</span>
                        <span className="ml-2 font-mono text-xs text-slate-500">{a.production.genome_hash}</span>
                      </span>
                    ) : (
                      <span className="text-slate-500">original baseline</span>
                    )}
                  </td>
                  <td className="py-2">
                    <Link href={`/genomes/${a.id}`} className="text-accent hover:underline">
                      view lineage →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-base font-semibold">Experiments</h2>
          <button onClick={load} className="text-xs text-accent hover:underline">
            refresh
          </button>
        </div>
        {experiments.length === 0 ? (
          <p className="text-sm text-slate-500">No experiments yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2">experiment</th>
                <th className="pb-2">strategy</th>
                <th className="pb-2">status</th>
                <th className="pb-2" />
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.experiment_id} className="border-t border-slate-100">
                  <td className="py-2 font-medium">{e.experiment_id}</td>
                  <td className="py-2">{e.strategy}</td>
                  <td className="py-2">
                    <span className={`badge ${STATUS_BADGE[e.status] ?? "bg-slate-200 text-slate-600"}`}>
                      {e.status}
                    </span>
                  </td>
                  <td className="py-2 text-right">
                    <Link href={`/experiments/${e.experiment_id}`} className="text-accent hover:underline">
                      details →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
