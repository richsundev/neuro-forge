"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  apiGet,
  apiPost,
  type ApplicationSummary,
  type DomainInfo,
  type ExperimentSummary,
} from "@/lib/api";

const STRATEGIES: { value: string; label: string; hint: string }[] = [
  { value: "evolutionary", label: "Evolutionary", hint: "population, selection, crossover — the default" },
  { value: "bayesian", label: "Bayesian", hint: "Gaussian-process surrogate; few, expensive evaluations" },
  { value: "bandit", label: "Bandit (UCB1)", hint: "explore/exploit over a bounded pool of configurations" },
  { value: "random", label: "Random", hint: "baseline for comparison" },
];

// Emphasis presets: pin one objective's share of the total; the rest keep their relative balance.
const PRIORITIES: { value: string; label: string; hint: string; weights: Record<string, number> | null }[] = [
  { value: "balanced", label: "Balanced", hint: "the default objective weights", weights: null },
  { value: "quality", label: "Quality first", hint: "quality is 60% of the objective", weights: { quality: 0.6 } },
  { value: "cost", label: "Cost first", hint: "cost is 40% of the objective", weights: { cost_usd: 0.4 } },
  { value: "latency", label: "Latency first", hint: "latency is 35% of the objective", weights: { latency_ms: 0.35 } },
];

const NEW_APP = "__new__";

export default function NewExperimentPage() {
  const router = useRouter();
  const [apps, setApps] = useState<ApplicationSummary[]>([]);
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [appChoice, setAppChoice] = useState<string>(NEW_APP);
  const [newAppId, setNewAppId] = useState("support-agent");
  const [newAppDomain, setNewAppDomain] = useState("forge-support");
  const [experimentId, setExperimentId] = useState("");
  const [baseline, setBaseline] = useState("champion");
  const [strategy, setStrategy] = useState("evolutionary");
  const [maxCandidates, setMaxCandidates] = useState(240);
  const [batchSize, setBatchSize] = useState(24);
  const [seed, setSeed] = useState(1);
  const [priority, setPriority] = useState("balanced");
  const [focus, setFocus] = useState<Record<string, boolean>>({});
  const [showGates, setShowGates] = useState(false);
  const [gates, setGates] = useState<{ maxCost: number; maxLatency: number; minQuality: number; maxViolation: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      apiGet<ApplicationSummary[]>("/api/v1/applications"),
      apiGet<DomainInfo[]>("/api/v1/domains"),
      apiGet<ExperimentSummary[]>("/api/v1/experiments"),
    ])
      .then(([a, d, e]) => {
        setApps(a);
        setDomains(d);
        setExperiments(e);
        setAppChoice(a.length > 0 ? a[0].id : NEW_APP);
        setExperimentId(`exp-${e.length + 1}`);
      })
      .catch((err) => setLoadError((err as Error).message));
  }, []);

  const selectedApp = apps.find((a) => a.id === appChoice) ?? null;
  const domainName = selectedApp ? selectedApp.domain : newAppDomain;
  const domain = useMemo(() => domains.find((d) => d.name === domainName) ?? null, [domains, domainName]);

  // Reset the per-domain controls whenever the domain changes: every section on, the domain's own gates.
  useEffect(() => {
    if (!domain) return;
    setFocus(Object.fromEntries(Object.keys(domain.search_dimensions).map((g) => [g, true])));
    setGates({
      maxCost: domain.promotion_gates.max_cost_increase * 100,
      maxLatency: domain.promotion_gates.max_latency_increase * 100,
      minQuality: domain.promotion_gates.min_quality,
      maxViolation: domain.safety_constraints.max_policy_violation_rate,
    });
  }, [domain]);

  const chosenSections = Object.entries(focus).filter(([, on]) => on).map(([g]) => g);
  const allSections = domain ? Object.keys(domain.search_dimensions) : [];
  const nothingChosen = domain !== null && chosenSections.length === 0;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!domain || !gates) return;
    setBusy(true);
    setSubmitError(null);
    try {
      const systemId = selectedApp ? selectedApp.id : newAppId.trim();
      if (!selectedApp) {
        await apiPost("/api/v1/applications", { id: systemId, name: systemId, domain: newAppDomain });
      }
      const body: Record<string, unknown> = {
        experiment_id: experimentId.trim(),
        system_id: systemId,
        domain: domainName,
        strategy,
        seed,
        batch_size: batchSize,
        max_batches: 100,
        baseline,
        budget: { max_candidates: maxCandidates, max_requests: 10_000_000, max_cost_usd: 1000, max_duration_minutes: 60 },
        promotion_gates: {
          min_quality: gates.minQuality,
          max_cost_increase: gates.maxCost / 100,
          max_latency_increase: gates.maxLatency / 100,
        },
        safety_constraints: { max_policy_violation_rate: gates.maxViolation },
      };
      if (chosenSections.length < allSections.length) body.search_dimensions = chosenSections;
      const weights = PRIORITIES.find((p) => p.value === priority)?.weights;
      if (weights) body.objective_weights = weights;
      await apiPost("/api/v1/experiments", body);
      await apiPost(`/api/v1/experiments/${body.experiment_id}/start`, {});
      router.push(`/experiments/${body.experiment_id}`);
    } catch (err) {
      setSubmitError((err as Error).message);
      setBusy(false);
    }
  }

  const label = "stat-label mb-1 block";
  const input = "w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm";

  return (
    <form onSubmit={submit} className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">New experiment</h1>
        <p className="text-sm text-slate-500">
          Choose what to evolve, which knobs the search may turn, and what &ldquo;better&rdquo; means. The
          experiment runs in the background; nothing here changes production &mdash; that stays an explicit,
          admin-only step on the Promotion page.
        </p>
      </div>

      {loadError && <div className="card text-sm text-bad">{loadError}</div>}

      <section className="card grid gap-4 md:grid-cols-2">
        <div>
          <label className={label}>Application</label>
          <select className={input} value={appChoice} onChange={(e) => setAppChoice(e.target.value)}>
            {apps.map((a) => (
              <option key={a.id} value={a.id}>
                {a.id} ({a.domain})
              </option>
            ))}
            <option value={NEW_APP}>+ new application…</option>
          </select>
          {!selectedApp && (
            <div className="mt-2 grid grid-cols-2 gap-2">
              <input className={input} value={newAppId} onChange={(e) => setNewAppId(e.target.value)} placeholder="application id" />
              <select className={input} value={newAppDomain} onChange={(e) => setNewAppDomain(e.target.value)}>
                {domains.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        <div>
          <label className={label}>Experiment id</label>
          <input className={input} value={experimentId} onChange={(e) => setExperimentId(e.target.value)} required />
        </div>
        <div>
          <label className={label}>Evolve from</label>
          <select className={input} value={baseline} onChange={(e) => setBaseline(e.target.value)}>
            <option value="champion">
              {selectedApp?.production ? `Champion (v${selectedApp.production.version}, in production)` : "Champion (nothing promoted yet → original)"}
            </option>
            <option value="original">Original baseline (v1)</option>
          </select>
        </div>
        <div>
          <label className={label}>Search strategy</label>
          <select className={input} value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label} — {s.hint}
              </option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-3 gap-3 md:col-span-2">
          <div>
            <label className={label}>Candidates</label>
            <input type="number" min={1} className={input} value={maxCandidates} onChange={(e) => setMaxCandidates(Number(e.target.value))} />
          </div>
          <div>
            <label className={label}>Batch size</label>
            <input type="number" min={1} max={500} className={input} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
          </div>
          <div>
            <label className={label}>Seed</label>
            <input type="number" className={input} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </div>
        </div>
      </section>

      <section className="card">
        <h2 className="text-base font-semibold">What may the search change?</h2>
        <p className="mb-3 text-xs text-slate-500">
          Fields in unchecked sections keep the baseline&rsquo;s values. Narrow this to ask &ldquo;what is the best{" "}
          <em>prompt</em> setup for this system as it stands?&rdquo;
        </p>
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          {domain &&
            Object.entries(domain.search_dimensions).map(([group, fields]) => (
              <label key={group} className="flex items-center gap-2 text-sm" title={fields.map((f) => f.path).join(", ")}>
                <input
                  type="checkbox"
                  checked={!!focus[group]}
                  onChange={(e) => setFocus((f) => ({ ...f, [group]: e.target.checked }))}
                />
                <span className="font-medium">{group}</span>
                <span className="text-xs text-slate-500">{fields.length} fields</span>
              </label>
            ))}
        </div>
        {nothingChosen && <p className="mt-2 text-sm text-bad">Choose at least one section.</p>}
      </section>

      <section className="card">
        <h2 className="text-base font-semibold">What does &ldquo;better&rdquo; mean?</h2>
        <p className="mb-3 text-xs text-slate-500">
          Cost and latency are scored relative to the baseline. Safety limits and the promotion gates still apply
          to whatever wins, whatever the emphasis.
        </p>
        <div className="grid gap-2 md:grid-cols-4">
          {PRIORITIES.map((p) => (
            <label
              key={p.value}
              className={`cursor-pointer rounded-md border p-3 text-sm ${
                priority === p.value ? "border-accent bg-accent/5" : "border-slate-200"
              }`}
            >
              <input type="radio" name="priority" className="mr-2" checked={priority === p.value} onChange={() => setPriority(p.value)} />
              <span className="font-medium">{p.label}</span>
              <span className="mt-1 block text-xs text-slate-500">{p.hint}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="card">
        <button type="button" className="text-sm font-semibold text-accent hover:underline" onClick={() => setShowGates((v) => !v)}>
          {showGates ? "▾" : "▸"} Promotion gates &amp; safety limits
        </button>
        {showGates && gates && (
          <div className="mt-3 grid gap-4 md:grid-cols-4">
            <div>
              <label className={label}>Max cost increase (%)</label>
              <input type="number" className={input} value={gates.maxCost} onChange={(e) => setGates((g) => g && { ...g, maxCost: Number(e.target.value) })} />
            </div>
            <div>
              <label className={label}>Max latency increase (%)</label>
              <input type="number" className={input} value={gates.maxLatency} onChange={(e) => setGates((g) => g && { ...g, maxLatency: Number(e.target.value) })} />
            </div>
            <div>
              <label className={label}>Min quality</label>
              <input type="number" step="0.01" className={input} value={gates.minQuality} onChange={(e) => setGates((g) => g && { ...g, minQuality: Number(e.target.value) })} />
            </div>
            <div>
              <label className={label}>Max policy-violation rate</label>
              <input type="number" step="0.01" className={input} value={gates.maxViolation} onChange={(e) => setGates((g) => g && { ...g, maxViolation: Number(e.target.value) })} />
            </div>
          </div>
        )}
      </section>

      {submitError && <div className="card text-sm text-bad">{submitError}</div>}
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={busy || !domain || nothingChosen || !experimentId.trim()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-40"
        >
          {busy ? "Starting…" : "Create and start"}
        </button>
        <span className="text-xs text-slate-500">
          {selectedApp?.production && baseline === "champion"
            ? `Candidates must beat the champion (v${selectedApp.production.version}) on the holdout to be promotable.`
            : "Candidates must beat the baseline on the holdout to be promotable."}
        </span>
      </div>
    </form>
  );
}
