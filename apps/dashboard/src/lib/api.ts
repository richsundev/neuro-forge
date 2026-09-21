const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function apiKey(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem("neuroforge_api_key") ?? "";
}

export function setApiKey(key: string): void {
  if (typeof window !== "undefined") {
    window.localStorage.setItem("neuroforge_api_key", key);
  }
}

/** FastAPI errors are `{"detail": "..."}` (HTTPException) or `{"detail": [{loc, msg}, ...]}`
 * (validation). Show the human part instead of a raw JSON blob. */
async function failure(method: string, path: string, res: Response): Promise<Error> {
  const raw = await res.text();
  let detail = raw;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (typeof parsed.detail === "string") {
      detail = parsed.detail;
    } else if (Array.isArray(parsed.detail)) {
      detail = parsed.detail
        .map((d: { loc?: unknown[]; msg?: string }) =>
          `${(d.loc ?? []).filter((p) => p !== "body").join(".")}: ${d.msg ?? "invalid"}`)
        .join("; ");
    }
  } catch {
    // not JSON — keep the raw text
  }
  return new Error(`${method} ${path} failed (${res.status}): ${detail}`);
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "X-API-Key": apiKey() },
    cache: "no-store",
  });
  if (!res.ok) throw await failure("GET", path, res);
  return res.json();
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "X-API-Key": apiKey(), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure("POST", path, res);
  return res.json();
}

export interface ApplicationSummary {
  id: string;
  name: string;
  domain: string;
  created_at: string;
}

export interface ExperimentSummary {
  experiment_id: string;
  application_id: string;
  dataset_id: string;
  domain: string;
  strategy: string;
  status: string;
  created_at: string;
  best_genome_hash: string | null;
  recommendation: string | null;
  comparison_summary: string | null;
}

export interface ExperimentDetail {
  experiment_id: string;
  status: string;
  config: Record<string, unknown>;
  result: ExperimentResult | null;
}

export interface ComparisonResult {
  mean_diff: number;
  relative_diff: number;
  ci_low: number;
  ci_high: number;
  effect_size: number;
  confidence: number;
  conclusion: string;
  summary: string;
}

export interface ExperimentResult {
  experiment_id: string;
  status: string;
  stop_reason: string;
  generations_completed: number;
  candidates_evaluated: number;
  baseline_genome: Record<string, unknown>;
  best_genome: Record<string, unknown>;
  baseline_metrics: Record<string, number>;
  best_metrics: Record<string, number>;
  comparison: ComparisonResult;
  budget_utilization: Record<string, number>;
  fitness_history: number[];
  recommendation: string;
}

export interface LineageNode {
  hash: string;
  version: number;
  generation: number;
  status: string;
  parent_hash: string | null;
  mutations: string[];
}

export interface LineageGraph {
  nodes: LineageNode[];
  edges: { from: string; to: string }[];
}

export interface CandidateEvaluation {
  genome_hash: string;
  version: number;
  generation: number;
  fitness: number;
  metrics: Record<string, number>;
  is_pareto_optimal: boolean;
}

export interface DatasetSummary {
  dataset_id: string;
  version: number;
  source: string;
  n_challenges: number;
  difficulty_score: number;
}

export interface PromotionDecision {
  approved: boolean;
  next_status: string;
  reasons: string[];
  evidence: string[];
}

export interface PromotionRecord {
  genome_hash: string;
  experiment_id: string | null;
  approved: boolean;
  next_status: string;
  reasons: string[];
  evidence: string[];
  created_at: string;
}

export interface PromotionApproval {
  genome_hash: string;
  experiment_id: string | null;
  approved_by: string;
  created_at: string;
}

export interface CanaryRecord {
  baseline_hash: string;
  candidate_hash: string;
  traffic_split: number;
  rollback_triggered: boolean;
  result: {
    baseline_metrics: Record<string, number>;
    candidate_metrics: Record<string, number>;
    reasons: string[];
    n_baseline_requests: number;
    n_candidate_requests: number;
  };
  created_at: string;
}
