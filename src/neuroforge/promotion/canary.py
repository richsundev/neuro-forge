"""Canary simulation (section 23): route deterministic mock traffic between baseline and
candidate, compare the two live slices, and roll back automatically if the candidate violates
safety or regresses badly — all without touching a real production system."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.domains.base import ApplicationDomain, Challenge
from neuroforge.evaluation.aggregate import AggregateMetrics, aggregate
from neuroforge.genomes.schema import SystemGenome
from neuroforge.promotion.safety import SafetyConstraints, check_safety
from neuroforge.providers.base import LLMProvider
from neuroforge.util import stable_unit_interval


@dataclass
class CanaryResult:
    baseline_metrics: AggregateMetrics
    candidate_metrics: AggregateMetrics
    traffic_split: float
    n_baseline_requests: int
    n_candidate_requests: int
    rollback_triggered: bool
    reasons: list[str]


def simulate_canary(
    domain: ApplicationDomain,
    provider: LLMProvider,
    baseline: SystemGenome,
    candidate: SystemGenome,
    traffic: list[Challenge],
    candidate_traffic_fraction: float = 0.10,
    safety_constraints: SafetyConstraints | None = None,
    max_quality_regression: float = 0.05,
) -> CanaryResult:
    safety_constraints = safety_constraints or SafetyConstraints()
    baseline_results = []
    candidate_results = []
    baseline_categories = []
    candidate_categories = []

    for challenge in traffic:
        route_to_candidate = (
            stable_unit_interval("canary-route", candidate.hash(), challenge.challenge_id)
            < candidate_traffic_fraction
        )
        if route_to_candidate:
            candidate_results.append(domain.evaluate(candidate, challenge, provider))
            candidate_categories.append(challenge.category)
        else:
            baseline_results.append(domain.evaluate(baseline, challenge, provider))
            baseline_categories.append(challenge.category)

    if not baseline_results or not candidate_results:
        raise ValueError(
            "canary traffic sample too small to produce both arms — increase traffic size"
        )

    baseline_agg = aggregate(baseline_results, baseline_categories)
    candidate_agg = aggregate(candidate_results, candidate_categories)

    reasons: list[str] = []
    safety_ok, violations = check_safety(candidate_agg.as_dict(), safety_constraints)
    if not safety_ok:
        reasons.extend(violations)

    quality_drop = baseline_agg.metrics.get("quality", 0.0) - candidate_agg.metrics.get("quality", 0.0)
    if quality_drop > max_quality_regression:
        reasons.append(
            f"candidate quality regressed by {quality_drop:.3f} "
            f"(> allowed {max_quality_regression:.3f}) during canary"
        )

    if candidate_agg.failure_rate > baseline_agg.failure_rate + 0.15:
        reasons.append(
            f"candidate failure_rate {candidate_agg.failure_rate:.3f} far exceeds baseline "
            f"{baseline_agg.failure_rate:.3f} during canary"
        )

    return CanaryResult(
        baseline_metrics=baseline_agg,
        candidate_metrics=candidate_agg,
        traffic_split=candidate_traffic_fraction,
        n_baseline_requests=len(baseline_results),
        n_candidate_requests=len(candidate_results),
        rollback_triggered=len(reasons) > 0,
        reasons=reasons or ["canary within safety and regression thresholds"],
    )
