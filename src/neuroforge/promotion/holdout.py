"""Holdout verification: the promotion pipeline's own, independent check.

Search only ever sees the train split and the experiment's recommendation uses validation, so
neither is a fresh sample by the time someone asks for promotion. The holdout split is: it is
reachable only through `HoldoutGuard.evaluate_holdout` (once per candidate — the caller persists the
result so a repeat request reuses it rather than re-measuring), and the promotion gates are applied
to *these* numbers, at confidence bounds, not to whatever the search happened to observe.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel

from neuroforge.datasets.dataset import DatasetVersion
from neuroforge.datasets.holdout import HoldoutGuard
from neuroforge.domains.base import ApplicationDomain, EvaluationResult
from neuroforge.evaluation.aggregate import AggregateMetrics, aggregate
from neuroforge.evaluation.statistics import ComparisonResult, Conclusion, compare
from neuroforge.genomes.schema import SystemGenome
from neuroforge.promotion.gates import PromotionDecision, PromotionGateConfig, evaluate_promotion
from neuroforge.promotion.safety import SafetyConstraints, SafetyEstimate, estimate_safety
from neuroforge.providers.base import LLMProvider

_AGGREGATE_ONLY_KEYS = ("cost_usd", "latency_ms", "failure_rate")
MIN_HOLDOUT_CHALLENGES = 10


class HoldoutEvidence(BaseModel):
    dataset_id: str
    dataset_version: int
    n_holdout: int
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    comparison: dict[str, Any]
    safety: SafetyEstimate


def per_challenge_metrics(results: list[EvaluationResult]) -> dict[str, list[float]]:
    """Regroup per-challenge results as metric name -> one value per challenge."""
    keys = sorted({k for r in results for k in r.metrics})
    return {k: [r.metrics[k] for r in results if k in r.metrics] for k in keys}


def evaluate_on_holdout(
    domain: ApplicationDomain,
    provider: LLMProvider,
    baseline: SystemGenome,
    candidate: SystemGenome,
    dataset: DatasetVersion,
    *,
    confidence: float = 0.95,
    min_relative_improvement: float = 0.02,
    seed: int = 0,
) -> HoldoutEvidence:
    challenges = HoldoutGuard(dataset.challenges).evaluate_holdout(candidate.hash())
    if len(challenges) < MIN_HOLDOUT_CHALLENGES:
        raise ValueError(
            f"dataset '{dataset.dataset_id}' v{dataset.version} has only {len(challenges)} holdout "
            f"challenges (need >= {MIN_HOLDOUT_CHALLENGES}) — seed a larger dataset before promoting"
        )
    categories = [c.category for c in challenges]
    baseline_results = [domain.evaluate(baseline, c, provider) for c in challenges]
    candidate_results = [domain.evaluate(candidate, c, provider) for c in challenges]
    comparison = compare(
        [r.metrics.get("quality", 0.0) for r in baseline_results],
        [r.metrics.get("quality", 0.0) for r in candidate_results],
        min_relative_improvement=min_relative_improvement,
        confidence=confidence,
        seed=seed,
    )
    comparison_fields = asdict(comparison)
    comparison_fields["conclusion"] = comparison.conclusion.value
    return HoldoutEvidence(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        n_holdout=len(challenges),
        baseline_metrics=aggregate(baseline_results, categories).as_dict(),
        candidate_metrics=aggregate(candidate_results, categories).as_dict(),
        comparison=comparison_fields,
        safety=estimate_safety(
            per_challenge_metrics(candidate_results), confidence=confidence, seed=seed
        ),
    )


def aggregate_from_metrics(genome_hash: str, metrics: dict[str, float], n: int) -> AggregateMetrics:
    """Inverse of `AggregateMetrics.as_dict()`: cost/latency/failure_rate are first-class fields
    on the aggregate, not entries in `metrics`, and the gates read them from there."""
    return AggregateMetrics(
        genome_hash=genome_hash,
        n_evaluations=n,
        metrics={k: v for k, v in metrics.items() if k not in _AGGREGATE_ONLY_KEYS},
        cost_usd=metrics.get("cost_usd", 0.0),
        latency_ms=metrics.get("latency_ms", 0.0),
        failure_rate=metrics.get("failure_rate", 0.0),
    )


def decide_from_evidence(
    evidence: HoldoutEvidence,
    candidate_hash: str,
    gates: PromotionGateConfig,
    safety_constraints: SafetyConstraints,
) -> PromotionDecision:
    fields = dict(evidence.comparison)
    fields["conclusion"] = Conclusion(fields["conclusion"])
    decision = evaluate_promotion(
        aggregate_from_metrics("baseline", evidence.baseline_metrics, evidence.n_holdout),
        aggregate_from_metrics(candidate_hash, evidence.candidate_metrics, evidence.n_holdout),
        ComparisonResult(**fields),
        gates,
        safety_constraints,
        evidence.safety,
    )
    decision.evidence.insert(
        0,
        f"holdout split of {evidence.dataset_id} v{evidence.dataset_version}, "
        f"n={evidence.n_holdout} (untouched by search and by the experiment's own validation)",
    )
    return decision
