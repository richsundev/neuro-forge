"""Promotion gates (section 22): a candidate never becomes production just for scoring higher —
it must clear an explicit, configurable set of gates, most importantly a regression guard
(section 24) that rejects a candidate whose gains on one metric come with disproportionate cost
elsewhere, even if the weighted multi-objective score looks better overall.

The metric gates (safety, quality floor, cost, latency) live in `metric_gate_findings`, shared by
two callers on purpose: `evaluate_promotion` applies them to holdout evidence to make the promotion
decision, and the experiment engine applies a margin-tightened copy of the *same* checks while
searching, so the search optimizes toward what promotion will actually accept instead of
discovering at the end that its winner was never promotable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from neuroforge.evaluation.aggregate import AggregateMetrics
from neuroforge.evaluation.statistics import ComparisonResult, Conclusion
from neuroforge.genomes.schema import PromotionStatus
from neuroforge.promotion.safety import (
    SafetyConstraints,
    SafetyEstimate,
    check_safety,
    violation_magnitude,
)

PROMOTION_ORDER: list[PromotionStatus] = [
    PromotionStatus.GENERATED,
    PromotionStatus.VALIDATED,
    PromotionStatus.BENCHMARKED,
    PromotionStatus.HOLDOUT_TESTED,
    PromotionStatus.APPROVED,
    PromotionStatus.CANARY,
    PromotionStatus.PROMOTED,
]


class PromotionGateConfig(BaseModel):
    min_quality: float = Field(default=0.62, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    max_cost_increase: float = Field(default=0.10, description="fractional, e.g. 0.10 = +10%")
    max_latency_increase: float = Field(default=0.15, description="fractional, e.g. 0.15 = +15%")
    require_statistically_significant_improvement: bool = True


class PromotionDecision(BaseModel):
    approved: bool
    next_status: PromotionStatus
    reasons: list[str]
    evidence: list[str] = Field(default_factory=list)


@dataclass
class GateFinding:
    reason: str
    magnitude: float  # how far outside the limit, in the metric's own units (always > 0)


def next_allowed_status(current: PromotionStatus) -> PromotionStatus:
    idx = PROMOTION_ORDER.index(current)
    return PROMOTION_ORDER[min(idx + 1, len(PROMOTION_ORDER) - 1)]


def metric_gate_findings(
    baseline: AggregateMetrics,
    candidate: AggregateMetrics,
    gates: PromotionGateConfig,
    safety_constraints: SafetyConstraints,
    safety_estimate: SafetyEstimate | None = None,
) -> list[GateFinding]:
    """Every metric-based gate the candidate currently fails (empty list = clears them all).
    With a `safety_estimate` the safety verdict uses confidence bounds; without one it falls back
    to a point-estimate check on the aggregate metrics."""
    findings: list[GateFinding] = []

    if safety_estimate is not None:
        safety_ok, safety_violations = safety_estimate.check(safety_constraints)
        safety_magnitude = safety_estimate.violation_magnitude(safety_constraints)
    else:
        safety_ok, safety_violations = check_safety(candidate.as_dict(), safety_constraints)
        safety_magnitude = violation_magnitude(candidate.as_dict(), safety_constraints)
    for v in safety_violations if not safety_ok else []:
        findings.append(GateFinding(f"SAFETY VIOLATION: {v}", safety_magnitude / len(safety_violations)))

    quality = candidate.metrics.get("quality", 0.0)
    if quality < gates.min_quality:
        findings.append(
            GateFinding(
                f"quality {quality:.4f} below minimum {gates.min_quality:.4f}",
                gates.min_quality - quality,
            )
        )

    cost_increase = _fractional_increase(baseline.cost_usd, candidate.cost_usd)
    if cost_increase > gates.max_cost_increase:
        findings.append(
            GateFinding(
                f"cost increased {cost_increase * 100:.1f}% > allowed "
                f"{gates.max_cost_increase * 100:.1f}%",
                cost_increase - gates.max_cost_increase,
            )
        )

    latency_increase = _fractional_increase(baseline.latency_ms, candidate.latency_ms)
    if latency_increase > gates.max_latency_increase:
        findings.append(
            GateFinding(
                f"latency increased {latency_increase * 100:.1f}% > allowed "
                f"{gates.max_latency_increase * 100:.1f}%",
                latency_increase - gates.max_latency_increase,
            )
        )
    return findings


def evaluate_promotion(
    baseline: AggregateMetrics,
    candidate: AggregateMetrics,
    comparison: ComparisonResult,
    gates: PromotionGateConfig,
    safety_constraints: SafetyConstraints,
    safety_estimate: SafetyEstimate | None = None,
) -> PromotionDecision:
    reasons = [f.reason for f in metric_gate_findings(
        baseline, candidate, gates, safety_constraints, safety_estimate
    )]

    quality = candidate.metrics.get("quality", 0.0)
    cost_increase = _fractional_increase(baseline.cost_usd, candidate.cost_usd)
    latency_increase = _fractional_increase(baseline.latency_ms, candidate.latency_ms)

    if (
        gates.require_statistically_significant_improvement
        and comparison.conclusion != Conclusion.LIKELY_IMPROVEMENT
    ):
        reasons.append(f"statistical comparison did not confirm improvement: {comparison.conclusion.value}")

    # Regression guard: a large single-metric gain paired with a disproportionate loss elsewhere.
    if quality - baseline.metrics.get("quality", 0.0) > 0 and (
        cost_increase > 2 * gates.max_cost_increase or latency_increase > 2 * gates.max_latency_increase
    ):
        reasons.append(
            "regression guard: quality gain does not justify the disproportionate cost/latency increase"
        )

    evidence = [f"quality {comparison.summary()}"]
    evidence.append(
        f"cost {cost_increase * 100:+.1f}% (limit +{gates.max_cost_increase * 100:.0f}%), "
        f"latency {latency_increase * 100:+.1f}% (limit +{gates.max_latency_increase * 100:.0f}%)"
    )
    if safety_estimate is not None:
        evidence.extend(safety_estimate.evidence(safety_constraints))

    approved = len(reasons) == 0
    return PromotionDecision(
        approved=approved,
        next_status=PromotionStatus.APPROVED if approved else PromotionStatus.REJECTED,
        reasons=reasons or ["all promotion gates passed"],
        evidence=evidence,
    )


def _fractional_increase(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0 if candidate <= 0 else 1.0
    return (candidate - baseline) / baseline
