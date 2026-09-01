"""Promotion gates (section 22): a candidate never becomes production just for scoring higher —
it must clear an explicit, configurable set of gates, most importantly a regression guard
(section 24) that rejects a candidate whose gains on one metric come with disproportionate cost
elsewhere, even if the weighted multi-objective score looks better overall.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from neuroforge.evaluation.aggregate import AggregateMetrics
from neuroforge.evaluation.statistics import ComparisonResult, Conclusion
from neuroforge.genomes.schema import PromotionStatus
from neuroforge.promotion.safety import SafetyConstraints, check_safety

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


def next_allowed_status(current: PromotionStatus) -> PromotionStatus:
    idx = PROMOTION_ORDER.index(current)
    return PROMOTION_ORDER[min(idx + 1, len(PROMOTION_ORDER) - 1)]


def evaluate_promotion(
    baseline: AggregateMetrics,
    candidate: AggregateMetrics,
    comparison: ComparisonResult,
    gates: PromotionGateConfig,
    safety_constraints: SafetyConstraints,
) -> PromotionDecision:
    reasons: list[str] = []

    safety_ok, safety_violations = check_safety(candidate.as_dict(), safety_constraints)
    if not safety_ok:
        reasons.extend(f"SAFETY VIOLATION: {v}" for v in safety_violations)

    quality = candidate.metrics.get("quality", 0.0)
    if quality < gates.min_quality:
        reasons.append(f"quality {quality:.3f} below minimum {gates.min_quality:.3f}")

    cost_increase = _fractional_increase(baseline.cost_usd, candidate.cost_usd)
    if cost_increase > gates.max_cost_increase:
        reasons.append(
            f"cost increased {cost_increase * 100:.1f}% > allowed {gates.max_cost_increase * 100:.1f}%"
        )

    latency_increase = _fractional_increase(baseline.latency_ms, candidate.latency_ms)
    if latency_increase > gates.max_latency_increase:
        reasons.append(
            f"latency increased {latency_increase * 100:.1f}% > allowed "
            f"{gates.max_latency_increase * 100:.1f}%"
        )

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

    approved = len(reasons) == 0
    return PromotionDecision(
        approved=approved,
        next_status=PromotionStatus.APPROVED if approved else PromotionStatus.REJECTED,
        reasons=reasons or ["all promotion gates passed"],
    )


def _fractional_increase(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0 if candidate <= 0 else 1.0
    return (candidate - baseline) / baseline
