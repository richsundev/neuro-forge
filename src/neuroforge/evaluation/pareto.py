"""Pareto-frontier computation (section 6). There is rarely one universally best candidate —
this module makes the tradeoff explicit and explains *why* a candidate is or isn't dominated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class ParetoPoint:
    candidate_id: str
    metrics: dict[str, float]


@dataclass
class ParetoResult:
    candidate_id: str
    is_pareto_optimal: bool
    dominated_by: list[str]
    explanation: str


def _dominates(
    a: dict[str, float], b: dict[str, float], directions: dict[str, Direction]
) -> bool:
    """True if `a` dominates `b`: at least as good on every objective, strictly better on one."""
    at_least_as_good_everywhere = True
    strictly_better_somewhere = False
    for key, direction in directions.items():
        av, bv = a[key], b[key]
        if direction == "maximize":
            better, worse_or_equal = av > bv, av >= bv
        else:
            better, worse_or_equal = av < bv, av <= bv
        if not worse_or_equal:
            at_least_as_good_everywhere = False
            break
        if better:
            strictly_better_somewhere = True
    return at_least_as_good_everywhere and strictly_better_somewhere


def pareto_frontier(
    points: list[ParetoPoint], directions: dict[str, Direction]
) -> list[ParetoResult]:
    results: list[ParetoResult] = []
    for p in points:
        dominators = [
            other.candidate_id
            for other in points
            if other.candidate_id != p.candidate_id
            and _dominates(other.metrics, p.metrics, directions)
        ]
        is_optimal = len(dominators) == 0
        if is_optimal:
            explanation = (
                "Pareto-optimal: no other candidate is at least as good on every objective "
                "and strictly better on at least one."
            )
        else:
            explanation = (
                f"Dominated by {', '.join(dominators)}: "
                "each of them matches or beats this candidate on every objective and beats it "
                "on at least one, so this candidate is strictly worse in every useful sense."
            )
        results.append(
            ParetoResult(
                candidate_id=p.candidate_id,
                is_pareto_optimal=is_optimal,
                dominated_by=dominators,
                explanation=explanation,
            )
        )
    return results
