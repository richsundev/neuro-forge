"""Aggregate per-challenge EvaluationResults into per-genome metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from neuroforge.domains.base import EvaluationResult


@dataclass
class AggregateMetrics:
    genome_hash: str
    n_evaluations: int
    metrics: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    failure_rate: float = 0.0
    per_category_quality: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        d = dict(self.metrics)
        d["cost_usd"] = self.cost_usd
        d["latency_ms"] = self.latency_ms
        d["failure_rate"] = self.failure_rate
        return d


def aggregate(
    results: list[EvaluationResult], categories: list[str] | None = None
) -> AggregateMetrics:
    if not results:
        raise ValueError("cannot aggregate zero evaluation results")
    genome_hash = results[0].genome_hash
    metric_keys_set: set[str] = set()
    for r in results:
        metric_keys_set.update(r.metrics.keys())
    # Sorted, not just deduplicated: `set` iteration order depends on PYTHONHASHSEED, which would
    # otherwise make dict/JSON key order vary run-to-run — see test_reproducibility.py.
    metric_keys = sorted(metric_keys_set)

    metrics = {
        key: sum(r.metrics.get(key, 0.0) for r in results) / len(results) for key in metric_keys
    }
    cost = sum(r.cost_usd for r in results) / len(results)
    latency = sum(r.latency_ms for r in results) / len(results)
    failure_rate = sum(1 for r in results if r.failed) / len(results)

    per_category: dict[str, float] = {}
    if categories:
        for cat in sorted(set(categories)):
            idxs = [i for i, c in enumerate(categories) if c == cat]
            if idxs:
                per_category[cat] = sum(results[i].metrics.get("quality", 0.0) for i in idxs) / len(
                    idxs
                )

    return AggregateMetrics(
        genome_hash=genome_hash,
        n_evaluations=len(results),
        metrics={k: round(v, 4) for k, v in metrics.items()},
        cost_usd=round(cost, 6),
        latency_ms=round(latency, 2),
        failure_rate=round(failure_rate, 4),
        per_category_quality={k: round(v, 4) for k, v in per_category.items()},
    )
