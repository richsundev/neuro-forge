# ADR-0003: Multi-objective optimization, not single-metric quality maximization

## Context

Optimizing an LLM application purely for output quality reliably finds configurations that are
also slower, more expensive, or less safe — a search that only sees "quality went up" has no
signal telling it that's a bad trade.

## Decision

`ObjectiveSpec` (`evaluation/objectives.py`) combines several weighted, directional objectives
(maximize quality/task_success/policy_compliance/safety_score; minimize latency/cost/failure_rate)
into the single scalar `fitness` that search strategies optimize, with weights fully configurable
per experiment. Independently, `evaluation/pareto.py` computes true Pareto dominance across
candidates so users aren't limited to whatever tradeoff the configured weights happened to encode
— see docs/optimization.md.

## Alternatives considered

- **Quality-only optimization with cost/latency as post-hoc filters.** Rejected: filtering after
  the fact means the search itself never gets steered away from expensive-but-marginally-better
  regions of the space — wasted budget.
- **Constrained optimization (quality subject to cost ≤ X).** Considered for a future iteration;
  more precise than weighted-sum for hard limits, but requires a feasibility-first search strategy
  design change across all five algorithms. Weighted-sum plus a separate hard-constraint gate
  (ADR-0012) gets most of the benefit with far less machinery.

## Tradeoffs

Weighted-sum scalarization can miss Pareto-optimal points that don't correspond to any convex
combination of weights (a known limitation of the approach) — mitigated by exposing the actual
Pareto frontier separately rather than only ever showing the fitness-ranked leaderboard.

## Consequences

Every search strategy shares one interface (`ask`/`tell` on a scalar fitness) regardless of how
many objectives are configured; the dashboard's Pareto chart and `neuroforge candidate compare`
give users the multi-objective view the scalar fitness necessarily throws away.
