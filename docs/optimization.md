# Optimization Methodology

## Multi-objective fitness

A single scalar `fitness` drives every search strategy (`SearchStrategy.tell`), computed by
`ObjectiveSpec.score()` (`src/neuroforge/evaluation/objectives.py`) from a weighted combination of
metrics, some maximized (quality, task_success, policy_compliance, safety_score) and some
minimized (latency, cost, failure_rate). Weights are fully configurable per experiment — the
default (`DEFAULT_OBJECTIVES`) intentionally weights `policy_compliance` and `safety_score`
alongside quality, not as an afterthought, because a search that only optimizes quality has no
signal pulling it toward safer configurations (see docs/safety.md).

Reducing everything to one scalar is what lets five very different algorithms (random, grid,
evolutionary, Bayesian, bandit) share one interface. It is *not* the last word on whether a
candidate should ship — see Pareto below, and docs/promotion.md for the separate, harder gate a
candidate must clear before promotion regardless of its fitness score.

## Search strategies

| strategy | file | good for |
|---|---|---|
| random | `optimization/random_search.py` | baseline; cheap, embarrassingly parallel |
| grid | `optimization/grid_search.py` | small bounded numeric spaces, exhaustive coverage |
| evolutionary | `optimization/evolutionary.py` | the default — population/selection/mutation/elitism |
| bayesian | `optimization/bayesian.py` | expensive evaluations, few dozen candidates |
| bandit | `optimization/bandit.py` | explicit explore/exploit control over a bounded arm set |

All five implement one `SearchStrategy` interface (`ask(n) -> points`, `tell(observations)`,
`convergence()`), so `experiment create --strategy <name>` swaps the algorithm without touching
anything else — see `neuroforge candidate compare` / `scripts/reproduce.py`'s strategy-comparison
section for a real, executed comparison on identical budgets.

**A note on the bandit strategy**: this is not a per-parameter grid search. An early
implementation *did* discretize the whole space via a Cartesian-product grid, which exploded
combinatorially on ForgeSupport's ~12-dimensional space (millions of arms, ~44s just to allocate
them). It now samples a bounded pool of `n_arms` distinct points and runs UCB1 over that pool —
see the regression test `test_bandit_arm_count_stays_bounded_in_high_dimensional_spaces`.

## Bayesian optimization, specifically

`BayesianSearchStrategy` is a from-scratch Gaussian Process (RBF kernel, closed-form posterior via
`numpy.linalg.pinv`) with Expected Improvement acquisition, not a wrapped third-party optimizer —
small enough to read in one sitting (`optimization/bayesian.py`), and it means the platform has no
heavyweight optimization dependency for something this size. `SearchSpace.encode()` handles mixed
numeric/categorical parameters (min-max scaling + one-hot) so the same GP works over genome-typed
search spaces without special-casing.

## Pareto frontier

`evaluation/pareto.py` computes true multi-objective dominance (not a weighted-sum leaderboard):
candidate A dominates B iff A is at least as good on every objective and strictly better on one.
The dashboard's Pareto chart (quality vs. cost) marks non-dominated candidates in green — every
`ParetoResult` also carries a plain-English `explanation` of *why* a candidate is or isn't
Pareto-optimal, not just a boolean.

## Convergence / early stopping

`detect_plateau()` (`optimization/strategy.py`) is shared by all five strategies: stop if the best
fitness in the last `patience` rounds hasn't improved by `min_delta` over the best in the rounds
before that. For evolutionary search a "round" is a generation; for the other four it's a batch
(`ask`/`tell` round) — tracked as best-of-round, not best-of-individual-sample, specifically so a
strategy comparison at equal budgets stops all strategies on a comparable basis (see the git
history / tests around `_round_best` for the bug this fixed: per-sample history made random
search's plateau detector trigger noise-sensitively after a single batch).
