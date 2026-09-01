# ADR-0004: Evolutionary search as the default strategy

## Context

NeuroForge needed one default search strategy for the ForgeSupport demo and for `experiment
create` when a user doesn't specify `--strategy`. The genome search space mixes numeric,
categorical, and list-valued fields, has ~12 dimensions in the ForgeSupport case, and the mock
provider makes evaluation cheap (thousands/sec) relative to optimizer bookkeeping.

## Decision

Default to `EvolutionarySearchStrategy` (`optimization/evolutionary.py`): population + tournament
selection + crossover + mutation + elitism. See docs/evolutionary-search.md for mechanics.

## Alternatives considered

- **Bayesian optimization as default.** Better sample efficiency in principle, but the GP
  surrogate (ADR-0005) degrades with mixed numeric/categorical/list-valued encodings and doesn't
  parallelize naturally — evolutionary search's population-per-generation structure maps directly
  onto "propose a batch, evaluate them all, tell the strategy," which is also what makes
  checkpointing clean (ADR-0008).
- **Random search as default.** Simpler, but strictly worse sample efficiency with no
  countervailing benefit given evaluation is cheap here.

## Tradeoffs

Evolutionary search needs a reasonably large population (`batch_size`) to have enough diversity
for crossover/mutation to be useful — with a very small budget, random search or a Bayesian
approach may reach a comparable result in fewer evaluations. `scripts/reproduce.py`'s
strategy-comparison section runs all four search-based strategies on identical budgets specifically
so this tradeoff is visible with real numbers rather than asserted.

## Consequences

Elitism guarantees monotonically non-decreasing best-fitness across generations, which is what
makes the fitness/generation charts in the dashboard readable, and what makes
`detect_plateau`-based convergence detection meaningful (see docs/optimization.md).
