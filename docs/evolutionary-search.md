# Evolutionary Search

`EvolutionarySearchStrategy` (`src/neuroforge/optimization/evolutionary.py`) is the default search
strategy and the one behind the Evolution Graph signature feature.

## Mechanics

- **Population**: `population_size` points sampled from the `SearchSpace` (defaults to the
  experiment's `batch_size`, so one `ask()`/`tell()` round is exactly one generation).
- **Fitness**: whatever `ObjectiveSpec.score()` returns for each candidate (see docs/optimization.md).
- **Elitism**: the top `elite_fraction` of each generation survives unchanged into the next —
  this is why fitness only ever plateaus, never regresses, across generations for this strategy.
- **Selection**: tournament selection (`k=3` by default) — pick 3 random candidates, keep the best.
- **Crossover**: `crossover_rate` chance to combine two parents field-by-field
  (`SearchSpace.crossover`, uniform per-field).
- **Mutation**: `mutation_rate` chance to perturb one field of the child
  (`SearchSpace.perturb` — numeric fields move by a random delta scaled by the field's span;
  categorical fields jump to a different value).
- Every `Generation` (population + fitness list) is retained in `strategy.generations`, which is
  what the checkpoint persists (see docs/reproducibility.md) and what generation-level status
  reporting (`neuroforge experiment status`) reads from.

## Why genome-typed candidates, not opaque vectors

`SearchSpace` parameter keys are genome field paths (`"retrieval.top_k"`, `"prompt.strategy"`,
...). `ExperimentEngine` turns every point a strategy proposes into a real `SystemGenome` via
`.derive()`, with a `MutationRecord` per changed field carrying a plain-English reason
("Proposed by evolutionary search to optimize the configured multi-objective fitness function.").
This means the evolutionary lineage graph you see in the dashboard is not a visualization of the
optimizer's internal state — it *is* the optimizer's actual output, genome by genome.

## Convergence

See docs/optimization.md's convergence section — for evolutionary search specifically, a
generation's "best" is the max fitness of its (post-elitism) population, and because of elitism
that sequence is non-decreasing, so `detect_plateau` is checking "have we stopped finding anything
better in the last 3 generations."

## What this is not

This is not a full genetic-programming system (no tree-structured genomes, no dynamic-length
representations) — every genome has the same fixed schema (`SystemGenome`), and "evolution" here
means search over that fixed-shape configuration space, which is the right level of complexity
for evolving an LLM application's configuration rather than its source code.
