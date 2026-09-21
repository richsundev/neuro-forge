# ADR-0007: Enforce holdout protection in code, not just in process

## Context

The entire premise of validating an optimization result statistically collapses if the search
loop can see the same data used to make the final promotion decision — a classic overfitting
trap, and one that's easy to reintroduce by accident (a future contributor wiring a new code path
that reaches for "all the challenges" instead of "the train split").

## Decision

`HoldoutGuard` (`datasets/holdout.py`) doesn't just document the train/validation/holdout
convention — it's the only object with access to the raw challenge list, and it exposes
`search_set()`, `validation_set()`, and `evaluate_holdout(genome_hash)` as the only three ways to
get challenges out. `evaluate_holdout` raises `HoldoutViolation` if called a second time for the
same genome hash, specifically to prevent "run holdout eval, don't like the number, tweak
something, run it again" — which would make the holdout set into a second validation set with
extra steps.

## Alternatives considered

- **A documented convention** ("don't pass holdout challenges to the search loop") enforced by
  code review only. Rejected: this is exactly the kind of invariant that survives review today and
  breaks silently in six months once nobody remembers why.

## Tradeoffs

A `HoldoutGuard` is per-instance, so its "only once" enforcement resets if a second guard is built
from the same `DatasetVersion`. That gap was real: the promotion pipeline is where the holdout is
actually consumed, and it constructs a guard per request. ADR-0014 closes it the way this ADR
anticipated — the one holdout measurement per (genome, dataset version) is persisted in
`holdout_evaluations` and reused, so repeat requests never re-measure, across processes.

## Consequences

`test_datasets.py::test_holdout_can_only_be_evaluated_once_per_genome` and
`test_holdout_guard_blocks_search_and_validation_access` make this a tested property, not just an
assertion in a docstring.
