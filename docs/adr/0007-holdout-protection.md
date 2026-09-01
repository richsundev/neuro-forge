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

A `HoldoutGuard` per experiment means holdout access is tied to one guard instance's lifetime —
if a caller constructs a second `HoldoutGuard` from the same `DatasetVersion`, the "only once"
enforcement resets. This is a real gap: the guarantee holds within one `ExperimentEngine` run, not
across arbitrary re-construction. Acceptable for this reference implementation's usage pattern
(the CLI/API always create one guard per experiment via `ExperimentEngine.__init__`); a hardened
version would persist "holdout already evaluated for genome X" in the database instead.

## Consequences

`test_datasets.py::test_holdout_can_only_be_evaluated_once_per_genome` and
`test_holdout_guard_blocks_search_and_validation_access` make this a tested property, not just an
assertion in a docstring.
