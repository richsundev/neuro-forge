# ADR-0013: Candidate generation is fully separate from candidate promotion

## Context

Section 69 of the original brief draws a hard line between "autonomous experimentation" and
"autonomous deployment." NeuroForge can propose and search over an enormous number of candidate
genomes; none of that activity should be able to reach production on its own.

## Decision

Every genome starts at `PromotionStatus.GENERATED` (`genomes/schema.py`). Advancing through
`VALIDATED → BENCHMARKED → HOLDOUT_TESTED → APPROVED → CANARY → PROMOTED` requires an explicit,
separate call at every stage (`neuroforge promotion request`, `neuroforge canary run`) — nothing
in `ExperimentEngine`, the mutation engine, or any search strategy ever advances a genome's status
past `GENERATED`. `promotion/gates.py:evaluate_promotion()` and `promotion/canary.py:
simulate_canary()` are the only code paths that produce an `APPROVED`/`PROMOTED`/`ROLLED_BACK`
outcome, and both require a human or a pipeline to invoke them.

## Alternatives considered

- **Auto-promote the best candidate found at the end of a search.** Rejected outright — this is
  precisely the "autonomous deployment" behavior the brief prohibits, and collapses the
  discovery/validation/safety distinctions the rest of the system (ADR-0007, ADR-0012) exists to
  preserve.

## Tradeoffs

Requiring an explicit call at every stage means a fully-automated CI/CD pipeline that wants to
promote winning candidates without a human in the loop has to compose these calls itself
(`experiment run` → `promotion request` → `canary run`) rather than getting one "auto-pilot"
command — a deliberate friction point, not an oversight. A production pipeline should keep at
least one human-approval gate somewhere in that chain regardless (see docs/development.md).

## Consequences

`ExperimentResult.recommendation` (e.g. `"PROMOTE TO CANARY — statistically significant
improvement on validation set"`) is exactly that — a recommendation, computed from real evidence,
never a side effect that changes system state on its own.
