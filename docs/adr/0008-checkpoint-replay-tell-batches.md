# ADR-0008: Checkpoint by replaying tell() batches, not by serializing strategy internals

## Context

Five different search strategies have five different kinds of internal state (evolutionary:
population + generation history; Bayesian: GP training data; bandit: per-arm pull/reward
counters; random/grid: minimal). An experiment must be resumable after a crash or manual
cancellation without redoing completed work or reconstructing strategy-specific serialization
code five times.

## Decision

`ExperimentCheckpoint` (`experiments/checkpoint.py`) stores every historical `(points, fitness)`
batch ever told to the strategy, in order. Resuming means: construct a *fresh* strategy instance
with the same config/seed, then call `strategy.tell(batch)` once per historical batch in original
order — which reproduces evolutionary population state, GP training data, and bandit statistics
exactly, because each strategy's `tell()` already knows how to fold one batch of observations into
its internal state. One resume mechanism, zero per-strategy serialization code.

## Alternatives considered

- **Pickle the strategy object.** Rejected: fragile across code changes (a field rename breaks
  old checkpoints), not human-readable/auditable, and doesn't compose with the JSON-based
  event log / API responses the rest of the system uses.
- **Bespoke `to_dict`/`from_dict` per strategy.** Rejected: five times the serialization code to
  write and keep in sync with each strategy's internals as they evolve.

## Tradeoffs

Replay cost grows linearly with the number of completed batches — for a very long-running
experiment (thousands of generations) resuming would mean replaying thousands of `tell()` calls
before the strategy is caught up. Not a real cost at the scale this system runs at (tens of
generations), and `tell()` for every strategy here is O(batch size), not O(all history), so replay
of N batches is O(N × batch_size), still fast.

## Consequences

`test_resume_reproduces_uninterrupted_run` verifies bit-for-bit equivalence between a stopped-and-
resumed run and a straight-through run — see docs/reproducibility.md. `scripts/failures/
worker_crash.py` demonstrates the same property against an actually-killed OS process, not just an
in-process simulation.
