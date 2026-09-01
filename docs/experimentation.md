# Experimentation

## ExperimentEngine

`src/neuroforge/experiments/engine.py:ExperimentEngine` is the orchestrator (section 34 of the
original spec). One `run()` call:

1. Loads or creates an `ExperimentCheckpoint` (resumable — see docs/reproducibility.md).
2. Reconstructs the search strategy and replays every historical `tell()` batch into it, so its
   internal state (evolutionary population, GP training data, bandit arm statistics) is exactly
   what it would be had the process never stopped.
3. Loops: check the cancel flag, check the budget, ask the strategy for a batch, turn each
   proposed point into a real genome, evaluate it against the dataset's **train** split only,
   score it, tell the strategy, checkpoint, check convergence.
4. On stop, re-evaluates baseline and best-candidate on the **validation** split, runs a paired
   bootstrap comparison, and produces a recommendation.

## Budget enforcement

`ExperimentBudget` (`experiments/budget.py`) caps `max_candidates`, `max_requests`, `max_cost_usd`,
`max_duration_minutes`. `BudgetTracker` checks all four before every batch; whichever is hit first
becomes the `stop_reason`. Duration is tracked cumulatively across resumes (`prior_elapsed_minutes`
persisted in the checkpoint), so a budget genuinely caps wall-clock spend across a resumed run, not
just the current process's uptime.

## Cancellation

`neuroforge experiment cancel <id>` (or `POST /api/v1/experiments/{id}/cancel`) touches a
`<id>.cancel` file in the experiment's state directory. The engine checks for it at the top of
every loop iteration — cooperative cancellation, so an in-flight batch finishes before stopping
rather than being killed mid-evaluation.

## Reproducibility metadata

Every `ExperimentConfig` records: search strategy, search space, objective weights, mutation
policy, budget, and seed. Every genome records its content hash, parent hash, and mutation records.
Every dataset version records its content hash and lineage. None of this is generated after the
fact for a report — it's the actual configuration the run executed with, serialized as-is.

## Events

`experiments/events.py:EventLog` is an append-only JSONL file per experiment
(`ExperimentCreated`, `CandidateGenerated`, `CandidateEvaluated`, `GenerationCompleted`,
`ExperimentStopped`, plus promotion/canary events emitted elsewhere). The API's
`/experiments/{id}/candidates` endpoint (which powers the dashboard's Pareto chart) reads this log
directly — per-candidate detail isn't duplicated into the database (see ADR-0008).

## Async execution path

`POST /experiments/{id}/enqueue` pushes the experiment ID onto a Redis list;
`scripts/worker.py` is an independent process that `BLPOP`s that list and runs
`ExperimentEngine` exactly as the API's synchronous `/run` endpoint does. Both paths call the
same `ExperimentEngine.run()` — the only difference is who calls it and when. See
docs/design-decisions.md ADR-0009 for why this is a plain Redis queue rather than Celery.
