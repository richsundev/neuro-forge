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
   bootstrap comparison and a confidence-bound safety check, and produces a recommendation. The
   **holdout** split is not touched here — the promotion review consumes it (docs/promotion.md).

## Constraint-aware search

"Score it" is not just the weighted fitness. Each candidate is checked against the same gates
promotion will apply — the safety limits, the quality floor, and the cost/latency caps relative to
the baseline (evaluated once up front on the train split) — and its fitness is divided by
`1 + constraint_penalty × violation`. `best` is tracked feasible-first: a candidate inside all limits
beats one outside them whatever its raw fitness. `search_safety_margin` (0.03) and
`search_quality_margin` (0.02) aim the search that far inside the safety limits and the quality
floor, so the winner has headroom against split-to-split noise. A plateau is not convergence until
a feasible candidate exists. Each `CandidateEvaluated` event records `feasible` and
`constraint_violation`; `ExperimentResult.selected_feasible` records whether the winner made it.
Rationale and measurements: ADR-0014.

## What the search may change, and what it optimizes

`search_dimensions` restricts the search space to whole sections (`prompt`) or single fields
(`retrieval.top_k`); anything not listed stays at the baseline's value. `objective_weights` pins
an objective's share of the total (`{"cost_usd": 0.4}`); the others split the rest in their default
proportions. Cost and latency are scored relative to the baseline (the baseline scores 0.5, half the
cost approaches 0.75), so a weight means the same thing whatever the absolute prices are. The CLI
takes `--focus`, `--weight` and the gate options; `GET /domains` lists the dimensions a domain has.
`POST /experiments/{id}/start` runs one in the background (202) and `GET /experiments/{id}/checkpoint`
reports live progress; production deployments use `/enqueue` instead. Rationale: ADR-0016.

## Which baseline

`ExperimentConfig.baseline_hash` records the genome the search evolves from, resolved once at
creation from the request's `baseline` (`"champion"` — the default — `"original"`, or a specific
genome). Candidates are derived from it, so lineage edges run from the champion and generation
numbers continue from its generation; a run and all its resumes start from the same genome. See
ADR-0015 and docs/promotion.md.

## Which dataset

A first run uses the dataset's latest version; a *resumed* run keeps the version recorded in its
checkpoint (`ExperimentCheckpoint.dataset_version`), even if the dataset has been evolved since —
batches evaluated on different versions aren't comparable. The engine refuses (`DatasetChanged`) to
resume on a different version.

`ExperimentConfig.dataset_id` records the dataset the experiment was created with (falling back to
`<application_id>-dataset` for older experiments), and `ExperimentResult` records the dataset
version it actually ran against, so a later promotion review measures the holdout of the same
version.

## Budget enforcement

`ExperimentBudget` (`experiments/budget.py`) caps `max_candidates`, `max_requests`, `max_cost_usd`,
`max_duration_minutes`. `BudgetTracker` checks all four before every batch; whichever is hit first
becomes the `stop_reason`. Duration is tracked cumulatively across resumes (`prior_elapsed_minutes`
persisted in the checkpoint), so a budget genuinely caps wall-clock spend across a resumed run, not
just the current process's uptime.

## Running safely

- **One run at a time.** `ExperimentEngine.run()` takes an exclusive advisory lock
  (`<id>.lock`); a second run of the same experiment raises `ExperimentAlreadyRunning` (HTTP 409)
  instead of interleaving checkpoint/event writes. The OS releases the lock if the holder dies, so
  a crashed worker never blocks its own resume.
- **Crash-safe files.** Checkpoints are written to a temp file and renamed into place; the event
  log skips a torn final line (from a kill mid-append) instead of refusing to load.
- **Failures are recorded.** `experiments/runner.py:run_recorded_experiment` — the single
  implementation behind `POST /experiments/{id}/run`, `neuroforge experiment run` and the worker —
  marks the experiment `running`, then `completed` or `failed`.
- **Budgets don't overshoot.** The last batch is truncated to the candidates the budget has left.
- **Early data check.** A dataset with no train data or fewer than 2 validation challenges is
  refused up front rather than after the whole search.

## Cancellation

`neuroforge experiment cancel <id>` (or `POST /api/v1/experiments/{id}/cancel`) touches a
`<id>.cancel` file in the experiment's state directory. The engine checks for it at the top of
every loop iteration — cooperative cancellation, so an in-flight batch finishes before stopping
rather than being killed mid-evaluation. Only `running` or `queued` experiments can be cancelled
(anything else is a 409): a flag left behind for an idle experiment used to cancel its *next* run.
A cancelled run finishes with status `cancelled` (a partial result, not `completed`) and can be
resumed.

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
docs/design-decisions.md ADR-0010 for why this is a plain Redis queue rather than Celery.
