# Architecture

## The core loop

```
Existing SystemGenome
        |
   generate cases (domain plugin)
        |
   search strategy proposes candidate genomes (random/grid/evolutionary/bayesian/bandit)
        |
   evaluate each candidate against the TRAIN split only
        |
   fitness = weighted multi-objective score
        |
   strategy.tell(fitness) -> next round of candidates
        |
   (repeat until budget exhausted / converged / cancelled)
        |
   pick best candidate, re-evaluate on VALIDATION split
        |
   paired bootstrap comparison vs. baseline -> LIKELY_IMPROVEMENT / REGRESSION / INCONCLUSIVE
        |
   promotion gates + hard safety constraints -> APPROVE / REJECT
        |
   canary simulation -> PROMOTE / ROLLBACK
```

Every step is real, executed code — see `src/neuroforge/experiments/engine.py:ExperimentEngine.run`.

## Why one Python package instead of nine

The brief's suggested layout lists `packages/core`, `packages/genomes`, `packages/mutations`,
`packages/optimization`, `packages/evaluation`, `packages/datasets`, `packages/promotion`,
`packages/providers`, `packages/domains` as separate installable packages. This repo implements
that same set of boundaries as **subpackages of one `neuroforge` distribution**
(`src/neuroforge/genomes`, `src/neuroforge/mutations`, ...) instead of nine separately-versioned
PyPI packages. See ADR-0002 for the reasoning — the module boundaries (and the discipline of only
importing "downward," e.g. `experiments` depends on `optimization` but never the reverse) are the
part that matters; nine `pyproject.toml` files for a single-deployable monorepo would be packaging
overhead without a corresponding benefit.

## Module map

```
src/neuroforge/
  genomes/       SystemGenome schema, content hashing, lineage store
  mutations/     MutationPolicy (safety boundary) + MutationEngine (genome-delta candidates)
  optimization/  SearchSpace + 5 search strategies (random/grid/evolutionary/bayesian/bandit)
  providers/     LLMProvider interface + MockLLMProvider (deterministic) + real adapters
  domains/       ApplicationDomain plugin interface + ForgeSupport/SQLAgent/ResearchAgent
  evaluation/    aggregation, Pareto frontier, statistical comparison, judge ensemble, objectives
  datasets/      DatasetVersion, holdout protection, benchmark evolution
  promotion/     gates, safety constraints (with confidence bounds), holdout verification, canary,
                 plus the DB-backed review.py (promotion decisions) and lifecycle.py (champion,
                 supersede, rollback) — not re-exported from the package, since they sit above db/
  experiments/   ExperimentEngine (the orchestrator), budget, checkpoint/resume, events, queue,
                 plus the DB-backed runner.py (run a stored experiment: API, CLI and worker share it)
  db/            SQLAlchemy models + repository functions (Postgres in prod, SQLite locally)
  cli/           Typer CLI — a thin wrapper over the same engine code the API uses

apps/
  api/           FastAPI app (auth, rate limiting, OpenAPI docs) — same thin-wrapper principle
  dashboard/     Next.js dashboard — Evolution Graph, Pareto/fitness charts, Challenge Evolution

workers via scripts/worker.py: pulls experiment IDs off a Redis queue, runs them independently
of the API process (see docs/design-decisions.md for why a plain BLPOP queue instead of Celery).
```

## Request flow (production-shaped path)

```
dashboard --HTTP--> api --enqueue--> Redis --BLPOP--> worker --writes--> Postgres
                      |                                              ^
                      \-------------------- reads results ------------/
```

The API also exposes a synchronous `/run` endpoint (used by the CLI and by CI) that executes the
experiment in-request — appropriate for small demo budgets, not for production traffic. See
docs/design-decisions.md ADR-0010.

## Data flow inside one experiment

`ExperimentConfig` names a `SearchSpace` whose parameter keys are literal genome field paths
(e.g. `"retrieval.top_k"`). Every point a strategy proposes is applied to the baseline genome via
`SystemGenome.derive()`, which means **every candidate any strategy ever produces is a real,
hashed, lineaged genome** — not an opaque parameter dict — regardless of which search algorithm
generated it. This is what makes the Evolution Graph work uniformly across strategies.

## Persistence

- `system_genomes`, `dataset_versions`, `experiments`, `promotion_decisions`, `canary_runs`,
  `applications`, `api_keys`, `audit_log` — see `src/neuroforge/db/models.py`.
- Nested domain objects (genome config, full experiment results) are stored as JSON columns keyed
  by content hash rather than fully normalized — see ADR-0008.
- Experiment checkpoints and event logs are files under `NEUROFORGE_STATE_DIR` (one JSON
  checkpoint + one JSONL event log per experiment), not database rows — see
  docs/reproducibility.md for why.
