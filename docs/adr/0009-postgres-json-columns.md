# ADR-0009: PostgreSQL with JSON columns for nested domain objects

## Context

NeuroForge's core domain objects (`SystemGenome`, `ExperimentResult`, `DatasetVersion`) are
already validated, versioned, content-hashed Pydantic models with rich nested structure
(genomes have 7 sub-configs; experiment results carry full baseline/candidate metric dicts).
The database needs to persist and query them.

## Decision

PostgreSQL (SQLite locally/CI — see `db/session.py`), with relational columns for everything the
API actually filters/sorts by (`system_id`, `version`, `status`, `created_at`, foreign keys for
lineage) and a `JSON` column (`data`/`config`/`result`) holding the full serialized Pydantic model.
See `db/models.py`.

## Alternatives considered

- **Fully normalized relational schema** (separate tables for `model_config`, `prompt_config`,
  etc., joined back together per genome). More "properly relational," but doubles as a second
  schema that has to be kept in sync with the Pydantic models by hand, for data that's never
  queried at the sub-field level (nobody runs "find all genomes with `retrieval.top_k > 5`" as a
  SQL query — that's a Python-level filter over `SystemGenome` objects already in memory).
- **A document database (MongoDB) instead of Postgres.** Would fit the JSON-heavy access pattern
  naturally, but loses the relational integrity NeuroForge does need (genome lineage foreign keys,
  experiment→application relationships) and adds a second database technology to operate for
  marginal benefit over Postgres's native `JSON`/`JSONB` support.

## Tradeoffs

Postgres's `JSON` column type doesn't get indexed field-level query support the way `JSONB` +
a GIN index would — acceptable because nothing currently queries into the JSON blobs from SQL;
everything is fetched by primary key/hash and deserialized back into a Pydantic model in Python.
Would need to move to `JSONB` + expression indexes if a future feature needs to filter genomes by
a nested field at the database level.

## Consequences

Why Postgres specifically (vs. staying SQLite-only): SQLite's single-writer-at-a-time model is
fine for local dev and CI (see docs/performance.md — ~2,000 writes/sec, single-threaded) but
doesn't hold up under the API + worker + (eventually) multiple worker replicas writing
concurrently, which the docker-compose/Kubernetes topology assumes.
