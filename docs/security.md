# Security

## Authentication and roles

Every `/api/v1/*` endpoint needs an `X-API-Key`. Keys are 256-bit random tokens stored only as
PBKDF2-SHA256 hashes (`apps/api/neuroforge_api/auth.py`); the raw key is shown once at creation.
Roles are ordered `viewer < operator < admin`:

| role | can |
|---|---|
| viewer | read everything |
| operator | create/run/cancel experiments, seed and evolve datasets, request promotion, run canaries |
| admin | everything above, create API keys, and take a candidate to `PROMOTED` |

## Throttling

- **Per key:** an in-memory token bucket (60 burst, 1/s refill) applies to authenticated requests.
- **Failed attempts:** 10 failed authentications burst, refilling at 1 per 5s, per client address —
  then `429`. A key that verified within the last five minutes is checked *before* this throttle, so
  a legitimate user behind a shared proxy isn't locked out by someone guessing at the same address.
- Both limiters are per-process; a multi-replica deployment needs them in Redis.

## Input validation

Identifiers (`experiment_id`, `system_id`, `dataset_id`, `id`) must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$` because they become file names under the state directory — a
`../` in an experiment id was a path-traversal write. Domains and strategies are checked against
their registries, sizes and fractions are bounded (an unbounded dataset `n` let one request exhaust
memory), and API-key roles are an enum. Invalid input is a `422` at the door, not a `500` later.

## Data at rest

The local SQLite database contains API-key hashes and every experiment. It, `.env` files and
`neuroforge_state/` are git-ignored and excluded from Docker build contexts (`.dockerignore`).
SQLite runs with `PRAGMA foreign_keys=ON` so local behavior matches Postgres.

## Not covered

No TLS termination (put it behind a proxy), no key revocation/rotation endpoints, CORS is open
(`*`) — acceptable only because authentication is a header, not a cookie — and `/metrics` and
`/health` are unauthenticated.
