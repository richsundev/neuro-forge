# ADR-0010: A plain Redis queue for the worker, not Celery

## Context

Section 30/31 of the original brief calls for "Celery or equivalent worker architecture" and a
`neuroforge-experiment-worker` service, so that experiment execution can happen asynchronously,
independent of the API request/response cycle.

## Decision

`experiments/queue.py` is a ~30-line wrapper around Redis `RPUSH`/`BLPOP`. `POST /experiments/
{id}/enqueue` pushes an experiment ID; `scripts/worker.py` is an independent process that blocks
on the queue and calls the exact same `ExperimentEngine.run()` the API's synchronous `/run`
endpoint calls.

## Alternatives considered

- **Celery.** The named option in the brief, and the more feature-complete choice (retries with
  backoff, task routing, priorities, distributed tracing, a mature ecosystem) — genuinely the
  right call for a production system with heterogeneous background jobs and complex retry
  semantics.

## Tradeoffs

This is a real, explicit scope cut, not a hidden one: no retry-with-backoff, no dead-letter queue,
no task priorities. `docs/development.md`'s "what's missing for production" section names this
directly. A queue this small doesn't need Celery's scheduling machinery to demonstrate the
architectural point — that experiment execution is decoupled from the API request lifecycle — and
building it from scratch keeps the entire async path (queue.py + worker.py) small enough to read
end to end in a few minutes, which matters more for a demonstration/reference codebase than
production-readiness would.

## Consequences

Swapping this for Celery later is a contained change: `enqueue_experiment`/`dequeue_experiment`
would become a Celery task dispatch/`@app.task`, and `ExperimentEngine.run()` — the thing that
actually matters — doesn't change at all. See `docker-compose.yml`'s `worker` service and
`infrastructure/kubernetes/31-worker.yaml` for how it's deployed today.
