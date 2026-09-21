# Development

## Local setup

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
uv pip install -e ./apps/api
cd apps/dashboard && npm install && cd ../..
```

## Running things

```bash
make test          # pytest, mock mode, no external services
make lint           # ruff + eslint
make typecheck       # mypy + tsc
make reproduce        # full reproducible experiment -> reproduce_output/
make docker-up          # full stack: postgres, redis, api, worker, dashboard
```

Or directly:

```bash
NEUROFORGE_DATABASE_URL=sqlite:///./neuroforge.db uvicorn neuroforge_api.main:app --reload
neuroforge experiment create exp-1 --system-id support-agent --domain forge-support
neuroforge experiment run exp-1
cd apps/dashboard && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Testing philosophy

- Every test uses seeded randomness and the mock provider — no network calls, no flakiness from
  external services, no paid API dependency (see docs/reproducibility.md).
- `tests/test_reproducibility.py` runs actual subprocesses with different `PYTHONHASHSEED` values
  — this is deliberate; it's the only way to catch cross-process determinism bugs (and it caught
  a real one — see docs/reproducibility.md).
- `tests/test_api.py` reloads the FastAPI app module per test (fresh SQLite file, fresh Prometheus
  `CollectorRegistry`) rather than sharing global state between tests.

## What's missing for real production

Being direct about the gap between this reference implementation and a production deployment:

- **Traffic-shifting canary**: `promotion/canary.py` simulates canary traffic against mock/replay
  data. A real deployment needs integration with an actual load balancer or feature-flag service
  to shift real traffic percentages.
- **Dashboard API URL is a build-time setting.** `NEXT_PUBLIC_API_URL` is inlined into the browser
  bundle by `next build`, so it must be passed as a Docker build arg (and must be a URL the *browser*
  can reach, not an in-cluster service name); setting it as a pod env var does nothing.
- **Shared filesystem for experiment state**: the Kubernetes manifests mount a `ReadWriteMany` PVC
  for `NEUROFORGE_STATE_DIR` (checkpoints + event logs), since both `api` and `worker` pods write
  to it. Most cloud block storage (EBS, GCE PD) is `ReadWriteOnce` only — a real deployment needs
  EFS/Filestore/NFS or should move checkpoints/events into the database instead.
- **Celery-grade worker features**: the Redis queue (`experiments/queue.py`) has no retry-with-
  backoff, no dead-letter queue, no task priorities, no distributed tracing across queue hops. See
  ADR-0009 for why a plain queue was chosen anyway, and what a production upgrade path looks like.
- **Human approval gate**: nothing currently blocks `APPROVED -> CANARY -> PROMOTED` on human
  sign-off; a real deployment pipeline should require it, at minimum for `CANARY -> PROMOTED`.
- **Multi-instance rate limiting**: `apps/api/neuroforge_api/auth.py`'s `RateLimiter` is an
  in-memory token bucket — correct for one process, not shared across horizontally-scaled API
  replicas. Production needs Redis-backed rate limiting.
- **Alembic migrations beyond the initial schema**: only one migration exists
  (`c8fc232e2891_initial_schema.py`); schema evolution beyond this point needs new migrations
  generated the normal way (`alembic revision --autogenerate`).
- **Real provider cost/latency calibration**: `providers/compatible.py`'s OpenAI/Anthropic
  adapters are structurally complete but don't populate `capability_signal`/cost the way the mock
  provider does (real providers return usage/cost data differently per vendor) — evaluators that
  lean on `capability_signal` specifically need real-provider-appropriate scoring, not a drop-in
  swap.

## Failure injection

```bash
make failure-worker-crash       # kill the worker mid-experiment, verify the API/checkpoint survive
make failure-evaluator-timeout   # simulate a slow domain.evaluate() call
make failure-model-timeout        # simulate the mock provider raising instead of responding
make failure-invalid-candidate      # feed the mutation policy a disallowed field path
make failure-budget-exceeded          # run with a budget of 1 candidate, verify it stops cleanly
make failure-canary-regression          # canary a deliberately-unsafe genome, verify rollback
```

See the Makefile for each target's implementation — most are `pytest -k` selections against
targeted test cases plus a couple of standalone scripts under `scripts/failures/`.
