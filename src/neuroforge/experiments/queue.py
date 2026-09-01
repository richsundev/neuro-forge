"""A minimal Redis-backed work queue for the experiment worker (section 30/34).

This is deliberately the smallest thing that is still a real, testable async worker
architecture — a `BLPOP`-based queue rather than a Celery app — because the project doesn't
need Celery's scheduling/retries machinery to demonstrate the separation between "API request
enqueues work" and "a worker process executes it independently." See docs/design-decisions.md.
"""

from __future__ import annotations

import os

import redis

QUEUE_KEY = "neuroforge:experiments:queue"


def get_redis_url() -> str:
    return os.environ.get("NEUROFORGE_REDIS_URL", "redis://localhost:6379/0")


def get_redis_client(socket_timeout: float | None = None) -> redis.Redis:
    """`socket_timeout` must exceed any BLPOP `timeout` used on this client — the client-side
    socket read timeout races the server-side blocking timeout otherwise, and (harmlessly but
    noisily) raises instead of returning None. See dequeue_experiment."""
    return redis.Redis.from_url(get_redis_url(), decode_responses=True, socket_timeout=socket_timeout)


def enqueue_experiment(experiment_id: str, client: redis.Redis | None = None) -> None:
    (client or get_redis_client()).rpush(QUEUE_KEY, experiment_id)


def dequeue_experiment(timeout: int = 5, client: redis.Redis | None = None) -> str | None:
    client = client or get_redis_client(socket_timeout=timeout + 5)
    try:
        result = client.blpop([QUEUE_KEY], timeout=timeout)
    except redis.exceptions.TimeoutError:
        return None
    if result is None:
        return None
    _key, experiment_id = result
    return str(experiment_id)
