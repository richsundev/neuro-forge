#!/usr/bin/env python3
"""neuroforge-experiment-worker (section 31): pulls experiment IDs off the Redis queue and runs
them with ExperimentEngine, independent of the API process. Start alongside the API and Redis:

    NEUROFORGE_DATABASE_URL=... NEUROFORGE_REDIS_URL=... python scripts/worker.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s worker: %(message)s")
logger = logging.getLogger("neuroforge.worker")


def _state_dir() -> Path:
    return Path(os.environ.get("NEUROFORGE_STATE_DIR", "./neuroforge_state"))


def run_one(experiment_id: str) -> None:
    from neuroforge.experiments.engine import ExperimentAlreadyRunning
    from neuroforge.experiments.runner import RunnerError, run_recorded_experiment

    logger.info("running experiment %s", experiment_id)
    try:
        result = run_recorded_experiment(experiment_id, _state_dir())
    except RunnerError as exc:
        logger.error("cannot run experiment %s: %s", experiment_id, exc)
        return
    except ExperimentAlreadyRunning:
        logger.warning("experiment %s is already running elsewhere, skipping", experiment_id)
        return
    logger.info("experiment %s finished: %s (%s)", experiment_id, result.status, result.stop_reason)


def main() -> None:
    from neuroforge.db.session import init_db
    from neuroforge.experiments.queue import dequeue_experiment
    from neuroforge.observability import configure_tracing

    configure_tracing("neuroforge-worker")
    init_db()
    logger.info("neuroforge-experiment-worker started, polling Redis queue")
    failures = 0
    while True:
        try:
            experiment_id = dequeue_experiment(timeout=5)
            failures = 0
            if experiment_id is None:
                continue
            run_one(experiment_id)
        except Exception:
            # Back off: with Redis down, `dequeue_experiment` raises immediately, and retrying with
            # no delay spun a CPU core and flooded the log.
            failures += 1
            delay = min(30.0, 0.5 * 2**min(failures, 6))
            logger.exception("worker loop iteration failed (%d in a row), retrying in %.1fs", failures, delay)
            time.sleep(delay)


if __name__ == "__main__":
    main()
