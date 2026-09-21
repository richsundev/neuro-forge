#!/usr/bin/env python3
"""neuroforge-experiment-worker (section 31): pulls experiment IDs off the Redis queue and runs
them with ExperimentEngine, independent of the API process. Start alongside the API and Redis:

    NEUROFORGE_DATABASE_URL=... NEUROFORGE_REDIS_URL=... python scripts/worker.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s worker: %(message)s")
logger = logging.getLogger("neuroforge.worker")


def _state_dir() -> Path:
    return Path(os.environ.get("NEUROFORGE_STATE_DIR", "./neuroforge_state"))


def run_one(experiment_id: str) -> None:
    from neuroforge.db.repository import (
        dataset_for_experiment,
        get_experiment,
        list_genomes_for_system,
        save_experiment,
        save_genome,
    )
    from neuroforge.db.session import session_scope
    from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
    from neuroforge.genomes.schema import SystemGenome

    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            logger.error("unknown experiment %s, skipping", experiment_id)
            return
        config = ExperimentConfig.model_validate(record.config)
        baseline = next(
            (
                SystemGenome.model_validate(r.data)
                for r in list_genomes_for_system(session, record.application_id)
                if r.version == 1
            ),
            None,
        )
        dataset = dataset_for_experiment(session, record.application_id, config)
        application_id = record.application_id
        save_experiment(session, application_id, config, status="running")

    if baseline is None or dataset is None:
        logger.error("experiment %s missing baseline/dataset, skipping", experiment_id)
        return

    logger.info("running experiment %s (strategy=%s)", experiment_id, config.search_strategy)
    engine = ExperimentEngine(config, baseline, dataset, _state_dir() / experiment_id)
    result = engine.run()

    with session_scope() as session:
        save_genome(session, SystemGenome.model_validate(result.best_genome))
        save_experiment(session, application_id, config, result=result, status=result.status)
    logger.info("experiment %s finished: %s (%s)", experiment_id, result.status, result.stop_reason)


def main() -> None:
    from neuroforge.db.session import init_db
    from neuroforge.experiments.queue import dequeue_experiment
    from neuroforge.observability import configure_tracing

    configure_tracing("neuroforge-worker")
    init_db()
    logger.info("neuroforge-experiment-worker started, polling Redis queue")
    while True:
        try:
            experiment_id = dequeue_experiment(timeout=5)
            if experiment_id is None:
                continue
            run_one(experiment_id)
        except Exception:
            logger.exception("worker loop iteration failed, continuing")


if __name__ == "__main__":
    main()
