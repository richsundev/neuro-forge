"""Run a stored experiment: the one implementation behind `POST /experiments/{id}/run`,
`neuroforge experiment run` and the queue worker.

Those three used to each re-implement "load config, baseline and dataset, run, save" and had
drifted apart (the API and worker ignored the experiment's `dataset_id`; none recorded a failure).
Not imported by `neuroforge.experiments.__init__`: it depends on the database layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from neuroforge.db.repository import (
    dataset_for_experiment,
    get_experiment,
    list_genomes_for_system,
    save_experiment,
    save_genome,
)
from neuroforge.db.session import session_scope
from neuroforge.experiments.engine import (
    DatasetTooSmall,
    ExperimentAlreadyRunning,
    ExperimentConfig,
    ExperimentEngine,
    ExperimentResult,
)
from neuroforge.genomes.schema import SystemGenome

logger = logging.getLogger("neuroforge.experiments.runner")


class RunnerError(ValueError):
    """The experiment can't be run (unknown id, missing baseline or dataset)."""


class UnknownExperiment(RunnerError):
    pass


def run_recorded_experiment(experiment_id: str, state_root: Path) -> ExperimentResult:
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise UnknownExperiment(f"unknown experiment '{experiment_id}'")
        config = ExperimentConfig.model_validate(record.config)
        application_id = record.application_id
        baseline = next(
            (
                SystemGenome.model_validate(r.data)
                for r in list_genomes_for_system(session, application_id)
                if r.version == 1
            ),
            None,
        )
        dataset = dataset_for_experiment(session, application_id, config)

    if baseline is None or dataset is None:
        raise RunnerError("experiment is missing its baseline genome or dataset")

    engine = ExperimentEngine(config, baseline, dataset, state_root / experiment_id)
    with session_scope() as session:
        save_experiment(session, application_id, config, status="running")
    try:
        result = engine.run()
    except ExperimentAlreadyRunning:
        raise
    except DatasetTooSmall as exc:
        with session_scope() as session:
            save_experiment(session, application_id, config, status="failed")
        raise RunnerError(str(exc)) from exc
    except Exception:
        # Without this the row stayed "running" (or "created") forever after a crash.
        logger.exception("experiment %s failed", experiment_id)
        with session_scope() as session:
            save_experiment(session, application_id, config, status="failed")
        raise

    with session_scope() as session:
        save_genome(session, SystemGenome.model_validate(result.best_genome))
        save_experiment(session, application_id, config, result=result, status=result.status)
    return result
