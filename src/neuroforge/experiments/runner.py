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
    get_dataset_version,
    get_experiment,
    get_genome,
    list_genomes_for_system,
    save_experiment,
    save_genome,
)
from neuroforge.db.session import session_scope
from neuroforge.experiments.checkpoint import ExperimentCheckpoint
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


def _baseline_for_run(session, application_id: str, config: ExperimentConfig) -> SystemGenome | None:
    """The genome the search starts from: the one recorded on the experiment, or — for experiments
    created before baselines were recorded — the application's v1."""
    if config.baseline_hash:
        return get_genome(session, config.baseline_hash)
    return next(
        (
            SystemGenome.model_validate(r.data)
            for r in list_genomes_for_system(session, application_id)
            if r.version == 1
        ),
        None,
    )


def _dataset_for_run(session, application_id: str, config: ExperimentConfig, experiment_id: str, state_root: Path):
    """A first run uses the dataset's latest version; a *resumed* run must keep the version its
    checkpoint started on, even if the dataset has been evolved since."""
    checkpoint = ExperimentCheckpoint.load_or_none(state_root / experiment_id / f"{experiment_id}.checkpoint.json")
    if checkpoint is not None and checkpoint.dataset_version:
        dataset_id = config.dataset_id or f"{application_id}-dataset"
        return get_dataset_version(session, dataset_id, checkpoint.dataset_version)
    return dataset_for_experiment(session, application_id, config)


def run_recorded_experiment(experiment_id: str, state_root: Path) -> ExperimentResult:
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise UnknownExperiment(f"unknown experiment '{experiment_id}'")
        config = ExperimentConfig.model_validate(record.config)
        application_id = record.application_id
        baseline = _baseline_for_run(session, application_id, config)
        dataset = _dataset_for_run(session, application_id, config, experiment_id, state_root)

    def mark_failed() -> None:
        with session_scope() as session:
            save_experiment(session, application_id, config, status="failed")

    if baseline is None or dataset is None:
        # `/enqueue` and the worker had already marked it queued; without this it stayed queued forever.
        mark_failed()
        raise RunnerError("experiment is missing its baseline genome or dataset")

    try:
        engine = ExperimentEngine(config, baseline, dataset, state_root / experiment_id)
    except Exception:
        logger.exception("experiment %s could not be set up", experiment_id)
        mark_failed()
        raise
    with session_scope() as session:
        save_experiment(session, application_id, config, status="running")
    cancel_flag = state_root / experiment_id / f"{experiment_id}.cancel"
    try:
        result = engine.run()
    except ExperimentAlreadyRunning:
        raise  # someone else is running it; their cancel flag is theirs
    except DatasetTooSmall as exc:
        mark_failed()
        raise RunnerError(str(exc)) from exc
    except Exception:
        # Without this the row stayed "running" (or "created") forever after a crash.
        logger.exception("experiment %s failed", experiment_id)
        mark_failed()
        raise
    finally:
        # The run is over, so a cancel request still on disk (made while it was finishing, or repeated
        # after the first was honoured) can only cancel the experiment's *next* run.
        cancel_flag.unlink(missing_ok=True)

    with session_scope() as session:
        save_genome(session, SystemGenome.model_validate(result.best_genome))
        save_experiment(session, application_id, config, result=result, status=result.status)
    return result
