"""Thin repository functions the API/CLI use to persist and query domain objects. Kept
deliberately simple (no repository-pattern ceremony) — one function per query the app needs."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from neuroforge.datasets.dataset import DatasetVersion
from neuroforge.db.models import (
    ApplicationRecord,
    CanaryRecord,
    DatasetVersionRecord,
    ExperimentRecord,
    GenomeRecord,
    PromotionRecord,
)
from neuroforge.experiments.engine import ExperimentConfig, ExperimentResult
from neuroforge.genomes.schema import SystemGenome
from neuroforge.promotion.canary import CanaryResult
from neuroforge.promotion.gates import PromotionDecision


def upsert_application(session: Session, app_id: str, name: str, domain_name: str, description: str = "") -> ApplicationRecord:
    record = session.get(ApplicationRecord, app_id)
    if record is None:
        record = ApplicationRecord(id=app_id, name=name, domain_name=domain_name, description=description)
        session.add(record)
    return record


def save_genome(session: Session, genome: SystemGenome) -> GenomeRecord:
    h = genome.hash()
    record = session.get(GenomeRecord, h)
    if record is not None:
        return record
    record = GenomeRecord(
        hash=h,
        system_id=genome.system_id,
        version=genome.version,
        parent_hash=genome.parent_hash,
        generation=genome.generation,
        status=genome.status.value,
        data=genome.model_dump(mode="json"),
    )
    session.add(record)
    return record


def get_genome(session: Session, genome_hash: str) -> SystemGenome | None:
    record = session.get(GenomeRecord, genome_hash)
    return SystemGenome.model_validate(record.data) if record else None


def list_genomes_for_system(session: Session, system_id: str) -> list[GenomeRecord]:
    stmt = select(GenomeRecord).where(GenomeRecord.system_id == system_id).order_by(GenomeRecord.version)
    return list(session.scalars(stmt))


def save_dataset_version(session: Session, dataset: DatasetVersion) -> DatasetVersionRecord:
    h = dataset.hash()
    existing = session.scalar(select(DatasetVersionRecord).where(DatasetVersionRecord.dataset_hash == h))
    if existing is not None:
        return existing
    record = DatasetVersionRecord(
        dataset_id=dataset.dataset_id,
        version=dataset.version,
        parent_version=dataset.parent_version,
        source=dataset.source,
        dataset_hash=h,
        difficulty_score=dataset.difficulty_score,
        n_challenges=len(dataset.challenges),
        data=dataset.model_dump(mode="json"),
    )
    session.add(record)
    return record


def get_dataset_version(session: Session, dataset_id: str, version: int) -> DatasetVersion | None:
    record = session.scalar(
        select(DatasetVersionRecord).where(
            DatasetVersionRecord.dataset_id == dataset_id, DatasetVersionRecord.version == version
        )
    )
    return DatasetVersion.model_validate(record.data) if record else None


def latest_dataset_version(session: Session, dataset_id: str) -> DatasetVersion | None:
    record = session.scalar(
        select(DatasetVersionRecord)
        .where(DatasetVersionRecord.dataset_id == dataset_id)
        .order_by(DatasetVersionRecord.version.desc())
        .limit(1)
    )
    return DatasetVersion.model_validate(record.data) if record else None


def save_experiment(
    session: Session,
    application_id: str,
    config: ExperimentConfig,
    result: ExperimentResult | None = None,
    status: str = "running",
) -> ExperimentRecord:
    record = session.get(ExperimentRecord, config.experiment_id)
    if record is None:
        record = ExperimentRecord(
            experiment_id=config.experiment_id,
            application_id=application_id,
            domain_name=config.domain_name,
            search_strategy=config.search_strategy,
            status=status,
            config=config.model_dump(mode="json"),
        )
        session.add(record)
    record.status = status
    if result is not None:
        record.result = result.model_dump(mode="json")
    return record


def get_experiment(session: Session, experiment_id: str) -> ExperimentRecord | None:
    return session.get(ExperimentRecord, experiment_id)


def list_experiments(session: Session) -> list[ExperimentRecord]:
    return list(session.scalars(select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())))


def save_promotion_decision(
    session: Session, genome_hash: str, experiment_id: str | None, decision: PromotionDecision
) -> PromotionRecord:
    record = PromotionRecord(
        genome_hash=genome_hash,
        experiment_id=experiment_id,
        approved=decision.approved,
        next_status=decision.next_status.value,
        reasons=decision.reasons,
    )
    session.add(record)
    return record


def save_canary_result(session: Session, candidate_hash: str, baseline_hash: str, result: CanaryResult) -> CanaryRecord:
    record = CanaryRecord(
        baseline_hash=baseline_hash,
        candidate_hash=candidate_hash,
        traffic_split=result.traffic_split,
        rollback_triggered=result.rollback_triggered,
        result={
            "baseline_metrics": result.baseline_metrics.as_dict(),
            "candidate_metrics": result.candidate_metrics.as_dict(),
            "reasons": result.reasons,
            "n_baseline_requests": result.n_baseline_requests,
            "n_candidate_requests": result.n_candidate_requests,
        },
    )
    session.add(record)
    return record
