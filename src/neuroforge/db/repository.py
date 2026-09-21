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
    HoldoutEvaluationRecord,
    PromotionApprovalRecord,
    PromotionRecord,
)
from neuroforge.experiments.engine import ExperimentConfig, ExperimentResult
from neuroforge.genomes.schema import PromotionStatus, SystemGenome
from neuroforge.promotion.canary import CanaryResult
from neuroforge.promotion.gates import PromotionDecision
from neuroforge.promotion.holdout import HoldoutEvidence


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


def genome_from_record(record: GenomeRecord) -> SystemGenome:
    """The stored `data` snapshot is taken when the genome is first saved; `status` is what the
    promotion pipeline updates afterwards, so it lives in the column and is applied here. (Status
    is metadata, not part of the content hash.)"""
    return SystemGenome.model_validate(record.data).model_copy(
        update={"status": PromotionStatus(record.status)}
    )


def get_genome(session: Session, genome_hash: str) -> SystemGenome | None:
    record = session.get(GenomeRecord, genome_hash)
    return genome_from_record(record) if record else None


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
    # Always persist the config that was passed: it used to be written only when the row was
    # created, so `experiment resume --extra-candidates` (which raises the budget in memory and
    # saves) silently kept the old budget and the resumed run stopped immediately.
    record.config = config.model_dump(mode="json")
    if result is not None:
        record.result = result.model_dump(mode="json")
    return record


def next_candidate_version(session: Session, application_id: str) -> int:
    """First version number not already used or reserved by this application's genomes and
    experiments, so candidate version labels are unique across experiments."""
    start = 2
    for genome in list_genomes_for_system(session, application_id):
        start = max(start, genome.version + 1)
    stmt = select(ExperimentRecord).where(ExperimentRecord.application_id == application_id)
    for record in session.scalars(stmt):
        config = record.config or {}
        first = config.get("first_candidate_version") or 2
        reserved = (config.get("budget") or {}).get("max_candidates", 0)
        start = max(start, first + reserved)
    return start


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
        evidence=decision.evidence,
    )
    session.add(record)
    _advance_status(session, genome_hash, decision.next_status.value)
    return record


def _advance_status(
    session: Session, genome_hash: str, new_status: str, only_from: set[str] | None = None
) -> None:
    """Keep `system_genomes.status` (what the Evolution Graph shows) in step with the pipeline.
    PROMOTED is terminal here: a repeat promotion request or canary never downgrades a genome that
    a human has already taken live."""
    record = session.get(GenomeRecord, genome_hash)
    if record is None or record.status == PromotionStatus.PROMOTED.value:
        return
    if only_from is not None and record.status not in only_from:
        return
    record.status = new_status


def get_holdout_evaluation(
    session: Session, genome_hash: str, dataset_id: str, dataset_version: int
) -> HoldoutEvaluationRecord | None:
    return session.scalar(
        select(HoldoutEvaluationRecord).where(
            HoldoutEvaluationRecord.genome_hash == genome_hash,
            HoldoutEvaluationRecord.dataset_id == dataset_id,
            HoldoutEvaluationRecord.dataset_version == dataset_version,
        )
    )


def save_holdout_evaluation(
    session: Session, genome_hash: str, experiment_id: str | None, evidence: HoldoutEvidence
) -> HoldoutEvaluationRecord:
    record = HoldoutEvaluationRecord(
        genome_hash=genome_hash,
        experiment_id=experiment_id,
        dataset_id=evidence.dataset_id,
        dataset_version=evidence.dataset_version,
        evidence=evidence.model_dump(mode="json"),
    )
    session.add(record)
    return record


def dataset_for_experiment(
    session: Session, application_id: str, config: ExperimentConfig
) -> DatasetVersion | None:
    """The dataset an experiment runs against: the one it was created with, falling back to the
    `<application_id>-dataset` convention for experiments created before `dataset_id` was recorded."""
    return latest_dataset_version(session, config.dataset_id or f"{application_id}-dataset")


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
    # A canary only means something for a genome that already cleared the promotion gates.
    _advance_status(
        session,
        candidate_hash,
        PromotionStatus.ROLLED_BACK.value if result.rollback_triggered else PromotionStatus.CANARY.value,
        only_from={
            PromotionStatus.APPROVED.value,
            PromotionStatus.CANARY.value,
            PromotionStatus.ROLLED_BACK.value,
        },
    )
    return record


def list_promotion_decisions(session: Session) -> list[PromotionRecord]:
    stmt = select(PromotionRecord).order_by(PromotionRecord.created_at.desc(), PromotionRecord.id.desc())
    return list(session.scalars(stmt))


def latest_promotion_decision(session: Session, genome_hash: str) -> PromotionRecord | None:
    stmt = (
        select(PromotionRecord)
        .where(PromotionRecord.genome_hash == genome_hash)
        .order_by(PromotionRecord.created_at.desc(), PromotionRecord.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def list_canary_runs(session: Session) -> list[CanaryRecord]:
    stmt = select(CanaryRecord).order_by(CanaryRecord.created_at.desc(), CanaryRecord.id.desc())
    return list(session.scalars(stmt))


def latest_canary_run(session: Session, candidate_hash: str) -> CanaryRecord | None:
    stmt = (
        select(CanaryRecord)
        .where(CanaryRecord.candidate_hash == candidate_hash)
        .order_by(CanaryRecord.created_at.desc(), CanaryRecord.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def set_genome_status(session: Session, genome_hash: str, status: str) -> GenomeRecord | None:
    record = session.get(GenomeRecord, genome_hash)
    if record is not None:
        record.status = status
    return record


def save_promotion_approval(
    session: Session, genome_hash: str, experiment_id: str | None, approved_by: str
) -> PromotionApprovalRecord:
    record = PromotionApprovalRecord(genome_hash=genome_hash, experiment_id=experiment_id, approved_by=approved_by)
    session.add(record)
    return record


def list_promotion_approvals(session: Session) -> list[PromotionApprovalRecord]:
    stmt = select(PromotionApprovalRecord).order_by(PromotionApprovalRecord.created_at.desc(), PromotionApprovalRecord.id.desc())
    return list(session.scalars(stmt))
