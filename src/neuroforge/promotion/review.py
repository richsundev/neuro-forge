"""Promotion review: the one implementation behind both `POST /api/v1/promotions` and
`neuroforge promotion request`.

A review takes a candidate genome and an experiment, measures the candidate against the experiment
baseline on the dataset's holdout split (once — later requests reuse the stored measurement),
applies the promotion gates and safety constraints *from the experiment's own config* to that
fresh evidence, and records the decision with the evidence that produced it.

Deliberately not imported by `neuroforge.promotion.__init__`: it depends on the database layer,
which itself imports the rest of this package.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from neuroforge.db.repository import (
    get_dataset_version,
    get_experiment,
    get_genome,
    get_holdout_evaluation,
    latest_dataset_version,
    save_holdout_evaluation,
    save_promotion_decision,
)
from neuroforge.domains import get_domain
from neuroforge.experiments.engine import ExperimentConfig
from neuroforge.genomes.schema import SystemGenome
from neuroforge.promotion.gates import PromotionDecision
from neuroforge.promotion.holdout import HoldoutEvidence, decide_from_evidence, evaluate_on_holdout
from neuroforge.providers.registry import get_provider


class ReviewError(ValueError):
    """The review can't be performed (unknown experiment/genome, missing dataset, ...)."""


@dataclass
class PromotionReview:
    decision: PromotionDecision
    evidence: HoldoutEvidence
    holdout_reused: bool


def review_promotion(session: Session, experiment_id: str, genome_hash: str) -> PromotionReview:
    record = get_experiment(session, experiment_id)
    if record is None or record.result is None:
        raise ReviewError(f"experiment '{experiment_id}' has no completed result yet")
    config = ExperimentConfig.model_validate(record.config)
    result = record.result

    dataset_id = result.get("dataset_id") or config.dataset_id or f"{record.application_id}-dataset"
    version = result.get("dataset_version") or 0
    dataset = (
        get_dataset_version(session, dataset_id, version)
        if version
        else latest_dataset_version(session, dataset_id)
    )
    if dataset is None:
        raise ReviewError(f"dataset '{dataset_id}' (v{version or 'latest'}) not found")

    candidate = get_genome(session, genome_hash)
    if candidate is None:
        raise ReviewError(f"unknown genome '{genome_hash}'")
    baseline = SystemGenome.model_validate(result["baseline_genome"])

    stored = get_holdout_evaluation(session, genome_hash, dataset.dataset_id, dataset.version)
    if stored is not None:
        evidence = HoldoutEvidence.model_validate(stored.evidence)
    else:
        try:
            evidence = evaluate_on_holdout(
                get_domain(config.domain_name),
                get_provider(config.provider_name),
                baseline,
                candidate,
                dataset,
                confidence=config.confidence,
                min_relative_improvement=config.min_relative_improvement,
                seed=config.seed,
            )
        except ValueError as exc:
            raise ReviewError(str(exc)) from exc
        try:
            # Savepoint: two simultaneous requests for one candidate both miss the lookup above and
            # both try to insert; the unique constraint makes the loser fail, and it should simply
            # use the winner's measurement (the whole point is that there is only one).
            with session.begin_nested():
                save_holdout_evaluation(session, genome_hash, experiment_id, evidence)
        except IntegrityError:
            winner = get_holdout_evaluation(session, genome_hash, dataset.dataset_id, dataset.version)
            if winner is None:
                raise
            evidence = HoldoutEvidence.model_validate(winner.evidence)
            stored = winner

    decision = decide_from_evidence(evidence, genome_hash, config.promotion_gates, config.safety_constraints)
    save_promotion_decision(session, genome_hash, experiment_id, decision)
    return PromotionReview(decision=decision, evidence=evidence, holdout_reused=stored is not None)
