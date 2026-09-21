"""Production lifecycle: which genome is the *champion* (in production) for an application, how a
new one replaces it, how it is rolled back, and which baseline the next experiment starts from.

The source of truth is the append-only approval log (`promotion_approvals`): every promotion and
every rollback is a row. The current champion and the rollback history are *replayed* from it, and
`system_genomes.status` is kept in step as a cache for the UI — so the two can never disagree about
what happened, only be out of date.

Deliberately not imported by `neuroforge.promotion.__init__`: it depends on the database layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from neuroforge.db.models import GenomeRecord
from neuroforge.db.repository import (
    get_experiment,
    get_genome_record,
    latest_canary_run,
    latest_promotion_decision,
    list_genomes_for_system,
    list_production_events,
    save_promotion_approval,
    set_genome_status,
)
from neuroforge.experiments.engine import ExperimentConfig
from neuroforge.genomes.schema import PromotionStatus


class ProductionError(ValueError):
    """The requested production change isn't allowed. `kind` lets each interface pick its own
    status: "not_found" (404), "conflict" (409), "precondition"/"invalid" (400)."""

    kind = "invalid"


class UnknownGenome(ProductionError):
    kind = "not_found"


class AlreadyPromoted(ProductionError):
    kind = "conflict"


class NotPromotable(ProductionError):
    """The approved-decision / passed-canary preconditions aren't met."""

    kind = "precondition"


class StaleBaseline(ProductionError):
    """The candidate was validated against something other than what is in production now."""

    kind = "conflict"


class NothingInProduction(ProductionError):
    kind = "conflict"


class UnknownBaseline(ProductionError):
    kind = "invalid"


@dataclass
class PromotionOutcome:
    promoted: GenomeRecord
    superseded: GenomeRecord | None


@dataclass
class RollbackOutcome:
    rolled_back: GenomeRecord
    restored: GenomeRecord | None


def production_stack(session: Session, system_id: str) -> list[GenomeRecord]:
    """Genomes that have been in production and not rolled back, oldest first; the last is the
    champion. Replayed from the approval log: a promotion pushes, a rollback removes the genome it
    names (which is always the top)."""
    stack: list[GenomeRecord] = []
    for approval, genome in list_production_events(session, system_id):
        stack = [g for g in stack if g.hash != genome.hash]
        if (approval.action or "promote") == "promote":
            stack.append(genome)
    return stack


def current_champion(session: Session, system_id: str) -> GenomeRecord | None:
    stack = production_stack(session, system_id)
    return stack[-1] if stack else None


def original_baseline(session: Session, system_id: str) -> GenomeRecord | None:
    """The application's first genome (v1) — what runs when nothing has been promoted."""
    for record in list_genomes_for_system(session, system_id):
        if record.version == 1:
            return record
    return None


def resolve_baseline(session: Session, system_id: str, spec: str) -> GenomeRecord | None:
    """Pick the genome an experiment starts from. `spec` is "champion" (what's in production, else
    the original), "original" (v1), or a specific genome as a hash or `<system>@v<N>`."""
    if spec == "champion":
        return current_champion(session, system_id) or original_baseline(session, system_id)
    if spec == "original":
        return original_baseline(session, system_id)
    if "@v" in spec:
        name, _, version = spec.partition("@v")
        if name != system_id or not version.isdigit():
            raise UnknownBaseline(f"'{spec}' is not a genome of '{system_id}'")
        for record in list_genomes_for_system(session, system_id):
            if record.version == int(version):
                return record
        raise UnknownBaseline(f"no genome '{spec}'")
    record = get_genome_record(session, spec)
    if record is None or record.system_id != system_id:
        raise UnknownBaseline(f"no genome '{spec}' for '{system_id}'")
    return record


def _validated_against(session: Session, genome: GenomeRecord) -> str | None:
    """The baseline hash this genome's latest promotion decision was measured against."""
    decision = latest_promotion_decision(session, genome.hash)
    if decision is None or not decision.experiment_id:
        return None
    experiment = get_experiment(session, decision.experiment_id)
    if experiment is None:
        return None
    config = ExperimentConfig.model_validate(experiment.config)
    if config.baseline_hash:
        return config.baseline_hash
    original = original_baseline(session, genome.system_id)  # experiments from before baselines were recorded
    return original.hash if original else None


def ensure_validated_against_production(session: Session, genome: GenomeRecord) -> None:
    """A candidate is only better than *what it was compared with*. If production has moved on since
    (another promotion, or a rollback), a promotion that was approved against the old baseline says
    nothing about the current one — promoting it could silently replace a better champion."""
    validated = _validated_against(session, genome)
    if validated is None:
        return
    champion = current_champion(session, genome.system_id)
    expected = champion or original_baseline(session, genome.system_id)
    if expected is not None and validated != expected.hash:
        which = "the current champion" if champion else "the original baseline"
        raise StaleBaseline(
            f"this candidate was validated against genome {validated}, but production is now "
            f"{which} ({expected.hash}); re-run the experiment from the current champion"
        )


def promote_to_production(
    session: Session, genome_hash: str, experiment_id: str | None, approved_by: str
) -> PromotionOutcome:
    """The human-approval gate: verify every precondition, then make the genome the champion.

    Requires an approved promotion decision, a canary that did not roll back, and that the decision
    was measured against what is in production *now*. The previous champion becomes SUPERSEDED."""
    genome = get_genome_record(session, genome_hash)
    if genome is None:
        raise UnknownGenome(f"unknown genome '{genome_hash}'")
    if genome.status == PromotionStatus.PROMOTED.value:
        # Otherwise a second click (or a retry) recorded a second approval for the same action.
        raise AlreadyPromoted("this genome is already promoted")
    decision = latest_promotion_decision(session, genome_hash)
    if decision is None or not decision.approved:
        raise NotPromotable("no approved promotion decision on record for this genome")
    canary = latest_canary_run(session, genome_hash)
    if canary is None:
        raise NotPromotable("no canary run on record for this genome — run a canary before promoting")
    if canary.rollback_triggered:
        raise NotPromotable("the latest canary run triggered a rollback — cannot promote")
    ensure_validated_against_production(session, genome)

    previous = current_champion(session, genome.system_id)
    save_promotion_approval(session, genome_hash, experiment_id, approved_by, action="promote")
    set_genome_status(session, genome_hash, PromotionStatus.PROMOTED.value)
    if previous is not None and previous.hash != genome_hash:
        set_genome_status(session, previous.hash, PromotionStatus.SUPERSEDED.value)
    return PromotionOutcome(promoted=genome, superseded=previous if previous and previous.hash != genome_hash else None)


def rollback_production(session: Session, system_id: str, approved_by: str) -> RollbackOutcome:
    stack = production_stack(session, system_id)
    if not stack:
        raise NothingInProduction(f"'{system_id}' has no promoted genome to roll back")
    current = stack[-1]
    restored = stack[-2] if len(stack) >= 2 else None
    save_promotion_approval(session, current.hash, None, approved_by, action="rollback")
    set_genome_status(session, current.hash, PromotionStatus.ROLLED_BACK.value)
    if restored is not None:
        set_genome_status(session, restored.hash, PromotionStatus.PROMOTED.value)
    return RollbackOutcome(rolled_back=current, restored=restored)

