"""NeuroForge CLI (section 45). A thin wrapper over the same engine/domain/db code the API uses —
no logic lives here that isn't also reachable programmatically.
"""

from __future__ import annotations

import json
import os
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from neuroforge.cli.paths import state_dir
from neuroforge.datasets.dataset import DatasetVersion
from neuroforge.datasets.evolution import evolve_from_failures, evolve_if_saturated, seed_dataset
from neuroforge.db.models import ApplicationRecord
from neuroforge.db.repository import (
    candidate_version_clash,
    genome_from_record,
    get_experiment,
    get_genome,
    latest_dataset_version,
    list_experiments,
    list_genomes_for_system,
    next_candidate_version,
    save_canary_result,
    save_dataset_version,
    save_experiment,
    save_genome,
    upsert_application,
)
from neuroforge.db.session import init_db, session_scope
from neuroforge.domains import get_domain
from neuroforge.evaluation.aggregate import aggregate
from neuroforge.evaluation.statistics import compare
from neuroforge.experiments.checkpoint import ExperimentCheckpoint
from neuroforge.experiments.configure import (
    ConfigError,
    build_objectives,
    build_search_space,
    parse_weight_options,
)
from neuroforge.experiments.engine import (
    ExperimentAlreadyRunning,
    ExperimentConfig,
    run_lock_is_held,
)
from neuroforge.experiments.runner import RunnerError, UnknownExperiment, run_recorded_experiment
from neuroforge.genomes.schema import PromotionStatus, SystemGenome
from neuroforge.optimization import STRATEGY_REGISTRY
from neuroforge.promotion.canary import fresh_traffic, simulate_canary
from neuroforge.promotion.gates import PromotionGateConfig
from neuroforge.promotion.lifecycle import (
    ProductionError,
    current_champion,
    production_lock,
    promote_to_production,
    resolve_baseline,
    rollback_production,
    system_of_genome,
)
from neuroforge.promotion.review import ReviewError, review_promotion
from neuroforge.promotion.safety import SafetyConstraints
from neuroforge.providers.registry import get_provider
from neuroforge.util import stable_int

app = typer.Typer(help="NeuroForge: autonomous LLM system evolution & experimentation engine.")
app_cmd = typer.Typer(help="Manage applications.")
genome_cmd = typer.Typer(help="Inspect system genomes.")
experiment_cmd = typer.Typer(help="Create, run, and inspect experiments.")
dataset_cmd = typer.Typer(help="Manage datasets and challenge sets.")
candidate_cmd = typer.Typer(help="Compare candidate genomes.")
promotion_cmd = typer.Typer(help="Request and inspect promotion decisions.")
canary_cmd = typer.Typer(help="Run canary simulations.")

app.add_typer(app_cmd, name="app")
app.add_typer(genome_cmd, name="genome")
app.add_typer(experiment_cmd, name="experiment")
app.add_typer(dataset_cmd, name="dataset")
app.add_typer(candidate_cmd, name="candidate")
app.add_typer(promotion_cmd, name="promotion")
app.add_typer(canary_cmd, name="canary")

console = Console()


@app.callback()
def _init() -> None:
    init_db()


def _resolve_genome(session, ident: str) -> SystemGenome:
    if "@v" in ident:
        system_id, version_str = ident.split("@v", 1)
        if not version_str.isdigit():
            raise typer.BadParameter(f"malformed genome identifier '{ident}' (expected <system>@v<number>)")
        records = list_genomes_for_system(session, system_id)
        for r in records:
            if r.version == int(version_str):
                return genome_from_record(r)
        raise typer.BadParameter(f"no genome found for {ident}")
    genome = get_genome(session, ident)
    if genome is None:
        raise typer.BadParameter(f"no genome found with hash {ident}")
    return genome


def _domain_or_error(name: str):
    try:
        return get_domain(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


# --- app ---------------------------------------------------------------


@app_cmd.command("create")
def app_create(app_id: str, name: str, domain: str = "forge-support", description: str = "") -> None:
    with session_scope() as session:
        upsert_application(session, app_id, name, domain, description)
    console.print(f"[green]created application[/] {app_id} (domain={domain})")


@app_cmd.command("list")
def app_list() -> None:
    from sqlalchemy import select

    from neuroforge.db.models import ApplicationRecord

    with session_scope() as session:
        rows = list(session.scalars(select(ApplicationRecord)))
        champions = {r.id: current_champion(session, r.id) for r in rows}
    table = Table("id", "name", "domain", "in production", "created_at")
    for r in rows:
        champion = champions[r.id]
        production = f"v{champion.version} ({champion.hash})" if champion else "original (v1)"
        table.add_row(r.id, r.name, r.domain_name, production, str(r.created_at))
    console.print(table)


# --- genome --------------------------------------------------------------


@genome_cmd.command("show")
def genome_show(ident: str) -> None:
    with session_scope() as session:
        genome = _resolve_genome(session, ident)
    console.print_json(json.dumps(genome.model_dump(mode="json")))


# --- dataset -------------------------------------------------------------


@dataset_cmd.command("seed")
def dataset_seed(
    dataset_id: str,
    domain: str = "forge-support",
    n: int = typer.Option(300, min=20, max=5000, help="challenges to generate (validation/holdout are ~15% each)"),
    seed: int = 1,
) -> None:
    d = _domain_or_error(domain)
    dataset = seed_dataset(d, dataset_id, n=n, seed=seed)
    with session_scope() as session:
        existing = latest_dataset_version(session, dataset_id)
        if existing is not None and existing.hash() != dataset.hash():
            raise typer.BadParameter(
                f"dataset '{dataset_id}' already exists (v{existing.version}); use `dataset evolve` "
                "to add a version, or pick a new id"
            )
        save_dataset_version(session, dataset)
    console.print_json(json.dumps(dataset.summary()))


@dataset_cmd.command("list")
def dataset_list() -> None:
    from sqlalchemy import select

    from neuroforge.db.models import DatasetVersionRecord

    with session_scope() as session:
        rows = list(session.scalars(select(DatasetVersionRecord).order_by(DatasetVersionRecord.dataset_id)))
    table = Table("dataset_id", "version", "source", "n_challenges", "difficulty")
    for r in rows:
        table.add_row(r.dataset_id, str(r.version), r.source, str(r.n_challenges), f"{r.difficulty_score:.3f}")
    console.print(table)


@dataset_cmd.command("evolve")
def dataset_evolve(
    dataset_id: str,
    mean_score: float | None = typer.Option(None, help="mean search-split score of the current best candidates (saturation-driven evolution)"),
    failing_category: Annotated[
        list[str] | None,
        typer.Option(help="add challenges concentrated on this category (repeatable; failure-driven evolution)"),
    ] = None,
    domain: str | None = typer.Option(None, help="defaults to the dataset's own domain; if given it must match"),
    n_new: int = 30,
    seed: int = 2,
) -> None:
    with session_scope() as session:
        current = latest_dataset_version(session, dataset_id)
        if current is None:
            raise typer.BadParameter(f"no dataset version found for '{dataset_id}' — run `dataset seed` first")
        if domain is not None and domain != current.domain_name:
            raise typer.BadParameter(f"dataset '{dataset_id}' belongs to domain '{current.domain_name}', not '{domain}'")
        d = _domain_or_error(current.domain_name)
        if failing_category:
            try:
                evolved = evolve_from_failures(d, current, failing_category, n_new, seed)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
        elif mean_score is None:
            raise typer.BadParameter("give --mean-score (saturation-driven) or --failing-category (failure-driven)")
        else:
            evolved = evolve_if_saturated(d, current, mean_score, n_new_challenges=n_new, seed=seed)
        if evolved is None:
            console.print(f"[yellow]not saturated[/] (mean_score={mean_score:.3f}) — benchmark unchanged")
            raise typer.Exit(0)
        save_dataset_version(session, evolved)
    console.print(f"[green]benchmark evolved[/] -> version {evolved.version}, difficulty {evolved.difficulty_score:.3f}")
    console.print_json(json.dumps(evolved.summary()))


# --- experiment ------------------------------------------------------------


@experiment_cmd.command("create")
def experiment_create(
    experiment_id: str,
    system_id: str = "support-agent",
    domain: str = "forge-support",
    dataset_id: str | None = None,
    strategy: str = "evolutionary",
    seed: int = 42,
    batch_size: Annotated[int, typer.Option(min=1, max=500)] = 16,
    max_batches: Annotated[int, typer.Option(min=1, max=1000)] = 20,
    max_candidates: Annotated[int, typer.Option(min=1)] = 200,
    baseline: str = typer.Option(
        "champion",
        help='what to evolve from: "champion" (current production genome, else the original), '
        '"original" (v1), or a genome hash / <system>@v<N>',
    ),
    focus: Annotated[
        list[str] | None,
        typer.Option(help="search only these genome sections/fields, e.g. prompt or retrieval.top_k (repeatable)"),
    ] = None,
    weight: Annotated[
        list[str] | None,
        typer.Option(help="pin an objective's share of the total, metric=value, e.g. cost_usd=0.4 (repeatable)"),
    ] = None,
    min_quality: Annotated[float | None, typer.Option(help="promotion gate: absolute quality floor")] = None,
    max_cost_increase: Annotated[float | None, typer.Option(help="promotion gate: fractional cost cap vs. baseline, 0.1 = +10%")] = None,
    max_latency_increase: Annotated[float | None, typer.Option(help="promotion gate: fractional latency cap vs. baseline")] = None,
    max_policy_violation: Annotated[float | None, typer.Option(help="safety limit on the policy-violation rate")] = None,
) -> None:
    dataset_id = dataset_id or f"{system_id}-dataset"
    d = _domain_or_error(domain)
    if strategy not in STRATEGY_REGISTRY:
        raise typer.BadParameter(f"unknown strategy '{strategy}'; known: {sorted(STRATEGY_REGISTRY)}")
    with session_scope() as session:
        if get_experiment(session, experiment_id) is not None:
            raise typer.BadParameter(f"experiment '{experiment_id}' already exists")
        existing_app = session.get(ApplicationRecord, system_id)
        if existing_app is not None and existing_app.domain_name != domain:
            raise typer.BadParameter(f"application '{system_id}' uses domain '{existing_app.domain_name}', not '{domain}'")
        upsert_application(session, system_id, system_id, domain)
        if get_genome_by_system_version(session, system_id, 1) is None:
            save_genome(session, SystemGenome(system_id=system_id, version=1))
            session.flush()  # sessions here don't autoflush; the baseline lookup below must see it
        try:
            baseline_record = resolve_baseline(session, system_id, baseline)
        except ProductionError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if baseline_record is None:
            raise typer.BadParameter(f"no baseline genome for '{system_id}'")
        dataset = latest_dataset_version(session, dataset_id)
        if dataset is None:
            dataset = seed_dataset(d, dataset_id, n=300, seed=seed)
            save_dataset_version(session, dataset)
        elif dataset.challenges and dataset.domain_name != domain:
            raise typer.BadParameter(f"dataset '{dataset_id}' belongs to domain '{dataset.domain_name}', not '{domain}'")

        from neuroforge.experiments.budget import ExperimentBudget

        try:
            search_space = build_search_space(domain, focus)
            objectives = build_objectives(parse_weight_options(weight))
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        gate_overrides = {
            k: v
            for k, v in {
                "min_quality": min_quality,
                "max_cost_increase": max_cost_increase,
                "max_latency_increase": max_latency_increase,
            }.items()
            if v is not None
        }
        extra: dict = {}
        try:
            if gate_overrides:
                extra["promotion_gates"] = PromotionGateConfig(**gate_overrides)
            if max_policy_violation is not None:
                extra["safety_constraints"] = SafetyConstraints(max_policy_violation_rate=max_policy_violation)
            config = ExperimentConfig(
                experiment_id=experiment_id,
                domain_name=domain,
                dataset_id=dataset_id,
                baseline_hash=baseline_record.hash,
                first_candidate_version=next_candidate_version(session, system_id),
                search_strategy=strategy,
                search_space=search_space,
                objectives=objectives,
                **extra,
                batch_size=batch_size,
                max_batches=max_batches,
                seed=seed,
                budget=ExperimentBudget(max_candidates=max_candidates, max_requests=1_000_000, max_cost_usd=100, max_duration_minutes=60),
            )
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            raise typer.BadParameter(problems) from exc
        save_experiment(session, system_id, config, status="created")
    console.print(
        f"[green]created experiment[/] {experiment_id} (system={system_id}, dataset={dataset_id}, "
        f"baseline=v{baseline_record.version} {baseline_record.hash})"
    )


def get_genome_by_system_version(session, system_id: str, version: int) -> SystemGenome | None:
    for r in list_genomes_for_system(session, system_id):
        if r.version == version:
            return SystemGenome.model_validate(r.data)
    return None


@experiment_cmd.command("run")
def experiment_run(experiment_id: str) -> None:
    try:
        result = run_recorded_experiment(experiment_id, state_dir())
    except UnknownExperiment as exc:
        raise typer.BadParameter(f"{exc} — run `experiment create` first") from exc
    except (RunnerError, ExperimentAlreadyRunning) as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(f"[bold]{experiment_id}[/] finished: {result.status} ({result.stop_reason})")
    console.print(f"generations={result.generations_completed} candidates={result.candidates_evaluated}")
    console.print(f"best genome: {SystemGenome.model_validate(result.best_genome).hash()}")
    console.print(f"comparison: {result.comparison['summary']}")
    console.print(f"[bold]{result.recommendation}[/]")


@experiment_cmd.command("status")
def experiment_status(experiment_id: str) -> None:
    ckpt_path = state_dir() / experiment_id / f"{experiment_id}.checkpoint.json"
    ckpt = ExperimentCheckpoint.load_or_none(ckpt_path)
    if ckpt is None:
        console.print(f"[yellow]no checkpoint yet for {experiment_id}[/] (has it been run?)")
        raise typer.Exit(1)
    console.print(
        f"status={ckpt.status} stop_reason={ckpt.stop_reason!r} "
        f"generations={ckpt.generations_completed()} candidates={ckpt.candidates_completed()} "
        f"best_fitness={ckpt.best_fitness:.4f}"
    )
    console.print_json(json.dumps(ckpt.budget_state))


@experiment_cmd.command("resume")
def experiment_resume(experiment_id: str, extra_candidates: Annotated[int, typer.Option(min=1)] = 100) -> None:
    if run_lock_is_held(state_dir() / experiment_id, experiment_id):
        raise typer.BadParameter(f"experiment '{experiment_id}' is already running")
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise typer.BadParameter(f"unknown experiment '{experiment_id}'")
        config = ExperimentConfig.model_validate(record.config)

    ckpt_path = state_dir() / experiment_id / f"{experiment_id}.checkpoint.json"
    ckpt = ExperimentCheckpoint.load_or_none(ckpt_path)
    already = ckpt.candidates_completed() if ckpt else 0
    config.budget.max_candidates = already + extra_candidates
    # A run that stopped on max_batches would otherwise stop again at once, however many candidates were added.
    generations = ckpt.generations_completed() if ckpt else 0
    config.max_batches = max(config.max_batches, generations + -(-extra_candidates // config.batch_size))
    if config.first_candidate_version:
        with session_scope() as session:
            clash = candidate_version_clash(
                session, experiment_id, record.application_id, config.first_candidate_version, config.budget.max_candidates
            )
        if clash:
            raise typer.BadParameter(
                f"raising '{experiment_id}' to {config.budget.max_candidates} candidates would reuse version "
                f"numbers already assigned to experiment '{clash}' — create a new experiment instead"
            )
    with session_scope() as session:
        save_experiment(session, record.application_id, config, status="running")
    experiment_run(experiment_id)


@experiment_cmd.command("cancel")
def experiment_cancel(experiment_id: str) -> None:
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise typer.BadParameter(f"unknown experiment '{experiment_id}'")
        if record.status not in ("running", "queued"):
            raise typer.BadParameter(f"experiment '{experiment_id}' is {record.status}, not running or queued")
    flag = state_dir() / experiment_id / f"{experiment_id}.cancel"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    console.print(f"[yellow]cancel requested[/] for {experiment_id} — it will stop at its next checkpoint")


@experiment_cmd.command("list")
def experiment_list() -> None:
    with session_scope() as session:
        rows = list_experiments(session)
    table = Table("experiment_id", "domain", "strategy", "status", "created_at")
    for r in rows:
        table.add_row(r.experiment_id, r.domain_name, r.search_strategy, r.status, str(r.created_at))
    console.print(table)


# --- candidate -------------------------------------------------------------


@candidate_cmd.command("compare")
def candidate_compare(genome_a: str, genome_b: str, dataset_id: str, n_challenges: int = 60, seed: int = 1) -> None:
    with session_scope() as session:
        a = _resolve_genome(session, genome_a)
        b = _resolve_genome(session, genome_b)
        dataset = latest_dataset_version(session, dataset_id)
        if dataset is None:
            raise typer.BadParameter(f"unknown dataset '{dataset_id}'")

    domain = get_domain_for_dataset(dataset)
    provider = get_provider("mock")
    challenges = dataset.split("validation") or dataset.split("train")
    challenges = challenges[:n_challenges]

    a_results = [domain.evaluate(a, c, provider) for c in challenges]
    b_results = [domain.evaluate(b, c, provider) for c in challenges]
    a_agg = aggregate(a_results, [c.category for c in challenges])
    b_agg = aggregate(b_results, [c.category for c in challenges])
    comparison = compare(
        [r.metrics["quality"] for r in a_results], [r.metrics["quality"] for r in b_results], seed=seed
    )

    console.print(f"[bold]{genome_a}[/] -> {json.dumps(a_agg.as_dict())}")
    console.print(f"[bold]{genome_b}[/] -> {json.dumps(b_agg.as_dict())}")
    console.print(f"comparison: {comparison.summary()}")


def get_domain_for_dataset(dataset: DatasetVersion):
    return get_domain(dataset.domain_name)


# --- promotion ---------------------------------------------------------


@promotion_cmd.command("request")
def promotion_request(genome_hash: str, experiment_id: str) -> None:
    """Review a candidate on the dataset's holdout split (measured once, reused on repeat) against
    the gates and safety limits in the experiment's own config."""
    with session_scope() as session:
        try:
            review = review_promotion(session, experiment_id, genome_hash)
        except ReviewError as exc:
            raise typer.BadParameter(str(exc)) from exc
    decision = review.decision
    console.print(f"approved={decision.approved} next_status={decision.next_status.value}")
    for reason in decision.reasons:
        console.print(f"- {reason}")
    for line in decision.evidence:
        console.print(f"  [dim]evidence:[/] {line}")


@promotion_cmd.command("promote")
def promotion_promote(genome_hash: str, experiment_id: str | None = None) -> None:
    """The human-approval gate between a passed canary and PROMOTED — requires both an approved
    promotion decision and a non-rollback canary run already on record (see docs/promotion.md).
    Mirrors POST /api/v1/promotions/finalize, which requires an admin-role API key for the same
    reason: this is the one step meant to need an explicit, attributable human action."""
    approved_by = os.environ.get("USER", "cli")
    try:
        with session_scope() as session:
            system_id = system_of_genome(session, genome_hash)
        with production_lock(system_id), session_scope() as session:
            outcome = promote_to_production(session, genome_hash, experiment_id, approved_by)
            superseded = outcome.superseded.hash if outcome.superseded else None
    except ProductionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"status={PromotionStatus.PROMOTED.value} approved_by={approved_by}")
    if superseded:
        console.print(f"superseded previous champion {superseded} (restore it with `promotion rollback`)")


@promotion_cmd.command("rollback")
def promotion_rollback(system_id: str) -> None:
    """Take the application's current champion out of production and restore the one it replaced
    (or the original baseline, if it was the first). Mirrors POST /api/v1/promotions/rollback."""
    approved_by = os.environ.get("USER", "cli")
    try:
        with production_lock(system_id), session_scope() as session:
            outcome = rollback_production(session, system_id, approved_by)
            rolled_back = outcome.rolled_back.hash
            restored = outcome.restored.hash if outcome.restored else None
    except ProductionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"rolled back {rolled_back} (by {approved_by})")
    console.print(f"production is now {restored or 'the original baseline'}")


# --- canary --------------------------------------------------------------


@canary_cmd.command("run")
def canary_run(
    baseline_ident: str,
    candidate_ident: str,
    dataset_id: str,
    traffic_fraction: float = 0.10,
    n_requests: int = 200,
) -> None:
    with session_scope() as session:
        baseline = _resolve_genome(session, baseline_ident)
        candidate = _resolve_genome(session, candidate_ident)
        dataset = latest_dataset_version(session, dataset_id)
        if dataset is None:
            raise typer.BadParameter(f"unknown dataset '{dataset_id}'")

        domain = get_domain_for_dataset(dataset)
        provider = get_provider("mock")
        traffic = fresh_traffic(domain, dataset, n_requests, stable_int("canary-traffic", candidate.hash()))
        try:
            result = simulate_canary(domain, provider, baseline, candidate, traffic, traffic_fraction)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        save_canary_result(session, candidate.hash(), baseline.hash(), result)

    console.print(f"rollback_triggered={result.rollback_triggered}")
    for reason in result.reasons:
        console.print(f"- {reason}")
    console.print(f"baseline: {json.dumps(result.baseline_metrics.as_dict())}")
    console.print(f"candidate: {json.dumps(result.candidate_metrics.as_dict())}")


if __name__ == "__main__":
    app()
