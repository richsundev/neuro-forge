"""Regression tests for bugs found in a whole-project review. Each test names the failure it pins;
every one of them failed before its fix."""

from __future__ import annotations

import importlib
import json
import random

import pytest

from neuroforge.datasets.evolution import evolve_if_saturated, seed_dataset
from neuroforge.domains import get_domain
from neuroforge.evaluation.aggregate import aggregate
from neuroforge.evaluation.statistics import Conclusion, compare
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.checkpoint import ExperimentCheckpoint
from neuroforge.experiments.engine import (
    ExperimentAlreadyRunning,
    ExperimentConfig,
    ExperimentEngine,
)
from neuroforge.experiments.events import Event, EventLog
from neuroforge.experiments.spaces import forge_support_search_space, search_space_for_domain
from neuroforge.genomes.schema import SystemGenome
from neuroforge.mutations.engine import MutationEngine
from neuroforge.mutations.policy import MutationPolicy
from neuroforge.optimization import get_strategy
from neuroforge.promotion.gates import PromotionGateConfig, evaluate_promotion, metric_gate_findings
from neuroforge.promotion.safety import SafetyConstraints
from neuroforge.providers.mock import MockLLMProvider


def _config(strategy: str, max_candidates: int, experiment_id: str = "x", **kw) -> ExperimentConfig:
    fields = {
        "experiment_id": experiment_id,
        "domain_name": "forge-support",
        "search_strategy": strategy,
        "search_space": forge_support_search_space(),
        "batch_size": 8,
        "max_batches": 100,
        "seed": 5,
        "budget": ExperimentBudget(
            max_candidates=max_candidates, max_requests=10**7, max_cost_usd=10**6, max_duration_minutes=30
        ),
    }
    return ExperimentConfig(**{**fields, **kw})


# --- search / engine ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["random", "evolutionary", "bayesian", "bandit", "grid"])
def test_resume_reproduces_an_uninterrupted_run_for_every_strategy(
    strategy, baseline_genome, small_dataset, tmp_path
):
    """Resume replayed only `tell()`, leaving ask-side state (RNG stream, grid cursor, bandit
    pulls) at its initial value: a resumed random search re-evaluated the points it had already
    evaluated, and a resumed bandit forgot every reward."""
    resumed_dir, straight_dir = tmp_path / "resumed", tmp_path / "straight"
    ExperimentEngine(_config(strategy, 8), baseline_genome, small_dataset, resumed_dir).run()
    resumed = ExperimentEngine(_config(strategy, 24), baseline_genome, small_dataset, resumed_dir).run()
    straight = ExperimentEngine(_config(strategy, 24), baseline_genome, small_dataset, straight_dir).run()
    assert resumed.fitness_history == straight.fitness_history

    ckpt = ExperimentCheckpoint.load(resumed_dir / "x.checkpoint.json")
    points = [json.dumps(p, sort_keys=True, default=str) for b in ckpt.tell_batches for p in b.points]
    if strategy in ("random", "grid", "bandit"):
        assert len(points) == 24
        assert len(set(points)) > 8  # not the first batch proposed again


def test_candidate_budget_is_never_overshot(baseline_genome, small_dataset, tmp_path):
    result = ExperimentEngine(_config("random", 10), baseline_genome, small_dataset, tmp_path).run()
    assert result.candidates_evaluated == 10
    assert result.budget_utilization["candidates"] == 1.0


def test_grid_strategy_never_materializes_the_grid():
    """ForgeSupport's space has ~22M grid points; `space.grid()` built them all as a list."""
    space = forge_support_search_space()
    assert space.grid_size(4) > 20_000_000
    strategy = get_strategy("grid", space)
    first, second = strategy.ask(3), strategy.ask(3)
    assert len(first) == len(second) == 3
    assert first != second
    assert not strategy.convergence().converged


def test_a_second_run_of_the_same_experiment_is_refused(baseline_genome, small_dataset, tmp_path):
    """Two runs interleave checkpoint/event writes and corrupt both."""
    holder = ExperimentEngine(_config("random", 8), baseline_genome, small_dataset, tmp_path)
    other = ExperimentEngine(_config("random", 8), baseline_genome, small_dataset, tmp_path)
    with holder._exclusive(), pytest.raises(ExperimentAlreadyRunning):
        other.run()
    other.run()  # the lock is released with the holder, so a later (resume) run works


def test_dataset_without_enough_validation_data_fails_before_the_search(baseline_genome, tmp_path):
    domain = get_domain("forge-support")
    dataset = seed_dataset(domain, "tiny", n=20, seed=1)
    starved = dataset.model_copy(
        update={"challenges": [c.model_copy(update={"split": "train"}) for c in dataset.challenges]}
    )
    with pytest.raises(ValueError, match="validation"):
        ExperimentEngine(_config("random", 8), baseline_genome, starved, tmp_path).run()


# --- crash safety ------------------------------------------------------------------------------


def test_event_log_survives_a_torn_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.emit(Event("ExperimentCreated", "e", {}))
    with path.open("a") as handle:
        handle.write('{"event_type": "CandidateEval')  # process killed mid-append
    reopened = EventLog(path)
    assert [e.event_type for e in reopened.all()] == ["ExperimentCreated"]
    reopened.emit(Event("ExperimentStopped", "e", {}))
    assert [e.event_type for e in EventLog(path).all()] == ["ExperimentCreated", "ExperimentStopped"]


def test_checkpoint_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "c.checkpoint.json"
    checkpoint = ExperimentCheckpoint(experiment_id="x", strategy_name="random", seed=1)
    checkpoint.save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["c.checkpoint.json"]


def test_checkpoint_with_no_best_candidate_yet_can_be_reloaded(tmp_path):
    """`best_fitness` starts at -inf, serialized as JSON null, which then failed to load — so an
    experiment cancelled before its first generation could never be resumed or inspected."""
    path = tmp_path / "c.checkpoint.json"
    ExperimentCheckpoint(experiment_id="x", strategy_name="random", seed=1).save(path)
    assert ExperimentCheckpoint.load(path).best_fitness == float("-inf")


def test_experiment_cancelled_before_it_starts_can_be_resumed(baseline_genome, small_dataset, tmp_path):
    (tmp_path / "x.cancel").touch()
    first = ExperimentEngine(_config("random", 16), baseline_genome, small_dataset, tmp_path).run()
    assert first.candidates_evaluated == 0
    second = ExperimentEngine(_config("random", 16), baseline_genome, small_dataset, tmp_path).run()
    assert second.candidates_evaluated == 16


# --- mutations / datasets ----------------------------------------------------------------------


def test_mutation_records_describe_only_changes_that_were_applied():
    base = SystemGenome(system_id="a", version=1)
    long_prompt = base.derive(
        mutations=[],
        overrides={"prompt.system_prompt": "You are helpful.\n- one\n- two\n- three\n- four"},
        new_version=2,
    )
    children = MutationEngine(MutationPolicy(), seed=3).propose(long_prompt, 300, 3)
    for child in children:
        paths = [m.field_path for m in child.mutations]
        assert len(paths) == len(set(paths))
        for m in child.mutations:
            assert getattr(getattr(child, m.field_path.split(".")[0]), m.field_path.split(".")[1]) == (
                tuple(m.new_value) if isinstance(m.new_value, list) else m.new_value
            )


def test_mutating_top_k_near_its_ceiling_does_not_crash():
    base = SystemGenome(system_id="a", version=1).derive(
        mutations=[], overrides={"retrieval.top_k": 49}, new_version=2
    )
    children = MutationEngine(MutationPolicy(), seed=1).propose(base, 100, 3)
    assert children and max(c.retrieval.top_k for c in children) <= 50


def test_evolving_a_dataset_keeps_existing_splits_and_ids_unique():
    """`deterministic_split` was re-run over everything with the evolve seed, moving ~45% of the
    existing challenges between train/validation/holdout; and evolving twice with one seed
    regenerated ids that already existed."""
    domain = get_domain("forge-support")
    v1 = seed_dataset(domain, "d", n=60, seed=1)
    v2 = evolve_if_saturated(domain, v1, 0.95, 30, seed=2)
    assert v2 is not None
    assert {c.challenge_id: c.split for c in v1.challenges} == {
        c.challenge_id: c.split for c in v2.challenges if c.challenge_id in {x.challenge_id for x in v1.challenges}
    }
    v3 = evolve_if_saturated(domain, v2, 0.95, 30, seed=2)
    assert v3 is not None
    ids = [c.challenge_id for c in v3.challenges]
    assert len(ids) == len(set(ids)) == 60 + 30 + 30


# --- domains / gates ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain_name", ["research-agent", "sql-agent"])
def test_default_gates_are_reachable_in_every_domain(domain_name):
    """ResearchAgent scores hallucination resistance from the system prompt, but its search space
    had no prompt dimension, so its safety score was stuck below the default limit and no
    candidate could ever be promoted."""
    domain = get_domain(domain_name)
    provider = MockLLMProvider()
    challenges = seed_dataset(domain, "cal", n=80, seed=1).challenges
    categories = [c.category for c in challenges]

    def population(genome):
        return aggregate([domain.evaluate(genome, c, provider) for c in challenges], categories)

    baseline = SystemGenome(system_id="cal", version=1)
    baseline_agg = population(baseline)
    space = search_space_for_domain(domain_name)
    rng = random.Random(0)
    gates, limits = PromotionGateConfig(), SafetyConstraints()
    feasible = sum(
        not metric_gate_findings(
            baseline_agg,
            population(baseline.derive(mutations=[], overrides=space.sample(rng), new_version=2 + i)),
            gates,
            limits,
        )
        for i in range(60)
    )
    assert feasible > 0


def test_comparison_summary_reports_the_confidence_it_was_run_at():
    result = compare([0.5, 0.6, 0.55, 0.5] * 5, [0.7, 0.8, 0.72, 0.75] * 5, confidence=0.9)
    assert "90% CI" in result.summary()
    assert "95%" not in result.summary()


def test_promotion_gate_enforces_min_confidence():
    """`PromotionGateConfig.min_confidence` was configurable but never consulted."""
    from neuroforge.evaluation.statistics import ComparisonResult

    baseline = aggregate_stub(quality=0.5)
    candidate = aggregate_stub(quality=0.8)
    lax = ComparisonResult(0.3, 0.6, 0.5, 0.7, 2.0, 0.80, Conclusion.LIKELY_IMPROVEMENT)
    decision = evaluate_promotion(baseline, candidate, lax, PromotionGateConfig(), SafetyConstraints())
    assert not decision.approved
    assert any("80% confidence" in r for r in decision.reasons)


def aggregate_stub(quality: float):
    from neuroforge.evaluation.aggregate import AggregateMetrics

    return AggregateMetrics(
        genome_hash="h",
        n_evaluations=10,
        metrics={"quality": quality, "safety_score": 0.95, "policy_compliance": 0.9},
        cost_usd=0.001,
        latency_ms=500.0,
    )


# --- database ----------------------------------------------------------------------------------


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROFORGE_DATABASE_URL", f"sqlite:///{tmp_path / 'nf.db'}")
    import neuroforge.db.session as session_module

    importlib.reload(session_module)
    return session_module


def test_sqlite_enforces_foreign_keys(fresh_db):
    """SQLite ignores foreign keys by default, so a bad genome hash passed every local test and
    would only fail on Postgres."""
    from sqlalchemy.exc import IntegrityError

    from neuroforge.db.models import PromotionRecord

    fresh_db.init_db()
    with pytest.raises(IntegrityError), fresh_db.session_scope() as session:
        session.add(PromotionRecord(genome_hash="nope", approved=False, next_status="REJECTED", reasons=[]))


def test_init_db_adds_columns_missing_from_an_older_database(fresh_db):
    """`create_all` never alters existing tables, so a database created before a column was added
    failed on the first query that selected it ("no such column")."""
    from sqlalchemy import inspect, text

    with fresh_db._engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE promotion_decisions (id INTEGER PRIMARY KEY, genome_hash VARCHAR(32), "
                "experiment_id VARCHAR(100), approved BOOLEAN, next_status VARCHAR(30), "
                "reasons JSON, created_at DATETIME)"
            )
        )
    fresh_db.init_db()
    columns = {c["name"] for c in inspect(fresh_db._engine).get_columns("promotion_decisions")}
    assert "evidence" in columns


def test_saving_an_experiment_persists_its_updated_config(fresh_db):
    """`experiment resume --extra-candidates` raised the budget in memory and saved, but the row's
    config was only ever written at creation, so the resumed run kept the old budget."""
    from neuroforge.db.repository import get_experiment, save_experiment

    fresh_db.init_db()
    config = _config("random", 10, experiment_id="resume-me")
    with fresh_db.session_scope() as session:
        save_experiment(session, "app", config, status="created")
    config.budget.max_candidates = 50
    with fresh_db.session_scope() as session:
        save_experiment(session, "app", config, status="running")
    with fresh_db.session_scope() as session:
        assert get_experiment(session, "resume-me").config["budget"]["max_candidates"] == 50


def test_a_failed_run_is_recorded_as_failed(fresh_db, tmp_path, monkeypatch):
    from neuroforge.db.repository import (
        get_experiment,
        save_dataset_version,
        save_experiment,
        save_genome,
    )
    from neuroforge.domains.forge_support import ForgeSupportDomain
    from neuroforge.experiments.runner import run_recorded_experiment

    fresh_db.init_db()
    domain = get_domain("forge-support")
    baseline = SystemGenome(system_id="app", version=1)
    config = _config("random", 8, experiment_id="boom", dataset_id="app-dataset")
    with fresh_db.session_scope() as session:
        save_genome(session, baseline)
        save_dataset_version(session, seed_dataset(domain, "app-dataset", n=40, seed=1))
        save_experiment(session, "app", config, status="created")

    calls = {"n": 0}
    original = ForgeSupportDomain.evaluate

    def flaky(self, genome, challenge, provider):
        calls["n"] += 1
        if genome.version > 1:
            raise RuntimeError("provider exploded")
        return original(self, genome, challenge, provider)

    monkeypatch.setattr(ForgeSupportDomain, "evaluate", flaky)
    with pytest.raises(RuntimeError, match="exploded"):
        run_recorded_experiment("boom", tmp_path)
    with fresh_db.session_scope() as session:
        assert get_experiment(session, "boom").status == "failed"


# --- API ---------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROFORGE_DATABASE_URL", f"sqlite:///{tmp_path / 'nf.db'}")
    monkeypatch.setenv("NEUROFORGE_STATE_DIR", str(tmp_path / "state"))
    import neuroforge.db.session as db_session_module

    importlib.reload(db_session_module)
    import neuroforge_api.auth as api_auth
    import neuroforge_api.main as api_main

    # Reload auth too: its rate limiters are module state that would otherwise leak between tests.
    importlib.reload(api_auth)
    importlib.reload(api_main)
    api_main.init_db()
    key = api_main.bootstrap_default_key()

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c, {"X-API-Key": key}


BUDGET = {"max_candidates": 8, "max_requests": 100000, "max_cost_usd": 100, "max_duration_minutes": 30}


def _experiment(eid="e1", **overrides):
    return {
        "experiment_id": eid,
        "system_id": "app",
        "strategy": "random",
        "batch_size": 8,
        "max_batches": 2,
        "budget": BUDGET,
        **overrides,
    }


def test_api_rejects_invalid_input_instead_of_failing_later(client):
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 40, "seed": 1}, headers=h)

    bad = [
        ("/api/v1/experiments", _experiment(strategy="nonsense")),  # used to 200, then 500 at run
        ("/api/v1/experiments", _experiment(domain="nope")),  # used to 500
        ("/api/v1/experiments", _experiment("../../escape")),  # path traversal through state files
        ("/api/v1/experiments", _experiment(batch_size=0)),
        ("/api/v1/api-keys", {"name": "x", "role": "superuser"}),  # used to be stored
        ("/api/v1/datasets", {"dataset_id": "z", "n": 0}),  # used to create an empty dataset
        ("/api/v1/datasets", {"dataset_id": "z", "n": 10**9}),  # used to allocate until it died
        ("/api/v1/datasets/app-dataset/evolve", {"mean_score": 4}),
        (
            "/api/v1/canaries",
            {"baseline_hash": "a", "candidate_hash": "b", "dataset_id": "app-dataset", "traffic_fraction": 5},
        ),
    ]
    for path, body in bad:
        assert c.post(path, json=body, headers=h).status_code == 422, (path, body)


def test_api_conflicts_and_lookups(client):
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 40, "seed": 1}, headers=h)
    assert c.post("/api/v1/experiments", json=_experiment(), headers=h).status_code == 200
    assert c.post("/api/v1/experiments/e1/run", headers=h).status_code == 200

    # Re-creating an existing id used to return 200 and reset the finished experiment to "created".
    assert c.post("/api/v1/experiments", json=_experiment(), headers=h).status_code == 409
    listed = next(e for e in c.get("/api/v1/experiments", headers=h).json() if e["experiment_id"] == "e1")
    assert listed["status"] == "completed"

    assert c.post("/api/v1/experiments/never-existed/cancel", headers=h).status_code == 404
    assert c.post("/api/v1/experiments/never-existed/run", headers=h).status_code == 404
    assert c.get("/api/v1/genomes/app@vX", headers=h).status_code == 400  # used to be a 500
    assert c.get("/api/v1/genomes/app@v99", headers=h).status_code == 404


def test_checkpoint_endpoint_handles_an_experiment_that_never_started(client, tmp_path):
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 40, "seed": 1}, headers=h)
    c.post("/api/v1/experiments", json=_experiment(), headers=h)
    flag = tmp_path / "state" / "e1" / "e1.cancel"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()  # a cancel request that arrived before the run started
    c.post("/api/v1/experiments/e1/run", headers=h)
    resp = c.get("/api/v1/experiments/e1/checkpoint", headers=h)
    assert resp.status_code == 200  # best_fitness was -inf: not JSON-serializable, and unreadable
    assert resp.json()["best_fitness"] is None


def test_lineage_labels_show_the_field_that_changed(client):
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 60, "seed": 1}, headers=h)
    c.post("/api/v1/experiments", json=_experiment(), headers=h)
    c.post("/api/v1/experiments/e1/run", headers=h)
    nodes = c.get("/api/v1/genomes/app/lineage", headers=h).json()["nodes"]
    labels = [m for n in nodes for m in n["mutations"]]
    assert labels and all("→" in m and ":" in m for m in labels)
    assert not any(m == "random.propose" for m in labels)


def test_failed_authentication_attempts_are_throttled(client):
    c, _ = client
    codes = [c.get("/api/v1/applications", headers={"X-API-Key": f"wrong-{i}"}).status_code for i in range(14)]
    assert codes[0] == 401
    assert codes[-1] == 429


def test_metrics_are_labelled_by_route_template_not_raw_path(client):
    """One label series per URL grows the registry without bound (and unauthenticated 404s counted)."""
    c, h = client
    for i in range(4):
        c.get(f"/nowhere-{i}")
        c.get(f"/api/v1/experiments/exp-{i}/checkpoint", headers=h)
    metrics = c.get("/metrics").text
    assert "/nowhere-" not in metrics and "exp-1" not in metrics
    assert 'path="/api/v1/experiments/{experiment_id}/checkpoint"' in metrics
    assert 'path="unmatched"' in metrics


def test_enqueue_failure_does_not_leave_the_experiment_marked_queued(client, monkeypatch):
    import neuroforge_api.main as api_main
    import redis

    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 40, "seed": 1}, headers=h)
    c.post("/api/v1/experiments", json=_experiment(), headers=h)

    def redis_down(_experiment_id):
        raise redis.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(api_main, "enqueue_experiment", redis_down)
    assert c.post("/api/v1/experiments/e1/enqueue", headers=h).status_code == 503
    listed = next(e for e in c.get("/api/v1/experiments", headers=h).json() if e["experiment_id"] == "e1")
    assert listed["status"] == "created"


def test_a_valid_key_keeps_working_while_its_client_address_is_throttled(client):
    """Throttling failed attempts per address must not lock out legitimate users behind the same
    address (a shared proxy)."""
    c, h = client
    assert c.get("/api/v1/applications", headers=h).status_code == 200  # verified and remembered
    for i in range(14):
        c.get("/api/v1/applications", headers={"X-API-Key": f"guess-{i}"})
    assert c.get("/api/v1/applications", headers={"X-API-Key": "one-more-guess"}).status_code == 429
    assert c.get("/api/v1/applications", headers=h).status_code == 200


def test_candidate_versions_are_unique_across_experiments_of_one_application(client):
    """Each experiment numbered its candidates from v2, so two experiments on one application both
    produced a "v141" — different genomes under one label."""
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 60, "seed": 1}, headers=h)
    versions = {}
    for eid, seed in (("e1", 1), ("e2", 2)):
        assert c.post("/api/v1/experiments", json=_experiment(eid, seed=seed), headers=h).status_code == 200
        assert c.post(f"/api/v1/experiments/{eid}/run", headers=h).status_code == 200
        versions[eid] = {x["version"] for x in c.get(f"/api/v1/experiments/{eid}/candidates", headers=h).json()["candidates"]}
    assert versions["e1"] and versions["e2"]
    assert not versions["e1"] & versions["e2"]


def test_api_timestamps_carry_a_utc_offset(client):
    """SQLite returns naive datetimes; without an offset browsers read the string as local time."""
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    created = c.get("/api/v1/applications", headers=h).json()[0]["created_at"]
    assert created.endswith("+00:00")


def test_cancelling_an_idle_experiment_is_refused(client):
    """A cancel flag left behind for an experiment that isn't running silently cancelled its next
    run and relabelled a finished experiment as cancelled."""
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 40, "seed": 1}, headers=h)
    c.post("/api/v1/experiments", json=_experiment(), headers=h)
    assert c.post("/api/v1/experiments/e1/cancel", headers=h).status_code == 409  # created, not running
    c.post("/api/v1/experiments/e1/run", headers=h)
    assert c.post("/api/v1/experiments/e1/cancel", headers=h).status_code == 409  # already completed


def test_a_cancelled_run_is_reported_as_cancelled_not_completed(baseline_genome, small_dataset, tmp_path):
    (tmp_path / "x.cancel").touch()
    ExperimentEngine(_config("random", 16), baseline_genome, small_dataset, tmp_path).run()
    assert ExperimentCheckpoint.load(tmp_path / "x.checkpoint.json").status == "cancelled"


def test_two_datasets_with_identical_parameters_are_both_created(client):
    """The unique content hash ignored the dataset id, so seeding "beta" with the same n and seed as
    "alpha" (two applications auto-seeding with the defaults) returned 200 and created nothing."""
    c, h = client
    for dataset_id in ("alpha", "beta"):
        assert c.post("/api/v1/datasets", json={"dataset_id": dataset_id, "n": 40, "seed": 1}, headers=h).status_code == 200
    assert {d["dataset_id"] for d in c.get("/api/v1/datasets", headers=h).json()} == {"alpha", "beta"}


def test_reseeding_a_dataset_id_is_idempotent_or_a_conflict(client):
    c, h = client
    body = {"dataset_id": "alpha", "n": 40, "seed": 1}
    assert c.post("/api/v1/datasets", json=body, headers=h).status_code == 200
    assert c.post("/api/v1/datasets", json=body, headers=h).status_code == 200  # identical: no-op
    assert c.post("/api/v1/datasets", json={**body, "seed": 2}, headers=h).status_code == 409
    assert len([d for d in c.get("/api/v1/datasets", headers=h).json() if d["dataset_id"] == "alpha"]) == 1


def test_a_resumed_experiment_keeps_the_dataset_version_it_started_on(client):
    """Resume always took the dataset's *latest* version, so batches evaluated on v1 and batches
    evaluated on an evolved v2 ended up in one fitness history."""
    c, h = client
    c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
    c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 60, "seed": 1}, headers=h)
    c.post("/api/v1/experiments", json=_experiment(), headers=h)
    first = c.post("/api/v1/experiments/e1/run", headers=h).json()
    assert first["dataset_version"] == 1
    evolved = c.post("/api/v1/datasets/app-dataset/evolve", json={"mean_score": 0.95, "n_new": 20}, headers=h)
    assert evolved.json()["version"] == 2
    again = c.post("/api/v1/experiments/e1/run", headers=h).json()
    assert again["dataset_version"] == 1
    # A *new* experiment does pick up the evolved dataset.
    c.post("/api/v1/experiments", json=_experiment("e2"), headers=h)
    assert c.post("/api/v1/experiments/e2/run", headers=h).json()["dataset_version"] == 2


def test_the_engine_refuses_to_resume_on_a_different_dataset_version(baseline_genome, small_dataset, tmp_path):
    from neuroforge.experiments.engine import DatasetChanged

    ExperimentEngine(_config("random", 8), baseline_genome, small_dataset, tmp_path).run()
    evolved = small_dataset.model_copy(update={"version": small_dataset.version + 1})
    with pytest.raises(DatasetChanged):
        ExperimentEngine(_config("random", 16), baseline_genome, evolved, tmp_path).run()
