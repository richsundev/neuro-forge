"""The production lifecycle: champion selection, evolving from the champion, superseding, rollback,
and refusing promotions that were validated against a baseline that is no longer in production."""

from __future__ import annotations

import importlib

import pytest

BUDGET = {"max_candidates": 16, "max_requests": 100000, "max_cost_usd": 100, "max_duration_minutes": 30}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROFORGE_DATABASE_URL", f"sqlite:///{tmp_path / 'nf.db'}")
    monkeypatch.setenv("NEUROFORGE_STATE_DIR", str(tmp_path / "state"))
    import neuroforge.db.session as db_session_module

    importlib.reload(db_session_module)
    import neuroforge_api.auth as api_auth
    import neuroforge_api.main as api_main

    importlib.reload(api_auth)
    importlib.reload(api_main)
    api_main.init_db()
    key = api_main.bootstrap_default_key()

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers={"X-API-Key": key})
        c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 60, "seed": 1}, headers={"X-API-Key": key})
        yield c, {"X-API-Key": key}


def _run(c, h, eid, seed, **extra) -> dict:
    body = {"experiment_id": eid, "system_id": "app", "strategy": "random", "batch_size": 8, "max_batches": 2, "seed": seed, "budget": BUDGET, **extra}
    created = c.post("/api/v1/experiments", json=body, headers=h)
    assert created.status_code == 200, created.text
    assert c.post(f"/api/v1/experiments/{eid}/run", headers=h).status_code == 200
    listed = next(e for e in c.get("/api/v1/experiments", headers=h).json() if e["experiment_id"] == eid)
    return {**created.json(), "best": listed["best_genome_hash"], "listed": listed}


def _approve_and_canary(best_hash: str, experiment_id: str) -> None:
    """Record an approved decision and a passing canary directly, so the lifecycle tests don't depend
    on whether a small random search happens to clear the real gates."""
    from neuroforge.db.repository import save_canary_result, save_promotion_decision
    from neuroforge.db.session import session_scope
    from neuroforge.evaluation.aggregate import AggregateMetrics
    from neuroforge.genomes.schema import PromotionStatus
    from neuroforge.promotion.canary import CanaryResult
    from neuroforge.promotion.gates import PromotionDecision

    agg = AggregateMetrics(genome_hash=best_hash, n_evaluations=10, metrics={"quality": 0.8})
    with session_scope() as session:
        save_promotion_decision(
            session, best_hash, experiment_id, PromotionDecision(approved=True, next_status=PromotionStatus.APPROVED, reasons=["ok"])
        )
        save_canary_result(
            session, best_hash, "app@v1",
            CanaryResult(baseline_metrics=agg, candidate_metrics=agg, traffic_split=0.1, n_baseline_requests=9, n_candidate_requests=1, rollback_triggered=False, reasons=["ok"]),
        )


def _promote(c, h, run: dict, eid: str):
    _approve_and_canary(run["best"], eid)
    return c.post("/api/v1/promotions/finalize", json={"genome_hash": run["best"], "experiment_id": eid}, headers=h)


def _statuses(c, h) -> dict[str, str]:
    nodes = c.get("/api/v1/genomes/app/lineage", headers=h).json()["nodes"]
    return {n["hash"]: n["status"] for n in nodes}


def _production(c, h):
    return next(a for a in c.get("/api/v1/applications", headers=h).json() if a["id"] == "app")["production"]


def test_experiments_start_from_the_champion_and_promotion_supersedes_it(client):
    c, h = client
    first = _run(c, h, "e1", 1)
    original = first["baseline_hash"]
    assert first["baseline_version"] == 1 and _production(c, h) is None

    promoted = _promote(c, h, first, "e1")
    assert promoted.status_code == 200 and promoted.json()["superseded"] is None
    assert _production(c, h)["genome_hash"] == first["best"]

    second = _run(c, h, "e2", 2)  # default baseline: "champion"
    assert second["baseline_hash"] == first["best"] != original
    detail = c.get("/api/v1/experiments/e2", headers=h).json()
    assert detail["config"]["baseline_hash"] == first["best"]
    assert detail["result"]["baseline_genome"]["parent_hash"] == original
    champion_generation = detail["result"]["baseline_genome"]["generation"]
    assert detail["result"]["best_genome"]["generation"] > champion_generation  # numbering continues
    assert second["listed"]["baseline"] == first["best"]

    if second["best"] != first["best"]:  # the second search found a different winner
        assert detail["result"]["best_genome"]["parent_hash"] == first["best"]
        assert _promote(c, h, second, "e2").json()["superseded"] == first["best"]
        statuses = _statuses(c, h)
        assert statuses[second["best"]] == "PROMOTED" and statuses[first["best"]] == "SUPERSEDED"
        assert _production(c, h)["genome_hash"] == second["best"]


def test_baseline_can_be_the_original_or_a_specific_genome_and_is_validated(client):
    c, h = client
    first = _run(c, h, "e1", 1)
    _promote(c, h, first, "e1")
    original = _run(c, h, "e2", 2, baseline="original")
    assert original["baseline_version"] == 1
    pinned = _run(c, h, "e3", 3, baseline=first["best"])
    assert pinned["baseline_hash"] == first["best"]
    by_ident = _run(c, h, "e4", 4, baseline="app@v1")
    assert by_ident["baseline_version"] == 1

    other = c.post("/api/v1/experiments", json={"experiment_id": "e5", "system_id": "app", "baseline": "nope", "budget": BUDGET}, headers=h)
    assert other.status_code == 400
    foreign = c.post("/api/v1/experiments", json={"experiment_id": "e6", "system_id": "app", "baseline": "other@v1", "budget": BUDGET}, headers=h)
    assert foreign.status_code == 400


def test_promoting_a_candidate_validated_against_stale_production_is_refused(client):
    c, h = client
    a = _run(c, h, "ea", 1, baseline="original")
    b = _run(c, h, "eb", 2, baseline="original")
    assert _promote(c, h, a, "ea").status_code == 200
    stale = _promote(c, h, b, "eb")  # b was measured against the original, but a is in production now
    assert stale.status_code == 409 and "champion" in stale.json()["detail"]
    assert _production(c, h)["genome_hash"] == a["best"]


def test_rollback_restores_the_previous_champion_then_the_original(client):
    c, h = client
    a = _run(c, h, "e1", 1)
    _promote(c, h, a, "e1")
    b = _run(c, h, "e2", 2)
    if b["best"] == a["best"]:
        pytest.skip("second search found no new winner; nothing to supersede")
    _promote(c, h, b, "e2")

    back = c.post("/api/v1/promotions/rollback", json={"system_id": "app"}, headers=h)
    assert back.status_code == 200
    assert back.json()["rolled_back"] == b["best"] and back.json()["restored"] == a["best"]
    statuses = _statuses(c, h)
    assert statuses[b["best"]] == "ROLLED_BACK" and statuses[a["best"]] == "PROMOTED"
    assert _production(c, h)["genome_hash"] == a["best"]

    again = c.post("/api/v1/promotions/rollback", json={"system_id": "app"}, headers=h)
    assert again.json()["restored"] is None  # back to the original baseline
    assert _production(c, h) is None
    assert c.post("/api/v1/promotions/rollback", json={"system_id": "app"}, headers=h).status_code == 409

    history = c.get("/api/v1/promotions/approvals", headers=h).json()
    assert [(e["action"], e["genome_hash"]) for e in history] == [
        ("rollback", a["best"]), ("rollback", b["best"]), ("promote", b["best"]), ("promote", a["best"]),
    ]
    assert all(e["system_id"] == "app" for e in history)


def test_only_an_admin_can_roll_back(client):
    c, h = client
    operator = c.post("/api/v1/api-keys", json={"name": "ops", "role": "operator"}, headers=h).json()["api_key"]
    assert c.post("/api/v1/promotions/rollback", json={"system_id": "app"}, headers={"X-API-Key": operator}).status_code == 403


def test_per_experiment_gates_can_be_overridden(client):
    c, h = client
    body = {
        "experiment_id": "e1", "system_id": "app", "budget": BUDGET, "strategy": "random", "batch_size": 8, "max_batches": 2,
        "promotion_gates": {"max_cost_increase": 2.5, "max_latency_increase": 3.0},
        "safety_constraints": {"max_policy_violation_rate": 0.3},
    }
    assert c.post("/api/v1/experiments", json=body, headers=h).status_code == 200
    config = c.get("/api/v1/experiments/e1", headers=h).json()["config"]
    assert config["promotion_gates"]["max_cost_increase"] == 2.5
    assert config["safety_constraints"]["max_policy_violation_rate"] == 0.3
    bad = {**body, "experiment_id": "e2", "promotion_gates": {"max_cost_increase": "lots"}}
    assert c.post("/api/v1/experiments", json=bad, headers=h).status_code == 422


def test_lineage_generations_continue_from_the_baseline():
    from neuroforge.genomes.schema import SystemGenome

    parent = SystemGenome(system_id="s", version=1, generation=7)
    child = parent.derive(mutations=[], overrides={"model.name": "fast-cheap"}, new_version=2)
    assert child.generation == 8


def test_experiments_report_whether_their_baseline_is_still_current(client):
    c, h = client
    a = _run(c, h, "ea", 1, baseline="original")
    _run(c, h, "eb", 2, baseline="original")
    current = {e["experiment_id"]: e["baseline_is_current"] for e in c.get("/api/v1/experiments", headers=h).json()}
    assert current == {"ea": True, "eb": True}
    _promote(c, h, a, "ea")
    current = {e["experiment_id"]: e["baseline_is_current"] for e in c.get("/api/v1/experiments", headers=h).json()}
    assert current == {"ea": False, "eb": False}  # production is now `a`, and neither started from it
    third = _run(c, h, "ec", 3)  # default baseline: the champion
    assert third["listed"]["baseline_is_current"] is True


def test_mutation_records_list_only_fields_that_differ_from_the_baseline(baseline_genome, small_dataset, tmp_path):
    """A search point sets every dimension, so most entries equal the baseline's value; recording them
    made every genome look like it had ~14 mutations, half of them no-ops."""
    from neuroforge.experiments.budget import ExperimentBudget
    from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
    from neuroforge.experiments.spaces import forge_support_search_space
    from neuroforge.genomes.schema import SystemGenome

    config = ExperimentConfig(
        experiment_id="m", domain_name="forge-support", search_strategy="random", search_space=forge_support_search_space(),
        batch_size=8, max_batches=2, seed=3,
        budget=ExperimentBudget(max_candidates=16, max_requests=10**6, max_cost_usd=100, max_duration_minutes=5),
    )
    result = ExperimentEngine(config, baseline_genome, small_dataset, tmp_path).run()
    best = SystemGenome.model_validate(result.best_genome)
    assert best.mutations
    for m in best.mutations:
        assert list(m.old_value) != list(m.new_value) if isinstance(m.old_value, list | tuple) else m.old_value != m.new_value
