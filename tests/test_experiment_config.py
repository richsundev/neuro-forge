"""User-steerable experiments: which dimensions the search may turn, what it optimizes, background
start with progress, and cost/latency scored relative to the baseline."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest

from neuroforge.evaluation.objectives import DEFAULT_OBJECTIVES
from neuroforge.experiments.configure import (
    ConfigError,
    build_objectives,
    build_search_space,
    describe_domain,
    parse_weight_options,
    search_groups,
)

BUDGET = {"max_candidates": 16, "max_requests": 100000, "max_cost_usd": 100, "max_duration_minutes": 30}


def test_search_dimensions_restrict_the_space_by_section_or_field():
    full = build_search_space("forge-support", None)
    assert {"model", "prompt", "retrieval", "tools", "agent"} <= set(search_groups(full))
    prompt_only = build_search_space("forge-support", ["prompt"])
    assert set(search_groups(prompt_only)) == {"prompt"}
    mixed = build_search_space("forge-support", ["retrieval.top_k", "model"])
    assert set(mixed.parameters) == {"retrieval.top_k"} | {p for p in full.parameters if p.startswith("model.")}
    with pytest.raises(ConfigError, match="unknown search dimension"):
        build_search_space("forge-support", ["prompt", "nonsense.field"])


def test_objective_weights_are_pinned_shares_and_the_rest_keep_their_balance():
    o = build_objectives({"cost_usd": 0.4})
    assert o.objectives["cost_usd"].weight == pytest.approx(0.4)
    assert sum(x.weight for x in o.objectives.values()) == pytest.approx(1.0)
    defaults = DEFAULT_OBJECTIVES.objectives
    ratio = o.objectives["quality"].weight / o.objectives["task_success"].weight
    assert ratio == pytest.approx(defaults["quality"].weight / defaults["task_success"].weight, rel=1e-3)
    assert o.objectives["cost_usd"].direction == "minimize" and o.objectives["quality"].direction == "maximize"
    # Weights that already fill the total leave nothing for the others.
    full = build_objectives({"quality": 1.0, "cost_usd": 1.0})
    assert full.objectives["quality"].weight == pytest.approx(0.5) and full.objectives["safety_score"].weight == 0.0
    assert build_objectives(None) is DEFAULT_OBJECTIVES
    for bad in ({"nope": 1.0}, {"quality": -1.0}, {"quality": 0.0}):
        with pytest.raises(ConfigError):
            build_objectives(bad)


def test_weight_options_parse_from_the_cli_form():
    assert parse_weight_options(["quality=0.4", "cost_usd=0.3"]) == {"quality": 0.4, "cost_usd": 0.3}
    assert parse_weight_options(None) is None
    for bad in (["quality"], ["quality=lots"]):
        with pytest.raises(ConfigError):
            parse_weight_options(bad)


def test_domain_description_lists_what_an_experiment_form_needs():
    info = describe_domain("forge-support")
    assert "prompt" in info["search_dimensions"]
    assert {"path", "type"} <= set(info["search_dimensions"]["prompt"][0])
    assert set(info["objectives"]) == set(DEFAULT_OBJECTIVES.objectives)
    assert info["promotion_gates"]["max_cost_increase"] == 0.10
    assert info["safety_constraints"]["max_policy_violation_rate"] == 0.28


def test_cost_is_scored_relative_to_the_baseline_not_a_fixed_dollar_scale(baseline_genome, small_dataset, tmp_path):
    """With the old fixed $0.05 scale a genome 3x cheaper than the baseline gained ~0.0008 fitness while
    +0.05 quality gained ~0.0125, so the cost weight could not steer the search. Relative to the
    baseline, halving cost is worth a lot."""
    from neuroforge.evaluation.objectives import ObjectiveSpec

    weights = build_objectives({"cost_usd": 0.5})
    metrics = {"quality": 0.6, "task_success": 0.6, "policy_compliance": 0.7, "safety_score": 0.9, "failure_rate": 0.1, "latency_ms": 500.0}
    scales = {"cost_usd": 0.0024, "latency_ms": 1500.0, "failure_rate": 1.0}  # 2x a $0.0012 / 750ms baseline
    cheap = weights.score({**metrics, "cost_usd": 0.0004}, scales)
    same = weights.score({**metrics, "cost_usd": 0.0012}, scales)
    pricey = weights.score({**metrics, "cost_usd": 0.0036}, scales)
    assert cheap - same > 0.1 and same - pricey > 0.1
    assert isinstance(weights, ObjectiveSpec)


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
        h = {"X-API-Key": key}
        c.post("/api/v1/applications", json={"id": "app", "name": "a", "domain": "forge-support"}, headers=h)
        c.post("/api/v1/datasets", json={"dataset_id": "app-dataset", "n": 60, "seed": 1}, headers=h)
        yield c, h


def _create(c, h, eid="e1", **extra):
    body = {"experiment_id": eid, "system_id": "app", "strategy": "random", "batch_size": 8, "max_batches": 2, "budget": BUDGET, **extra}
    return c.post("/api/v1/experiments", json=body, headers=h)


def test_domains_endpoint_describes_every_domain(client):
    c, h = client
    domains = {d["name"]: d for d in c.get("/api/v1/domains", headers=h).json()}
    assert {"forge-support", "sql-agent", "research-agent"} <= set(domains)
    assert "prompt" in domains["research-agent"]["search_dimensions"]


def test_dimensions_and_weights_are_validated_stored_and_honoured(client):
    c, h = client
    assert _create(c, h, "bad1", search_dimensions=["nope"]).status_code == 400
    assert _create(c, h, "bad2", objective_weights={"nope": 0.5}).status_code == 400
    assert _create(c, h, "bad3", objective_weights={"quality": -1}).status_code == 422

    assert _create(c, h, "e1", search_dimensions=["prompt"], objective_weights={"cost_usd": 0.4}).status_code == 200
    config = c.get("/api/v1/experiments/e1", headers=h).json()["config"]
    assert {p.split(".")[0] for p in config["search_space"]["parameters"]} == {"prompt"}
    assert config["objectives"]["objectives"]["cost_usd"]["weight"] == pytest.approx(0.4)

    assert c.post("/api/v1/experiments/e1/run", headers=h).status_code == 200
    detail = c.get("/api/v1/experiments/e1", headers=h).json()["result"]
    changed = {m["field_path"] for m in detail["best_genome"]["mutations"]}
    assert changed and all(path.startswith("prompt.") for path in changed)  # nothing outside the chosen section moved


def test_an_experiment_can_be_started_in_the_background_and_followed(client):
    c, h = client
    assert _create(c, h, "e1").status_code == 200
    started = c.post("/api/v1/experiments/e1/start", headers=h)
    assert started.status_code == 202 and started.json()["status"] == "queued"
    deadline = time.time() + 30
    status = ""
    while time.time() < deadline:
        status = c.get("/api/v1/experiments/e1", headers=h).json()["status"]
        if status == "completed":
            break
        time.sleep(0.2)
    assert status == "completed"
    detail = c.get("/api/v1/experiments/e1", headers=h).json()
    assert detail["result"]["candidates_evaluated"] == 16
    progress = c.get("/api/v1/experiments/e1/checkpoint", headers=h).json()
    assert progress["candidates_completed"] == 16 and progress["fitness_history"]


def test_starting_twice_or_an_unknown_experiment_is_refused(client):
    c, h = client
    assert c.post("/api/v1/experiments/nope/start", headers=h).status_code == 404
    _create(c, h, "e1")
    first = c.post("/api/v1/experiments/e1/start", headers=h)
    assert first.status_code == 202
    second = c.post("/api/v1/experiments/e1/start", headers=h)
    assert second.status_code in (409, 202)  # 409 while queued/running; 202 only if it already finished
    if second.status_code == 202:
        pytest.skip("the first run finished before the second start")
    viewer = c.post("/api/v1/api-keys", json={"name": "v", "role": "viewer"}, headers=h).json()["api_key"]
    assert c.post("/api/v1/experiments/e1/start", headers={"X-API-Key": viewer}).status_code == 403


def _cli(tmp_path, *args):
    env = {
        **os.environ,
        "NEUROFORGE_DATABASE_URL": f"sqlite:///{tmp_path / 'cli.db'}",
        "NEUROFORGE_STATE_DIR": str(tmp_path / "state"),
    }
    return subprocess.run(
        [sys.executable, "-m", "neuroforge.cli.main", *args], env=env, capture_output=True, text=True, timeout=120
    )


def test_cli_experiment_create_takes_focus_weights_and_gate_options(tmp_path):
    ok = _cli(
        tmp_path, "experiment", "create", "cli-exp", "--focus", "prompt", "--focus", "retrieval.top_k",
        "--weight", "cost_usd=0.4", "--min-quality", "0.5", "--max-cost-increase", "0.2",
        "--max-policy-violation", "0.3", "--max-candidates", "8",
    )
    assert ok.returncode == 0, ok.stderr
    with sqlite3.connect(tmp_path / "cli.db") as db:
        (raw,) = db.execute("select config from experiments where experiment_id = 'cli-exp'").fetchone()
    config = json.loads(raw)
    params = config["search_space"]["parameters"]
    assert {p.split(".")[0] for p in params} == {"prompt", "retrieval"}
    assert config["objectives"]["objectives"]["cost_usd"]["weight"] == pytest.approx(0.4)
    assert config["promotion_gates"]["min_quality"] == 0.5 and config["promotion_gates"]["max_cost_increase"] == 0.2
    assert config["safety_constraints"]["max_policy_violation_rate"] == 0.3

    bad = _cli(tmp_path, "experiment", "create", "cli-bad", "--focus", "nonsense.field")
    assert bad.returncode != 0 and "unknown search dimension" in (bad.stdout + bad.stderr)
    bad = _cli(tmp_path, "experiment", "create", "cli-bad2", "--weight", "quality=lots")
    assert bad.returncode != 0


def test_cli_rejects_invalid_options_with_a_message_not_a_traceback(tmp_path):
    for args, needle in (
        (["--min-quality", "2"], "min_quality"),
        (["--max-cost-increase", "nan"], "max_cost_increase"),
        (["--max-policy-violation", "5"], "max_policy_violation_rate"),
        (["--weight", "quality=nan"], "finite"),
        (["--batch-size", "0"], "batch-size"),
        (["--max-candidates", "0"], "max-candidates"),
    ):
        out = _cli(tmp_path, "experiment", "create", "bad", *args)
        assert out.returncode != 0 and needle in (out.stdout + out.stderr), args
        assert "Traceback" not in out.stderr, args


def test_cli_refuses_a_domain_or_dataset_that_does_not_match(tmp_path):
    assert _cli(tmp_path, "experiment", "create", "e1").returncode == 0
    wrong_domain = _cli(tmp_path, "experiment", "create", "e2", "--domain", "sql-agent")
    assert wrong_domain.returncode != 0 and "uses domain" in " ".join((wrong_domain.stdout + wrong_domain.stderr).split())
    wrong_dataset = _cli(
        tmp_path, "experiment", "create", "e3", "--domain", "sql-agent", "--system-id", "sql", "--dataset-id", "support-agent-dataset"
    )
    assert wrong_dataset.returncode != 0 and "belongs to domain" in " ".join((wrong_dataset.stdout + wrong_dataset.stderr).split())


def test_cli_resume_refuses_to_grow_a_budget_into_another_experiments_versions(tmp_path):
    for eid in ("e1", "e2"):
        assert _cli(tmp_path, "experiment", "create", eid, "--max-candidates", "8", "--batch-size", "4").returncode == 0
    out = _cli(tmp_path, "experiment", "resume", "e1", "--extra-candidates", "100")
    assert out.returncode != 0 and "would reuse version" in " ".join((out.stdout + out.stderr).split())
    assert _cli(tmp_path, "experiment", "resume", "e1", "--extra-candidates", "0").returncode != 0
