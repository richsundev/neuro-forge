"""End-to-end API tests against an isolated SQLite DB (per-test tmp file)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "nf.db"
    monkeypatch.setenv("NEUROFORGE_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("NEUROFORGE_STATE_DIR", str(tmp_path / "state"))

    import neuroforge.db.session as db_session_module

    importlib.reload(db_session_module)

    import neuroforge_api.auth as api_auth
    import neuroforge_api.main as api_main

    # Reload auth too: its rate limiters are module state that would otherwise leak between tests.
    importlib.reload(api_auth)
    importlib.reload(api_main)

    api_main.init_db()
    admin_key = api_main.bootstrap_default_key()
    assert admin_key is not None

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c, admin_key


def test_health_requires_no_auth(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_missing_api_key_is_rejected(client):
    c, _ = client
    resp = c.get("/api/v1/applications")
    assert resp.status_code == 401


def test_full_workflow_via_api(client):
    c, admin_key = client
    headers = {"X-API-Key": admin_key}

    resp = c.post(
        "/api/v1/applications",
        json={"id": "support-agent", "name": "Forge Support", "domain": "forge-support"},
        headers=headers,
    )
    assert resp.status_code == 200

    resp = c.post(
        "/api/v1/datasets",
        json={"dataset_id": "support-agent-dataset", "domain": "forge-support", "n": 40, "seed": 1},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["n_challenges"] == 40

    resp = c.post(
        "/api/v1/experiments",
        json={
            "experiment_id": "exp-api-test",
            "system_id": "support-agent",
            "domain": "forge-support",
            "strategy": "random",
            "seed": 3,
            "batch_size": 8,
            "max_batches": 3,
            "budget": {
                "max_candidates": 16,
                "max_requests": 100000,
                "max_cost_usd": 100,
                "max_duration_minutes": 30,
            },
        },
        headers=headers,
    )
    assert resp.status_code == 200

    resp = c.post("/api/v1/experiments/exp-api-test/run", headers=headers)
    assert resp.status_code == 200
    result = resp.json()
    assert result["candidates_evaluated"] == 16
    assert "conclusion" in result["comparison"]

    resp = c.get("/api/v1/experiments/exp-api-test/checkpoint", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["candidates_completed"] == 16

    resp = c.get("/api/v1/genomes/support-agent/lineage", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["nodes"]) >= 1

    # The enriched experiment list should surface enough to drive a Promotion page without
    # the frontend having to fetch every experiment's full detail.
    resp = c.get("/api/v1/experiments", headers=headers)
    assert resp.status_code == 200
    listed = next(e for e in resp.json() if e["experiment_id"] == "exp-api-test")
    from neuroforge.genomes.schema import SystemGenome

    assert listed["best_genome_hash"] == SystemGenome.model_validate(result["best_genome"]).hash()
    assert listed["recommendation"] == result["recommendation"]
    assert listed["comparison_summary"] == result["comparison"]["summary"]

    best_hash = listed["best_genome_hash"]
    resp = c.post(
        "/api/v1/promotions",
        json={"genome_hash": best_hash, "experiment_id": "exp-api-test"},
        headers=headers,
    )
    assert resp.status_code == 200
    decision = resp.json()
    assert "approved" in decision and "reasons" in decision

    resp = c.get("/api/v1/promotions", headers=headers)
    assert resp.status_code == 200
    promotions = resp.json()
    assert any(p["genome_hash"] == best_hash and p["experiment_id"] == "exp-api-test" for p in promotions)

    resp = c.post(
        "/api/v1/canaries",
        json={
            "baseline_hash": "support-agent@v1",
            "candidate_hash": best_hash,
            "dataset_id": "support-agent-dataset",
            "traffic_fraction": 0.3,
            "n_requests": 40,
        },
        headers=headers,
    )
    assert resp.status_code == 200
    canary_result = resp.json()
    assert "rollback_triggered" in canary_result

    resp = c.get("/api/v1/canaries", headers=headers)
    assert resp.status_code == 200
    canaries = resp.json()
    assert any(cn["candidate_hash"] == best_hash for cn in canaries)


def test_promotion_finalize_requires_admin_approval_and_passed_canary(client):
    """The human-approval gate (docs/promotion.md): PROMOTED must be unreachable without both an
    approved promotion decision and a non-rollback canary on record, and only an admin-role key
    may cross it — mirrors the CLI's `neuroforge promotion promote`."""
    c, admin_key = client
    headers = {"X-API-Key": admin_key}

    c.post(
        "/api/v1/applications",
        json={"id": "support-agent", "name": "Forge Support", "domain": "forge-support"},
        headers=headers,
    ).raise_for_status()
    c.post(
        "/api/v1/datasets",
        json={"dataset_id": "support-agent-dataset", "domain": "forge-support", "n": 40, "seed": 1},
        headers=headers,
    ).raise_for_status()
    resp = c.post(
        "/api/v1/experiments",
        json={
            "experiment_id": "exp-finalize-test",
            "system_id": "support-agent",
            "domain": "forge-support",
            "strategy": "random",
            "seed": 3,
            "batch_size": 8,
            "max_batches": 3,
            "budget": {
                "max_candidates": 16,
                "max_requests": 100000,
                "max_cost_usd": 100,
                "max_duration_minutes": 30,
            },
        },
        headers=headers,
    )
    resp.raise_for_status()
    resp = c.post("/api/v1/experiments/exp-finalize-test/run", headers=headers)
    resp.raise_for_status()

    from neuroforge.genomes.schema import PromotionStatus, SystemGenome

    best_hash = SystemGenome.model_validate(resp.json()["best_genome"]).hash()

    # No promotion decision on record yet.
    resp = c.post(
        "/api/v1/promotions/finalize",
        json={"genome_hash": best_hash, "experiment_id": "exp-finalize-test"},
        headers=headers,
    )
    assert resp.status_code == 400

    # Seed a deterministic approved decision and a passed canary directly, rather than relying on
    # this random-strategy experiment's own (uncertain) gate/canary outcome for a positive-path test.
    from neuroforge.db.models import GenomeRecord
    from neuroforge.db.repository import save_canary_result, save_promotion_decision
    from neuroforge.db.session import session_scope
    from neuroforge.evaluation.aggregate import AggregateMetrics
    from neuroforge.promotion.canary import CanaryResult
    from neuroforge.promotion.gates import PromotionDecision as GateDecision

    with session_scope() as session:
        save_promotion_decision(
            session,
            best_hash,
            "exp-finalize-test",
            GateDecision(approved=True, next_status=PromotionStatus.APPROVED, reasons=["ok"]),
        )

    def genome_status() -> str:
        with session_scope() as session:
            return session.get(GenomeRecord, best_hash).status

    assert genome_status() == "APPROVED"

    # Decision approved, but no canary yet.
    resp = c.post(
        "/api/v1/promotions/finalize",
        json={"genome_hash": best_hash, "experiment_id": "exp-finalize-test"},
        headers=headers,
    )
    assert resp.status_code == 400

    agg = AggregateMetrics(genome_hash=best_hash, n_evaluations=10, metrics={"quality": 0.8})
    with session_scope() as session:
        save_canary_result(
            session,
            best_hash,
            "support-agent@v1",
            CanaryResult(
                baseline_metrics=agg,
                candidate_metrics=agg,
                traffic_split=0.1,
                n_baseline_requests=9,
                n_candidate_requests=1,
                rollback_triggered=True,
                reasons=["contrived rollback"],
            ),
        )

    assert genome_status() == "ROLLED_BACK"

    # Decision approved, but the latest canary triggered a rollback.
    resp = c.post(
        "/api/v1/promotions/finalize",
        json={"genome_hash": best_hash, "experiment_id": "exp-finalize-test"},
        headers=headers,
    )
    assert resp.status_code == 400

    with session_scope() as session:
        save_canary_result(
            session,
            best_hash,
            "support-agent@v1",
            CanaryResult(
                baseline_metrics=agg,
                candidate_metrics=agg,
                traffic_split=0.1,
                n_baseline_requests=9,
                n_candidate_requests=1,
                rollback_triggered=False,
                reasons=["canary within thresholds"],
            ),
        )

    assert genome_status() == "CANARY"

    # A repeat approved review must not demote a genome whose canary already passed.
    with session_scope() as session:
        save_promotion_decision(
            session,
            best_hash,
            "exp-finalize-test",
            GateDecision(approved=True, next_status=PromotionStatus.APPROVED, reasons=["ok again"]),
        )
    assert genome_status() == "CANARY"

    # Viewer/operator keys cannot finalize even once the gates are satisfied — admin only.
    resp = c.post(
        "/api/v1/api-keys", json={"name": "ops", "role": "operator"}, headers=headers
    )
    operator_key = resp.json()["api_key"]
    resp = c.post(
        "/api/v1/promotions/finalize",
        json={"genome_hash": best_hash, "experiment_id": "exp-finalize-test"},
        headers={"X-API-Key": operator_key},
    )
    assert resp.status_code == 403

    resp = c.post(
        "/api/v1/promotions/finalize",
        json={"genome_hash": best_hash, "experiment_id": "exp-finalize-test"},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PROMOTED"
    assert body["approved_by"] == "bootstrap-admin"
    assert genome_status() == "PROMOTED"
    again = c.post(
        "/api/v1/promotions/finalize",
        json={"genome_hash": best_hash, "experiment_id": "exp-finalize-test"},
        headers=headers,
    )
    assert again.status_code == 409
    approvals_after = c.get("/api/v1/promotions/approvals", headers=headers).json()
    assert len([a for a in approvals_after if a["genome_hash"] == best_hash]) == 1

    # PROMOTED is terminal: a later canary or repeat review must not downgrade a live genome.
    with session_scope() as session:
        save_canary_result(
            session,
            best_hash,
            "support-agent@v1",
            CanaryResult(
                baseline_metrics=agg,
                candidate_metrics=agg,
                traffic_split=0.1,
                n_baseline_requests=9,
                n_candidate_requests=1,
                rollback_triggered=True,
                reasons=["late rollback"],
            ),
        )
    assert genome_status() == "PROMOTED"

    resp = c.get("/api/v1/promotions/approvals", headers=headers)
    assert resp.status_code == 200
    approvals = resp.json()
    assert any(a["genome_hash"] == best_hash and a["approved_by"] == "bootstrap-admin" for a in approvals)


def test_viewer_role_cannot_create_application(client):
    c, admin_key = client
    resp = c.post(
        "/api/v1/api-keys",
        json={"name": "read-only", "role": "viewer"},
        headers={"X-API-Key": admin_key},
    )
    assert resp.status_code == 200
    viewer_key = resp.json()["api_key"]

    resp = c.post(
        "/api/v1/applications",
        json={"id": "x", "name": "x", "domain": "forge-support"},
        headers={"X-API-Key": viewer_key},
    )
    assert resp.status_code == 403


def _create_and_run(c, headers, experiment_id: str, dataset_id: str, n: int, seed: int = 3) -> dict:
    c.post(
        "/api/v1/applications",
        json={"id": "support-agent", "name": "Forge Support", "domain": "forge-support"},
        headers=headers,
    ).raise_for_status()
    c.post(
        "/api/v1/datasets",
        json={"dataset_id": dataset_id, "domain": "forge-support", "n": n, "seed": seed},
        headers=headers,
    ).raise_for_status()
    c.post(
        "/api/v1/experiments",
        json={
            "experiment_id": experiment_id,
            "system_id": "support-agent",
            "dataset_id": dataset_id,
            "strategy": "random",
            "seed": seed,
            "batch_size": 8,
            "max_batches": 2,
            "budget": {
                "max_candidates": 16,
                "max_requests": 100000,
                "max_cost_usd": 100,
                "max_duration_minutes": 30,
            },
        },
        headers=headers,
    ).raise_for_status()
    resp = c.post(f"/api/v1/experiments/{experiment_id}/run", headers=headers)
    resp.raise_for_status()
    return resp.json()


def test_experiment_runs_against_the_dataset_it_was_created_with(client):
    """Regression: `dataset_id` was accepted at creation but ignored at run time, which always used
    `<application>-dataset` — so experiments silently ran on a different dataset than requested."""
    c, admin_key = client
    headers = {"X-API-Key": admin_key}
    result = _create_and_run(c, headers, "exp-alt-dataset", "a-different-dataset", n=60)
    assert result["dataset_id"] == "a-different-dataset"
    assert result["dataset_version"] == 1

    listed = next(e for e in c.get("/api/v1/experiments", headers=headers).json() if e["experiment_id"] == "exp-alt-dataset")
    assert listed["dataset_id"] == "a-different-dataset"


def test_promotion_review_measures_the_holdout_once_and_reuses_it(client):
    from neuroforge.db.models import GenomeRecord, HoldoutEvaluationRecord
    from neuroforge.db.session import session_scope

    c, admin_key = client
    headers = {"X-API-Key": admin_key}
    result = _create_and_run(c, headers, "exp-holdout-once", "support-agent-dataset", n=120)
    best_hash = next(e for e in c.get("/api/v1/experiments", headers=headers).json())["best_genome_hash"]

    body = {"genome_hash": best_hash, "experiment_id": "exp-holdout-once"}
    first = c.post("/api/v1/promotions", json=body, headers=headers)
    second = c.post("/api/v1/promotions", json=body, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["evidence"][0].startswith("holdout split of support-agent-dataset v1")

    with session_scope() as session:
        rows = session.query(HoldoutEvaluationRecord).filter_by(genome_hash=best_hash).all()
        status = session.get(GenomeRecord, best_hash).status
    assert len(rows) == 1
    assert rows[0].evidence["dataset_version"] == result["dataset_version"]
    assert status == ("APPROVED" if first.json()["approved"] else "REJECTED")

    # The Evolution Graph reads statuses through the lineage endpoint: it must show the pipeline's
    # current state, not the snapshot taken when the genome was first saved.
    lineage = c.get("/api/v1/genomes/support-agent/lineage", headers=headers).json()
    node = next(n for n in lineage["nodes"] if n["hash"] == best_hash)
    assert node["status"] == status

    recorded = c.get("/api/v1/promotions", headers=headers).json()
    assert all(p["evidence"] for p in recorded)


def test_promotion_request_without_a_finished_experiment_is_a_client_error(client):
    c, admin_key = client
    resp = c.post(
        "/api/v1/promotions",
        json={"genome_hash": "0" * 16, "experiment_id": "does-not-exist"},
        headers={"X-API-Key": admin_key},
    )
    assert resp.status_code == 400
