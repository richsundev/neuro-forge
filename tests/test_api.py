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

    import neuroforge_api.main as api_main

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
