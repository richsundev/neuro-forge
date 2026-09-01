"""NeuroForge REST API (section 46). Every endpoint here is a thin wrapper over the same
neuroforge.* engine/domain/db code the CLI uses — the API adds auth, HTTP shape, and OpenAPI docs,
nothing else. Two ways to execute an experiment: POST .../run executes synchronously in-request
(used by the CLI and CI); POST .../enqueue hands it to the Redis-queued neuroforge-experiment-
worker (scripts/worker.py) instead — see docs/adr/0010-redis-queue-not-celery.md."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from sqlalchemy import select

from neuroforge.datasets.evolution import evolve_if_saturated, seed_dataset
from neuroforge.db.models import ApiKeyRecord, ApplicationRecord, DatasetVersionRecord
from neuroforge.db.repository import (
    get_experiment,
    get_genome,
    latest_dataset_version,
    list_experiments,
    list_genomes_for_system,
    save_canary_result,
    save_dataset_version,
    save_experiment,
    save_genome,
    save_promotion_decision,
    upsert_application,
)
from neuroforge.db.session import init_db, session_scope
from neuroforge.domains import get_domain
from neuroforge.evaluation.aggregate import AggregateMetrics, aggregate
from neuroforge.evaluation.pareto import ParetoPoint, pareto_frontier
from neuroforge.evaluation.statistics import ComparisonResult, Conclusion, compare
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.checkpoint import ExperimentCheckpoint
from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
from neuroforge.experiments.events import EventLog
from neuroforge.experiments.queue import enqueue_experiment
from neuroforge.experiments.spaces import search_space_for_domain
from neuroforge.genomes.lineage import GenomeStore
from neuroforge.genomes.schema import SystemGenome
from neuroforge.observability import configure_tracing, get_tracer
from neuroforge.promotion.canary import simulate_canary
from neuroforge.promotion.gates import PromotionGateConfig, evaluate_promotion
from neuroforge.promotion.safety import SafetyConstraints
from neuroforge.providers.registry import get_provider
from neuroforge_api.auth import (
    Principal,
    audit,
    bootstrap_default_key,
    generate_api_key,
    hash_key,
    require_role,
)
from neuroforge_api.schemas import (
    ApiKeyCreateRequest,
    ApplicationCreate,
    CanaryRequest,
    DatasetEvolveRequest,
    DatasetSeedRequest,
    ExperimentCreateRequest,
    PromotionRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("neuroforge.api")

STATE_DIR_ENV = "NEUROFORGE_STATE_DIR"

METRICS_REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "neuroforge_api_requests_total", "API requests", ["method", "path", "status"], registry=METRICS_REGISTRY
)
REQUEST_LATENCY = Histogram(
    "neuroforge_api_request_latency_seconds", "API request latency", ["path"], registry=METRICS_REGISTRY
)

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_tracing("neuroforge-api")
    init_db()
    key = bootstrap_default_key()
    if key:
        logger.warning("Bootstrap admin API key created (store it now, shown only once): %s", key)
    yield


app = FastAPI(title="NeuroForge API", version="1.0.0", description=__doc__, lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _metrics_and_logging(request: Request, call_next):
    tracer = get_tracer("neuroforge.api")
    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}",
        attributes={"http.method": request.method, "http.path": request.url.path},
    ) as span:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        span.set_attribute("http.status_code", response.status_code)
    REQUEST_LATENCY.labels(path=request.url.path).observe(duration)
    REQUEST_COUNT.labels(method=request.method, path=request.url.path, status=response.status_code).inc()
    logger.info(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            }
        )
    )
    return response


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
def metrics() -> str:
    return generate_latest(METRICS_REGISTRY).decode()


def _state_dir():
    import os
    from pathlib import Path

    return Path(os.environ.get(STATE_DIR_ENV, "./neuroforge_state"))


def _resolve_genome(session, ident: str) -> SystemGenome:
    if "@v" in ident:
        system_id, version_str = ident.split("@v", 1)
        for r in list_genomes_for_system(session, system_id):
            if r.version == int(version_str):
                return SystemGenome.model_validate(r.data)
        raise HTTPException(404, f"no genome found for {ident}")
    genome = get_genome(session, ident)
    if genome is None:
        raise HTTPException(404, f"no genome found with hash {ident}")
    return genome


def _domain_for_dataset(dataset) -> str:
    prefix = dataset.challenges[0].challenge_id.split("-")[0]
    return {"fs": "forge-support", "sql": "sql-agent", "ra": "research-agent"}.get(prefix, "forge-support")


# --- applications ------------------------------------------------------


@app.post("/api/v1/applications", tags=["applications"])
def create_application(body: ApplicationCreate, principal: Principal = Depends(require_role("operator"))) -> dict:
    with session_scope() as session:
        upsert_application(session, body.id, body.name, body.domain, body.description)
    audit(principal.name, "create_application", {"id": body.id})
    return {"id": body.id, "created": True}


@app.get("/api/v1/applications", tags=["applications"])
def list_applications(principal: Principal = Depends(require_role("viewer"))) -> list[dict]:
    with session_scope() as session:
        rows = list(session.scalars(select(ApplicationRecord)))
        return [
            {"id": r.id, "name": r.name, "domain": r.domain_name, "created_at": r.created_at.isoformat()}
            for r in rows
        ]


# --- genomes -------------------------------------------------------------


@app.get("/api/v1/genomes/{ident}", tags=["genomes"])
def get_genome_endpoint(ident: str, principal: Principal = Depends(require_role("viewer"))) -> dict:
    with session_scope() as session:
        genome = _resolve_genome(session, ident)
    return genome.model_dump(mode="json")


@app.get("/api/v1/genomes", tags=["genomes"])
def list_genomes(system_id: str, principal: Principal = Depends(require_role("viewer"))) -> list[dict]:
    with session_scope() as session:
        rows = list_genomes_for_system(session, system_id)
        return [SystemGenome.model_validate(r.data).model_dump(mode="json") for r in rows]


@app.get("/api/v1/genomes/{system_id}/lineage", tags=["genomes"])
def genome_lineage(system_id: str, principal: Principal = Depends(require_role("viewer"))) -> dict:
    """Powers the Evolution Graph signature feature (section 62)."""
    with session_scope() as session:
        rows = list_genomes_for_system(session, system_id)
    store = GenomeStore()
    for r in sorted(rows, key=lambda r: r.version):
        store.add(SystemGenome.model_validate(r.data))
    return store.lineage_graph(system_id)


# --- datasets --------------------------------------------------------------


@app.post("/api/v1/datasets", tags=["datasets"])
def seed_dataset_endpoint(body: DatasetSeedRequest, principal: Principal = Depends(require_role("operator"))) -> dict:
    domain = get_domain(body.domain)
    dataset = seed_dataset(domain, body.dataset_id, n=body.n, seed=body.seed)
    with session_scope() as session:
        save_dataset_version(session, dataset)
    audit(principal.name, "seed_dataset", {"dataset_id": body.dataset_id})
    return dataset.summary()


@app.get("/api/v1/datasets", tags=["datasets"])
def list_datasets(principal: Principal = Depends(require_role("viewer"))) -> list[dict]:
    with session_scope() as session:
        rows = list(session.scalars(select(DatasetVersionRecord)))
        return [
            {
                "dataset_id": r.dataset_id,
                "version": r.version,
                "source": r.source,
                "n_challenges": r.n_challenges,
                "difficulty_score": r.difficulty_score,
            }
            for r in rows
        ]


@app.post("/api/v1/datasets/{dataset_id}/evolve", tags=["datasets"])
def evolve_dataset_endpoint(
    dataset_id: str, body: DatasetEvolveRequest, principal: Principal = Depends(require_role("operator"))
) -> dict:
    domain = get_domain(body.domain)
    with session_scope() as session:
        current = latest_dataset_version(session, dataset_id)
        if current is None:
            raise HTTPException(404, f"no dataset version found for '{dataset_id}'")
        evolved = evolve_if_saturated(domain, current, body.mean_score, body.n_new, body.seed)
        if evolved is None:
            return {"evolved": False, "reason": "benchmark not yet saturated"}
        save_dataset_version(session, evolved)
    audit(principal.name, "evolve_dataset", {"dataset_id": dataset_id, "new_version": evolved.version})
    return {"evolved": True, **evolved.summary()}


# --- experiments -----------------------------------------------------------


@app.post("/api/v1/experiments", tags=["experiments"])
def create_experiment(body: ExperimentCreateRequest, principal: Principal = Depends(require_role("operator"))) -> dict:
    dataset_id = body.dataset_id or f"{body.system_id}-dataset"
    domain = get_domain(body.domain)
    with session_scope() as session:
        upsert_application(session, body.system_id, body.system_id, body.domain)
        baseline = None
        for r in list_genomes_for_system(session, body.system_id):
            if r.version == 1:
                baseline = SystemGenome.model_validate(r.data)
        if baseline is None:
            baseline = SystemGenome(system_id=body.system_id, version=1)
            save_genome(session, baseline)
        dataset = latest_dataset_version(session, dataset_id)
        if dataset is None:
            dataset = seed_dataset(domain, dataset_id, n=100, seed=body.seed)
            save_dataset_version(session, dataset)

        config = ExperimentConfig(
            experiment_id=body.experiment_id,
            domain_name=body.domain,
            search_strategy=body.strategy,
            search_space=search_space_for_domain(body.domain),
            batch_size=body.batch_size,
            max_batches=body.max_batches,
            seed=body.seed,
            budget=body.budget or ExperimentBudget(max_candidates=200, max_requests=1_000_000, max_cost_usd=100, max_duration_minutes=60),
        )
        save_experiment(session, body.system_id, config, status="created")
    audit(principal.name, "create_experiment", {"experiment_id": body.experiment_id})
    return {"experiment_id": body.experiment_id, "created": True}


@app.get("/api/v1/experiments", tags=["experiments"])
def list_experiments_endpoint(principal: Principal = Depends(require_role("viewer"))) -> list[dict]:
    with session_scope() as session:
        rows = list_experiments(session)
        return [
            {
                "experiment_id": r.experiment_id,
                "domain": r.domain_name,
                "strategy": r.search_strategy,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


@app.get("/api/v1/experiments/{experiment_id}", tags=["experiments"])
def get_experiment_endpoint(experiment_id: str, principal: Principal = Depends(require_role("viewer"))) -> dict:
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise HTTPException(404, f"unknown experiment '{experiment_id}'")
        return {
            "experiment_id": record.experiment_id,
            "status": record.status,
            "config": record.config,
            "result": record.result,
        }


@app.post("/api/v1/experiments/{experiment_id}/run", tags=["experiments"])
def run_experiment_endpoint(experiment_id: str, principal: Principal = Depends(require_role("operator"))) -> dict:
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise HTTPException(404, f"unknown experiment '{experiment_id}'")
        config = ExperimentConfig.model_validate(record.config)
        baseline = None
        for r in list_genomes_for_system(session, record.application_id):
            if r.version == 1:
                baseline = SystemGenome.model_validate(r.data)
        dataset = latest_dataset_version(session, f"{record.application_id}-dataset")
        application_id = record.application_id

    if baseline is None or dataset is None:
        raise HTTPException(400, "experiment is missing its baseline genome or dataset")

    engine = ExperimentEngine(config, baseline, dataset, _state_dir() / experiment_id)
    result = engine.run()

    with session_scope() as session:
        save_genome(session, SystemGenome.model_validate(result.best_genome))
        save_experiment(session, application_id, config, result=result, status=result.status)
    audit(principal.name, "run_experiment", {"experiment_id": experiment_id, "status": result.status})
    return result.model_dump(mode="json")


@app.post("/api/v1/experiments/{experiment_id}/cancel", tags=["experiments"])
def cancel_experiment_endpoint(experiment_id: str, principal: Principal = Depends(require_role("operator"))) -> dict:
    flag = _state_dir() / experiment_id / f"{experiment_id}.cancel"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    audit(principal.name, "cancel_experiment", {"experiment_id": experiment_id})
    return {"cancelled": True}


@app.post("/api/v1/experiments/{experiment_id}/enqueue", tags=["experiments"])
def enqueue_experiment_endpoint(experiment_id: str, principal: Principal = Depends(require_role("operator"))) -> dict:
    """Hand the experiment to the neuroforge-experiment-worker via Redis instead of running it
    synchronously in this request — the async path used by the dashboard/production deployments."""
    with session_scope() as session:
        record = get_experiment(session, experiment_id)
        if record is None:
            raise HTTPException(404, f"unknown experiment '{experiment_id}'")
        config = ExperimentConfig.model_validate(record.config)
        save_experiment(session, record.application_id, config, status="queued")
    enqueue_experiment(experiment_id)
    audit(principal.name, "enqueue_experiment", {"experiment_id": experiment_id})
    return {"queued": True}


@app.get("/api/v1/experiments/{experiment_id}/checkpoint", tags=["experiments"])
def experiment_checkpoint_endpoint(experiment_id: str, principal: Principal = Depends(require_role("viewer"))) -> dict:
    ckpt = ExperimentCheckpoint.load_or_none(_state_dir() / experiment_id / f"{experiment_id}.checkpoint.json")
    if ckpt is None:
        raise HTTPException(404, "no checkpoint yet — has this experiment been run?")
    return {
        "status": ckpt.status,
        "stop_reason": ckpt.stop_reason,
        "generations_completed": ckpt.generations_completed(),
        "candidates_completed": ckpt.candidates_completed(),
        "best_fitness": ckpt.best_fitness,
        "budget_state": ckpt.budget_state,
        "fitness_history": [f for b in ckpt.tell_batches for f in b.fitness],
    }


@app.get("/api/v1/experiments/{experiment_id}/candidates", tags=["experiments"])
def experiment_candidates(experiment_id: str, principal: Principal = Depends(require_role("viewer"))) -> dict:
    """Every evaluated candidate this run, with a quality-vs-cost Pareto frontier flag — powers
    the Pareto frontier chart (section 6/40)."""
    log_path = _state_dir() / experiment_id / f"{experiment_id}.events.jsonl"
    if not log_path.exists():
        raise HTTPException(404, "no events recorded yet for this experiment")
    events = EventLog(log_path).of_type("CandidateEvaluated")
    points = [
        ParetoPoint(
            candidate_id=f"v{e.payload['version']}",
            metrics={
                "quality": e.payload["metrics"].get("quality", 0.0),
                "cost_usd": e.payload["metrics"].get("cost_usd", 0.0),
            },
        )
        for e in events
    ]
    pareto = {r.candidate_id: r for r in pareto_frontier(points, {"quality": "maximize", "cost_usd": "minimize"})}
    return {
        "candidates": [
            {
                "genome_hash": e.payload["genome_hash"],
                "version": e.payload["version"],
                "generation": e.payload["generation"],
                "fitness": e.payload["fitness"],
                "metrics": e.payload["metrics"],
                "is_pareto_optimal": pareto[f"v{e.payload['version']}"].is_pareto_optimal,
            }
            for e in events
        ]
    }


# --- candidates --------------------------------------------------------


@app.get("/api/v1/candidates/compare", tags=["candidates"])
def compare_candidates(
    genome_a: str, genome_b: str, dataset_id: str, principal: Principal = Depends(require_role("viewer"))
) -> dict:
    with session_scope() as session:
        a = _resolve_genome(session, genome_a)
        b = _resolve_genome(session, genome_b)
        dataset = latest_dataset_version(session, dataset_id)
        if dataset is None:
            raise HTTPException(404, f"unknown dataset '{dataset_id}'")

    domain = get_domain(_domain_for_dataset(dataset))
    provider = get_provider("mock")
    challenges = dataset.split("validation") or dataset.split("train")

    a_results = [domain.evaluate(a, c, provider) for c in challenges]
    b_results = [domain.evaluate(b, c, provider) for c in challenges]
    a_agg = aggregate(a_results, [c.category for c in challenges])
    b_agg = aggregate(b_results, [c.category for c in challenges])
    comparison = compare([r.metrics["quality"] for r in a_results], [r.metrics["quality"] for r in b_results])

    return {
        "genome_a": a_agg.as_dict(),
        "genome_b": b_agg.as_dict(),
        "comparison": _comparison_dict(comparison),
    }


# --- promotions ----------------------------------------------------------


@app.post("/api/v1/promotions", tags=["promotions"])
def request_promotion(body: PromotionRequest, principal: Principal = Depends(require_role("operator"))) -> dict:
    with session_scope() as session:
        record = get_experiment(session, body.experiment_id)
        if record is None or record.result is None:
            raise HTTPException(400, f"experiment '{body.experiment_id}' has no completed result yet")
        result = record.result
        baseline_agg = AggregateMetrics(genome_hash="baseline", n_evaluations=0, metrics=result["baseline_metrics"])
        candidate_agg = AggregateMetrics(genome_hash=body.genome_hash, n_evaluations=0, metrics=result["best_metrics"])
        comparison = ComparisonResult(
            mean_diff=result["comparison"]["mean_diff"],
            relative_diff=result["comparison"]["relative_diff"],
            ci_low=result["comparison"]["ci_low"],
            ci_high=result["comparison"]["ci_high"],
            effect_size=result["comparison"]["effect_size"],
            confidence=result["comparison"]["confidence"],
            conclusion=Conclusion(result["comparison"]["conclusion"]),
        )
        decision = evaluate_promotion(baseline_agg, candidate_agg, comparison, PromotionGateConfig(), SafetyConstraints())
        save_promotion_decision(session, body.genome_hash, body.experiment_id, decision)
    audit(principal.name, "request_promotion", {"genome_hash": body.genome_hash, "approved": decision.approved})
    return decision.model_dump(mode="json")


# --- canaries --------------------------------------------------------------


@app.post("/api/v1/canaries", tags=["canaries"])
def run_canary(body: CanaryRequest, principal: Principal = Depends(require_role("operator"))) -> dict:
    with session_scope() as session:
        baseline = _resolve_genome(session, body.baseline_hash)
        candidate = _resolve_genome(session, body.candidate_hash)
        dataset = latest_dataset_version(session, body.dataset_id)
        if dataset is None:
            raise HTTPException(404, f"unknown dataset '{body.dataset_id}'")
        domain = get_domain(_domain_for_dataset(dataset))
        provider = get_provider("mock")
        traffic = (dataset.split("validation") + dataset.split("train"))[: body.n_requests]
        result = simulate_canary(domain, provider, baseline, candidate, traffic, body.traffic_fraction)
        save_canary_result(session, candidate.hash(), baseline.hash(), result)
    audit(principal.name, "run_canary", {"candidate_hash": body.candidate_hash, "rollback": result.rollback_triggered})
    return {
        "baseline_metrics": result.baseline_metrics.as_dict(),
        "candidate_metrics": result.candidate_metrics.as_dict(),
        "rollback_triggered": result.rollback_triggered,
        "reasons": result.reasons,
    }


# --- api keys (admin only) --------------------------------------------


@app.post("/api/v1/api-keys", tags=["admin"])
def create_api_key(body: ApiKeyCreateRequest, principal: Principal = Depends(require_role("admin"))) -> dict:
    raw_key = generate_api_key()
    with session_scope() as session:
        session.add(ApiKeyRecord(name=body.name, key_hash=hash_key(raw_key), role=body.role))
    audit(principal.name, "create_api_key", {"name": body.name, "role": body.role})
    return {"name": body.name, "role": body.role, "api_key": raw_key}


def _comparison_dict(comparison: ComparisonResult) -> dict:
    return {
        "mean_diff": comparison.mean_diff,
        "relative_diff": comparison.relative_diff,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "effect_size": comparison.effect_size,
        "confidence": comparison.confidence,
        "conclusion": comparison.conclusion.value,
        "summary": comparison.summary(),
    }


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
