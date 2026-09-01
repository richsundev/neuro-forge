from __future__ import annotations

from pydantic import BaseModel

from neuroforge.experiments.budget import ExperimentBudget


class ApplicationCreate(BaseModel):
    id: str
    name: str
    domain: str = "forge-support"
    description: str = ""


class DatasetSeedRequest(BaseModel):
    dataset_id: str
    domain: str = "forge-support"
    n: int = 100
    seed: int = 1


class DatasetEvolveRequest(BaseModel):
    mean_score: float
    domain: str = "forge-support"
    n_new: int = 30
    seed: int = 2


class ExperimentCreateRequest(BaseModel):
    experiment_id: str
    system_id: str
    domain: str = "forge-support"
    dataset_id: str | None = None
    strategy: str = "evolutionary"
    seed: int = 42
    batch_size: int = 16
    max_batches: int = 20
    budget: ExperimentBudget | None = None


class PromotionRequest(BaseModel):
    genome_hash: str
    experiment_id: str


class CanaryRequest(BaseModel):
    baseline_hash: str
    candidate_hash: str
    dataset_id: str
    traffic_fraction: float = 0.10
    n_requests: int = 200


class ApiKeyCreateRequest(BaseModel):
    name: str
    role: str = "viewer"
