from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from neuroforge.domains import DOMAIN_REGISTRY
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.optimization import STRATEGY_REGISTRY

# Identifiers become file names (an experiment's checkpoint, events and lock live under
# `<state_dir>/<experiment_id>/`), so they must not be able to carry path separators or `..`.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
Identifier = Annotated[str, StringConstraints(pattern=ID_PATTERN)]


def _known_domain(value: str) -> str:
    if value not in DOMAIN_REGISTRY:
        raise ValueError(f"unknown domain '{value}'; known: {sorted(DOMAIN_REGISTRY)}")
    return value


class ApplicationCreate(BaseModel):
    id: Identifier
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    domain: str = "forge-support"
    description: str = ""

    _domain = field_validator("domain")(_known_domain)


class DatasetSeedRequest(BaseModel):
    dataset_id: Identifier
    domain: str = "forge-support"
    # Bounded on both sides: a dataset needs enough challenges for its validation/holdout splits to
    # be usable, and an unbounded `n` let one request allocate arbitrary memory.
    n: int = Field(default=300, ge=20, le=5000)
    seed: int = 1

    _domain = field_validator("domain")(_known_domain)


class DatasetEvolveRequest(BaseModel):
    mean_score: float = Field(ge=0.0, le=1.0)
    domain: str = "forge-support"
    n_new: int = Field(default=30, ge=1, le=1000)
    seed: int = 2

    _domain = field_validator("domain")(_known_domain)


class ExperimentCreateRequest(BaseModel):
    experiment_id: Identifier
    system_id: Identifier
    domain: str = "forge-support"
    dataset_id: Identifier | None = None
    strategy: str = "evolutionary"
    seed: int = 42
    batch_size: int = Field(default=16, ge=1, le=500)
    max_batches: int = Field(default=20, ge=1, le=1000)
    budget: ExperimentBudget | None = None

    _domain = field_validator("domain")(_known_domain)

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in STRATEGY_REGISTRY:
            raise ValueError(f"unknown strategy '{value}'; known: {sorted(STRATEGY_REGISTRY)}")
        return value


class PromotionRequest(BaseModel):
    genome_hash: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    experiment_id: Identifier


class PromoteRequest(BaseModel):
    genome_hash: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    experiment_id: Identifier | None = None


class CanaryRequest(BaseModel):
    baseline_hash: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    candidate_hash: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    dataset_id: Identifier
    traffic_fraction: float = Field(default=0.10, gt=0.0, lt=1.0)
    n_requests: int = Field(default=200, ge=20, le=20000)
    # Defaults to a seed derived from the candidate hash, so a canary is deterministic per candidate.
    traffic_seed: int | None = None


class ApiKeyCreateRequest(BaseModel):
    name: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    role: Literal["viewer", "operator", "admin"] = "viewer"
