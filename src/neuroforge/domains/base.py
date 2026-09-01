"""ApplicationDomain plugin interface. Every domain plugs the same genome + provider into
its own case generation, evaluation, and safety-constraint logic."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from neuroforge.genomes.schema import SystemGenome
from neuroforge.providers.base import LLMProvider


class Challenge(BaseModel):
    challenge_id: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    input: str
    expected_behavior: str
    evaluation_rules: list[str] = Field(default_factory=list)
    split: str = "train"  # train | validation | holdout


class EvaluationResult(BaseModel):
    challenge_id: str
    genome_hash: str
    metrics: dict[str, float]
    cost_usd: float
    latency_ms: float
    failed: bool
    notes: str = ""


class ApplicationDomain(ABC):
    """A pluggable task domain. NeuroForge's search/eval/promotion machinery is domain-agnostic;
    everything domain-specific (what a good answer looks like, what "unsafe" means) lives here."""

    name: str

    @abstractmethod
    def generate_cases(
        self, n: int, difficulty_range: tuple[float, float], seed: int
    ) -> list[Challenge]: ...

    @abstractmethod
    def evaluate(
        self, genome: SystemGenome, challenge: Challenge, provider: LLMProvider
    ) -> EvaluationResult: ...

    @abstractmethod
    def validate(self, challenge: Challenge) -> bool:
        """Sanity-check a generated challenge before it enters a dataset (well-formed, has a
        non-trivial expected behavior, difficulty is plausible)."""

    @abstractmethod
    def safety_constraints(self) -> dict[str, float]:
        """Hard constraints this domain enforces, e.g. {'safety_score': 0.85}. These override the
        optimization objective — see promotion.safety."""
