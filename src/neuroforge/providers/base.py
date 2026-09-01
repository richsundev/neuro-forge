"""Provider abstraction. The optimization/evaluation engines depend only on this interface —
swapping mock -> OpenAI-compatible -> Anthropic-compatible never touches search or eval code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from neuroforge.genomes.schema import ModelConfig


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_input: str
    context_documents: tuple[str, ...] = ()
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    capability_signal: float
    """A deterministic [0,1] proxy for "how good the underlying model+config combo is" at this
    request. Only the mock provider exposes it directly (real providers must have evaluators infer
    quality from `text` instead) — it exists so the mock backend can produce *meaningfully
    different* behavior across genomes without needing a real model in CI."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, model: ModelConfig, request: GenerationRequest) -> GenerationResult: ...
