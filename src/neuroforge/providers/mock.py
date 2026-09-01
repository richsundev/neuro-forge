"""Deterministic mock LLM provider.

Real evaluation science requires a backend whose behavior *actually changes* with configuration
so the optimizer has real signal to search over, while remaining reproducible without paid APIs
(section 28/64 of the project brief: no fabricated improvements). This provider derives every
output from a stable hash of (seed, prompt content, model config) — same input, same output,
forever — while genuinely varying quality/cost/latency across models, temperatures, and effort
levels so different SystemGenomes are measurably different in evaluation.
"""

from __future__ import annotations

from neuroforge.genomes.schema import ModelConfig
from neuroforge.providers.base import GenerationRequest, GenerationResult, LLMProvider
from neuroforge.util import stable_rng


class ModelProfile:
    def __init__(
        self, capability: float, cost_per_1k_tokens: float, latency_ms: float, precision: float
    ) -> None:
        self.capability = capability
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.latency_ms = latency_ms
        self.precision = precision  # resistance to temperature-induced errors


MODEL_TABLE: dict[str, ModelProfile] = {
    "reasoning-large": ModelProfile(
        capability=0.90, cost_per_1k_tokens=0.030, latency_ms=1600, precision=0.9
    ),
    "reasoning-small": ModelProfile(
        capability=0.78, cost_per_1k_tokens=0.008, latency_ms=700, precision=0.75
    ),
    "fast-cheap": ModelProfile(
        capability=0.62, cost_per_1k_tokens=0.002, latency_ms=300, precision=0.55
    ),
}

EFFORT_MULTIPLIER = {"low": 0.6, "medium": 1.0, "high": 1.6}
EFFORT_BONUS = {"low": -0.03, "medium": 0.0, "high": 0.05}


class MockLLMProvider(LLMProvider):
    name = "mock"

    def generate(self, model: ModelConfig, request: GenerationRequest) -> GenerationResult:
        profile = MODEL_TABLE.get(model.name, MODEL_TABLE["reasoning-small"])
        rng = stable_rng(
            request.seed,
            model.name,
            model.temperature,
            model.reasoning_effort,
            request.system_prompt,
            request.user_input,
            len(request.context_documents),
        )

        temperature_penalty = model.temperature * (1.0 - profile.precision) * 0.25
        effort_bonus = EFFORT_BONUS[model.reasoning_effort]
        jitter = rng.uniform(-0.03, 0.03)
        capability = profile.capability + effort_bonus - temperature_penalty + jitter
        capability = min(1.0, max(0.0, capability))

        tokens_in = len(request.system_prompt.split()) + len(request.user_input.split())
        tokens_in += sum(len(d.split()) for d in request.context_documents)
        tokens_out = min(model.max_tokens, int(40 + capability * 120 + rng.uniform(-10, 10)))
        tokens_out = max(1, tokens_out)

        latency_ms = (
            profile.latency_ms
            * EFFORT_MULTIPLIER[model.reasoning_effort]
            * (1 + rng.uniform(-0.08, 0.08))
            * (1 + 0.02 * len(request.context_documents))
        )
        cost_usd = (tokens_in + tokens_out) / 1000.0 * profile.cost_per_1k_tokens

        text = (
            f"[{model.name}|t={model.temperature}] "
            f"response to: {request.user_input[:80]!r} "
            f"(context_docs={len(request.context_documents)})"
        )

        return GenerationResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6),
            capability_signal=round(capability, 4),
        )
