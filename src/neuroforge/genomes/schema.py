"""SystemGenome: the versioned, hashable configuration of an LLM application."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["mock", "openai_compatible", "anthropic_compatible", "local"] = "mock"
    name: str = "reasoning-small"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32000)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"


class PromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str = "You are a helpful assistant."
    strategy: Literal[
        "direct",
        "structured_reasoning",
        "chain_of_thought",
        "few_shot",
        "constrained",
    ] = "direct"
    include_examples: bool = False
    max_instructions: int = Field(default=6, ge=1, le=30)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=600, ge=50, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    reranker: Literal["none", "cross_encoder", "mmr"] = "none"
    query_transform: Literal["none", "expansion", "hyde"] = "none"


class ToolsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: tuple[str, ...] = ("search",)
    selection_policy: Literal["greedy", "risk_aware", "conservative"] = "greedy"
    max_tool_calls: int = Field(default=3, ge=0, le=20)


class ContextConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    compression: Literal["none", "summarize", "truncate"] = "none"
    deduplicate: bool = False
    ordering: Literal["relevance", "recency", "source_priority"] = "relevance"


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    planning_strategy: Literal["single_shot", "react", "plan_and_execute"] = "single_shot"
    max_iterations: int = Field(default=1, ge=1, le=20)
    stopping_criteria: Literal["first_answer", "confidence_threshold", "max_iterations"] = (
        "first_answer"
    )


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["free_text", "json_schema", "markdown"] = "free_text"


class MutationRecord(BaseModel):
    """A single mutation applied to produce this genome from its parent."""

    model_config = ConfigDict(frozen=True)

    mutation_type: str
    field_path: str
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    generator_version: str = "1.0.0"


class PromotionStatus(str, Enum):
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    BENCHMARKED = "BENCHMARKED"
    HOLDOUT_TESTED = "HOLDOUT_TESTED"
    APPROVED = "APPROVED"
    CANARY = "CANARY"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    # Was in production; replaced by a newer champion (still restorable by a rollback).
    SUPERSEDED = "SUPERSEDED"


class SystemGenome(BaseModel):
    """Immutable, content-addressed configuration of an LLM application version."""

    model_config = ConfigDict(frozen=True)

    system_id: str
    version: int
    parent_hash: str | None = None
    mutations: tuple[MutationRecord, ...] = ()
    generation: int = 0
    status: PromotionStatus = PromotionStatus.GENERATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model: ModelConfig = Field(default_factory=ModelConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def content_dict(self) -> dict[str, Any]:
        """Fields that participate in the content hash (identity), excluding metadata."""
        return {
            "system_id": self.system_id,
            "model": self.model.model_dump(),
            "prompt": self.prompt.model_dump(),
            "retrieval": self.retrieval.model_dump(),
            "tools": self.tools.model_dump(),
            "context": self.context.model_dump(),
            "agent": self.agent.model_dump(),
            "output": self.output.model_dump(),
        }

    def hash(self) -> str:
        """Deterministic content hash — identical configuration always hashes identically."""
        payload = json.dumps(self.content_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def short_id(self) -> str:
        return f"{self.system_id}@v{self.version}#{self.hash()[:8]}"

    def derive(
        self,
        *,
        mutations: list[MutationRecord],
        overrides: dict[str, Any],
        new_version: int,
        generation: int | None = None,
    ) -> SystemGenome:
        """Produce a new immutable child genome with the given field overrides applied."""
        data = self.model_dump()
        for key in ("version", "parent_hash", "mutations", "created_at", "status", "generation"):
            data.pop(key, None)
        _deep_set(data, overrides)
        return SystemGenome(
            **data,
            version=new_version,
            parent_hash=self.hash(),
            mutations=tuple(mutations),
            generation=generation if generation is not None else self.generation + 1,
            status=PromotionStatus.GENERATED,
        )


def _deep_set(data: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply dotted-path overrides like {'retrieval.top_k': 7} onto a nested dict."""
    for path, value in overrides.items():
        parts = path.split(".")
        cursor = data
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
