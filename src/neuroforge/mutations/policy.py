"""MutationPolicy: the safety boundary around what the mutation engine may touch."""

from __future__ import annotations

from pydantic import BaseModel, Field

DEFAULT_ALLOWED_FIELDS: tuple[str, ...] = (
    "prompt.system_prompt",
    "prompt.strategy",
    "prompt.include_examples",
    "prompt.max_instructions",
    "model.name",
    "model.temperature",
    "model.max_tokens",
    "model.reasoning_effort",
    "retrieval.chunk_size",
    "retrieval.chunk_overlap",
    "retrieval.top_k",
    "retrieval.reranker",
    "retrieval.query_transform",
    "tools.enabled",
    "tools.selection_policy",
    "tools.max_tool_calls",
    "context.compression",
    "context.deduplicate",
    "context.ordering",
    "agent.planning_strategy",
    "agent.max_iterations",
    "agent.stopping_criteria",
    "output.format",
)

DEFAULT_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "system_id",
    "security_policy",
    "authentication",
)


class MutationPolicy(BaseModel):
    """Bounds on what a single mutation pass may change.

    Never let the mutation engine touch fields outside `allowed`, and never allow more than
    `max_changes_per_candidate` field changes per generated candidate — this keeps every
    candidate a small, explainable delta from its parent instead of a random new system.
    """

    max_changes_per_candidate: int = Field(default=3, ge=1, le=10)
    allowed: tuple[str, ...] = DEFAULT_ALLOWED_FIELDS
    forbidden: tuple[str, ...] = DEFAULT_FORBIDDEN_FIELDS
    forbidden_combinations: tuple[tuple[str, str], ...] = (
        ("model.temperature", "output.format"),
    )

    def is_allowed(self, field_path: str) -> bool:
        if field_path in self.forbidden:
            return False
        return field_path in self.allowed

    def validate_change_set(self, field_paths: list[str]) -> tuple[bool, str]:
        if len(field_paths) > self.max_changes_per_candidate:
            return False, (
                f"candidate touches {len(field_paths)} fields, "
                f"policy allows at most {self.max_changes_per_candidate}"
            )
        for f in field_paths:
            if not self.is_allowed(f):
                return False, f"field '{f}' is not permitted by mutation policy"
        for a, b in self.forbidden_combinations:
            if a in field_paths and b in field_paths:
                return False, f"combination ({a}, {b}) is forbidden by mutation policy"
        return True, "ok"
