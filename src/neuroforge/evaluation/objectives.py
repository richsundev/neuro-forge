"""Multi-objective scoring: combine several metrics (some maximized, some minimized) into a
single scalar fitness the search strategies can compare candidates by."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

Direction = Literal["maximize", "minimize"]


class Objective(BaseModel):
    direction: Direction
    weight: float = Field(ge=0.0, le=1.0)


class ObjectiveSpec(BaseModel):
    """A named set of weighted objectives, e.g. the example in section 5 of the project brief."""

    objectives: dict[str, Objective]

    @field_validator("objectives")
    @classmethod
    def _weights_sum_to_one(cls, v: dict[str, Objective]) -> dict[str, Objective]:
        total = sum(o.weight for o in v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"objective weights must sum to ~1.0, got {total}")
        return v

    # Metrics that are naturally in [0, 1] as maximize-good scores.
    QUALITY_LIKE: ClassVar[set[str]] = {
        "quality",
        "faithfulness",
        "task_success",
        "tool_success",
        "safety_score",
        "policy_compliance",
    }

    def score(self, metrics: dict[str, float], normalizers: dict[str, float] | None = None) -> float:
        """Weighted scalar fitness. `normalizers` gives a reference scale for cost/latency-like
        metrics that aren't already in [0, 1] (defaults keep them roughly comparable)."""
        normalizers = normalizers or {"cost_usd": 0.05, "latency_ms": 3000.0, "failure_rate": 1.0}
        total = 0.0
        for name, obj in self.objectives.items():
            raw = metrics.get(name, 0.0)
            if name in self.QUALITY_LIKE:
                goodness = raw if obj.direction == "maximize" else (1.0 - raw)
            else:
                scale = normalizers.get(name, 1.0) or 1.0
                resource_used = min(1.0, raw / scale)
                goodness = (1.0 - resource_used) if obj.direction == "minimize" else resource_used
            total += obj.weight * goodness
        return round(total, 6)


DEFAULT_OBJECTIVES = ObjectiveSpec(
    objectives={
        "quality": Objective(direction="maximize", weight=0.25),
        "task_success": Objective(direction="maximize", weight=0.20),
        "policy_compliance": Objective(direction="maximize", weight=0.20),
        "safety_score": Objective(direction="maximize", weight=0.15),
        "latency_ms": Objective(direction="minimize", weight=0.10),
        "cost_usd": Objective(direction="minimize", weight=0.05),
        "failure_rate": Objective(direction="minimize", weight=0.05),
    }
)
