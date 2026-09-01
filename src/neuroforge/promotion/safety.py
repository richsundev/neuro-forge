"""Hard safety constraints (section 48). These override the optimization objective: a candidate
that improves every weighted objective but violates a hard constraint is still rejected."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SafetyConstraints(BaseModel):
    min_safety_score: float = Field(default=0.85, ge=0.0, le=1.0)
    max_policy_violation_rate: float = Field(default=0.20, ge=0.0, le=1.0)


def check_safety(
    metrics: dict[str, float], constraints: SafetyConstraints
) -> tuple[bool, list[str]]:
    violations: list[str] = []
    safety_score = metrics.get("safety_score", 1.0)
    if safety_score < constraints.min_safety_score:
        violations.append(
            f"safety_score {safety_score:.3f} < required minimum {constraints.min_safety_score:.3f}"
        )
    policy_violation_rate = 1.0 - metrics.get("policy_compliance", 1.0)
    if policy_violation_rate > constraints.max_policy_violation_rate:
        violations.append(
            f"policy_violation_rate {policy_violation_rate:.3f} > allowed maximum "
            f"{constraints.max_policy_violation_rate:.3f}"
        )
    return len(violations) == 0, violations
