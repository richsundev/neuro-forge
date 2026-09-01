"""Resource-aware experimentation (section 18): the optimizer must physically stop when the
budget runs out, not just report that it "should have"."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class ExperimentBudget(BaseModel):
    max_candidates: int = Field(default=50, ge=1)
    max_requests: int = Field(default=5000, ge=1)
    max_cost_usd: float = Field(default=10.0, ge=0.0)
    max_duration_minutes: float = Field(default=30.0, ge=0.0)


@dataclass
class BudgetTracker:
    budget: ExperimentBudget
    candidates_used: int = 0
    requests_used: int = 0
    cost_used: float = 0.0
    prior_elapsed_minutes: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    def record(self, candidates: int, requests: int, cost: float) -> None:
        self.candidates_used += candidates
        self.requests_used += requests
        self.cost_used += cost

    def elapsed_minutes(self) -> float:
        return self.prior_elapsed_minutes + (time.monotonic() - self._start_time) / 60.0

    def exhausted(self) -> tuple[bool, str]:
        if self.candidates_used >= self.budget.max_candidates:
            return True, f"max_candidates ({self.budget.max_candidates}) reached"
        if self.requests_used >= self.budget.max_requests:
            return True, f"max_requests ({self.budget.max_requests}) reached"
        if self.cost_used >= self.budget.max_cost_usd:
            return True, f"max_cost_usd ({self.budget.max_cost_usd}) reached"
        if self.elapsed_minutes() >= self.budget.max_duration_minutes:
            return True, f"max_duration_minutes ({self.budget.max_duration_minutes}) reached"
        return False, ""

    def utilization(self) -> dict[str, float]:
        return {
            "candidates": self.candidates_used / self.budget.max_candidates,
            "requests": self.requests_used / self.budget.max_requests,
            "cost": self.cost_used / self.budget.max_cost_usd if self.budget.max_cost_usd else 0.0,
            "duration": self.elapsed_minutes() / self.budget.max_duration_minutes
            if self.budget.max_duration_minutes
            else 0.0,
        }

    def to_state(self) -> dict[str, float]:
        return {
            "candidates_used": self.candidates_used,
            "requests_used": self.requests_used,
            "cost_used": self.cost_used,
            "elapsed_minutes": self.elapsed_minutes(),
        }

    @classmethod
    def from_state(cls, budget: ExperimentBudget, state: dict[str, float]) -> BudgetTracker:
        tracker = cls(budget=budget)
        tracker.candidates_used = int(state.get("candidates_used", 0))
        tracker.requests_used = int(state.get("requests_used", 0))
        tracker.cost_used = float(state.get("cost_used", 0.0))
        tracker.prior_elapsed_minutes = float(state.get("elapsed_minutes", 0.0))
        return tracker
