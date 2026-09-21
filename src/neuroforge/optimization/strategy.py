"""Common interface every search strategy implements, so the orchestrator (and the strategy
comparison feature in section 42) can swap one for another without touching calling code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    point: dict[str, Any]
    fitness: float


@dataclass
class ConvergenceState:
    converged: bool = False
    reason: str = ""
    best_fitness_history: list[float] = field(default_factory=list)


class SearchStrategy(ABC):
    name: str

    @abstractmethod
    def ask(self, n: int) -> list[dict[str, Any]]:
        """Propose up to n new points to evaluate."""

    @abstractmethod
    def tell(self, observations: list[Observation]) -> None:
        """Report fitness for previously-asked points so the strategy can adapt."""

    def replay_ask(self, n: int) -> None:
        """Advance ask-side state (RNG stream, cursor, pull counts) exactly as `ask(n)` did, when
        resuming from a checkpoint whose points are already known. Defaults to calling `ask`;
        strategies whose `ask` is expensive but whose state change is cheap override it."""
        self.ask(n)

    def convergence(self) -> ConvergenceState:
        return ConvergenceState()


def detect_plateau(history: list[float], patience: int = 3, min_delta: float = 0.005) -> bool:
    """Shared plateau detector: true if the best-so-far hasn't improved by min_delta over the
    last `patience` observations. Used for early stopping (section 19)."""
    if len(history) < patience + 1:
        return False
    recent_best = max(history[-patience:])
    prior_best = max(history[: -patience])
    return (recent_best - prior_best) < min_delta
