from __future__ import annotations

import itertools
from typing import Any

from neuroforge.optimization.search_space import SearchSpace
from neuroforge.optimization.strategy import ConvergenceState, Observation, SearchStrategy


class GridSearchStrategy(SearchStrategy):
    """Exhaustive (bounded) sweep over a discretized grid — good for small, bounded numeric
    search spaces where completeness matters more than sample efficiency. The grid is walked
    lazily: it is never materialized, so a large space costs time proportional to the candidates
    actually evaluated, not memory proportional to the whole grid."""

    name = "grid_search"

    def __init__(self, space: SearchSpace, steps: int = 4) -> None:
        self.space = space
        self._total = space.grid_size(steps)
        self._points = space.iter_grid(steps)
        self._cursor = 0
        self._history: list[float] = []

    def ask(self, n: int) -> list[dict[str, Any]]:
        batch = list(itertools.islice(self._points, max(0, n)))
        self._cursor += len(batch)
        return batch

    def tell(self, observations: list[Observation]) -> None:
        self._history.extend(o.fitness for o in observations)

    def exhausted(self) -> bool:
        return self._cursor >= self._total

    def convergence(self) -> ConvergenceState:
        return ConvergenceState(
            converged=self.exhausted(),
            reason="search space exhausted" if self.exhausted() else "",
            best_fitness_history=list(self._history),
        )
