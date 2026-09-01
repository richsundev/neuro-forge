from __future__ import annotations

import random
from typing import Any

from neuroforge.optimization.search_space import SearchSpace
from neuroforge.optimization.strategy import (
    ConvergenceState,
    Observation,
    SearchStrategy,
    detect_plateau,
)


class RandomSearchStrategy(SearchStrategy):
    """Baseline: independent uniform samples from the search space."""

    name = "random_search"

    def __init__(self, space: SearchSpace, seed: int = 0) -> None:
        self.space = space
        self._rng = random.Random(seed)
        self._round_best: list[float] = []

    def ask(self, n: int) -> list[dict[str, Any]]:
        return [self.space.sample(self._rng) for _ in range(n)]

    def tell(self, observations: list[Observation]) -> None:
        # Track best-of-round (not every individual sample) so convergence is judged on the same
        # "is another batch still buying us anything" basis as the evolutionary/bandit strategies,
        # instead of being noise-sensitive to single random draws within one batch.
        self._round_best.append(max(o.fitness for o in observations))

    def convergence(self) -> ConvergenceState:
        converged = detect_plateau(self._round_best)
        return ConvergenceState(
            converged=converged,
            reason="fitness plateau detected across rounds" if converged else "",
            best_fitness_history=list(self._round_best),
        )
