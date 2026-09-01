"""Bandit strategy (section 9): UCB1 over a fixed, bounded set of discrete arms, for problems
where you want explicit explore/exploit control rather than a global surrogate model."""

from __future__ import annotations

import math
import random
from typing import Any

from neuroforge.optimization.search_space import SearchSpace
from neuroforge.optimization.strategy import (
    ConvergenceState,
    Observation,
    SearchStrategy,
    detect_plateau,
)


def _hashable_key(point: dict[str, Any]) -> tuple:
    """Categorical values can themselves be lists (e.g. genome tools.enabled) — make them hashable
    so arms can be deduplicated by value."""
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in point.items()))


class BanditSearchStrategy(SearchStrategy):
    """Builds a bounded pool of candidate arms and plays UCB1 over them.

    A full grid (every combination of every parameter's discretized values) is the textbook
    description of "discretize the space," but it's combinatorially explosive for a realistic
    genome search space (12+ fields, even 3 values each is 3^12 ≈ 500K+ arms) — so instead we
    sample a fixed-size pool of `n_arms` distinct points, which keeps genuine bandit semantics
    (a small number of arms, each pulled and scored repeatedly) at any dimensionality.
    """

    name = "bandit_ucb1"

    def __init__(self, space: SearchSpace, seed: int = 0, n_arms: int = 24) -> None:
        self.space = space
        self._rng = random.Random(seed)
        self.arms = self._build_arms(n_arms)
        self._pulls = [0] * len(self.arms)
        self._reward_sum = [0.0] * len(self.arms)
        self._total_pulls = 0
        self._round_best: list[float] = []
        self._last_batch: list[int] = []

    def _build_arms(self, n_arms: int) -> list[dict[str, Any]]:
        arms: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        attempts = 0
        while len(arms) < n_arms and attempts < n_arms * 50:
            attempts += 1
            point = self.space.sample(self._rng)
            key = _hashable_key(point)
            if key in seen:
                continue
            seen.add(key)
            arms.append(point)
        return arms

    def ask(self, n: int) -> list[dict[str, Any]]:
        chosen_idxs = []
        for _ in range(n):
            unplayed = [i for i in range(len(self.arms)) if self._pulls[i] == 0]
            if unplayed:
                idx = self._rng.choice(unplayed)
            else:
                idx = max(range(len(self.arms)), key=self._ucb_score)
            chosen_idxs.append(idx)
            self._pulls[idx] += 1  # reserve optimistically until `tell` reconciles
        self._last_batch = chosen_idxs
        return [self.arms[i] for i in chosen_idxs]

    def _ucb_score(self, idx: int) -> float:
        if self._pulls[idx] == 0:
            return float("inf")
        mean = self._reward_sum[idx] / self._pulls[idx]
        bonus = math.sqrt(2 * math.log(max(1, self._total_pulls)) / self._pulls[idx])
        return mean + bonus

    def tell(self, observations: list[Observation]) -> None:
        for obs, idx in zip(observations, self._last_batch, strict=False):
            self._reward_sum[idx] += obs.fitness
            self._total_pulls += 1
        if observations:
            self._round_best.append(max(o.fitness for o in observations))

    def convergence(self) -> ConvergenceState:
        # Judged on best-per-round (see random_search.py) for a fair cross-strategy comparison.
        converged = detect_plateau(self._round_best, patience=5, min_delta=0.003)
        return ConvergenceState(
            converged=converged,
            reason="reward plateaued across rounds" if converged else "",
            best_fitness_history=list(self._round_best),
        )
