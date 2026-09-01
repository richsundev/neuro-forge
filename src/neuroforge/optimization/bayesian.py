"""Lightweight Bayesian optimization (section 9): a from-scratch Gaussian Process surrogate with
an Expected Improvement acquisition function, built on plain NumPy so the platform doesn't need a
heavyweight optimizer dependency for something this small. Categorical and numeric parameters are
both encoded via `SearchSpace.encode` (min-max scaling + one-hot) before entering the GP kernel.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy.stats import norm

from neuroforge.optimization.search_space import SearchSpace
from neuroforge.optimization.strategy import (
    ConvergenceState,
    Observation,
    SearchStrategy,
    detect_plateau,
)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, length_scale: float = 0.6) -> np.ndarray:
    dists = np.sum(x**2, axis=1)[:, None] + np.sum(y**2, axis=1)[None, :] - 2 * x @ y.T
    return np.exp(-0.5 * np.maximum(dists, 0) / length_scale**2)


class BayesianSearchStrategy(SearchStrategy):
    """Sequential model-based optimization: fit a GP on observed (point, fitness) pairs, then pick
    the untried candidate with the highest Expected Improvement over the current best."""

    name = "bayesian_optimization"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        n_initial_random: int = 4,
        candidate_pool_size: int = 200,
        noise: float = 1e-4,
    ) -> None:
        self.space = space
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.n_initial_random = n_initial_random
        self.candidate_pool_size = candidate_pool_size
        self.noise = noise
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._points: list[dict[str, Any]] = []
        self._round_best: list[float] = []

    def ask(self, n: int) -> list[dict[str, Any]]:
        if len(self._points) < self.n_initial_random:
            return [self.space.sample(self._rng) for _ in range(n)]

        pool = [self.space.sample(self._rng) for _ in range(self.candidate_pool_size)]
        X_train = np.vstack(self._X)
        y_train = np.array(self._y)
        best_y = y_train.max()

        mu, std = self._posterior(X_train, y_train, [self.space.encode(p) for p in pool])
        with np.errstate(divide="ignore"):
            z = (mu - best_y) / np.where(std > 1e-9, std, 1e-9)
            ei = (mu - best_y) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std > 1e-9, ei, 0.0)

        order = np.argsort(ei)[::-1]
        chosen: list[dict[str, Any]] = []
        seen = set()
        for idx in order:
            # Categorical values can themselves be lists (e.g. tools.enabled) — make them hashable.
            key = tuple(
                sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in pool[idx].items())
            )
            if key in seen:
                continue
            seen.add(key)
            chosen.append(pool[idx])
            if len(chosen) >= n:
                break
        return chosen

    def _posterior(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        Xt = np.vstack(X_test)
        K = _rbf_kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        K_s = _rbf_kernel(X_train, Xt)
        K_ss = _rbf_kernel(Xt, Xt)
        K_inv = np.linalg.pinv(K)
        mu = K_s.T @ K_inv @ y_train
        cov = K_ss - K_s.T @ K_inv @ K_s
        std = np.sqrt(np.maximum(np.diag(cov), 0))
        return mu, std

    def tell(self, observations: list[Observation]) -> None:
        for obs in observations:
            self._X.append(self.space.encode(obs.point))
            self._y.append(obs.fitness)
            self._points.append(obs.point)
        self._round_best.append(max(o.fitness for o in observations))

    def convergence(self) -> ConvergenceState:
        # Judged on best-per-round (see random_search.py) so a strategy comparison at equal
        # budgets stops all strategies on a comparable basis rather than random-search-style noise.
        converged = detect_plateau(self._round_best, patience=5, min_delta=0.003)
        return ConvergenceState(
            converged=converged,
            reason="GP expected-improvement plateaued across rounds" if converged else "",
            best_fitness_history=list(self._round_best),
        )
