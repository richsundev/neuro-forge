"""Evolutionary search (section 10): population, fitness, tournament selection, crossover,
mutation, and elitism, with each generation persisted so the Evolution Graph and generation
history views can render real, executed generations rather than a synthetic animation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from neuroforge.optimization.search_space import SearchSpace
from neuroforge.optimization.strategy import (
    ConvergenceState,
    Observation,
    SearchStrategy,
    detect_plateau,
)


@dataclass
class Generation:
    index: int
    population: list[dict[str, Any]]
    fitness: list[float]

    def best(self) -> tuple[dict[str, Any], float]:
        best_i = max(range(len(self.fitness)), key=lambda i: self.fitness[i])
        return self.population[best_i], self.fitness[best_i]

    def mean_fitness(self) -> float:
        return sum(self.fitness) / len(self.fitness)


@dataclass
class EvolutionarySearchStrategy(SearchStrategy):
    space: SearchSpace
    population_size: int = 12
    elite_fraction: float = 0.2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    seed: int = 0

    name: str = field(default="evolutionary_search", init=False)
    generations: list[Generation] = field(default_factory=list, init=False)
    _rng: random.Random = field(init=False, repr=False)
    _pending: list[dict[str, Any]] = field(default_factory=list, init=False)
    _current_gen_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._pending = [self.space.sample(self._rng) for _ in range(self.population_size)]

    def ask(self, n: int) -> list[dict[str, Any]]:
        return self._pending[:n]

    def tell(self, observations: list[Observation]) -> None:
        population = [o.point for o in observations]
        fitness = [o.fitness for o in observations]
        gen = Generation(index=self._current_gen_index, population=population, fitness=fitness)
        self.generations.append(gen)
        self._current_gen_index += 1

        ranked = sorted(zip(population, fitness, strict=True), key=lambda pf: pf[1], reverse=True)
        n_elite = max(1, int(len(ranked) * self.elite_fraction))
        elite = [p for p, _ in ranked[:n_elite]]

        next_pop: list[dict[str, Any]] = list(elite)  # elitism
        while len(next_pop) < self.population_size:
            parent_a = self._tournament_select(ranked)
            parent_b = self._tournament_select(ranked)
            child = (
                self.space.crossover(parent_a, parent_b, self._rng)
                if self._rng.random() < self.crossover_rate
                else dict(parent_a)
            )
            if self._rng.random() < self.mutation_rate:
                child = self.space.perturb(child, self._rng)
            next_pop.append(child)

        self._pending = next_pop

    def _tournament_select(
        self, ranked: list[tuple[dict[str, Any], float]], k: int = 3
    ) -> dict[str, Any]:
        contenders = self._rng.sample(ranked, k=min(k, len(ranked)))
        return max(contenders, key=lambda pf: pf[1])[0]

    def convergence(self) -> ConvergenceState:
        history = [g.best()[1] for g in self.generations]
        converged = detect_plateau(history, patience=3, min_delta=0.005)
        return ConvergenceState(
            converged=converged,
            reason="fitness plateaued across generations" if converged else "",
            best_fitness_history=history,
        )
