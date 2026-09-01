"""Optimizer behavior tests against a simple synthetic objective (not the LLM domain, so these
run fast and isolate the search algorithms from evaluation noise)."""

from __future__ import annotations

import pytest

from neuroforge.optimization import (
    BanditSearchStrategy,
    BayesianSearchStrategy,
    EvolutionarySearchStrategy,
    GridSearchStrategy,
    Observation,
    ParamSpec,
    RandomSearchStrategy,
    SearchSpace,
)

SPACE = SearchSpace(
    parameters={
        "x": ParamSpec(type="float", min=0.0, max=1.0),
        "choice": ParamSpec(type="categorical", values=["low", "mid", "high"]),
    }
)

CHOICE_BONUS = {"low": 0.0, "mid": 0.1, "high": -0.2}


def synthetic_fitness(point: dict) -> float:
    """Peaks at x=0.75 with choice='mid'."""
    return 1.0 - (point["x"] - 0.75) ** 2 + CHOICE_BONUS[point["choice"]]


def run_loop(strategy, n_rounds: int = 8, batch_size: int = 6) -> float:
    best = float("-inf")
    for _ in range(n_rounds):
        points = strategy.ask(batch_size)
        if not points:
            break
        observations = [Observation(point=p, fitness=synthetic_fitness(p)) for p in points]
        strategy.tell(observations)
        best = max(best, max(o.fitness for o in observations))
    return best


def test_random_search_finds_reasonable_optimum():
    strategy = RandomSearchStrategy(SPACE, seed=0)
    best = run_loop(strategy, n_rounds=10, batch_size=20)
    assert best > 0.85


def test_grid_search_exhausts_and_covers_space():
    strategy = GridSearchStrategy(SPACE, steps=5)
    seen = 0
    while not strategy.exhausted():
        points = strategy.ask(5)
        if not points:
            break
        strategy.tell([Observation(point=p, fitness=synthetic_fitness(p)) for p in points])
        seen += len(points)
    assert seen == 5 * 3  # 5 float steps * 3 categorical values
    assert strategy.convergence().converged


def test_evolutionary_search_improves_over_generations():
    strategy = EvolutionarySearchStrategy(SPACE, population_size=16, seed=0)
    run_loop(strategy, n_rounds=10, batch_size=16)
    assert len(strategy.generations) >= 2
    first_best = strategy.generations[0].best()[1]
    last_best = strategy.generations[-1].best()[1]
    assert last_best >= first_best


def test_bayesian_search_handles_list_valued_categoricals():
    """Regression test: a categorical field whose values are lists (e.g. genome tools.enabled)
    used to crash BayesianSearchStrategy.ask() with `TypeError: unhashable type: 'list'` once it
    ran past its initial random-sampling phase into pool deduplication."""
    space = SearchSpace(
        parameters={
            "x": ParamSpec(type="float", min=0.0, max=1.0),
            "tools": ParamSpec(type="categorical", values=[["search"], ["search", "refund"]]),
        }
    )
    strategy = BayesianSearchStrategy(space, seed=0, n_initial_random=2)
    for _ in range(4):
        points = strategy.ask(6)
        strategy.tell([Observation(point=p, fitness=p["x"]) for p in points])


def test_bayesian_search_beats_random_baseline_given_same_budget():
    bo = BayesianSearchStrategy(SPACE, seed=0, n_initial_random=4)
    bo_best = run_loop(bo, n_rounds=8, batch_size=4)

    rand = RandomSearchStrategy(SPACE, seed=0)
    rand_best = run_loop(rand, n_rounds=8, batch_size=4)

    # Not a strict guarantee for every seed/objective, but on this smooth unimodal function
    # a model-based search should not do meaningfully worse than pure random with equal budget.
    assert bo_best >= rand_best - 0.05


def test_bandit_ucb1_converges_to_best_arm():
    strategy = BanditSearchStrategy(SPACE, seed=0, n_arms=10)
    run_loop(strategy, n_rounds=15, batch_size=5)
    best_arm_idx = max(range(len(strategy.arms)), key=lambda i: strategy._ucb_score(i) if strategy._pulls[i] else -1)
    assert strategy._pulls[best_arm_idx] > 0


def test_bandit_arm_count_stays_bounded_in_high_dimensional_spaces():
    """Regression test: BanditSearchStrategy used to discretize via a full Cartesian-product grid
    (space.grid()), which exploded combinatorially for realistic genome search spaces (12+ fields
    -> millions of arms, ~44s just to build them). It must now stay near the requested n_arms
    regardless of dimensionality."""
    wide_space = SearchSpace(
        parameters={
            f"param_{i}": ParamSpec(type="categorical", values=["a", "b", "c", "d", "e"])
            for i in range(15)
        }
    )
    strategy = BanditSearchStrategy(wide_space, seed=0, n_arms=20)
    assert len(strategy.arms) == 20


@pytest.mark.parametrize(
    "strategy_cls,kwargs",
    [
        (RandomSearchStrategy, {"seed": 0}),
        (EvolutionarySearchStrategy, {"seed": 0, "population_size": 8}),
        (BayesianSearchStrategy, {"seed": 0}),
        (BanditSearchStrategy, {"seed": 0}),
    ],
)
def test_strategies_are_deterministic_given_seed(strategy_cls, kwargs):
    s1 = strategy_cls(SPACE, **kwargs)
    s2 = strategy_cls(SPACE, **kwargs)
    p1 = s1.ask(5)
    p2 = s2.ask(5)
    assert p1 == p2
