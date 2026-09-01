from neuroforge.optimization.bandit import BanditSearchStrategy
from neuroforge.optimization.bayesian import BayesianSearchStrategy
from neuroforge.optimization.evolutionary import EvolutionarySearchStrategy, Generation
from neuroforge.optimization.grid_search import GridSearchStrategy
from neuroforge.optimization.random_search import RandomSearchStrategy
from neuroforge.optimization.search_space import ParamSpec, SearchSpace
from neuroforge.optimization.strategy import ConvergenceState, Observation, SearchStrategy

STRATEGY_REGISTRY = {
    "random": RandomSearchStrategy,
    "grid": GridSearchStrategy,
    "evolutionary": EvolutionarySearchStrategy,
    "bayesian": BayesianSearchStrategy,
    "bandit": BanditSearchStrategy,
}


def get_strategy(
    name: str, space: SearchSpace, seed: int = 0, batch_size: int | None = None
) -> SearchStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown search strategy '{name}'. Known: {list(STRATEGY_REGISTRY)}")
    if name == "grid":
        return cls(space)
    if name == "evolutionary" and batch_size:
        return cls(space, seed=seed, population_size=batch_size)
    return cls(space, seed=seed)


__all__ = [
    "STRATEGY_REGISTRY",
    "BanditSearchStrategy",
    "BayesianSearchStrategy",
    "ConvergenceState",
    "EvolutionarySearchStrategy",
    "Generation",
    "GridSearchStrategy",
    "Observation",
    "ParamSpec",
    "RandomSearchStrategy",
    "SearchSpace",
    "SearchStrategy",
    "get_strategy",
]
