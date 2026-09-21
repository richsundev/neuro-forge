"""Turning user-facing experiment choices into engine configuration.

An experiment is defined by three things the user should be able to steer without editing code:
*which knobs the search may turn* (search dimensions), *what "better" means* (objective weights),
and *what a candidate must satisfy to ship* (gates and safety limits, which `ExperimentConfig` already
carries). The API, the CLI and the dashboard all go through here so they validate identically.
"""

from __future__ import annotations

from typing import Any

from neuroforge.evaluation.objectives import DEFAULT_OBJECTIVES, Objective, ObjectiveSpec
from neuroforge.experiments.spaces import search_space_for_domain
from neuroforge.optimization.search_space import SearchSpace


class ConfigError(ValueError):
    """The requested experiment configuration isn't valid."""


def search_groups(space: SearchSpace) -> dict[str, list[str]]:
    """The space's field paths grouped by their genome section (`model`, `prompt`, `retrieval`, ...)."""
    groups: dict[str, list[str]] = {}
    for path in space.parameters:
        groups.setdefault(path.split(".")[0], []).append(path)
    return groups


def build_search_space(domain_name: str, dimensions: list[str] | None) -> SearchSpace:
    """The domain's default search space, optionally restricted to `dimensions`: a genome section
    name (`"prompt"`) or a full field path (`"retrieval.top_k"`). Fields left out are not searched, so
    every candidate keeps the baseline's value for them — restricting to `["prompt"]` asks "what is the
    best prompt configuration for this system as it stands?"."""
    space = search_space_for_domain(domain_name)
    if not dimensions:
        return space
    groups = search_groups(space)
    chosen: set[str] = set()
    unknown: list[str] = []
    for item in dimensions:
        if item in groups:
            chosen.update(groups[item])
        elif item in space.parameters:
            chosen.add(item)
        else:
            unknown.append(item)
    if unknown:
        raise ConfigError(
            f"unknown search dimension(s) {sorted(unknown)} for '{domain_name}'; "
            f"sections: {sorted(groups)}; fields: {sorted(space.parameters)}"
        )
    return SearchSpace(parameters={p: spec for p, spec in space.parameters.items() if p in chosen})


def build_objectives(weights: dict[str, float] | None) -> ObjectiveSpec:
    """The default objectives with the listed metrics pinned to `weights` (as final shares of the
    total) and every other metric sharing what is left in proportion to its default weight. So
    `{"cost_usd": 0.4}` means "cost is 40% of what I care about" and the rest keep their relative
    balance; if the listed weights already sum to 1 or more they are rescaled to 1 and the rest drop to
    0. Directions are fixed by the metric (quality is maximized, cost and latency minimized); only the
    emphasis is configurable."""
    if not weights:
        return DEFAULT_OBJECTIVES
    defaults = DEFAULT_OBJECTIVES.objectives
    unknown = sorted(set(weights) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown objective(s) {unknown}; known: {sorted(defaults)}")
    if any(w < 0 for w in weights.values()):
        raise ConfigError("objective weights must be >= 0")
    pinned_total = sum(weights.values())
    if pinned_total <= 0:
        raise ConfigError("at least one objective weight must be positive")
    if pinned_total >= 1.0:
        final = {name: (weights[name] / pinned_total if name in weights else 0.0) for name in defaults}
    else:
        others = {name: obj.weight for name, obj in defaults.items() if name not in weights}
        others_total = sum(others.values())
        room = 1.0 - pinned_total
        final = {
            name: weights[name] if name in weights else (others[name] / others_total * room if others_total else 0.0)
            for name in defaults
        }
    # Rounding can leave the total a hair off 1.0; put the remainder on the largest weight the user did
    # not pin, so the shares they asked for stay exact.
    final = {name: round(w, 6) for name, w in final.items()}
    drift = round(1.0 - sum(final.values()), 6)
    adjustable = [n for n in final if n not in weights] or list(final)
    final[max(adjustable, key=lambda n: final[n])] += drift
    return ObjectiveSpec(
        objectives={name: Objective(direction=defaults[name].direction, weight=w) for name, w in final.items()}
    )


def parse_weight_options(items: list[str] | None) -> dict[str, float] | None:
    """`["quality=0.4", "cost_usd=0.3"]` (the CLI's `--weight` option) -> `{"quality": 0.4, ...}`."""
    if not items:
        return None
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"weight '{item}' must look like metric=value")
        try:
            out[name.strip()] = float(value)
        except ValueError as exc:
            raise ConfigError(f"weight '{item}' has a non-numeric value") from exc
    return out


def describe_domain(domain_name: str) -> dict[str, Any]:
    """What a client needs to build an experiment form for `domain_name`."""
    from neuroforge.promotion.gates import PromotionGateConfig
    from neuroforge.promotion.safety import SafetyConstraints

    space = search_space_for_domain(domain_name)
    return {
        "name": domain_name,
        "search_dimensions": {
            group: [{"path": path, **space.parameters[path].model_dump(exclude_none=True)} for path in paths]
            for group, paths in search_groups(space).items()
        },
        "objectives": {
            name: {"direction": obj.direction, "weight": obj.weight}
            for name, obj in DEFAULT_OBJECTIVES.objectives.items()
        },
        "promotion_gates": PromotionGateConfig().model_dump(),
        "safety_constraints": SafetyConstraints().model_dump(),
    }
