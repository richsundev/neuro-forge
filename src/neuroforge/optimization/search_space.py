"""Search space definitions (section 36): the parameter ranges an optimizer is allowed to explore.

Independent of `MutationPolicy` (which bounds genome-lineage mutations), a `SearchSpace` describes
a classic hyperparameter-optimization view over a fixed set of named fields — used by the
random/grid/evolutionary/Bayesian/bandit strategies so they can be compared on equal footing
(section 42) without needing a parent genome to mutate from.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ParamSpec(BaseModel):
    type: Literal["float", "integer", "categorical"]
    min: float | None = None
    max: float | None = None
    values: list[Any] | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> ParamSpec:
        if self.type in ("float", "integer"):
            if self.min is None or self.max is None:
                raise ValueError(f"{self.type} parameter requires min and max")
            if self.min >= self.max:
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        elif self.type == "categorical":
            if not self.values:
                raise ValueError("categorical parameter requires a non-empty `values` list")
        return self

    def bounds(self) -> tuple[float, float]:
        assert self.min is not None and self.max is not None
        return self.min, self.max

    def categories(self) -> list[Any]:
        assert self.values is not None
        return self.values

    def sample(self, rng: random.Random) -> Any:
        if self.type == "float":
            lo, hi = self.bounds()
            return round(rng.uniform(lo, hi), 4)
        if self.type == "integer":
            lo, hi = self.bounds()
            return rng.randint(int(lo), int(hi))
        return rng.choice(self.categories())

    def grid_values(self, steps: int = 4) -> list[Any]:
        if self.type == "categorical":
            return list(self.categories())
        lo, hi = self.bounds()
        if self.type == "integer":
            span = int(hi) - int(lo)
            n = min(steps, span + 1)
            return sorted({int(lo + round(i * span / max(1, n - 1))) for i in range(n)})
        if steps < 2:
            return [round((lo + hi) / 2, 4)]
        return [round(lo + i * (hi - lo) / (steps - 1), 4) for i in range(steps)]

    def clamp(self, value: Any) -> Any:
        if self.type == "categorical":
            categories = self.categories()
            return value if value in categories else categories[0]
        lo, hi = self.bounds()
        v = max(lo, min(hi, value))
        return int(round(v)) if self.type == "integer" else round(v, 4)


class SearchSpace(BaseModel):
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {name: spec.sample(rng) for name, spec in self.parameters.items()}

    def perturb(self, point: dict[str, Any], rng: random.Random, strength: float = 0.25) -> dict[str, Any]:
        """Local move used by the evolutionary strategy's mutation step."""
        new_point = dict(point)
        name = rng.choice(list(self.parameters))
        spec = self.parameters[name]
        if spec.type == "categorical":
            new_point[name] = rng.choice(spec.categories())
        else:
            lo, hi = spec.bounds()
            delta = rng.uniform(-strength, strength) * (hi - lo)
            new_point[name] = spec.clamp(point[name] + delta)
        return new_point

    def crossover(self, a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        return {name: (a[name] if rng.random() < 0.5 else b[name]) for name in self.parameters}

    def encode(self, point: dict[str, Any]) -> np.ndarray:
        """Numeric encoding used by the Bayesian surrogate: min-max scale numerics, one-hot
        categoricals. Deterministic ordering (sorted parameter names) so vectors are comparable."""
        vec: list[float] = []
        for name in sorted(self.parameters):
            spec = self.parameters[name]
            value = point[name]
            if spec.type == "categorical":
                vec.extend(1.0 if v == value else 0.0 for v in spec.categories())
            else:
                lo, hi = spec.bounds()
                span = (hi - lo) or 1.0
                vec.append((value - lo) / span)
        return np.array(vec, dtype=float)

    def grid_size(self, steps: int = 4) -> int:
        """Number of points in the full grid, without materializing it."""
        return math.prod(len(spec.grid_values(steps)) for spec in self.parameters.values())

    def iter_grid(self, steps: int = 4) -> Iterator[dict[str, Any]]:
        """Lazily walk the full grid. A realistic genome space has tens of millions of grid points
        (ForgeSupport's 14 fields at 4 steps is ~22M), so building the list is an out-of-memory
        crash, not a slow search."""
        names = list(self.parameters)
        value_lists = [self.parameters[n].grid_values(steps) for n in names]
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo, strict=True))

    def grid(self, steps: int = 4) -> list[dict[str, Any]]:
        return list(self.iter_grid(steps))
