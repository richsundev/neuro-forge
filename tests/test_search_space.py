import random

import pytest
from pydantic import ValidationError

from neuroforge.optimization.search_space import ParamSpec, SearchSpace


def test_param_spec_requires_bounds_for_numeric():
    with pytest.raises(ValidationError):
        ParamSpec(type="float")


def test_param_spec_requires_values_for_categorical():
    with pytest.raises(ValidationError):
        ParamSpec(type="categorical")


def test_param_spec_min_must_be_less_than_max():
    with pytest.raises(ValidationError):
        ParamSpec(type="integer", min=5, max=5)


def test_sample_within_bounds():
    space = SearchSpace(
        parameters={
            "x": ParamSpec(type="float", min=0.0, max=1.0),
            "y": ParamSpec(type="integer", min=1, max=10),
            "z": ParamSpec(type="categorical", values=["a", "b", "c"]),
        }
    )
    rng = random.Random(0)
    for _ in range(50):
        point = space.sample(rng)
        assert 0.0 <= point["x"] <= 1.0
        assert 1 <= point["y"] <= 10
        assert point["z"] in ("a", "b", "c")


def test_grid_is_bounded_and_covers_categoricals():
    space = SearchSpace(
        parameters={
            "x": ParamSpec(type="float", min=0.0, max=1.0),
            "z": ParamSpec(type="categorical", values=["a", "b"]),
        }
    )
    grid = space.grid(steps=3)
    assert len(grid) == 3 * 2
    assert {p["z"] for p in grid} == {"a", "b"}


def test_encode_is_consistent_length():
    space = SearchSpace(
        parameters={
            "x": ParamSpec(type="float", min=0.0, max=1.0),
            "z": ParamSpec(type="categorical", values=["a", "b", "c"]),
        }
    )
    rng = random.Random(1)
    p1 = space.sample(rng)
    p2 = space.sample(rng)
    assert space.encode(p1).shape == space.encode(p2).shape
