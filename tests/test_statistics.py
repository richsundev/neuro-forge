import pytest

from neuroforge.evaluation.statistics import Conclusion, compare


def test_clear_improvement_is_detected():
    baseline = [0.5, 0.5, 0.5, 0.5, 0.5] * 10
    candidate = [0.65, 0.65, 0.65, 0.65, 0.65] * 10
    result = compare(baseline, candidate, min_relative_improvement=0.02, seed=0)
    assert result.conclusion == Conclusion.LIKELY_IMPROVEMENT
    assert result.ci_low > 0


def test_clear_regression_is_detected():
    baseline = [0.7] * 40
    candidate = [0.5] * 40
    result = compare(baseline, candidate, min_relative_improvement=0.02, seed=0)
    assert result.conclusion == Conclusion.LIKELY_REGRESSION
    assert result.ci_high < 0


def test_noisy_tiny_difference_is_inconclusive():
    import random

    rng = random.Random(0)
    baseline = [0.5 + rng.uniform(-0.2, 0.2) for _ in range(20)]
    candidate = [b + rng.uniform(-0.01, 0.01) for b in baseline]
    result = compare(baseline, candidate, min_relative_improvement=0.05, seed=0)
    assert result.conclusion == Conclusion.INCONCLUSIVE


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compare([0.1, 0.2], [0.1], seed=0)


def test_too_few_samples_raise():
    with pytest.raises(ValueError):
        compare([0.1], [0.2], seed=0)


def test_never_forces_a_winner_from_a_single_point_style_gap():
    """A tiny, noisy gap (like 0.91 vs 0.92 from the brief) must not be called a winner."""
    import random

    rng = random.Random(1)
    baseline = [0.91 + rng.uniform(-0.05, 0.05) for _ in range(15)]
    candidate = [0.92 + rng.uniform(-0.05, 0.05) for _ in range(15)]
    result = compare(baseline, candidate, min_relative_improvement=0.02, seed=0)
    assert result.conclusion == Conclusion.INCONCLUSIVE
