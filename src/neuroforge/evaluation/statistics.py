"""Statistical comparison between a baseline and a candidate (section 16).

We refuse to declare a winner off a single point estimate. `compare` runs a paired bootstrap over
per-challenge score pairs and only calls an improvement "likely" when the confidence interval for
the mean difference clears both a minimum-effect-size bar and a confidence threshold; otherwise it
reports INCONCLUSIVE rather than forcing a winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class Conclusion(str, Enum):
    LIKELY_IMPROVEMENT = "LIKELY_IMPROVEMENT"
    LIKELY_REGRESSION = "LIKELY_REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ComparisonResult:
    mean_diff: float
    relative_diff: float
    ci_low: float
    ci_high: float
    effect_size: float
    confidence: float
    conclusion: Conclusion

    def summary(self) -> str:
        pct = self.relative_diff * 100
        return (
            f"{'+' if pct >= 0 else ''}{pct:.1f}% "
            f"(95% CI [{self.ci_low * 100:+.1f}%, {self.ci_high * 100:+.1f}%]) -> "
            f"{self.conclusion.value}"
        )


def bootstrap_ci(
    diffs: np.ndarray, n_resamples: int = 5000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_mean_bound(
    values: list[float],
    *,
    side: Literal["upper", "lower"],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided percentile-bootstrap bound on the mean: with `confidence` probability the true
    mean is <= the "upper" bound (or >= the "lower" bound). This is what turns a safety *point
    estimate* on a few dozen challenges into a claim you can actually defend."""
    if not values:
        raise ValueError("cannot bound the mean of zero values")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    q = confidence if side == "upper" else 1.0 - confidence
    return float(np.quantile(means, q))


def compare(
    baseline_scores: list[float],
    candidate_scores: list[float],
    *,
    min_relative_improvement: float = 0.02,
    confidence: float = 0.95,
    seed: int = 0,
) -> ComparisonResult:
    """Paired comparison: baseline_scores[i] and candidate_scores[i] must be scores on the *same*
    challenge i, so the bootstrap is over paired differences (removes per-challenge variance)."""
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("baseline and candidate score lists must be the same length (paired)")
    if len(baseline_scores) < 2:
        raise ValueError("need at least 2 paired samples for a statistical comparison")

    b = np.asarray(baseline_scores, dtype=float)
    c = np.asarray(candidate_scores, dtype=float)
    diffs = c - b
    mean_diff = float(diffs.mean())
    baseline_mean = float(b.mean()) or 1e-9
    relative_diff = mean_diff / abs(baseline_mean)

    pooled_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    effect_size = mean_diff / pooled_std if pooled_std > 1e-9 else 0.0

    ci_low, ci_high = bootstrap_ci(diffs, confidence=confidence, seed=seed)
    ci_low_rel, ci_high_rel = ci_low / abs(baseline_mean), ci_high / abs(baseline_mean)

    meaningfully_positive = ci_low_rel > 0 and relative_diff >= min_relative_improvement
    meaningfully_negative = ci_high_rel < 0 and relative_diff <= -min_relative_improvement

    if meaningfully_positive:
        conclusion = Conclusion.LIKELY_IMPROVEMENT
    elif meaningfully_negative:
        conclusion = Conclusion.LIKELY_REGRESSION
    else:
        conclusion = Conclusion.INCONCLUSIVE

    return ComparisonResult(
        mean_diff=round(mean_diff, 6),
        relative_diff=round(relative_diff, 6),
        ci_low=round(ci_low_rel, 6),
        ci_high=round(ci_high_rel, 6),
        effect_size=round(effect_size, 4),
        confidence=confidence,
        conclusion=conclusion,
    )
