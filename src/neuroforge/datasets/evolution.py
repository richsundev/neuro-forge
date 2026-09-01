"""Benchmark evolution: the dataset itself gets harder as candidates saturate it (section 14/63).

Without this, an optimizer can "win" simply by overfitting a fixed, eventually-too-easy benchmark.
`evolve_if_saturated` inspects the mean score candidates are achieving on the *search* split only
(never holdout — see holdout.py) and, if it crosses a saturation threshold, generates a new dataset
version at a higher difficulty band, carrying dataset lineage forward.
"""

from __future__ import annotations

from neuroforge.datasets.dataset import DatasetVersion
from neuroforge.datasets.holdout import deterministic_split
from neuroforge.domains.base import ApplicationDomain

SATURATION_THRESHOLD = 0.90
DIFFICULTY_STEP = 0.18


def is_saturated(mean_search_score: float, threshold: float = SATURATION_THRESHOLD) -> bool:
    return mean_search_score >= threshold


def evolve_if_saturated(
    domain: ApplicationDomain,
    current: DatasetVersion,
    mean_search_score: float,
    n_new_challenges: int,
    seed: int,
) -> DatasetVersion | None:
    """Return a new, harder DatasetVersion if the current one is saturated, else None."""
    if not is_saturated(mean_search_score):
        return None

    lo = min(0.95, current.difficulty_score + DIFFICULTY_STEP * 0.5)
    hi = min(1.0, current.difficulty_score + DIFFICULTY_STEP)
    new_challenges = domain.generate_cases(n_new_challenges, (lo, hi), seed)
    valid = [c for c in new_challenges if domain.validate(c)]
    combined = current.challenges + valid
    combined = deterministic_split(combined, seed)

    difficulty_score = sum(c.difficulty for c in combined) / max(1, len(combined))
    return DatasetVersion(
        dataset_id=current.dataset_id,
        version=current.version + 1,
        parent_version=current.version,
        source="benchmark_evolution",
        challenges=combined,
        difficulty_score=round(difficulty_score, 4),
        validated=True,
    )


def seed_dataset(
    domain: ApplicationDomain, dataset_id: str, n: int, seed: int, difficulty: tuple[float, float] = (0.15, 0.45)
) -> DatasetVersion:
    challenges = domain.generate_cases(n, difficulty, seed)
    valid = [c for c in challenges if domain.validate(c)]
    split = deterministic_split(valid, seed)
    difficulty_score = sum(c.difficulty for c in split) / max(1, len(split))
    return DatasetVersion(
        dataset_id=dataset_id,
        version=1,
        parent_version=None,
        source="seed",
        challenges=split,
        difficulty_score=round(difficulty_score, 4),
        validated=True,
    )


def failure_driven_challenges(
    domain: ApplicationDomain,
    failing_categories: list[str],
    n: int,
    seed: int,
) -> list:
    """Generate targeted new challenges around categories where candidates are failing (section
    12). Falls back to the domain's full category mix if no categories are specified."""
    all_new = domain.generate_cases(n * max(1, len(failing_categories) or 1), (0.5, 0.95), seed)
    if not failing_categories:
        return all_new[:n]
    targeted = [c for c in all_new if c.category in failing_categories]
    return targeted[:n] or all_new[:n]
