"""Benchmark evolution: the dataset itself gets harder as candidates saturate it (section 14/63).

Without this, an optimizer can "win" simply by overfitting a fixed, eventually-too-easy benchmark.
`evolve_if_saturated` inspects the mean score candidates are achieving on the *search* split only
(never holdout — see holdout.py) and, if it crosses a saturation threshold, generates a new dataset
version at a higher difficulty band, carrying dataset lineage forward.
"""

from __future__ import annotations

from neuroforge.datasets.dataset import DatasetVersion
from neuroforge.datasets.holdout import deterministic_split
from neuroforge.domains.base import ApplicationDomain, Challenge

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
    return _extend(
        current,
        [c for c in domain.generate_cases(n_new_challenges, (lo, hi), seed) if domain.validate(c)],
        seed,
        "benchmark_evolution",
    )


def _extend(
    current: DatasetVersion, new_challenges: list[Challenge], seed: int, source: str
) -> DatasetVersion:
    """The next version of `current` with `new_challenges` added."""
    new_version = current.version + 1

    # Generated ids come from the seed, so evolving twice with the same seed (the API's default)
    # regenerates ids that already exist: duplicated challenges, silently double-counted.
    taken = {c.challenge_id for c in current.challenges}
    unique = []
    for c in new_challenges:
        cid = c.challenge_id
        if cid in taken:
            cid = f"{cid}-v{new_version}"
        taken.add(cid)
        unique.append(c.model_copy(update={"challenge_id": cid}))

    # Only the *new* challenges get a split assignment. Existing challenges keep theirs: re-running
    # `deterministic_split` over everything with a different seed moved ~45% of v1's challenges
    # between train/validation/holdout, so a challenge that had been a candidate's held-out
    # evidence could become search data in the next version.
    combined = current.challenges + deterministic_split(unique, seed)

    difficulty_score = sum(c.difficulty for c in combined) / max(1, len(combined))
    return DatasetVersion(
        dataset_id=current.dataset_id,
        version=new_version,
        parent_version=current.version,
        source=source,
        challenges=combined,
        difficulty_score=round(difficulty_score, 4),
        validated=True,
    )


def seed_dataset(
    domain: ApplicationDomain, dataset_id: str, n: int, seed: int, difficulty: tuple[float, float] = (0.15, 0.45)
) -> DatasetVersion:
    challenges = domain.generate_cases(n, difficulty, seed)
    valid = [c for c in challenges if domain.validate(c)]
    if not valid:
        raise ValueError(f"no valid challenges could be generated for '{dataset_id}' (n={n})")
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
) -> list[Challenge]:
    """Generate exactly `n` new challenges in the categories where candidates are failing (section
    12), at the hard end of the difficulty range. With no categories, the domain's normal mix.

    Generated cases cycle through *all* of a domain's categories, so asking for `n` and filtering
    kept only ~n/8 of them — `n=5` for one category returned one challenge, and when nothing matched
    the old code silently returned untargeted challenges instead. This over-generates (varying the
    seed so ids stay distinct) until it has `n` of the requested categories, and fails loudly if a
    category doesn't exist in the domain."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not failing_categories:
        return domain.generate_cases(n, (0.5, 0.95), seed)
    wanted = set(failing_categories)
    found: list[Challenge] = []
    for round_ in range(50):
        batch = domain.generate_cases(max(n * 8, 16), (0.5, 0.95), seed + round_)
        found.extend(c for c in batch if c.category in wanted and domain.validate(c))
        if len(found) >= n:
            return found[:n]
    known = sorted({c.category for c in domain.generate_cases(64, (0.5, 0.95), seed)})
    raise ValueError(f"could not generate challenges for categories {sorted(wanted)}; the domain has {known}")


def evolve_from_failures(
    domain: ApplicationDomain,
    current: DatasetVersion,
    failing_categories: list[str],
    n_new: int,
    seed: int,
) -> DatasetVersion:
    """A new dataset version with extra challenges concentrated on the categories a candidate is
    failing (`source="failure_driven"`), turning an observed failure mode into evaluation material
    without waiting for the whole benchmark to saturate."""
    if not failing_categories:
        raise ValueError("failing_categories must name at least one category")
    return _extend(current, failure_driven_challenges(domain, failing_categories, n_new, seed), seed, "failure_driven")
