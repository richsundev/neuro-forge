import pytest

from neuroforge.datasets.evolution import evolve_if_saturated, seed_dataset
from neuroforge.datasets.holdout import HoldoutGuard, HoldoutViolation, deterministic_split


def test_split_ratios_are_roughly_correct(forge_support_domain):
    dataset = seed_dataset(forge_support_domain, "ds", n=300, seed=1)
    summary = dataset.summary()
    total = summary["n_challenges"]
    splits = summary["splits"]
    assert splits["train"] / total == pytest.approx(0.70, abs=0.06)
    assert splits["validation"] / total == pytest.approx(0.15, abs=0.06)
    assert splits["holdout"] / total == pytest.approx(0.15, abs=0.06)


def test_split_is_deterministic(forge_support_domain):
    challenges = forge_support_domain.generate_cases(50, (0.1, 0.5), seed=3)
    a = deterministic_split(challenges, seed=3)
    b = deterministic_split(challenges, seed=3)
    assert [c.split for c in a] == [c.split for c in b]


def test_holdout_guard_blocks_search_and_validation_access(small_dataset):
    guard = HoldoutGuard(small_dataset.challenges)
    search = guard.search_set()
    validation = guard.validation_set()
    assert all(c.split == "train" for c in search)
    assert all(c.split == "validation" for c in validation)
    holdout_ids = {c.challenge_id for c in small_dataset.challenges if c.split == "holdout"}
    assert not (holdout_ids & {c.challenge_id for c in search})
    assert not (holdout_ids & {c.challenge_id for c in validation})


def test_holdout_can_only_be_evaluated_once_per_genome(small_dataset):
    guard = HoldoutGuard(small_dataset.challenges)
    guard.evaluate_holdout("genome-hash-1")
    with pytest.raises(HoldoutViolation):
        guard.evaluate_holdout("genome-hash-1")
    # a different genome hash is fine
    guard.evaluate_holdout("genome-hash-2")


def test_benchmark_evolves_when_saturated(forge_support_domain, small_dataset):
    harder = evolve_if_saturated(
        forge_support_domain, small_dataset, mean_search_score=0.95, n_new_challenges=20, seed=5
    )
    assert harder is not None
    assert harder.version == small_dataset.version + 1
    assert harder.parent_version == small_dataset.version
    assert harder.difficulty_score > small_dataset.difficulty_score
    assert len(harder.challenges) > len(small_dataset.challenges)


def test_benchmark_does_not_evolve_when_not_saturated(forge_support_domain, small_dataset):
    result = evolve_if_saturated(
        forge_support_domain, small_dataset, mean_search_score=0.5, n_new_challenges=20, seed=5
    )
    assert result is None
