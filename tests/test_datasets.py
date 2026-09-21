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


def test_failure_driven_generation_returns_exactly_n_of_the_requested_categories(forge_support_domain):
    """It over-filtered: asking for 5 of one category returned 1, and when nothing matched it silently
    returned untargeted challenges."""
    from neuroforge.datasets.evolution import failure_driven_challenges

    out = failure_driven_challenges(forge_support_domain, ["refund_request"], 5, seed=3)
    assert len(out) == 5 and {c.category for c in out} == {"refund_request"}
    out = failure_driven_challenges(forge_support_domain, ["escalation_case", "duplicate_charge"], 9, seed=3)
    assert len(out) == 9 and {c.category for c in out} <= {"escalation_case", "duplicate_charge"}
    assert len({c.challenge_id for c in out}) == 9
    assert all(c.difficulty >= 0.5 for c in out)
    with pytest.raises(ValueError, match="categories"):
        failure_driven_challenges(forge_support_domain, ["no_such_category"], 3, seed=1)


def test_failure_driven_evolution_adds_a_version_and_keeps_existing_splits(forge_support_domain):
    from neuroforge.datasets.evolution import evolve_from_failures, seed_dataset

    v1 = seed_dataset(forge_support_domain, "d", n=60, seed=1)
    v2 = evolve_from_failures(forge_support_domain, v1, ["refund_request"], 12, seed=5)
    assert v2.version == 2 and v2.source == "failure_driven" and len(v2.challenges) == 72
    old_splits = {c.challenge_id: c.split for c in v1.challenges}
    assert all(old_splits[c.challenge_id] == c.split for c in v2.challenges if c.challenge_id in old_splits)
    new = [c for c in v2.challenges if c.challenge_id not in old_splits]
    assert len(new) == 12 and {c.category for c in new} == {"refund_request"}
    v3 = evolve_from_failures(forge_support_domain, v2, ["refund_request"], 12, seed=5)  # same seed again
    assert len({c.challenge_id for c in v3.challenges}) == 84
