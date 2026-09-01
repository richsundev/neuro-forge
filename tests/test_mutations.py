import pytest

from neuroforge.mutations.engine import MutationEngine
from neuroforge.mutations.policy import MutationPolicy


def test_policy_rejects_forbidden_field():
    policy = MutationPolicy()
    ok, reason = policy.validate_change_set(["security_policy"])
    assert not ok
    assert "not permitted" in reason


def test_policy_rejects_too_many_changes():
    policy = MutationPolicy(max_changes_per_candidate=2)
    ok, reason = policy.validate_change_set(
        ["prompt.strategy", "model.temperature", "retrieval.top_k"]
    )
    assert not ok
    assert "at most 2" in reason


def test_policy_allows_within_bounds():
    policy = MutationPolicy()
    ok, _ = policy.validate_change_set(["prompt.strategy", "model.temperature"])
    assert ok


def test_engine_respects_max_changes(baseline_genome):
    policy = MutationPolicy(max_changes_per_candidate=1)
    engine = MutationEngine(policy, seed=0)
    candidates = engine.propose(baseline_genome, n_candidates=5, next_version_start=2)
    for c in candidates:
        assert len(c.mutations) <= 1


def test_engine_produces_valid_lineage(baseline_genome):
    policy = MutationPolicy()
    engine = MutationEngine(policy, seed=1)
    candidates = engine.propose(baseline_genome, n_candidates=6, next_version_start=2)
    assert len(candidates) > 0
    for c in candidates:
        assert c.parent_hash == baseline_genome.hash()
        assert c.hash() != baseline_genome.hash()
        for m in c.mutations:
            assert policy.is_allowed(m.field_path)


def test_engine_is_deterministic_given_seed(baseline_genome):
    policy = MutationPolicy()
    e1 = MutationEngine(policy, seed=42)
    e2 = MutationEngine(policy, seed=42)
    c1 = e1.propose(baseline_genome, n_candidates=4, next_version_start=2)
    c2 = e2.propose(baseline_genome, n_candidates=4, next_version_start=2)
    assert [c.hash() for c in c1] == [c.hash() for c in c2]


def test_rationale_is_human_readable(baseline_genome):
    policy = MutationPolicy()
    engine = MutationEngine(policy, seed=2)
    candidates = engine.propose(baseline_genome, n_candidates=1, next_version_start=2)
    assert candidates
    text = engine.rationale(candidates[0])
    assert "Mutation rationale" in text
    assert "->" in text


@pytest.mark.parametrize("forbidden_combo", [("model.temperature", "output.format")])
def test_forbidden_combination_rejected(forbidden_combo):
    policy = MutationPolicy()
    ok, reason = policy.validate_change_set(list(forbidden_combo))
    assert not ok
    assert "forbidden" in reason
