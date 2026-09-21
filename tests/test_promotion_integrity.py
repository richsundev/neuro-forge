"""Regression tests for the promotion pipeline's statistical integrity: constraint-aware search,
confidence-bound safety checks, holdout verification, and the gates actually seeing cost/latency.

Each of these guards a specific failure found while validating the pipeline end to end:

- the default safety limit was below the reachable floor of the search space, so every "pass" was
  small-sample noise (`test_default_gates_are_reachable_in_forge_support`);
- the API/CLI rebuilt aggregates with cost/latency zeroed, so those gates never fired
  (`test_cost_and_latency_gates_see_the_holdout_numbers`);
- the holdout split was never evaluated anywhere (`test_holdout_*`);
- the canary replayed train/validation data (`test_canary_traffic_is_fresh`).
"""

from __future__ import annotations

import random

import pytest

from neuroforge.datasets.evolution import seed_dataset
from neuroforge.datasets.holdout import HoldoutGuard
from neuroforge.evaluation.aggregate import aggregate
from neuroforge.evaluation.statistics import bootstrap_mean_bound
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
from neuroforge.experiments.spaces import forge_support_search_space
from neuroforge.genomes.schema import SystemGenome
from neuroforge.promotion.canary import fresh_traffic
from neuroforge.promotion.gates import PromotionGateConfig, metric_gate_findings
from neuroforge.promotion.holdout import HoldoutEvidence, decide_from_evidence, evaluate_on_holdout
from neuroforge.promotion.safety import (
    SafetyConstraints,
    check_safety,
    estimate_safety,
    violation_magnitude,
)
from neuroforge.providers.mock import MockLLMProvider


def test_bootstrap_bounds_bracket_the_mean():
    values = [0.1, 0.2, 0.3, 0.25, 0.15] * 10
    mean = sum(values) / len(values)
    assert bootstrap_mean_bound(values, side="lower", seed=1) < mean
    assert bootstrap_mean_bound(values, side="upper", seed=1) > mean


def test_safety_check_uses_the_upper_bound_not_the_point_estimate():
    # Mean violation rate 0.25 passes a 0.28 limit as a point estimate, but on only 12 noisy
    # challenges the 95% upper bound is well above it — which is exactly the borderline candidate
    # a point-estimate check waves through by luck.
    compliance = [0.6, 0.9] * 6
    ok_point, _ = check_safety(
        {"policy_compliance": sum(compliance) / len(compliance), "safety_score": 0.95},
        SafetyConstraints(),
    )
    assert ok_point
    estimate = estimate_safety({"policy_compliance": compliance, "safety_score": [0.95] * 12}, seed=0)
    assert estimate.policy_violation_upper > SafetyConstraints().max_policy_violation_rate
    ok_bound, violations = estimate.check(SafetyConstraints())
    assert not ok_bound
    assert "upper 95% bound" in violations[0]


def test_safety_estimate_skips_policy_rate_for_domains_without_the_metric():
    estimate = estimate_safety({"safety_score": [0.97] * 30}, seed=0)
    assert not estimate.has_policy_metric
    assert estimate.check(SafetyConstraints())[0]


def test_violation_magnitude_is_zero_inside_and_grows_outside():
    limits = SafetyConstraints()
    assert violation_magnitude({"policy_compliance": 0.8, "safety_score": 0.95}, limits) == 0.0
    mild = violation_magnitude({"policy_compliance": 0.65, "safety_score": 0.95}, limits)
    severe = violation_magnitude({"policy_compliance": 0.4, "safety_score": 0.95}, limits)
    assert 0.0 < mild < severe


def test_default_gates_are_reachable_in_forge_support(forge_support_domain, baseline_genome):
    """Calibration guard: at least one configuration in the search space must clear *every* default
    gate on a full population, or every promotion verdict against those gates is noise. The
    original 0.20 policy-violation limit failed this — no configuration in ForgeSupport's space
    scores below ~0.216."""
    provider = MockLLMProvider()
    dataset = seed_dataset(forge_support_domain, "calibration", n=100, seed=1)
    challenges = dataset.challenges
    categories = [c.category for c in challenges]

    def population_metrics(genome):
        return aggregate([forge_support_domain.evaluate(genome, c, provider) for c in challenges], categories)

    baseline_agg = population_metrics(baseline_genome)
    space = forge_support_search_space()
    rng = random.Random(0)
    gates, limits = PromotionGateConfig(), SafetyConstraints()

    assert metric_gate_findings(baseline_agg, baseline_agg, gates, limits), "baseline must fail the gates"

    feasible = 0
    for i in range(250):
        genome = baseline_genome.derive(mutations=[], overrides=space.sample(rng), new_version=2 + i)
        if not metric_gate_findings(baseline_agg, population_metrics(genome), gates, limits):
            feasible += 1
    assert feasible > 0


def test_cost_and_latency_gates_see_the_holdout_numbers():
    """Regression: aggregates rebuilt from stored metric dicts used to have cost/latency zeroed, so
    a candidate 3x as expensive as baseline sailed through the cost gate."""
    evidence = HoldoutEvidence(
        dataset_id="d",
        dataset_version=1,
        n_holdout=50,
        baseline_metrics={"quality": 0.5, "cost_usd": 0.001, "latency_ms": 700.0, "failure_rate": 0.5},
        candidate_metrics={
            "quality": 0.75,
            "policy_compliance": 0.85,
            "safety_score": 0.95,
            "cost_usd": 0.003,
            "latency_ms": 1500.0,
            "failure_rate": 0.1,
        },
        comparison={
            "mean_diff": 0.25,
            "relative_diff": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
            "effect_size": 2.0,
            "confidence": 0.95,
            "conclusion": "LIKELY_IMPROVEMENT",
        },
        safety=estimate_safety({"policy_compliance": [0.85] * 50, "safety_score": [0.95] * 50}),
    )
    decision = decide_from_evidence(evidence, "abc", PromotionGateConfig(), SafetyConstraints())
    assert not decision.approved
    assert any("cost increased" in r for r in decision.reasons)
    assert any("latency increased" in r for r in decision.reasons)
    assert decision.evidence[0].startswith("holdout split of d v1")


def test_holdout_evaluation_matches_the_holdout_split_only(forge_support_domain, baseline_genome):
    dataset = seed_dataset(forge_support_domain, "holdout-check", n=100, seed=3)
    candidate = baseline_genome.derive(
        mutations=[], overrides={"tools.selection_policy": "risk_aware"}, new_version=2
    )
    evidence = evaluate_on_holdout(
        forge_support_domain, MockLLMProvider(), baseline_genome, candidate, dataset, seed=1
    )
    assert evidence.n_holdout == len(dataset.split("holdout"))
    assert evidence.dataset_version == dataset.version
    assert evidence.safety.n == evidence.n_holdout


def test_holdout_needs_enough_challenges(forge_support_domain, baseline_genome):
    dataset = seed_dataset(forge_support_domain, "tiny", n=12, seed=1)
    candidate = baseline_genome.derive(mutations=[], overrides={"model.name": "fast-cheap"}, new_version=2)
    with pytest.raises(ValueError, match="holdout"):
        evaluate_on_holdout(forge_support_domain, MockLLMProvider(), baseline_genome, candidate, dataset)


def test_canary_traffic_is_fresh(forge_support_domain):
    dataset = seed_dataset(forge_support_domain, "traffic-check", n=100, seed=1)
    traffic = fresh_traffic(forge_support_domain, dataset, 200, seed=99)
    dataset_ids = {c.challenge_id for c in dataset.challenges}
    assert len(traffic) == 200
    assert not dataset_ids & {c.challenge_id for c in traffic}
    lo = min(c.difficulty for c in dataset.challenges)
    hi = max(c.difficulty for c in dataset.challenges)
    assert all(lo <= c.difficulty <= hi for c in traffic)
    assert [c.input for c in traffic] == [
        c.input for c in fresh_traffic(forge_support_domain, dataset, 200, seed=99)
    ]


def _config(experiment_id: str, max_candidates: int, **overrides) -> ExperimentConfig:
    fields = {
        "experiment_id": experiment_id,
        "domain_name": "forge-support",
        "search_strategy": "evolutionary",
        "search_space": forge_support_search_space(),
        "batch_size": 8,
        "max_batches": 100,
        "seed": 5,
        "budget": ExperimentBudget(
            max_candidates=max_candidates, max_requests=1_000_000, max_cost_usd=1000, max_duration_minutes=30
        ),
    }
    return ExperimentConfig(**{**fields, **overrides})


def test_search_never_reads_the_holdout(baseline_genome, small_dataset, tmp_path, monkeypatch):
    def forbidden(self, genome_hash):
        raise AssertionError("the experiment engine must never evaluate the holdout split")

    monkeypatch.setattr(HoldoutGuard, "evaluate_holdout", forbidden)
    ExperimentEngine(_config("exp-no-holdout", 16), baseline_genome, small_dataset, tmp_path).run()


def test_unsatisfiable_constraints_yield_no_feasible_selection(baseline_genome, small_dataset, tmp_path):
    cfg = _config(
        "exp-infeasible",
        max_candidates=48,
        safety_constraints=SafetyConstraints(max_policy_violation_rate=0.0),
    )
    result = ExperimentEngine(cfg, baseline_genome, small_dataset, tmp_path).run()
    assert not result.selected_feasible
    assert result.recommendation.startswith("DO NOT PROMOTE")
    # While nothing is feasible a flat fitness curve is "still stuck", not "converged": the search
    # must run its full budget rather than stopping on a plateau.
    assert result.candidates_evaluated == 48


def test_search_finds_a_feasible_winner_and_it_verifies_on_holdout(forge_support_domain, baseline_genome, tmp_path):
    dataset = seed_dataset(forge_support_domain, "feasible-search", n=300, seed=1)
    cfg = _config("exp-feasible", max_candidates=240, batch_size=24)
    result = ExperimentEngine(cfg, baseline_genome, dataset, tmp_path).run()
    assert result.selected_feasible
    assert result.dataset_id == "feasible-search" and result.dataset_version == 1

    best = SystemGenome.model_validate(result.best_genome)
    evidence = evaluate_on_holdout(
        forge_support_domain, MockLLMProvider(), baseline_genome, best, dataset, seed=cfg.seed
    )
    decision = decide_from_evidence(evidence, best.hash(), cfg.promotion_gates, cfg.safety_constraints)
    assert decision.approved, decision.reasons
