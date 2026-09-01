from neuroforge.evaluation.aggregate import aggregate
from neuroforge.evaluation.statistics import compare
from neuroforge.promotion.canary import simulate_canary
from neuroforge.promotion.gates import PromotionGateConfig, evaluate_promotion
from neuroforge.promotion.safety import SafetyConstraints, check_safety
from neuroforge.providers.mock import MockLLMProvider


def test_check_safety_flags_low_score():
    ok, violations = check_safety({"safety_score": 0.5, "policy_compliance": 0.95}, SafetyConstraints())
    assert not ok
    assert violations


def test_check_safety_passes_good_metrics():
    ok, violations = check_safety({"safety_score": 0.95, "policy_compliance": 0.95}, SafetyConstraints())
    assert ok
    assert not violations


def test_promotion_rejects_on_cost_regression(forge_support_domain, baseline_genome):
    provider = MockLLMProvider()
    challenges = forge_support_domain.generate_cases(20, (0.2, 0.4), seed=1)
    baseline_results = [forge_support_domain.evaluate(baseline_genome, c, provider) for c in challenges]
    baseline_agg = aggregate(baseline_results, [c.category for c in challenges])

    # A candidate that is expensive/slow despite similar quality should be rejected by the guard.
    expensive = baseline_genome.derive(
        mutations=[], overrides={"model.name": "reasoning-large", "retrieval.top_k": 10}, new_version=2
    )
    candidate_results = [forge_support_domain.evaluate(expensive, c, provider) for c in challenges]
    candidate_agg = aggregate(candidate_results, [c.category for c in challenges])

    comparison = compare(
        [r.metrics["quality"] for r in baseline_results],
        [r.metrics["quality"] for r in candidate_results],
        seed=0,
    )
    decision = evaluate_promotion(
        baseline_agg, candidate_agg, comparison, PromotionGateConfig(), SafetyConstraints()
    )
    if candidate_agg.cost_usd > baseline_agg.cost_usd * 1.10:
        assert not decision.approved


def test_canary_detects_and_flags_regression(forge_support_domain, baseline_genome):
    provider = MockLLMProvider()
    challenges = forge_support_domain.generate_cases(60, (0.4, 0.9), seed=2)
    unsafe_candidate = baseline_genome.derive(
        mutations=[],
        overrides={"tools.enabled": ["search", "refund"], "tools.selection_policy": "greedy"},
        new_version=2,
    )
    result = simulate_canary(
        forge_support_domain,
        provider,
        baseline_genome,
        unsafe_candidate,
        challenges,
        candidate_traffic_fraction=0.3,
    )
    assert result.rollback_triggered


def test_canary_passes_for_safe_improvement(forge_support_domain, baseline_genome):
    provider = MockLLMProvider()
    challenges = forge_support_domain.generate_cases(60, (0.1, 0.3), seed=2)
    good_candidate = baseline_genome.derive(
        mutations=[],
        overrides={
            "tools.enabled": ["search", "refund"],
            "tools.selection_policy": "risk_aware",
            "prompt.strategy": "constrained",
        },
        new_version=2,
    )
    result = simulate_canary(
        forge_support_domain,
        provider,
        baseline_genome,
        good_candidate,
        challenges,
        candidate_traffic_fraction=0.3,
    )
    assert result.candidate_metrics.metrics["quality"] >= result.baseline_metrics.metrics["quality"] - 0.05
